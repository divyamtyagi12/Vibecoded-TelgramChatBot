"""Telegram adapter with reply-chain reconstruction, quote understanding, and general vision."""

from __future__ import annotations

from collections import OrderedDict
import json
import logging
import re
from typing import Any

from telegram import BotCommand, Message, Update
from telegram.constants import ChatMemberStatus, ChatType
from telegram.error import TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from telegram_ai.context import assemble_context, find_image_in_conversation
from telegram_ai.domain import (
    ConversationMessage,
    GroupSettings,
    ParticipationMode,
    is_explain_context_intent,
    is_photo_rating_intent,
    is_retry_intent,
    should_reply,
)
from telegram_ai.openrouter import OpenRouterClient, QuotaExhaustedError
from telegram_ai.storage import PostgresRepository, SQLiteRepository
from telegram_ai.vision import MAX_IMAGE_BYTES, download_telegram_image

logger = logging.getLogger(__name__)
MAX_PHOTO_BYTES = MAX_IMAGE_BYTES  # 10 MB limit

_GENERIC_IMAGE_QUERY_RE = re.compile(
    r"^(?:please\s+)?(?:"
    r"explain(?:\s+(?:this|image|photo|pic|screenshot|scene|post|discussion|conversation))?|"
    r"what\s+is\s+this\??|"
    r"what(?:'s|\s+is)\s+happening\??|"
    r"what\s+does\s+this\s+mean\??|"
    r"context\??|"
    r"details\??|"
    r"tell\s+me\s+about\s+this\??|"
    r"what\??|"
    r"look\s+at\s+this\??|"
    r"describe(?:\s+this)?|"
    r"meaning\??"
    r")\.?$",
    re.IGNORECASE,
)


_IMAGE_FOLLOWUP_QUERY_RE = re.compile(
    r"\b("
    r"this|it|image|photo|pic|screenshot|poster|meme|picture|scene|character|person|"
    r"explain|describe|details?|more|elaborate|summarize|context|meaning|translate|read|transcribe|"
    r"fix|solve|debug|error|bug|code|line|rate|score"
    r")\b",
    re.IGNORECASE,
)

_UNRELATED_TOPIC_SWITCH_RE = re.compile(
    r"\b("
    r"next\s+(?:test\s+)?match|test\s+match|live\s+score|scorecard|cricket|ipl|world\s+cup|ind\s+vs|vs\s+nz|ind\s+nz|"
    r"train\s+\d+|pnr|running\s+status|weather\s+in|current\s+time|latest\s+news|stock\s+price|"
    r"other\s+questions?|different\s+question|unrelated|change\s+topic"
    r")\b",
    re.IGNORECASE,
)


class TelegramAIHandlers:
    def __init__(
        self,
        repository: SQLiteRepository | PostgresRepository,
        ai: OpenRouterClient,
        max_context: int,
        max_tokens: int = 3000,
    ) -> None:
        self.repository = repository
        self.ai = ai
        self.max_context = max_context
        self.max_tokens = max_tokens
        self._bot_user = None
        self._processed_messages: OrderedDict[tuple[int, int], None] = OrderedDict()

    def _claim_message(self, chat_id: int, message_id: int | None) -> bool:
        """Return True if this (chat_id, message_id) has not been processed yet; prevents double replies."""
        if not isinstance(message_id, int):
            return True
        key = (chat_id, message_id)
        if key in self._processed_messages:
            logger.info("Skipping duplicate message delivery: chat=%s msg=%s", chat_id, message_id)
            return False
        self._processed_messages[key] = None
        if len(self._processed_messages) > 500:
            self._processed_messages.popitem(last=False)
        return True

    async def configure(self, application: Application) -> None:
        await application.bot.set_my_commands(
            [
                BotCommand("help", "Show how to use TelegramAI"),
                BotCommand("info", "Show this group's AI settings"),
                BotCommand("feedback", "Rate the bot: /feedback 1-5 comment"),
                BotCommand("settings", "Admin: show configuration"),
                BotCommand("style", "Admin: set answer style"),
                BotCommand("mode", "Admin: set participation mode"),
                BotCommand("memory", "Admin: enable/disable context"),
            ]
        )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if not message:
            return
        await message.reply_text(
            "Mention me or reply to one of my messages to ask a question.\n\n"
            "Send or reply to a photo/screenshot with a question (code analysis, math, memes, anime, general questions) "
            "or ask for a visual rating.\n\n"
            "Everyone: /help, /info, /feedback 1-5 optional comment\n"
            "Admins: /settings, /style <instruction>, /mode mention|reply|always|off, /memory on|off"
        )

    async def info(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        message = update.effective_message
        if not chat or not message:
            return
        settings = self.repository.get_settings(chat.id)
        await message.reply_text(self._settings_summary(settings))

    async def settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._is_admin(update):
            return
        await self.info(update, context)

    async def style(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._is_admin(update):
            return
        chat = update.effective_chat
        message = update.effective_message
        if not chat or not message:
            return
        style = " ".join(context.args or []).strip()
        if not style:
            await message.reply_text("Usage: /style Be concise and explain technical terms.")
            return
        if len(style) > 500:
            await message.reply_text("Keep the style instruction to 500 characters or fewer.")
            return
        current = self.repository.get_settings(chat.id)
        self.repository.save_settings(
            GroupSettings(current.chat_id, style, current.participation_mode, current.memory_enabled)
        )
        await message.reply_text("Response style saved.")

    async def mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._is_admin(update):
            return
        chat = update.effective_chat
        message = update.effective_message
        if not chat or not message:
            return
        requested = " ".join(context.args or []).lower().strip()
        try:
            mode = ParticipationMode(requested)
        except ValueError:
            await message.reply_text("Usage: /mode mention|reply|always|off")
            return
        current = self.repository.get_settings(chat.id)
        self.repository.save_settings(
            GroupSettings(current.chat_id, current.response_style, mode, current.memory_enabled)
        )
        await message.reply_text(f"Participation mode is now: {mode.value}.")

    async def memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._is_admin(update):
            return
        chat = update.effective_chat
        message = update.effective_message
        if not chat or not message:
            return
        requested = " ".join(context.args or []).lower().strip()
        if requested not in {"on", "off"}:
            await message.reply_text("Usage: /memory on|off")
            return
        current = self.repository.get_settings(chat.id)
        self.repository.save_settings(
            GroupSettings(current.chat_id, current.response_style, current.participation_mode, requested == "on")
        )
        status = "enabled" if requested == "on" else "disabled"
        await message.reply_text(
            f"Short conversation context is {status}. Existing local context is not sent while disabled."
        )

    async def feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        user = update.effective_user
        message = update.effective_message
        if not chat or not user or not message:
            return
        args = context.args or []
        if not args or not args[0].isdigit() or not 1 <= int(args[0]) <= 5:
            await message.reply_text("Usage: /feedback 1-5 optional comment")
            return
        rating = int(args[0])
        comment = " ".join(args[1:]).strip() or "No comment"
        self.repository.add_feedback(
            chat.id, user.id, rating, comment
        )
        await message.reply_text("Thanks—your feedback has been recorded.")

    # ------------------------------------------------------------------ #
    # Message Handlers                                                   #
    # ------------------------------------------------------------------ #

    async def respond(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        chat = update.effective_chat
        if not message or not chat or not message.text or message.from_user is None or message.from_user.is_bot:
            return

        chat_id = chat.id
        settings = self.repository.get_settings(chat_id)
        bot = self._bot_user or await context.bot.get_me()
        self._bot_user = bot

        # Index replied-to message if it exists
        self._record_parent_message_if_missing(chat_id, message.reply_to_message, bot)

        # Extract quote if present
        quote_obj = getattr(message, "quote", None)
        quote_text = quote_obj.text if quote_obj and isinstance(getattr(quote_obj, "text", None), str) else None

        is_mentioned = self._is_mentioned(message.text, message.entities, bot.username)
        is_reply_to_bot = bool(
            message.reply_to_message
            and message.reply_to_message.from_user
            and message.reply_to_message.from_user.id == bot.id
        )

        logger.info(
            "chat=%s mention=%s reply=%s mode=%s",
            chat_id,
            is_mentioned,
            is_reply_to_bot,
            settings.participation_mode,
        )

        # Extract user prompt clean of mention
        prompt = self._request_text(message.text, bot.username)

        # If user quoted text, format prompt to explicitly include it
        if quote_text:
            prompt_with_quote = f'[Quoting: "{quote_text}"]\n{prompt}' if prompt else f'[Quoting: "{quote_text}"]'
        else:
            prompt_with_quote = prompt

        msg_id = message.message_id if isinstance(getattr(message, "message_id", None), int) else None
        s_id = message.from_user.id if message.from_user and isinstance(getattr(message.from_user, "id", None), int) else None
        r_id = message.reply_to_message.message_id if message.reply_to_message and isinstance(getattr(message.reply_to_message, "message_id", None), int) else None
        t_id = message.message_thread_id if isinstance(getattr(message, "message_thread_id", None), int) else None

        current_msg = ConversationMessage(
            author=message.from_user.full_name,
            text=prompt_with_quote or message.text,
            message_id=msg_id,
            sender_id=s_id,
            reply_to_message_id=r_id,
            thread_id=t_id,
            is_bot=False,
            media_type="text",
            quote_text=quote_text,
        )

        if not should_reply(settings, is_mentioned=is_mentioned, is_reply_to_bot=is_reply_to_bot):
            # Still record message for future reply chain context if memory enabled
            if settings.memory_enabled:
                self.repository.add_context(chat_id, current_msg)
            return

        if not prompt_with_quote:
            await message.reply_text("What would you like help with?")
            return

        if not self._claim_message(chat_id, msg_id):
            return

        # -------------------------------------------------------------- #
        # Check for Contextual Retry ("Try", "Try again", "Retry")       #
        # -------------------------------------------------------------- #
        if is_retry_intent(prompt):
            retried = await self._handle_retry_request(
                message=message,
                context=context,
                chat_id=chat_id,
                r_id=r_id,
                settings=settings,
                current_msg=current_msg,
                bot=bot,
            )
            if retried:
                return

        # -------------------------------------------------------------- #
        # Discover referenced image (direct reply or in ancestor chain)   #
        # -------------------------------------------------------------- #
        replied_image_file_id, image_media_type, image_source_msg = find_image_in_conversation(
            message, chat_id, self.repository
        )
        if replied_image_file_id:
            logger.info(
                "Image detected: chat_id=%s, msg_id=%s, file_id=%s, media_type=%s",
                chat_id,
                msg_id,
                replied_image_file_id,
                image_media_type,
            )

        # -------------------------------------------------------------- #
        # Check for Context Explanation ("Explain context")              #
        # -------------------------------------------------------------- #
        if is_explain_context_intent(prompt) and (r_id or replied_image_file_id):
            prompt_with_quote = (
                "[INSTRUCTION: The user is asking you to explain and summarize the context of the ongoing conversation, "
                "including any image and messages referenced so far. Do NOT give a dictionary definition of the word 'context'. "
                "Provide a helpful, direct explanation of what is being discussed in this conversation.] "
                "Explain the context of our discussion so far."
            )

        # Build conversation context
        history = assemble_context(
            self.repository,
            chat_id,
            current_msg,
            memory_enabled=settings.memory_enabled,
            max_messages=self.max_context,
            max_tokens=self.max_tokens,
        )

        await context.bot.send_chat_action(chat_id, "typing")

        reply: str
        failed_op_data: str | None = None
        image_summary: str | None = None

        # Guard: Check if the user's request requires an image that could not be found/accessed
        r_photo = getattr(message.reply_to_message, "photo", None) if message.reply_to_message else None
        r_doc = getattr(message.reply_to_message, "document", None) if message.reply_to_message else None
        is_reply_to_image = bool(
            (isinstance(r_photo, (list, tuple)) and len(r_photo) > 0)
            or (
                r_doc is not None
                and isinstance(getattr(r_doc, "mime_type", None), str)
                # pyrefly: ignore[missing-attribute]
                and r_doc.mime_type.startswith("image/")
            )
        )
        stored_parent = self.repository.get_message(chat_id, r_id) if r_id else None
        is_stored_parent_image = bool(
            stored_parent
            and not stored_parent.is_bot
            and stored_parent.media_type in ("photo", "document_image")
        )
        prompt_requests_image = bool(
            re.search(r"\b(image|photo|pic|screenshot|meme|picture|poster)\b", prompt, re.IGNORECASE)
        )

        # Determine whether to treat as an image-grounded response:
        # - Always treat as image if directly replying to an actual image message or explicitly mentioning the image.
        # - If the image was only inherited from an ancestor in the reply chain (e.g. user replied to a bot text message),
        #   only treat as an image query if the user is still asking a follow-up about the image and NOT switching to a new topic.
        if replied_image_file_id:
            if is_reply_to_image or is_stored_parent_image or prompt_requests_image:
                should_treat_as_image = True
            elif _UNRELATED_TOPIC_SWITCH_RE.search(prompt):
                should_treat_as_image = False
                replied_image_file_id = None
            elif _GENERIC_IMAGE_QUERY_RE.match(prompt.strip()) or _IMAGE_FOLLOWUP_QUERY_RE.search(prompt):
                should_treat_as_image = True
            else:
                should_treat_as_image = False
                replied_image_file_id = None
        else:
            should_treat_as_image = False

        if not replied_image_file_id and (is_reply_to_image or is_stored_parent_image or prompt_requests_image):
            logger.warning(
                "Required image context missing: chat_id=%s, msg_id=%s, prompt=%s",
                chat_id,
                msg_id,
                prompt,
            )
            logger.warning("Fallback activated: reason=missing_required_image_context")
            await message.reply_text(
                "I couldn't find or access the referenced image. Please upload or reply directly to the photo."
            )
            return

        if should_treat_as_image and replied_image_file_id:
            # Extract caption/text from the replied-to message so vision always has that context
            replied_caption = ""
            if message.reply_to_message:
                raw_rc = getattr(message.reply_to_message, "caption", None) or getattr(message.reply_to_message, "text", None)
                if isinstance(raw_rc, str) and raw_rc.strip():
                    replied_caption = raw_rc.strip()

            # If prompt is a generic query ("explain", "what is this", "context?"), expand it for vision model
            if _GENERIC_IMAGE_QUERY_RE.match(prompt.strip()):
                base_q = "Explain what is shown and discussed in this image in detail."
                effective_image_prompt = f'[Quoting: "{quote_text}"]\n{base_q}' if quote_text else base_q
            else:
                effective_image_prompt = prompt_with_quote

            if replied_caption and replied_caption not in effective_image_prompt:
                effective_image_prompt = (
                    f'[Replied message caption/text: "{replied_caption}"]\n{effective_image_prompt}'
                )

            reply, failed_op_data, image_summary = await self._handle_image_response(
                context=context,
                chat_id=chat_id,
                file_id=replied_image_file_id,
                prompt=effective_image_prompt,
                settings=settings,
                history=history,
                parent_message_id=r_id,
            )
        else:
            try:
                reply = await self.ai.answer(prompt_with_quote, settings, history)
            except QuotaExhaustedError:
                logger.warning("OpenRouter free quota/rate-limit reached for chat %s", chat_id)
                await message.reply_text(
                    "The free AI quota is temporarily rate-limited or exhausted. Please try again in a minute."
                )
                return
            except RuntimeError:
                logger.exception("AI response failed for chat %s", chat_id)
                await message.reply_text(
                    "I couldn't reach the AI service just now. Please try again shortly."
                )
                return

        # Send reply and track bot response
        sent = await message.reply_text(reply)
        if settings.memory_enabled:
            msg_to_store = current_msg
            if replied_image_file_id:
                msg_to_store = ConversationMessage(
                    author=current_msg.author,
                    text=current_msg.text,
                    message_id=current_msg.message_id,
                    sender_id=current_msg.sender_id,
                    reply_to_message_id=current_msg.reply_to_message_id,
                    thread_id=current_msg.thread_id,
                    is_bot=current_msg.is_bot,
                    media_type=image_media_type or "photo",
                    media_file_id=replied_image_file_id,
                    media_summary=image_summary,
                    quote_text=current_msg.quote_text,
                )

            self.repository.add_context(chat_id, msg_to_store)
            sent_msg_id = sent.message_id if sent and isinstance(getattr(sent, "message_id", None), int) else None
            self.repository.add_context(
                chat_id,
                ConversationMessage(
                    author="TelegramAI",
                    text=reply,
                    message_id=sent_msg_id,
                    sender_id=bot.id if isinstance(getattr(bot, "id", None), int) else None,
                    reply_to_message_id=msg_id,
                    thread_id=t_id,
                    is_bot=True,
                    media_type="photo" if replied_image_file_id else "text",
                    media_file_id=replied_image_file_id,
                    media_summary=image_summary,
                    failed_operation=failed_op_data,
                ),
            )

    async def photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        chat = update.effective_chat
        if not message or not chat or message.from_user is None or message.from_user.is_bot:
            return

        # Check for photo or document image
        photos = getattr(message, "photo", None)
        photo = photos[-1] if isinstance(photos, (list, tuple)) and len(photos) > 0 else None
        raw_doc = getattr(message, "document", None)
        raw_mime = getattr(raw_doc, "mime_type", None) if raw_doc is not None else None
        doc = (
            raw_doc
            if raw_doc is not None and isinstance(raw_mime, str) and raw_mime.startswith("image/")
            else None
        )
        if not photo and not doc:
            return

        chat_id = chat.id
        settings = self.repository.get_settings(chat_id)
        bot = self._bot_user or await context.bot.get_me()
        self._bot_user = bot

        self._record_parent_message_if_missing(chat_id, message.reply_to_message, bot)

        caption = message.caption or ""
        caption_entities = message.caption_entities or []
        is_mentioned = self._is_mentioned(caption, caption_entities, bot.username)
        is_reply_to_bot = bool(
            message.reply_to_message
            and message.reply_to_message.from_user
            and message.reply_to_message.from_user.id == bot.id
        )

        quote_obj = getattr(message, "quote", None)
        quote_text = quote_obj.text if quote_obj and isinstance(getattr(quote_obj, "text", None), str) else None
        file_id = photo.file_id if photo else (doc.file_id if doc else None)
        file_unique_id = photo.file_unique_id if photo else (doc.file_unique_id if doc else None)
        file_size = photo.file_size if photo else (doc.file_size if doc else None)

        msg_id = message.message_id if isinstance(getattr(message, "message_id", None), int) else None
        s_id = message.from_user.id if message.from_user and isinstance(getattr(message.from_user, "id", None), int) else None
        r_id = message.reply_to_message.message_id if message.reply_to_message and isinstance(getattr(message.reply_to_message, "message_id", None), int) else None
        t_id = message.message_thread_id if isinstance(getattr(message, "message_thread_id", None), int) else None

        current_msg = ConversationMessage(
            author=message.from_user.full_name,
            text=caption or "[Photo]",
            message_id=msg_id,
            sender_id=s_id,
            reply_to_message_id=r_id,
            thread_id=t_id,
            is_bot=False,
            media_type="photo" if photo else "document_image",
            media_file_id=file_id,
            media_unique_id=file_unique_id,
            quote_text=quote_text,
        )

        if not should_reply(settings, is_mentioned=is_mentioned, is_reply_to_bot=is_reply_to_bot):
            if settings.memory_enabled:
                self.repository.add_context(chat_id, current_msg)
            return

        if not self._claim_message(chat_id, msg_id):
            return

        prompt = self._request_text(caption, bot.username)
        if quote_text:
            prompt = f'[Quoting: "{quote_text}"]\n{prompt}' if prompt else f'[Quoting: "{quote_text}"]'

        is_rating = is_photo_rating_intent(prompt)

        # File size check before downloading
        if isinstance(file_size, int) and file_size > MAX_PHOTO_BYTES:
            if is_rating:
                await message.reply_text("Please send a photo no larger than 10 MB for a rating.")
            else:
                await message.reply_text("Please send an image no larger than 10 MB.")
            return

        await context.bot.send_chat_action(chat_id, "typing")

        image_data, mime_type, download_err = await download_telegram_image(
            context.bot, file_id or ""
        )

        if download_err or image_data is None:
            if download_err == "file_too_large":
                if is_rating:
                    await message.reply_text("Please send a photo no larger than 10 MB for a rating.")
                else:
                    await message.reply_text("Please send an image no larger than 10 MB.")
            elif download_err == "invalid_format":
                await message.reply_text(
                    "The image format could not be decoded. Please send a standard JPEG, PNG, or WEBP image."
                )
            else:
                await message.reply_text("I couldn't download that photo. Please try again.")
            return

        # Route between photo rating and general vision
        reply: str
        logger.info("Vision model selected: %s", getattr(self.ai, "vision_model", "nvidia/nemotron-3.5-super-vl:free"))
        try:
            if is_rating:
                reply = await self.ai.rate_photo(
                    image_data, mime_type or "image/jpeg", prompt, settings, file_id=file_id
                )
            else:
                history = assemble_context(
                    self.repository,
                    chat_id,
                    current_msg,
                    memory_enabled=settings.memory_enabled,
                    max_messages=self.max_context,
                    max_tokens=self.max_tokens,
                )
                reply = await self.ai.analyze_image(
                    image_data, mime_type or "image/jpeg", prompt, settings, context=history, file_id=file_id
                )
        except QuotaExhaustedError:
            logger.warning("OpenRouter free quota/rate-limit reached during photo processing for chat %s", chat_id)
            await message.reply_text(
                "The free AI quota is temporarily rate-limited or exhausted. Please try again in a minute."
            )
            return
        except RuntimeError:
            logger.exception("Photo processing failed for chat %s", chat_id)
            err_text = (
                "I couldn't rate that photo just now. Please try again shortly."
                if is_rating
                else "I couldn't reach the AI service just now. Please try again shortly."
            )
            sent = await message.reply_text(err_text)
            if settings.memory_enabled:
                failed_op = json.dumps({
                    "action": "rate_photo" if is_rating else "analyze_image",
                    "file_id": file_id,
                    "prompt": prompt,
                })
                self.repository.add_context(chat_id, current_msg)
                sent_msg_id = sent.message_id if sent and isinstance(getattr(sent, "message_id", None), int) else None
                self.repository.add_context(
                    chat_id,
                    ConversationMessage(
                        author="TelegramAI",
                        text=err_text,
                        message_id=sent_msg_id,
                        sender_id=bot.id if isinstance(getattr(bot, "id", None), int) else None,
                        reply_to_message_id=msg_id,
                        thread_id=t_id,
                        is_bot=True,
                        media_type="text",
                        failed_operation=failed_op,
                    ),
                )
            return

        # Generate and cache image description for follow-up conversation memory
        image_summary = (
            reply[:300]
            if reply
            else await self.ai.describe_image(image_data, mime_type or "image/jpeg")
        )
        current_msg = ConversationMessage(
            author=current_msg.author,
            text=current_msg.text,
            message_id=current_msg.message_id,
            sender_id=current_msg.sender_id,
            reply_to_message_id=current_msg.reply_to_message_id,
            thread_id=current_msg.thread_id,
            is_bot=False,
            media_type=current_msg.media_type,
            media_file_id=current_msg.media_file_id,
            media_unique_id=current_msg.media_unique_id,
            media_summary=image_summary,
            quote_text=current_msg.quote_text,
        )

        sent = await message.reply_text(reply)
        if settings.memory_enabled:
            self.repository.add_context(chat_id, current_msg)
            sent_msg_id = sent.message_id if sent and isinstance(getattr(sent, "message_id", None), int) else None
            self.repository.add_context(
                chat_id,
                ConversationMessage(
                    author="TelegramAI",
                    text=reply,
                    message_id=sent_msg_id,
                    sender_id=bot.id if isinstance(getattr(bot, "id", None), int) else None,
                    reply_to_message_id=msg_id,
                    thread_id=t_id,
                    is_bot=True,
                    media_type="text",
                ),
            )

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    async def _handle_image_response(
        self,
        *,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        file_id: str,
        prompt: str,
        settings: GroupSettings,
        history: list[ConversationMessage],
        parent_message_id: int | None,
    ) -> tuple[str, str | None, str | None]:
        """Download image with retries, execute vision model, and update cached summary.
        
        Returns:
            (reply_text, failed_operation_json_or_None, image_summary_or_None)
        """
        is_rating = is_photo_rating_intent(prompt)
        image_data, mime_type, download_err = await download_telegram_image(
            context.bot, file_id
        )

        if download_err or image_data is None:
            logger.warning("Failed to fetch referenced image %s: reason=%s", file_id, download_err)
            # Check if we already have a cached summary in the parent message
            if parent_message_id:
                parent = self.repository.get_message(chat_id, parent_message_id)
                if parent and parent.media_summary:
                    prompt_with_ref = (
                        f"[Referencing earlier image: {parent.media_summary}]\n{prompt}"
                    )
                    try:
                        logger.warning("Fallback activated: reason=using_cached_media_summary")
                        ans = await self.ai.answer(prompt_with_ref, settings, history)
                        return ans, None, parent.media_summary
                    except RuntimeError:
                        pass

            logger.warning("Fallback activated: reason=download_failed_%s", download_err)
            if download_err == "file_too_large":
                msg = (
                    "Please send a photo no larger than 10 MB for a rating."
                    if is_rating
                    else "Please send an image no larger than 10 MB."
                )
                return msg, None, None

            if download_err in ("not_found", "forbidden", "invalid_file_id"):
                return (
                    "The referenced photo has expired or is no longer accessible from Telegram. Please re-upload the image.",
                    None,
                    None,
                )

            return (
                "I couldn't download the referenced image due to a network issue. Please try again.",
                None,
                None,
            )

        logger.info("Vision model selected: %s", getattr(self.ai, "vision_model", "nvidia/nemotron-3.5-super-vl:free"))
        try:
            if is_rating:
                reply = await self.ai.rate_photo(
                    image_data, mime_type or "image/jpeg", prompt, settings, file_id=file_id
                )
            else:
                reply = await self.ai.analyze_image(
                    image_data, mime_type or "image/jpeg", prompt, settings, context=history, file_id=file_id
                )
        except QuotaExhaustedError:
            logger.warning("OpenRouter free quota/rate-limit reached during vision reply for chat %s", chat_id)
            err_msg = "The free AI quota is temporarily rate-limited or exhausted. Please try again in a minute."
            failed_op = json.dumps({
                "action": "rate_photo" if is_rating else "analyze_image",
                "file_id": file_id,
                "prompt": prompt,
            })
            return err_msg, failed_op, None
        except RuntimeError:
            logger.exception("Vision processing failed for replied image in chat %s", chat_id)
            logger.warning("Fallback activated: reason=vision_processing_failed")
            err_msg = (
                "I couldn't rate that photo just now. Please try again shortly."
                if is_rating
                else "I couldn't reach the AI service just now. Please try again shortly."
            )
            failed_op = json.dumps({
                "action": "rate_photo" if is_rating else "analyze_image",
                "file_id": file_id,
                "prompt": prompt,
            })
            return err_msg, failed_op, None

        # Update cached media summary for parent image and return summary
        summary: str | None = None
        if not is_rating:
            summary = reply[:300] if reply else await self.ai.describe_image(image_data, mime_type or "image/jpeg")
            if parent_message_id and summary:
                self.repository.update_media_summary(chat_id, parent_message_id, summary)

        return reply, None, summary

    async def _handle_retry_request(
        self,
        *,
        message: Message,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        r_id: int | None,
        settings: GroupSettings,
        current_msg: ConversationMessage,
        bot: Any,
    ) -> bool:
        """Handle 'Try' / 'Again' follow-ups by attempting the previous failed operation."""
        failed_op_raw = None
        target_msg_id = None

        # Check directly replied-to message for failed_operation
        if r_id is not None:
            stored = self.repository.get_message(chat_id, r_id)
            if stored and stored.failed_operation:
                failed_op_raw = stored.failed_operation
                target_msg_id = stored.message_id
            else:
                # Walk up reply chain to look for failed_operation
                chain = self.repository.get_reply_chain(chat_id, r_id, max_depth=5)
                for ancestor in reversed(chain):
                    if ancestor.failed_operation:
                        failed_op_raw = ancestor.failed_operation
                        target_msg_id = ancestor.message_id
                        break

        # If no explicit failed operation found, check if an image is in the reply chain
        if not failed_op_raw:
            image_fid, _, _ = find_image_in_conversation(message, chat_id, self.repository)
            if image_fid:
                failed_op_raw = json.dumps({
                    "action": "analyze_image",
                    "file_id": image_fid,
                    "prompt": "Analyze and explain this image.",
                })

        if not failed_op_raw:
            return False

        try:
            op_data = json.loads(failed_op_raw)
        except Exception:
            return False

        file_id = op_data.get("file_id")
        orig_prompt = op_data.get("prompt") or "Explain this image"
        action = op_data.get("action") or "analyze_image"

        if not file_id:
            return False

        await context.bot.send_chat_action(chat_id, "typing")

        image_data, mime_type, download_err = await download_telegram_image(
            context.bot, file_id
        )

        if download_err or image_data is None:
            if download_err in ("not_found", "forbidden", "invalid_file_id"):
                await message.reply_text(
                    "I tried to re-process the image, but the Telegram file has expired or is no longer accessible. Please re-upload the photo."
                )
            else:
                await message.reply_text(
                    "I couldn't download the referenced photo due to a network issue. Please try again."
                )
            return True

        history = assemble_context(
            self.repository,
            chat_id,
            current_msg,
            memory_enabled=settings.memory_enabled,
            max_messages=self.max_context,
            max_tokens=self.max_tokens,
        )

        logger.info("Vision model selected: %s", getattr(self.ai, "vision_model", "nvidia/nemotron-3.5-super-vl:free"))
        try:
            if action == "rate_photo":
                reply = await self.ai.rate_photo(
                    image_data, mime_type or "image/jpeg", orig_prompt, settings, file_id=file_id
                )
            else:
                reply = await self.ai.analyze_image(
                    image_data, mime_type or "image/jpeg", orig_prompt, settings, context=history, file_id=file_id
                )
        except RuntimeError:
            await message.reply_text("I couldn't reach the AI service just now. Please try again shortly.")
            return True

        sent = await message.reply_text(reply)
        if settings.memory_enabled:
            # Clear failed operation on previous message and record new response
            if target_msg_id:
                self.repository.update_failed_operation(chat_id, target_msg_id, None)

            self.repository.add_context(chat_id, current_msg)
            sent_msg_id = sent.message_id if sent and isinstance(getattr(sent, "message_id", None), int) else None
            self.repository.add_context(
                chat_id,
                ConversationMessage(
                    author="TelegramAI",
                    text=reply,
                    message_id=sent_msg_id,
                    sender_id=bot.id if isinstance(getattr(bot, "id", None), int) else None,
                    reply_to_message_id=current_msg.message_id,
                    thread_id=current_msg.thread_id,
                    is_bot=True,
                    media_type="text",
                ),
            )
        return True

    def _record_parent_message_if_missing(
        self, chat_id: int, parent_msg: Message | None, bot: Any
    ) -> None:
        """Ensure the parent message exists in SQLite even if privacy mode masked its initial delivery."""
        if not parent_msg:
            return
        p_msg_id = parent_msg.message_id if isinstance(getattr(parent_msg, "message_id", None), int) else None
        if p_msg_id is None:
            return

        existing = self.repository.get_message(chat_id, p_msg_id)
        if existing is not None:
            return

        is_bot = bool(parent_msg.from_user and getattr(parent_msg.from_user, "id", None) == getattr(bot, "id", None))
        author = parent_msg.from_user.full_name if parent_msg.from_user and getattr(parent_msg.from_user, "full_name", None) else ("TelegramAI" if is_bot else "User")
        sender_id = parent_msg.from_user.id if parent_msg.from_user and isinstance(getattr(parent_msg.from_user, "id", None), int) else None
        text = parent_msg.text or parent_msg.caption or ""

        photo = parent_msg.photo[-1] if getattr(parent_msg, "photo", None) else None
        raw_p_doc = getattr(parent_msg, "document", None)
        raw_p_mime = getattr(raw_p_doc, "mime_type", None) if raw_p_doc is not None else None
        doc = (
            raw_p_doc
            if raw_p_doc is not None and isinstance(raw_p_mime, str) and raw_p_mime.startswith("image/")
            else None
        )
        media_type = "photo" if photo else ("document_image" if doc else "text")
        file_id = photo.file_id if photo and isinstance(getattr(photo, "file_id", None), str) else (doc.file_id if doc and isinstance(getattr(doc, "file_id", None), str) else None)
        file_unique_id = photo.file_unique_id if photo and isinstance(getattr(photo, "file_unique_id", None), str) else (doc.file_unique_id if doc and isinstance(getattr(doc, "file_unique_id", None), str) else None)

        p_r_id = parent_msg.reply_to_message.message_id if parent_msg.reply_to_message and isinstance(getattr(parent_msg.reply_to_message, "message_id", None), int) else None
        p_t_id = parent_msg.message_thread_id if isinstance(getattr(parent_msg, "message_thread_id", None), int) else None

        self.repository.add_context(
            chat_id,
            ConversationMessage(
                author=author,
                text=text,
                message_id=p_msg_id,
                sender_id=sender_id,
                reply_to_message_id=p_r_id,
                thread_id=p_t_id,
                is_bot=is_bot,
                media_type=media_type,
                media_file_id=file_id,
                media_unique_id=file_unique_id,
            ),
        )

    @staticmethod
    def _extract_image_file_id(msg: Message | None) -> str | None:
        if not msg:
            return None
        photos = getattr(msg, "photo", None)
        if isinstance(photos, (list, tuple)) and len(photos) > 0:
            last = photos[-1]
            fid = getattr(last, "file_id", None)
            if isinstance(fid, str):
                return fid
        doc = getattr(msg, "document", None)
        # pyrefly: ignore[missing-attribute]
        if doc and isinstance(getattr(doc, "mime_type", None), str) and doc.mime_type.startswith("image/"):
            fid = getattr(doc, "file_id", None)
            if isinstance(fid, str):
                return fid
        return None

    @staticmethod
    def _request_text(text: str, bot_username: str | None = None) -> str:
        """Remove the Telegram routing mention before sending the request to the AI."""
        if not text:
            return ""
        normalized = text.strip()
        if not bot_username:
            return normalized
        clean_user = bot_username.lstrip("@")
        mention = re.escape(clean_user)
        without_mentions = re.sub(
            rf"(?<!\w)@{mention}\b", "", normalized, flags=re.IGNORECASE
        )
        cleaned = re.sub(r"\s+", " ", without_mentions)
        return re.sub(r"\s+([,.:;!?])", r"\1", cleaned).strip(" ,:-\n\t")

    @staticmethod
    def _is_mentioned(text: str, entities, bot_username: str | None) -> bool:
        if not bot_username or not text:
            return False
        mention = f"@{bot_username.lower()}"
        return any(
            entity.type == "mention"
            and text[entity.offset : entity.offset + entity.length].lower() == mention
            for entity in (entities or [])
        ) or mention in text.lower()

    async def _is_admin(self, update: Update) -> bool:
        chat = update.effective_chat
        user = update.effective_user
        if not chat or not user:
            return False
        if chat.type == ChatType.PRIVATE:
            return True
        member = await chat.get_member(user.id)
        if member.status in {ChatMemberStatus.OWNER, ChatMemberStatus.ADMINISTRATOR}:
            return True
        if update.effective_message:
            await update.effective_message.reply_text("Only group admins can change bot settings.")
        return False

    @staticmethod
    def _settings_summary(settings: GroupSettings) -> str:
        return (
            "TelegramAI group settings\n"
            f"• Participation: {settings.participation_mode.value}\n"
            f"• Short-term context: {'on' if settings.memory_enabled else 'off'}\n"
            f"• Response style: {settings.response_style}"
        )


def register_handlers(application: Application, handlers: TelegramAIHandlers) -> None:
    application.add_handler(CommandHandler("help", handlers.help))
    application.add_handler(CommandHandler("start", handlers.help))
    application.add_handler(CommandHandler("info", handlers.info))
    application.add_handler(CommandHandler("settings", handlers.settings))
    application.add_handler(CommandHandler("style", handlers.style))
    application.add_handler(CommandHandler("mode", handlers.mode))
    application.add_handler(CommandHandler("memory", handlers.memory))
    application.add_handler(CommandHandler("feedback", handlers.feedback))
    application.add_handler(
        MessageHandler(filters.PHOTO | filters.Document.IMAGE, handlers.photo)
    )
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.respond))

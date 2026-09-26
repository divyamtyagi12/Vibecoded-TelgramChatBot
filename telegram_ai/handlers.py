"""Telegram adapter and group-administration commands."""

from __future__ import annotations

import logging
import re
from io import BytesIO

from telegram import BotCommand, Update
from telegram.constants import ChatMemberStatus
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from telegram_ai.domain import ConversationMessage, GroupSettings, ParticipationMode, should_reply
from telegram_ai.gemini import GeminiClient
from telegram_ai.storage import SQLiteRepository

logger = logging.getLogger(__name__)


class TelegramAIHandlers:
    def __init__(
        self,
        repository: SQLiteRepository,
        ai: GeminiClient,
        max_context: int,
    ) -> None:
        self.repository = repository
        self.ai = ai
        self.max_context = max_context
        self._bot_user = None

    async def configure(self, application: Application) -> None:
        await application.bot.set_my_commands(
            [
                BotCommand("help", "Show how to use TelegramAI"),
                BotCommand("image", "Generate an image: /image prompt"),
                BotCommand("info", "Show this group's AI settings"),
                BotCommand("feedback", "Rate the bot: /feedback 1-5 comment"),
                BotCommand("settings", "Admin: show configuration"),
                BotCommand("style", "Admin: set answer style"),
                BotCommand("mode", "Admin: set participation mode"),
                BotCommand("memory", "Admin: enable/disable context"),
            ]
        )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_message.reply_text(
            "Mention me or reply to one of my messages to ask a question.\n\n"
            "Everyone: /help, /image <prompt>, /info, /feedback 1-5 optional comment\n"
            "Admins: /settings, /style <instruction>, /mode mention|reply|always|off, /memory on|off"
        )

    async def image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        prompt = " ".join(context.args).strip()
        if not prompt:
            await update.effective_message.reply_text("Usage: /image a cozy cabin in a snowy forest at sunset")
            return
        await self._send_image(update.effective_message, context, prompt)

    async def info(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        settings = self.repository.get_settings(update.effective_chat.id)
        await update.effective_message.reply_text(self._settings_summary(settings))

    async def settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._is_admin(update):
            return
        await self.info(update, context)

    async def style(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._is_admin(update):
            return
        style = " ".join(context.args).strip()
        if not style:
            await update.effective_message.reply_text("Usage: /style Be concise and explain technical terms.")
            return
        if len(style) > 500:
            await update.effective_message.reply_text("Keep the style instruction to 500 characters or fewer.")
            return
        current = self.repository.get_settings(update.effective_chat.id)
        self.repository.save_settings(
            GroupSettings(current.chat_id, style, current.participation_mode, current.memory_enabled)
        )
        await update.effective_message.reply_text("Response style saved.")

    async def mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._is_admin(update):
            return
        requested = " ".join(context.args).lower().strip()
        try:
            mode = ParticipationMode(requested)
        except ValueError:
            await update.effective_message.reply_text("Usage: /mode mention|reply|always|off")
            return
        current = self.repository.get_settings(update.effective_chat.id)
        self.repository.save_settings(
            GroupSettings(current.chat_id, current.response_style, mode, current.memory_enabled)
        )
        await update.effective_message.reply_text(f"Participation mode is now: {mode.value}.")

    async def memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._is_admin(update):
            return
        requested = " ".join(context.args).lower().strip()
        if requested not in {"on", "off"}:
            await update.effective_message.reply_text("Usage: /memory on|off")
            return
        current = self.repository.get_settings(update.effective_chat.id)
        self.repository.save_settings(
            GroupSettings(current.chat_id, current.response_style, current.participation_mode, requested == "on")
        )
        status = "enabled" if requested == "on" else "disabled"
        await update.effective_message.reply_text(
            f"Short conversation context is {status}. Existing local context is not sent while disabled."
        )

    async def feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not context.args or not context.args[0].isdigit() or not 1 <= int(context.args[0]) <= 5:
            await update.effective_message.reply_text("Usage: /feedback 1-5 optional comment")
            return
        rating = int(context.args[0])
        comment = " ".join(context.args[1:]).strip() or "No comment"
        self.repository.add_feedback(
            update.effective_chat.id, update.effective_user.id, rating, comment
        )
        await update.effective_message.reply_text("Thanks—your feedback has been recorded.")

    async def respond(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if not message or not message.text or message.from_user is None or message.from_user.is_bot:
            return
        settings = self.repository.get_settings(update.effective_chat.id)
        bot = self._bot_user or await context.bot.get_me()
        self._bot_user = bot

        # Check entities first (the reliable Telegram way for @mentions)
        mentioned_in_entities = any(
            entity.type == "mention"
            and message.text[entity.offset : entity.offset + entity.length].lower()
            == f"@{bot.username.lower()}"
            for entity in (message.entities or [])
        )
        # Fallback: plain text scan (covers edge cases)
        mentioned_in_text = bool(
            bot.username and f"@{bot.username.lower()}" in message.text.lower()
        )
        is_mentioned = mentioned_in_entities or mentioned_in_text
        is_reply_to_bot = bool(
            message.reply_to_message
            and message.reply_to_message.from_user
            and message.reply_to_message.from_user.id == bot.id
        )
        logger.info(
            "chat=%s mention=%s reply=%s mode=%s",
            update.effective_chat.id,
            is_mentioned,
            is_reply_to_bot,
            settings.participation_mode,
        )
        if not should_reply(settings, is_mentioned=is_mentioned, is_reply_to_bot=is_reply_to_bot):
            return

        prompt = self._request_text(message.text, bot.username)
        if not prompt:
            await message.reply_text("What would you like help with?")
            return

        image_prompt = self._image_prompt(prompt)
        if image_prompt:
            await self._send_image(message, context, image_prompt)
            return

        if settings.memory_enabled:
            self.repository.add_context(
                update.effective_chat.id,
                ConversationMessage(message.from_user.full_name, prompt),
            )
            recent = self.repository.get_context(update.effective_chat.id, self.max_context)
        else:
            recent = []
        await context.bot.send_chat_action(update.effective_chat.id, "typing")
        try:
            reply = await self.ai.answer(prompt, settings, recent)
        except RuntimeError:
            logger.exception("AI response failed for chat %s", update.effective_chat.id)
            await message.reply_text("I couldn't reach the AI service just now. Please try again shortly.")
            return
        if settings.memory_enabled:
            self.repository.add_context(
                update.effective_chat.id, ConversationMessage("TelegramAI", reply)
            )
        await message.reply_text(reply)

    async def _send_image(self, message, context: ContextTypes.DEFAULT_TYPE, prompt: str) -> None:
        await context.bot.send_chat_action(message.chat_id, "upload_photo")
        try:
            image_data, mime_type = await self.ai.generate_image(prompt)
        except RuntimeError:
            logger.exception("Image generation failed for chat %s", message.chat_id)
            await message.reply_text("I couldn't generate that image just now. Please try again shortly.")
            return
        image_file = BytesIO(image_data)
        image_file.name = "generated." + mime_type.split("/", 1)[1]
        await message.reply_photo(image_file, caption="Generated image")

    @staticmethod
    def _request_text(text: str, bot_username: str | None = None) -> str:
        """Remove the Telegram routing mention before sending the request to Gemini."""
        normalized = text.strip()
        if not bot_username:
            return normalized
        mention = re.escape(f"@{bot_username}")
        without_mentions = re.sub(
            rf"(?<!\w){mention}\b", "", normalized, flags=re.IGNORECASE
        )
        return re.sub(r"\s+([,.:;!?])", r"\1", without_mentions).strip(" ,:-")

    @staticmethod
    def _image_prompt(text: str) -> str | None:
        normalized = text.strip()
        for prefix in (
            "generate an image of ",
            "generate an image ",
            "generate image of ",
            "generate image ",
            "create an image of ",
            "create an image ",
            "make an image of ",
            "make an image ",
            "draw ",
            "illustrate ",
        ):
            if normalized.lower().startswith(prefix):
                return normalized[len(prefix) :].strip() or None
        return None

    async def _is_admin(self, update: Update) -> bool:
        chat = update.effective_chat
        user = update.effective_user
        if not chat or not user:
            return False
        # In private chats there are no admin roles; treat the user as owner.
        from telegram.constants import ChatType
        if chat.type == ChatType.PRIVATE:
            return True
        member = await chat.get_member(user.id)
        if member.status in {ChatMemberStatus.OWNER, ChatMemberStatus.ADMINISTRATOR}:
            return True
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
    application.add_handler(CommandHandler("image", handlers.image))
    application.add_handler(CommandHandler("info", handlers.info))
    application.add_handler(CommandHandler("settings", handlers.settings))
    application.add_handler(CommandHandler("style", handlers.style))
    application.add_handler(CommandHandler("mode", handlers.mode))
    application.add_handler(CommandHandler("memory", handlers.memory))
    application.add_handler(CommandHandler("feedback", handlers.feedback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.respond))

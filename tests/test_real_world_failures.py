"""Automated reproduction and verification of real-world scenarios A-F.

Scenario A: Pratik sends photo. Dre replies: "@workflow012_bot explain this image" -> Analyzes photo.
Scenario B: Dre replies to a photo: "Explain this" -> Analyzes replied photo.
Scenario C: Image analysis fails. Dre replies: "Try" -> Bot retries original image request using stored context.
Scenario D: Dre replies to bot: "Explain context" -> Explains discussion/image context, not dictionary definition.
Scenario E: Dre sends a photo without requesting a rating -> No auto-rating.
Scenario F: Several users discuss unrelated topics simultaneously -> Contexts isolated.
"""

from __future__ import annotations

import io
import unittest
from unittest.mock import AsyncMock, MagicMock

# pyrefly: ignore[missing-import]
from PIL import Image

from telegram_ai.domain import ConversationMessage, GroupSettings, ParticipationMode
from telegram_ai.groq import GroqClient
from telegram_ai.handlers import TelegramAIHandlers
from telegram_ai.storage import SQLiteRepository


def _make_sample_image_bytes(color: str = "green") -> bytes:
    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestRealWorldFailures(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.repo = SQLiteRepository(":memory:")
        self.ai = MagicMock(spec=GroqClient)
        self.ai.answer = AsyncMock(return_value="Detailed discussion context explanation.")
        self.ai.rate_photo = AsyncMock(return_value="8/10 rating.")
        self.ai.analyze_image = AsyncMock(return_value="This image shows a green diagram.")
        self.ai.describe_image = AsyncMock(return_value="A green diagram with technical flowchart nodes.")
        self.ai.vision_model = "qwen/qwen3.8-27b"
        self.handlers = TelegramAIHandlers(self.repo, self.ai, max_context=10)
        self.handlers._bot_user = MagicMock(id=123, username="workflow012_bot")

    # ------------------------------------------------------------------
    # Scenario A: Pratik sends a photo. Dre replies: "@workflow012_bot explain this image"
    # Expected: Bot analyzes Pratik's photo (never says "Which image? Upload it").
    # ------------------------------------------------------------------
    async def test_scenario_a_pratik_photo_dre_replies_explain(self):
        chat_id = -5001
        self.repo.save_settings(GroupSettings(chat_id=chat_id, participation_mode=ParticipationMode.MENTION))
        img_bytes = _make_sample_image_bytes()

        # Pratik's photo message (msg 501)
        pratik_user = MagicMock(id=201, full_name="Pratik", is_bot=False)
        photo_size = MagicMock(file_id="pratik_photo_file_id", file_unique_id="u501", file_size=len(img_bytes))
        pratik_msg = MagicMock()
        pratik_msg.message_id = 501
        pratik_msg.chat_id = chat_id
        pratik_msg.from_user = pratik_user
        pratik_msg.text = None
        pratik_msg.caption = "Check this out"
        pratik_msg.photo = [photo_size]
        pratik_msg.document = None
        pratik_msg.reply_to_message = None

        # Dre replies to Pratik's photo (msg 502)
        dre_user = MagicMock(id=202, full_name="Dre", is_bot=False)
        dre_msg = MagicMock()
        dre_msg.message_id = 502
        dre_msg.chat_id = chat_id
        dre_msg.from_user = dre_user
        dre_msg.text = "@workflow012_bot explain this image"
        entity = MagicMock(type="mention", offset=0, length=17)
        dre_msg.entities = [entity]
        dre_msg.caption = None
        dre_msg.photo = None
        dre_msg.document = None
        dre_msg.quote = None
        dre_msg.reply_to_message = pratik_msg
        dre_msg.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = MagicMock(id=chat_id)
        update.effective_user = dre_user
        update.effective_message = dre_msg

        fake_file = MagicMock()
        fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(img_bytes))
        ctx = MagicMock()
        ctx.bot = MagicMock()
        ctx.bot.get_file = AsyncMock(return_value=fake_file)
        ctx.bot.get_me = AsyncMock(return_value=self.handlers._bot_user)
        ctx.bot.send_chat_action = AsyncMock()

        await self.handlers.respond(update, ctx)

        # Verified: analyze_image called with Pratik's photo
        self.ai.analyze_image.assert_awaited_once()
        call_prompt = self.ai.analyze_image.call_args[0][2]
        self.assertIn("explain this image", call_prompt.lower())
        dre_msg.reply_text.assert_awaited_once_with("This image shows a green diagram.")

    # ------------------------------------------------------------------
    # Scenario B: Dre replies to a photo: "Explain this"
    # Expected: Bot analyzes the replied-to photo.
    # ------------------------------------------------------------------
    async def test_scenario_b_dre_replies_explain_this(self):
        chat_id = -5002
        self.repo.save_settings(GroupSettings(chat_id=chat_id, participation_mode=ParticipationMode.ALWAYS))
        img_bytes = _make_sample_image_bytes(color="blue")

        # Photo message (msg 601)
        author_user = MagicMock(id=301, full_name="User", is_bot=False)
        photo_size = MagicMock(file_id="blue_photo_file_id", file_unique_id="u601", file_size=len(img_bytes))
        photo_msg = MagicMock()
        photo_msg.message_id = 601
        photo_msg.chat_id = chat_id
        photo_msg.from_user = author_user
        photo_msg.text = None
        photo_msg.caption = None
        photo_msg.photo = [photo_size]
        photo_msg.document = None
        photo_msg.reply_to_message = None

        # Dre replies "Explain this" (msg 602)
        dre_user = MagicMock(id=302, full_name="Dre", is_bot=False)
        dre_msg = MagicMock()
        dre_msg.message_id = 602
        dre_msg.chat_id = chat_id
        dre_msg.from_user = dre_user
        dre_msg.text = "Explain this"
        dre_msg.entities = []
        dre_msg.caption = None
        dre_msg.photo = None
        dre_msg.document = None
        dre_msg.quote = None
        dre_msg.reply_to_message = photo_msg
        dre_msg.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = MagicMock(id=chat_id)
        update.effective_user = dre_user
        update.effective_message = dre_msg

        fake_file = MagicMock()
        fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(img_bytes))
        ctx = MagicMock()
        ctx.bot = MagicMock()
        ctx.bot.get_file = AsyncMock(return_value=fake_file)
        ctx.bot.get_me = AsyncMock(return_value=self.handlers._bot_user)
        ctx.bot.send_chat_action = AsyncMock()

        await self.handlers.respond(update, ctx)

        self.ai.analyze_image.assert_awaited_once()
        self.ai.rate_photo.assert_not_awaited()

    # ------------------------------------------------------------------
    # Scenario C: Image analysis fails. Dre replies: "Try"
    # Expected: Bot retries the original image request using stored context.
    # ------------------------------------------------------------------
    async def test_scenario_c_failed_analysis_dre_replies_try(self):
        chat_id = -5003
        self.repo.save_settings(GroupSettings(chat_id=chat_id, participation_mode=ParticipationMode.MENTION))
        img_bytes = _make_sample_image_bytes()

        # Step 1: Simulated previous failed attempt
        # User sends photo and asks question -> AI throws RuntimeError
        self.ai.analyze_image = AsyncMock(side_effect=RuntimeError("Groq temporarily unavailable"))

        dre_user = MagicMock(id=401, full_name="Dre", is_bot=False)
        photo_size = MagicMock(file_id="recoverable_img_id", file_unique_id="u701", file_size=len(img_bytes))
        photo_msg = MagicMock()
        photo_msg.message_id = 701
        photo_msg.chat_id = chat_id
        photo_msg.from_user = dre_user
        photo_msg.caption = "@workflow012_bot Find the error in this code"
        mention_entity = MagicMock(type="mention", offset=0, length=17)
        photo_msg.caption_entities = [mention_entity]
        photo_msg.photo = [photo_size]
        photo_msg.document = None
        photo_msg.quote = None
        photo_msg.reply_to_message = None

        bot_error_sent = MagicMock()
        bot_error_sent.message_id = 702
        photo_msg.reply_text = AsyncMock(return_value=bot_error_sent)

        update1 = MagicMock()
        update1.effective_chat = MagicMock(id=chat_id)
        update1.effective_user = dre_user
        update1.effective_message = photo_msg

        fake_file = MagicMock()
        fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(img_bytes))
        ctx = MagicMock()
        ctx.bot = MagicMock()
        ctx.bot.get_file = AsyncMock(return_value=fake_file)
        ctx.bot.get_me = AsyncMock(return_value=self.handlers._bot_user)
        ctx.bot.send_chat_action = AsyncMock()

        await self.handlers.photo(update1, ctx)
        photo_msg.reply_text.assert_awaited_with(
            "I couldn't reach the AI service just now. Please try again shortly."
        )

        # Step 2: Now Dre replies "Try" to the bot's failed message (msg 702)
        # AI service is restored
        self.ai.analyze_image = AsyncMock(return_value="The syntax error was missing colon on line 4.")

        bot_failed_msg_mock = MagicMock()
        bot_failed_msg_mock.message_id = 702
        bot_failed_msg_mock.from_user = MagicMock(id=123, is_bot=True)
        bot_failed_msg_mock.text = "I couldn't reach the AI service just now. Please try again shortly."
        bot_failed_msg_mock.photo = None
        bot_failed_msg_mock.document = None

        retry_msg = MagicMock()
        retry_msg.message_id = 703
        retry_msg.chat_id = chat_id
        retry_msg.from_user = dre_user
        retry_msg.text = "Try"
        retry_msg.entities = []
        retry_msg.reply_to_message = bot_failed_msg_mock
        retry_msg.quote = None
        retry_msg.photo = None
        retry_msg.document = None
        retry_msg.reply_text = AsyncMock()

        update2 = MagicMock()
        update2.effective_chat = MagicMock(id=chat_id)
        update2.effective_user = dre_user
        update2.effective_message = retry_msg

        await self.handlers.respond(update2, ctx)

        # Verified: analyze_image was automatically retried on the previous image!
        self.ai.analyze_image.assert_awaited_once()
        retry_msg.reply_text.assert_awaited_with("The syntax error was missing colon on line 4.")

    # ------------------------------------------------------------------
    # Scenario D: Dre replies to bot: "Explain context"
    # Expected: Bot explains discussion/image context, not dictionary definition.
    # ------------------------------------------------------------------
    async def test_scenario_d_dre_replies_explain_context(self):
        chat_id = -5004
        self.repo.save_settings(GroupSettings(chat_id=chat_id, participation_mode=ParticipationMode.REPLY))

        # Ongoing discussion about an anime screenshot
        self.repo.add_context(
            chat_id,
            ConversationMessage(
                author="Dre",
                text="Who is this character?",
                message_id=801,
                media_type="photo",
                media_summary="Screenshot of Gojo Satoru in Jujutsu Kaisen",
            ),
        )
        self.repo.add_context(
            chat_id,
            ConversationMessage(
                author="TelegramAI",
                text="That's Gojo Satoru activating his Domain Expansion.",
                message_id=802,
                reply_to_message_id=801,
                is_bot=True,
            ),
        )

        bot_msg = MagicMock()
        bot_msg.message_id = 802
        bot_msg.from_user = MagicMock(id=123, is_bot=True)
        bot_msg.text = "That's Gojo Satoru activating his Domain Expansion."
        bot_msg.photo = None
        bot_msg.document = None

        dre_user = MagicMock(id=402, full_name="Dre", is_bot=False)
        dre_msg = MagicMock()
        dre_msg.message_id = 803
        dre_msg.chat_id = chat_id
        dre_msg.from_user = dre_user
        dre_msg.text = "Explain context"
        dre_msg.entities = []
        dre_msg.reply_to_message = bot_msg
        dre_msg.quote = None
        dre_msg.photo = None
        dre_msg.document = None
        dre_msg.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = MagicMock(id=chat_id)
        update.effective_user = dre_user
        update.effective_message = dre_msg

        ctx = MagicMock()
        ctx.bot = MagicMock()
        ctx.bot.get_me = AsyncMock(return_value=self.handlers._bot_user)
        ctx.bot.send_chat_action = AsyncMock()

        await self.handlers.respond(update, ctx)

        self.ai.answer.assert_awaited_once()
        prompt_passed = self.ai.answer.call_args[0][0]
        # Verified: Prompt specifically instructs Groq to explain discussion context, not dictionary definition
        self.assertIn("Do NOT give a dictionary definition of the word 'context'", prompt_passed)
        self.assertIn("ongoing conversation", prompt_passed)

    # ------------------------------------------------------------------
    # Scenario E: Dre sends a photo without requesting a rating
    # Expected: Bot does not automatically rate the photo.
    # ------------------------------------------------------------------
    async def test_scenario_e_photo_without_rating_request(self):
        chat_id = -5005
        # Group in mention mode
        self.repo.save_settings(GroupSettings(chat_id=chat_id, participation_mode=ParticipationMode.MENTION))
        img_bytes = _make_sample_image_bytes()

        dre_user = MagicMock(id=501, full_name="Dre", is_bot=False)
        photo_size = MagicMock(file_id="photo_e_501", file_unique_id="uE", file_size=len(img_bytes))
        msg = MagicMock()
        msg.message_id = 901
        msg.chat_id = chat_id
        msg.from_user = dre_user
        # Mention bot, but asks to explain, NOT rate
        msg.caption = "@workflow012_bot What is in this picture?"
        entity = MagicMock(type="mention", offset=0, length=17)
        msg.caption_entities = [entity]
        msg.photo = [photo_size]
        msg.document = None
        msg.quote = None
        msg.reply_to_message = None
        msg.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = MagicMock(id=chat_id)
        update.effective_user = dre_user
        update.effective_message = msg

        fake_file = MagicMock()
        fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(img_bytes))
        ctx = MagicMock()
        ctx.bot = MagicMock()
        ctx.bot.get_file = AsyncMock(return_value=fake_file)
        ctx.bot.get_me = AsyncMock(return_value=self.handlers._bot_user)
        ctx.bot.send_chat_action = AsyncMock()

        await self.handlers.photo(update, ctx)

        # Verified: analyze_image was called, NOT rate_photo
        self.ai.analyze_image.assert_awaited_once()
        self.ai.rate_photo.assert_not_awaited()

    # ------------------------------------------------------------------
    # Scenario F: Several users discuss unrelated topics simultaneously
    # Expected: Bot keeps reply chains separate and does not mix conversations.
    # ------------------------------------------------------------------
    def test_scenario_f_unrelated_topics_isolation(self):
        chat_id = -5006
        # Topic 1: Naruto
        self.repo.add_context(chat_id, ConversationMessage("Pratik", "Naruto mastered Rasengan", message_id=1001))
        self.repo.add_context(chat_id, ConversationMessage("TelegramAI", "Jiraiya taught him Rasengan", message_id=1002, reply_to_message_id=1001, is_bot=True))

        # Topic 2: Database indexing
        self.repo.add_context(chat_id, ConversationMessage("Dre", "How do B-trees work?", message_id=2001))
        self.repo.add_context(chat_id, ConversationMessage("TelegramAI", "B-trees maintain sorted data for logarithmic searches", message_id=2002, reply_to_message_id=2001, is_bot=True))

        # Topic 3: Recipe
        self.repo.add_context(chat_id, ConversationMessage("Alice", "Best way to bake sourdough?", message_id=3001))
        self.repo.add_context(chat_id, ConversationMessage("TelegramAI", "Maintain active starter and fold every 30 mins", message_id=3002, reply_to_message_id=3001, is_bot=True))

        # Pratik replies to Topic 1
        follow_up_pratik = ConversationMessage("Pratik", "What about Rasenshuriken?", message_id=1003, reply_to_message_id=1002)
        from telegram_ai.context import assemble_context
        ctx_pratik = assemble_context(self.repo, chat_id, follow_up_pratik, memory_enabled=True)

        pratik_ids = [m.message_id for m in ctx_pratik]
        self.assertIn(1001, pratik_ids)
        self.assertIn(1002, pratik_ids)
        # Sourdough and B-tree messages must be absent
        self.assertNotIn(2001, pratik_ids)
        self.assertNotIn(2002, pratik_ids)
        self.assertNotIn(3001, pratik_ids)
        self.assertNotIn(3002, pratik_ids)

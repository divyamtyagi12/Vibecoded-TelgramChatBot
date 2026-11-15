"""Comprehensive test suite for Advanced Conversation Memory and Image Understanding.

Covers all 20 required scenarios:
1. Direct replies to the bot.
2. Replies to other users.
3. Telegram quoted messages.
4. Nested reply chains.
5. Old conversation continuation.
6. Simultaneous unrelated group discussions.
7. Direct image questions.
8. Questions about replied-to images.
9. Code screenshot analysis.
10. Mathematics screenshot analysis.
11. Meme explanation.
12. Image-based follow-up questions.
13. Photo-rating requests.
14. Missing images / download failure.
15. Missing reply parents.
16. Duplicate message prevention.
17. Memory-disabled behavior.
18. Concurrent database operations.
19. Database migration compatibility.
20. API failures and rate limits.
"""

from __future__ import annotations

import asyncio
import io
import sqlite3
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from PIL import Image

from telegram_ai.config import Settings
from telegram_ai.context import assemble_context, estimate_message_tokens
from telegram_ai.domain import (
    ConversationMessage,
    GroupSettings,
    ParticipationMode,
    is_photo_rating_intent,
)
from telegram_ai.groq import GroqClient
from telegram_ai.handlers import TelegramAIHandlers
from telegram_ai.storage import SQLiteRepository
from telegram_ai.vision import validate_and_process_image


def _create_test_image_bytes(w: int = 64, h: int = 64, color: str = "red") -> bytes:
    img = Image.new("RGB", (w, h), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestConversationMemoryAndVision(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.repo = SQLiteRepository(":memory:")
        self.ai = MagicMock(spec=GroqClient)
        self.ai.answer = AsyncMock(return_value="AI text reply")
        self.ai.rate_photo = AsyncMock(return_value="9/10 - Great lighting and composition")
        self.ai.analyze_image = AsyncMock(return_value="Analysis: This is a test image")
        self.ai.describe_image = AsyncMock(return_value="A solid red test square")
        self.ai.vision_model = "qwen/qwen3.8-27b"
        self.handlers = TelegramAIHandlers(self.repo, self.ai, max_context=10)
        self.handlers._bot_user = MagicMock(id=999, username="TestBot")

    # ------------------------------------------------------------------
    # 1. Direct replies to the bot
    # ------------------------------------------------------------------
    async def test_01_direct_reply_to_bot(self):
        chat_id = -1001
        self.repo.save_settings(GroupSettings(chat_id=chat_id, participation_mode=ParticipationMode.REPLY))
        # Bot's earlier message
        self.repo.add_context(
            chat_id,
            ConversationMessage(
                author="TelegramAI",
                text="The speed of light is ~300,000 km/s.",
                message_id=10,
                is_bot=True,
            ),
        )

        bot_msg = MagicMock()
        bot_msg.message_id = 10
        bot_msg.from_user = MagicMock(id=999, is_bot=True)
        bot_msg.text = "The speed of light is ~300,000 km/s."
        bot_msg.photo = None
        bot_msg.document = None

        user_msg = MagicMock()
        user_msg.message_id = 11
        user_msg.chat_id = chat_id
        user_msg.text = "Can anything travel faster?"
        user_msg.entities = []
        user_msg.from_user = MagicMock(id=42, full_name="Alice", is_bot=False)
        user_msg.reply_to_message = bot_msg
        user_msg.quote = None
        user_msg.photo = None
        user_msg.document = None
        user_msg.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = MagicMock(id=chat_id)
        update.effective_user = user_msg.from_user
        update.effective_message = user_msg

        ctx = MagicMock()
        ctx.bot = MagicMock()
        ctx.bot.get_me = AsyncMock(return_value=self.handlers._bot_user)
        ctx.bot.send_chat_action = AsyncMock()

        await self.handlers.respond(update, ctx)
        self.ai.answer.assert_awaited_once()
        call_context = self.ai.answer.call_args[0][2]
        self.assertTrue(any("speed of light" in m.text for m in call_context))

    # ------------------------------------------------------------------
    # 2. Replies to other users
    # ------------------------------------------------------------------
    async def test_02_reply_to_other_user(self):
        chat_id = -1002
        self.repo.save_settings(GroupSettings(chat_id=chat_id, participation_mode=ParticipationMode.MENTION))
        # User A's message
        self.repo.add_context(
            chat_id,
            ConversationMessage(
                author="UserA",
                text="Reinhard defeats Asta easily.",
                message_id=20,
                sender_id=101,
                is_bot=False,
            ),
        )

        user_a_msg = MagicMock()
        user_a_msg.message_id = 20
        user_a_msg.from_user = MagicMock(id=101, full_name="UserA", is_bot=False)
        user_a_msg.text = "Reinhard defeats Asta easily."
        user_a_msg.photo = None
        user_a_msg.document = None

        # User B replies to User A and mentions bot
        user_b_msg = MagicMock()
        user_b_msg.message_id = 21
        user_b_msg.chat_id = chat_id
        user_b_msg.text = "@TestBot Is this true?"
        mention_entity = MagicMock(type="mention", offset=0, length=8)
        user_b_msg.entities = [mention_entity]
        user_b_msg.from_user = MagicMock(id=102, full_name="UserB", is_bot=False)
        user_b_msg.reply_to_message = user_a_msg
        user_b_msg.quote = None
        user_b_msg.photo = None
        user_b_msg.document = None
        user_b_msg.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = MagicMock(id=chat_id)
        update.effective_user = user_b_msg.from_user
        update.effective_message = user_b_msg

        ctx = MagicMock()
        ctx.bot = MagicMock()
        ctx.bot.get_me = AsyncMock(return_value=self.handlers._bot_user)
        ctx.bot.send_chat_action = AsyncMock()

        await self.handlers.respond(update, ctx)
        self.ai.answer.assert_awaited_once()
        prompt_arg = self.ai.answer.call_args[0][0]
        context_arg = self.ai.answer.call_args[0][2]
        self.assertEqual(prompt_arg, "Is this true?")
        self.assertTrue(any("Reinhard defeats Asta" in m.text for m in context_arg))

    # ------------------------------------------------------------------
    # 3. Telegram quoted messages
    # ------------------------------------------------------------------
    async def test_03_telegram_quoted_messages(self):
        chat_id = -1003
        self.repo.save_settings(GroupSettings(chat_id=chat_id, participation_mode=ParticipationMode.ALWAYS))

        user_msg = MagicMock()
        user_msg.message_id = 30
        user_msg.chat_id = chat_id
        user_msg.text = "What does this specific term mean?"
        user_msg.entities = []
        user_msg.from_user = MagicMock(id=50, full_name="Dave", is_bot=False)
        user_msg.reply_to_message = None

        quote_mock = MagicMock()
        quote_mock.text = "quantum entanglement"
        quote_mock.position = 12
        user_msg.quote = quote_mock
        user_msg.photo = None
        user_msg.document = None
        user_msg.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = MagicMock(id=chat_id)
        update.effective_user = user_msg.from_user
        update.effective_message = user_msg

        ctx = MagicMock()
        ctx.bot = MagicMock()
        ctx.bot.get_me = AsyncMock(return_value=self.handlers._bot_user)
        ctx.bot.send_chat_action = AsyncMock()

        await self.handlers.respond(update, ctx)
        self.ai.answer.assert_awaited_once()
        prompt_arg = self.ai.answer.call_args[0][0]
        self.assertIn('[Quoting: "quantum entanglement"]', prompt_arg)

    # ------------------------------------------------------------------
    # 4. Nested reply chains
    # ------------------------------------------------------------------
    def test_04_nested_reply_chains(self):
        chat_id = -1004
        # Add 4-level nested conversation
        self.repo.add_context(chat_id, ConversationMessage("Alice", "Root topic", message_id=1, reply_to_message_id=None))
        self.repo.add_context(chat_id, ConversationMessage("Bob", "Reply to root", message_id=2, reply_to_message_id=1))
        self.repo.add_context(chat_id, ConversationMessage("TelegramAI", "Bot answer to Bob", message_id=3, reply_to_message_id=2, is_bot=True))
        self.repo.add_context(chat_id, ConversationMessage("Charlie", "Reply to bot", message_id=4, reply_to_message_id=3))

        chain = self.repo.get_reply_chain(chat_id, 4, max_depth=10)
        self.assertEqual(len(chain), 4)
        # Chronological order
        self.assertEqual([m.message_id for m in chain], [1, 2, 3, 4])

    # ------------------------------------------------------------------
    # 5. Old conversation continuation
    # ------------------------------------------------------------------
    def test_05_old_conversation_continuation(self):
        chat_id = -1005
        # Message saved with older timestamp
        self.repo.add_context(
            chat_id,
            ConversationMessage("Alice", "Old discussion from last week", message_id=50, created_at="2026-10-01 10:00:00"),
        )
        self.repo.add_context(
            chat_id,
            ConversationMessage("TelegramAI", "Old bot answer", message_id=51, reply_to_message_id=50, is_bot=True),
        )

        # Later reply to msg 51
        curr = ConversationMessage("Alice", "Resuming this old topic", message_id=52, reply_to_message_id=51)
        ctx = assemble_context(self.repo, chat_id, curr, memory_enabled=True)
        self.assertEqual(len(ctx), 2)
        self.assertEqual(ctx[0].message_id, 50)
        self.assertEqual(ctx[1].message_id, 51)

    # ------------------------------------------------------------------
    # 6. Simultaneous unrelated group discussions
    # ------------------------------------------------------------------
    def test_06_simultaneous_unrelated_group_discussions(self):
        chat_id = -1006
        # Discussion A: Naruto
        self.repo.add_context(chat_id, ConversationMessage("UserA", "Naruto vs Sasuke", message_id=101))
        self.repo.add_context(chat_id, ConversationMessage("TelegramAI", "Naruto has Sage of Six Paths", message_id=102, reply_to_message_id=101, is_bot=True))

        # Discussion B: Python programming
        self.repo.add_context(chat_id, ConversationMessage("UserB", "How to use asyncio?", message_id=201))
        self.repo.add_context(chat_id, ConversationMessage("TelegramAI", "Use asyncio.run()", message_id=202, reply_to_message_id=201, is_bot=True))

        # User A continues Discussion A
        user_a_followup = ConversationMessage("UserA", "What about Kurama?", message_id=103, reply_to_message_id=102)
        ctx_a = assemble_context(self.repo, chat_id, user_a_followup, memory_enabled=True)

        # Context for A must NOT contain Python messages 201 or 202
        msg_ids_a = [m.message_id for m in ctx_a]
        self.assertIn(101, msg_ids_a)
        self.assertIn(102, msg_ids_a)
        self.assertNotIn(201, msg_ids_a)
        self.assertNotIn(202, msg_ids_a)

    # ------------------------------------------------------------------
    # 7. Direct image questions
    # ------------------------------------------------------------------
    async def test_07_direct_image_questions(self):
        chat_id = -1007
        self.repo.save_settings(GroupSettings(chat_id=chat_id, participation_mode=ParticipationMode.ALWAYS))
        img_bytes = _create_test_image_bytes()

        msg = MagicMock()
        msg.message_id = 70
        msg.chat_id = chat_id
        msg.caption = "What color is this square?"
        msg.caption_entities = []
        msg.reply_to_message = None
        msg.from_user = MagicMock(id=60, full_name="User", is_bot=False)
        photo_size = MagicMock(file_id="img70", file_unique_id="u70", file_size=len(img_bytes))
        msg.photo = [photo_size]
        msg.document = None
        msg.quote = None
        msg.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = MagicMock(id=chat_id)
        update.effective_user = msg.from_user
        update.effective_message = msg

        fake_file = MagicMock()
        fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(img_bytes))
        ctx = MagicMock()
        ctx.bot = MagicMock()
        ctx.bot.get_file = AsyncMock(return_value=fake_file)
        ctx.bot.get_me = AsyncMock(return_value=self.handlers._bot_user)
        ctx.bot.send_chat_action = AsyncMock()

        await self.handlers.photo(update, ctx)
        self.ai.analyze_image.assert_awaited_once()
        self.ai.rate_photo.assert_not_awaited()

    # ------------------------------------------------------------------
    # 8. Questions about replied-to images
    # ------------------------------------------------------------------
    async def test_08_questions_about_replied_to_images(self):
        chat_id = -1008
        self.repo.save_settings(GroupSettings(chat_id=chat_id, participation_mode=ParticipationMode.ALWAYS))
        img_bytes = _create_test_image_bytes()

        # Image message sent by User A earlier
        parent_msg = MagicMock()
        parent_msg.message_id = 80
        parent_msg.from_user = MagicMock(id=81, full_name="UserA", is_bot=False)
        parent_msg.text = None
        parent_msg.caption = "Here is my code"
        photo_size = MagicMock(file_id="code_img_80", file_unique_id="u80", file_size=len(img_bytes))
        parent_msg.photo = [photo_size]
        parent_msg.document = None
        parent_msg.reply_to_message = None

        # User B replies to User A's photo
        user_b_msg = MagicMock()
        user_b_msg.message_id = 81
        user_b_msg.chat_id = chat_id
        user_b_msg.text = "What is wrong with this code?"
        user_b_msg.entities = []
        user_b_msg.from_user = MagicMock(id=82, full_name="UserB", is_bot=False)
        user_b_msg.reply_to_message = parent_msg
        user_b_msg.quote = None
        user_b_msg.photo = None
        user_b_msg.document = None
        user_b_msg.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = MagicMock(id=chat_id)
        update.effective_user = user_b_msg.from_user
        update.effective_message = user_b_msg

        fake_file = MagicMock()
        fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(img_bytes))
        ctx = MagicMock()
        ctx.bot = MagicMock()
        ctx.bot.get_file = AsyncMock(return_value=fake_file)
        ctx.bot.get_me = AsyncMock(return_value=self.handlers._bot_user)
        ctx.bot.send_chat_action = AsyncMock()

        await self.handlers.respond(update, ctx)
        self.ai.analyze_image.assert_awaited_once()

    # ------------------------------------------------------------------
    # 9. Code screenshot analysis
    # ------------------------------------------------------------------
    def test_09_code_screenshot_intent_routing(self):
        self.assertFalse(is_photo_rating_intent("Find the error in this C++ snippet"))
        self.assertFalse(is_photo_rating_intent("Explain what is wrong with this code"))

    # ------------------------------------------------------------------
    # 10. Mathematics screenshot analysis
    # ------------------------------------------------------------------
    def test_10_mathematics_screenshot_intent_routing(self):
        self.assertFalse(is_photo_rating_intent("Solve this integral step by step"))
        self.assertFalse(is_photo_rating_intent("What is the value of x in this geometry problem?"))

    # ------------------------------------------------------------------
    # 11. Meme explanation
    # ------------------------------------------------------------------
    def test_11_meme_explanation_intent_routing(self):
        self.assertFalse(is_photo_rating_intent("Explain this meme"))
        self.assertFalse(is_photo_rating_intent("Why is this joke funny?"))

    # ------------------------------------------------------------------
    # 12. Image-based follow-up questions
    # ------------------------------------------------------------------
    def test_12_image_based_follow_up_questions(self):
        chat_id = -1012
        # Message 120 had an image with cached summary
        self.repo.add_context(
            chat_id,
            ConversationMessage(
                author="User",
                text="Here is a binary tree problem",
                message_id=120,
                media_type="photo",
                media_summary="Binary search tree inversion problem in Python",
            ),
        )
        self.repo.add_context(
            chat_id,
            ConversationMessage(
                author="TelegramAI",
                text="You can invert the tree by swapping left and right child pointers.",
                message_id=121,
                reply_to_message_id=120,
                is_bot=True,
            ),
        )

        # Follow-up: User asks about recursion
        follow_up = ConversationMessage(
            author="User",
            text="Can you write the recursive implementation?",
            message_id=122,
            reply_to_message_id=121,
        )
        ctx = assemble_context(self.repo, chat_id, follow_up, memory_enabled=True)
        self.assertEqual(len(ctx), 2)
        # Verify the image summary is preserved in the context
        self.assertIn("Binary search tree inversion", ctx[0].media_summary or "")

    # ------------------------------------------------------------------
    # 13. Photo-rating requests
    # ------------------------------------------------------------------
    async def test_13_photo_rating_requests(self):
        chat_id = -1013
        self.repo.save_settings(GroupSettings(chat_id=chat_id, participation_mode=ParticipationMode.ALWAYS))
        img_bytes = _create_test_image_bytes()

        msg = MagicMock()
        msg.message_id = 130
        msg.chat_id = chat_id
        msg.caption = "Rate this photo out of 10"
        msg.caption_entities = []
        msg.reply_to_message = None
        msg.from_user = MagicMock(id=70, full_name="User", is_bot=False)
        photo_size = MagicMock(file_id="pic130", file_unique_id="u130", file_size=len(img_bytes))
        msg.photo = [photo_size]
        msg.document = None
        msg.quote = None
        msg.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = MagicMock(id=chat_id)
        update.effective_user = msg.from_user
        update.effective_message = msg

        fake_file = MagicMock()
        fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(img_bytes))
        ctx = MagicMock()
        ctx.bot = MagicMock()
        ctx.bot.get_file = AsyncMock(return_value=fake_file)
        ctx.bot.get_me = AsyncMock(return_value=self.handlers._bot_user)
        ctx.bot.send_chat_action = AsyncMock()

        await self.handlers.photo(update, ctx)
        self.ai.rate_photo.assert_awaited_once()
        self.ai.analyze_image.assert_not_awaited()

    # ------------------------------------------------------------------
    # 14. Missing images / download failure
    # ------------------------------------------------------------------
    async def test_14_missing_images_handled_gracefully(self):
        from telegram.error import TelegramError

        chat_id = -1014
        self.repo.save_settings(GroupSettings(chat_id=chat_id, participation_mode=ParticipationMode.ALWAYS))

        msg = MagicMock()
        msg.message_id = 140
        msg.chat_id = chat_id
        msg.caption = "Rate this"
        msg.caption_entities = []
        msg.reply_to_message = None
        msg.from_user = MagicMock(id=80, full_name="User", is_bot=False)
        photo_size = MagicMock(file_id="bad_file_id", file_unique_id="u140", file_size=5000)
        msg.photo = [photo_size]
        msg.document = None
        msg.quote = None
        msg.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = MagicMock(id=chat_id)
        update.effective_user = msg.from_user
        update.effective_message = msg

        ctx = MagicMock()
        ctx.bot = MagicMock()
        ctx.bot.get_file = AsyncMock(side_effect=TelegramError("File expired or unavailable"))
        ctx.bot.get_me = AsyncMock(return_value=self.handlers._bot_user)
        ctx.bot.send_chat_action = AsyncMock()

        await self.handlers.photo(update, ctx)
        msg.reply_text.assert_awaited_with("I couldn't download that photo. Please try again.")

    # ------------------------------------------------------------------
    # 15. Missing reply parents
    # ------------------------------------------------------------------
    def test_15_missing_reply_parents(self):
        chat_id = -1015
        # Current message points to non-existent parent 9999
        curr = ConversationMessage("User", "Replying into the void", message_id=150, reply_to_message_id=9999)
        ctx = assemble_context(self.repo, chat_id, curr, memory_enabled=True)
        # Should not crash, returns empty or available history
        self.assertIsInstance(ctx, list)

    # ------------------------------------------------------------------
    # 16. Duplicate message prevention
    # ------------------------------------------------------------------
    def test_16_duplicate_message_prevention(self):
        chat_id = -1016
        msg = ConversationMessage("Alice", "Hello once", message_id=160)
        self.repo.add_context(chat_id, msg)

        # Attempt to insert same message_id with modified text
        msg_duplicate = ConversationMessage("Alice", "Hello updated", message_id=160)
        self.repo.add_context(chat_id, msg_duplicate)

        # Verify only 1 entry exists
        rows = self.repo.connection.execute(
            "SELECT count(*) FROM conversation_context WHERE chat_id = ? AND message_id = 160",
            (chat_id,),
        ).fetchone()[0]
        self.assertEqual(rows, 1)

        saved = self.repo.get_message(chat_id, 160)
        self.assertIsNotNone(saved)
        self.assertEqual(saved.text, "Hello updated")

    # ------------------------------------------------------------------
    # 17. Memory-disabled behavior
    # ------------------------------------------------------------------
    def test_17_memory_disabled_behavior(self):
        chat_id = -1017
        self.repo.add_context(chat_id, ConversationMessage("Alice", "History msg 1", message_id=171))
        self.repo.add_context(chat_id, ConversationMessage("Bob", "History msg 2", message_id=172))

        # When memory is off and not a reply, context must be empty
        curr = ConversationMessage("Charlie", "New question", message_id=173)
        ctx = assemble_context(self.repo, chat_id, curr, memory_enabled=False)
        self.assertEqual(ctx, [])

    # ------------------------------------------------------------------
    # 18. Concurrent database operations
    # ------------------------------------------------------------------
    async def test_18_concurrent_database_handling(self):
        chat_id = -1018

        async def worker(idx: int):
            for i in range(10):
                m_id = idx * 100 + i
                self.repo.add_context(
                    chat_id,
                    ConversationMessage(
                        f"Worker{idx}",
                        f"Message {i}",
                        message_id=m_id,
                    ),
                )
                await asyncio.sleep(0.001)

        await asyncio.gather(*(worker(w) for w in range(5)))
        count = self.repo.connection.execute(
            "SELECT count(*) FROM conversation_context WHERE chat_id = ?", (chat_id,)
        ).fetchone()[0]
        self.assertEqual(count, 50)

    # ------------------------------------------------------------------
    # 19. Database migration compatibility
    # ------------------------------------------------------------------
    def test_19_database_migration_compatibility(self):
        # Create a raw SQLite in-memory DB with ONLY the old schema
        raw_conn = sqlite3.connect(":memory:")
        raw_conn.executescript(
            """
            CREATE TABLE group_settings (
                chat_id INTEGER PRIMARY KEY,
                response_style TEXT NOT NULL,
                participation_mode TEXT NOT NULL CHECK (participation_mode IN ('mention', 'reply', 'always', 'off')),
                memory_enabled INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE conversation_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                author TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX context_by_chat ON conversation_context(chat_id, id DESC);
            INSERT INTO conversation_context (chat_id, author, text) VALUES (1, 'LegacyUser', 'Legacy message');
            """
        )
        raw_conn.commit()

        # Instantiate SQLiteRepository with this connection
        repo = SQLiteRepository.__new__(SQLiteRepository)
        repo.connection = raw_conn
        repo.connection.row_factory = sqlite3.Row
        repo._create_schema()

        # Verify legacy row is still readable
        ctx = repo.get_context(1, 10)
        self.assertEqual(len(ctx), 1)
        self.assertEqual(ctx[0].author, "LegacyUser")
        self.assertEqual(ctx[0].text, "Legacy message")

        # Verify new columns were added and can be written to
        repo.add_context(
            1,
            ConversationMessage(
                "ModernUser",
                "New multimodal msg",
                message_id=999,
                media_type="photo",
                media_summary="A cute dog",
            ),
        )
        modern = repo.get_message(1, 999)
        self.assertIsNotNone(modern)
        self.assertEqual(modern.media_summary, "A cute dog")

    # ------------------------------------------------------------------
    # 20. API failures and rate limits
    # ------------------------------------------------------------------
    async def test_20_api_failures_and_rate_limits(self):
        import httpx

        client = GroqClient(
            api_key="TEST_KEY",
            model="openai/gpt-oss-120b",
            timeout_seconds=2.0,
        )

        req = MagicMock()
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.request = req
        resp_429.headers = {"retry-after": "0.1"}

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Recovered response"}}]
        }

        call_count = 0

        async def fake_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return resp_429
            return resp_200

        with patch("httpx.AsyncClient.post", side_effect=fake_post):
            msg = await client._chat_completion([{"role": "user", "content": "Hello"}])
            self.assertEqual(msg.get("content"), "Recovered response")
            self.assertEqual(call_count, 2)


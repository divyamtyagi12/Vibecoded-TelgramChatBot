"""Multi-group isolation, composite deduplication, and simultaneous webhook tests."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from telegram.constants import ChatMemberStatus, ChatType

from telegram_ai.domain import ConversationMessage, GroupSettings, ParticipationMode
from telegram_ai.handlers import TelegramAIHandlers
from telegram_ai.storage import SQLiteRepository


def _make_bot(bot_id: int = 9109775211, username: str = "AIChatBot_12_bot") -> MagicMock:
    bot = MagicMock()
    bot.id = bot_id
    bot.username = username
    bot.get_me = AsyncMock(return_value=bot)
    bot.send_chat_action = AsyncMock()
    return bot


def _make_update(
    *,
    chat_id: int,
    message_id: int,
    user_id: int,
    user_name: str,
    text: str,
    reply_to_message: MagicMock | None = None,
    is_bot: bool = False,
    sender_chat: MagicMock | None = None,
    chat_type: str = ChatType.SUPERGROUP,
) -> MagicMock:
    update = MagicMock()
    chat = MagicMock()
    chat.id = chat_id
    chat.type = chat_type
    chat.get_member = AsyncMock(return_value=MagicMock(status=ChatMemberStatus.ADMINISTRATOR))
    update.effective_chat = chat

    user = MagicMock()
    user.id = user_id
    user.full_name = user_name
    user.is_bot = is_bot
    update.effective_user = user

    msg = MagicMock()
    msg.message_id = message_id
    msg.text = text
    msg.caption = None
    msg.entities = []
    msg.from_user = user
    msg.sender_chat = sender_chat
    msg.reply_to_message = reply_to_message
    msg.message_thread_id = None
    msg.quote = None
    msg.photo = []
    msg.document = None

    sent_reply = MagicMock()
    sent_reply.message_id = message_id + 1000
    msg.reply_text = AsyncMock(return_value=sent_reply)

    update.effective_message = msg
    return update


class TestMultiGroupSupport(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.repo = SQLiteRepository(":memory:")
        self.ai = MagicMock()
        self.bot = _make_bot()
        self.context = MagicMock()
        self.context.bot = self.bot
        self.handlers = TelegramAIHandlers(self.repo, self.ai, max_context=8)

        async def _echo_answer(prompt: str, settings: GroupSettings, history: list[ConversationMessage]) -> str:
            await asyncio.sleep(0.01)
            hist_texts = [m.text for m in history]
            return f"[chat={settings.chat_id}|hist={len(hist_texts)}] {prompt}"

        self.ai.answer = AsyncMock(side_effect=_echo_answer)

    async def test_01_group_a_and_group_b_receive_independent_responses(self) -> None:
        upd_a = _make_update(
            chat_id=-100111, message_id=10, user_id=1, user_name="Alice",
            text="@AIChatBot_12_bot What is Python?",
        )
        upd_b = _make_update(
            chat_id=-100222, message_id=20, user_id=2, user_name="Bob",
            text="@AIChatBot_12_bot What is Rust?",
        )

        await self.handlers.respond(upd_a, self.context)
        await self.handlers.respond(upd_b, self.context)

        upd_a.effective_message.reply_text.assert_awaited_once()
        upd_b.effective_message.reply_text.assert_awaited_once()
        reply_a = upd_a.effective_message.reply_text.call_args.args[0]
        reply_b = upd_b.effective_message.reply_text.call_args.args[0]
        self.assertIn("chat=-100111", reply_a)
        self.assertIn("What is Python?", reply_a)
        self.assertIn("chat=-100222", reply_b)
        self.assertIn("What is Rust?", reply_b)

    async def test_02_same_message_id_in_different_groups_do_not_block_each_other(self) -> None:
        # Both Group A and Group B send message_id=42
        upd_a = _make_update(
            chat_id=-100111, message_id=42, user_id=1, user_name="Alice",
            text="@AIChatBot_12_bot Group A message 42",
        )
        upd_b = _make_update(
            chat_id=-100222, message_id=42, user_id=2, user_name="Bob",
            text="@AIChatBot_12_bot Group B message 42",
        )

        await self.handlers.respond(upd_a, self.context)
        await self.handlers.respond(upd_b, self.context)

        upd_a.effective_message.reply_text.assert_awaited_once()
        upd_b.effective_message.reply_text.assert_awaited_once()

        # Verify storage kept both (chat_id, 42) records separately
        msg_a = self.repo.get_message(-100111, 42)
        msg_b = self.repo.get_message(-100222, 42)
        self.assertIsNotNone(msg_a)
        self.assertIsNotNone(msg_b)
        self.assertIn("Group A message 42", msg_a.text)
        self.assertIn("Group B message 42", msg_b.text)

    async def test_03_simultaneous_replies_in_both_groups(self) -> None:
        bot_msg_a = MagicMock()
        bot_msg_a.message_id = 500
        bot_msg_a.text = "Group A bot prior answer about cricket"
        bot_msg_a.from_user = self.bot
        bot_msg_a.reply_to_message = None
        bot_msg_a.message_thread_id = None
        bot_msg_a.photo = []
        bot_msg_a.document = None

        bot_msg_b = MagicMock()
        bot_msg_b.message_id = 500  # Same message_id in Group B!
        bot_msg_b.text = "Group B bot prior answer about anime"
        bot_msg_b.from_user = self.bot
        bot_msg_b.reply_to_message = None
        bot_msg_b.message_thread_id = None
        bot_msg_b.photo = []
        bot_msg_b.document = None

        upd_a = _make_update(
            chat_id=-100111, message_id=501, user_id=10, user_name="Dre",
            text="What about Kohli?", reply_to_message=bot_msg_a,
        )
        upd_b = _make_update(
            chat_id=-100222, message_id=501, user_id=20, user_name="Pratik",
            text="What about season 2?", reply_to_message=bot_msg_b,
        )

        # Fire both simultaneously
        await asyncio.gather(
            self.handlers.respond(upd_a, self.context),
            self.handlers.respond(upd_b, self.context),
        )

        upd_a.effective_message.reply_text.assert_awaited_once()
        upd_b.effective_message.reply_text.assert_awaited_once()

        # Verify each AI call only saw its own group's history
        calls_by_chat = {
            call.args[1].chat_id: [m.text for m in call.args[2]]
            for call in self.ai.answer.call_args_list
        }
        self.assertIn("Group A bot prior answer about cricket", " ".join(calls_by_chat[-100111]))
        self.assertNotIn("anime", " ".join(calls_by_chat[-100111]))
        self.assertIn("Group B bot prior answer about anime", " ".join(calls_by_chat[-100222]))
        self.assertNotIn("cricket", " ".join(calls_by_chat[-100222]))

    async def test_04_independent_group_settings_and_no_memory_leak(self) -> None:
        # Group A sets mode=off, memory=off; Group B sets mode=always, memory=on
        self.repo.save_settings(
            GroupSettings(chat_id=-100111, participation_mode=ParticipationMode.OFF, memory_enabled=False)
        )
        self.repo.save_settings(
            GroupSettings(chat_id=-100222, participation_mode=ParticipationMode.ALWAYS, memory_enabled=True)
        )

        # Seed secret message in Group A
        self.repo.add_context(
            -100111,
            ConversationMessage(author="Alice", text="SECRET_GROUP_A_TOKEN", message_id=77, sender_id=1),
        )

        upd_a = _make_update(
            chat_id=-100111, message_id=78, user_id=1, user_name="Alice",
            text="@AIChatBot_12_bot Hello from A",
        )
        upd_b = _make_update(
            chat_id=-100222, message_id=78, user_id=2, user_name="Bob",
            text="Casual message in B without mention",
        )

        await self.handlers.respond(upd_a, self.context)
        await self.handlers.respond(upd_b, self.context)

        # Group A is OFF -> no reply; Group B is ALWAYS -> replies without mention
        upd_a.effective_message.reply_text.assert_not_awaited()
        upd_b.effective_message.reply_text.assert_awaited_once()

        # Verify Group A memory did NOT leak into Group B's context
        b_history = self.ai.answer.call_args.args[2]
        self.assertNotIn("SECRET_GROUP_A_TOKEN", " ".join(m.text for m in b_history))

    async def test_05_duplicate_message_delivery_in_same_group_is_deduplicated(self) -> None:
        upd_1 = _make_update(
            chat_id=-100111, message_id=99, user_id=1, user_name="Alice",
            text="@AIChatBot_12_bot Hello once",
        )
        upd_dup = _make_update(
            chat_id=-100111, message_id=99, user_id=1, user_name="Alice",
            text="@AIChatBot_12_bot Hello once",
        )

        await self.handlers.respond(upd_1, self.context)
        await self.handlers.respond(upd_dup, self.context)

        upd_1.effective_message.reply_text.assert_awaited_once()
        upd_dup.effective_message.reply_text.assert_not_awaited()

    async def test_06_anonymous_group_admin_message_is_accepted_while_other_bots_ignored(self) -> None:
        anon_upd = _make_update(
            chat_id=-100333, message_id=5, user_id=1087968824, user_name="Group",
            text="@AIChatBot_12_bot Status check", is_bot=True, sender_chat=MagicMock(id=-100333),
        )
        other_bot_upd = _make_update(
            chat_id=-100333, message_id=6, user_id=555555, user_name="OtherBot",
            text="@AIChatBot_12_bot Spam", is_bot=True, sender_chat=None,
        )

        await self.handlers.respond(anon_upd, self.context)
        await self.handlers.respond(other_bot_upd, self.context)

        anon_upd.effective_message.reply_text.assert_awaited_once()
        other_bot_upd.effective_message.reply_text.assert_not_awaited()

    async def test_07_webhook_deduplicates_update_id_and_accepts_multiple_groups(self) -> None:
        import api.telegram as tg_api

        tg_api._seen_update_ids.clear()
        processed_chat_ids: list[int] = []

        async def _fake_process_update(update) -> None:
            if update and update.effective_chat:
                processed_chat_ids.append(update.effective_chat.id)

        payload_a = {
            "update_id": 9001,
            "message": {
                "message_id": 12,
                "date": 1795531320,
                "chat": {"id": -100111, "type": "supergroup", "title": "Group A"},
                "from": {"id": 1, "is_bot": False, "first_name": "Alice"},
                "text": "@AIChatBot_12_bot hi from A",
            },
        }
        payload_b = {
            "update_id": 9002,
            "message": {
                "message_id": 12,  # Same message_id in Group B
                "date": 1795531321,
                "chat": {"id": -100222, "type": "supergroup", "title": "Group B"},
                "from": {"id": 2, "is_bot": False, "first_name": "Bob"},
                "text": "@AIChatBot_12_bot hi from B",
            },
        }

        def _make_req(body: dict) -> MagicMock:
            req = MagicMock()
            req.json = AsyncMock(return_value=body)
            return req

        secret = tg_api.settings.telegram_webhook_secret or "test-secret"
        fake_app = MagicMock()
        fake_app.bot = tg_api.telegram_application.bot
        fake_app.process_update = AsyncMock(side_effect=_fake_process_update)
        with (
            patch.object(tg_api, "_ensure_initialized", AsyncMock()),
            patch.object(tg_api, "telegram_application", fake_app),
            patch.object(tg_api, "settings", MagicMock(telegram_webhook_secret=secret)),
        ):
            await tg_api.receive_update(_make_req(payload_a), telegram_secret=secret)
            # Duplicate webhook delivery of 9001 must be ignored
            await tg_api.receive_update(_make_req(payload_a), telegram_secret=secret)
            # Distinct group update 9002 (same message_id=12) must be processed
            await tg_api.receive_update(_make_req(payload_b), telegram_secret=secret)

        self.assertEqual(processed_chat_ids, [-100111, -100222])


if __name__ == "__main__":
    unittest.main(verbosity=2)

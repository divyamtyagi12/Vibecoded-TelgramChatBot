"""Strict group-isolation regression tests (Group A vs Group B)."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from telegram.constants import ChatType

from telegram_ai.context import assemble_context, filter_messages_for_chat
from telegram_ai.domain import ConversationMessage, GroupSettings
from telegram_ai.handlers import TelegramAIHandlers
from telegram_ai.openrouter import OpenRouterClient
from telegram_ai.storage import SQLiteRepository

GROUP_A_ID = -1001001
GROUP_B_ID = -1002002

GROUP_A_MEMBERS = ("JD", "Nikhil", "Hardik", "Dre")
GROUP_B_MEMBERS = ("Pratik", "Akash", "Manish", "Ekansh")


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
) -> MagicMock:
    update = MagicMock()
    chat = MagicMock()
    chat.id = chat_id
    chat.type = ChatType.SUPERGROUP
    update.effective_chat = chat

    user = MagicMock()
    user.id = user_id
    user.full_name = user_name
    user.is_bot = False
    update.effective_user = user

    msg = MagicMock()
    msg.message_id = message_id
    msg.text = text
    msg.caption = None
    msg.entities = []
    msg.from_user = user
    msg.sender_chat = None
    msg.reply_to_message = reply_to_message
    msg.message_thread_id = None
    msg.quote = None
    msg.photo = []
    msg.document = None

    sent_reply = MagicMock()
    sent_reply.message_id = message_id + 5000
    msg.reply_text = AsyncMock(return_value=sent_reply)

    update.effective_message = msg
    return update


class TestStrictGroupIsolation(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.repo = SQLiteRepository(":memory:")
        self.client = OpenRouterClient(
            api_key="gsk_test_key",
            model="openai/gpt-oss-120b",
            timeout_seconds=10.0,
        )
        self.bot = _make_bot()
        self.tg_ctx = MagicMock()
        self.tg_ctx.bot = self.bot
        self.handlers = TelegramAIHandlers(self.repo, self.client, max_context=6)

        # Seed Group A messages (JD, Nikhil, Hardik, Dre)
        for idx, (uid, name, text) in enumerate(
            [
                (101, "JD", "Group A crypto update"),
                (102, "Nikhil", "Group A gaming night"),
                (103, "Hardik", "Group A gym plan"),
                (104, "Dre", "Group A stock talk"),
            ],
            start=1,
        ):
            self.repo.add_context(
                GROUP_A_ID,
                ConversationMessage(
                    author=name, text=text, message_id=idx, sender_id=uid, chat_id=GROUP_A_ID
                ),
            )

        # Seed Group B messages (Pratik, Akash, Manish, Ekansh) with IDENTICAL message_ids 1..4
        for idx, (uid, name, text) in enumerate(
            [
                (201, "Pratik", "Group B trip plan"),
                (202, "Akash", "Group B movie suggestion"),
                (203, "Manish", "Group B dinner spot"),
                (204, "Ekansh", "Group B cricket match"),
            ],
            start=1,
        ):
            self.repo.add_context(
                GROUP_B_ID,
                ConversationMessage(
                    author=name, text=text, message_id=idx, sender_id=uid, chat_id=GROUP_B_ID
                ),
            )

    async def test_01_group_a_ranking_contains_only_group_a_members(self) -> None:
        captured: list[list[dict]] = []

        async def _fake(messages, **kwargs):
            captured.append(messages)
            return {"content": "Group A ranking response"}

        upd = _make_update(
            chat_id=GROUP_A_ID, message_id=50, user_id=104, user_name="Dre",
            text="@AIChatBot_12_bot Rank people according to their humbleness here.",
        )
        with patch.object(self.client, "_chat_completion", AsyncMock(side_effect=_fake)):
            await self.handlers.respond(upd, self.tg_ctx)

        payload_text = " ".join(m["content"] for m in captured[0])
        for name in GROUP_A_MEMBERS:
            self.assertIn(name, payload_text)
        for name in GROUP_B_MEMBERS:
            self.assertNotIn(name, payload_text)

    async def test_02_group_b_ranking_contains_zero_group_a_members(self) -> None:
        captured: list[list[dict]] = []

        async def _fake(messages, **kwargs):
            captured.append(messages)
            return {"content": "Group B ranking response"}

        upd = _make_update(
            chat_id=GROUP_B_ID, message_id=50, user_id=201, user_name="Pratik",
            text="@AIChatBot_12_bot Rank people according to their humbleness here.",
        )
        with patch.object(self.client, "_chat_completion", AsyncMock(side_effect=_fake)):
            await self.handlers.respond(upd, self.tg_ctx)

        payload_text = " ".join(m["content"] for m in captured[0])
        for name in GROUP_B_MEMBERS:
            self.assertIn(name, payload_text)
        # Critical: NOT A SINGLE Group A member name may appear anywhere (not even in system prompt!)
        for name in GROUP_A_MEMBERS:
            self.assertNotIn(name, payload_text)

    async def test_03_simultaneous_messages_with_identical_message_ids(self) -> None:
        captured_by_prompt: dict[str, str] = {}

        async def _fake(messages, **kwargs):
            await asyncio.sleep(0.01)
            last_user = messages[-1]["content"]
            captured_by_prompt[last_user] = " ".join(m["content"] for m in messages)
            return {"content": f"Ack: {last_user}"}

        upd_a = _make_update(
            chat_id=GROUP_A_ID, message_id=77, user_id=101, user_name="JD",
            text="@AIChatBot_12_bot Simultaneous from A",
        )
        upd_b = _make_update(
            chat_id=GROUP_B_ID, message_id=77, user_id=201, user_name="Pratik",
            text="@AIChatBot_12_bot Simultaneous from B",
        )
        with patch.object(self.client, "_chat_completion", AsyncMock(side_effect=_fake)):
            await asyncio.gather(
                self.handlers.respond(upd_a, self.tg_ctx),
                self.handlers.respond(upd_b, self.tg_ctx),
            )

        upd_a.effective_message.reply_text.assert_awaited_once()
        upd_b.effective_message.reply_text.assert_awaited_once()
        for name in GROUP_B_MEMBERS:
            self.assertNotIn(name, captured_by_prompt["Simultaneous from A"])
        for name in GROUP_A_MEMBERS:
            self.assertNotIn(name, captured_by_prompt["Simultaneous from B"])

    async def test_04_separate_conversation_summaries_per_group(self) -> None:
        # Overflow max_messages=3 in both groups to trigger summary generation
        for i in range(10, 16):
            self.repo.add_context(
                GROUP_A_ID,
                ConversationMessage(author="Hardik", text=f"AlphaA_{i}", message_id=i, sender_id=103, chat_id=GROUP_A_ID),
            )
            self.repo.add_context(
                GROUP_B_ID,
                ConversationMessage(author="Ekansh", text=f"BetaB_{i}", message_id=i, sender_id=204, chat_id=GROUP_B_ID),
            )

        ctx_a = assemble_context(
            self.repo, GROUP_A_ID,
            ConversationMessage(author="JD", text="Next", message_id=99, chat_id=GROUP_A_ID),
            memory_enabled=True, max_messages=3, max_tokens=3000,
        )
        ctx_b = assemble_context(
            self.repo, GROUP_B_ID,
            ConversationMessage(author="Pratik", text="Next", message_id=99, chat_id=GROUP_B_ID),
            memory_enabled=True, max_messages=3, max_tokens=3000,
        )

        self.assertTrue(all(m.chat_id == GROUP_A_ID for m in ctx_a))
        self.assertTrue(all(m.chat_id == GROUP_B_ID for m in ctx_b))
        self.assertEqual(ctx_a[0].media_type, "summary")
        self.assertEqual(ctx_b[0].media_type, "summary")
        self.assertNotIn("BetaB_", ctx_a[0].text)
        self.assertNotIn("Pratik", ctx_a[0].text)
        self.assertNotIn("AlphaA_", ctx_b[0].text)
        self.assertNotIn("Hardik", ctx_b[0].text)

    async def test_05_shared_user_in_both_groups_stays_isolated(self) -> None:
        # Same Telegram user_id=999 participates in both Group A and Group B
        self.repo.add_context(
            GROUP_A_ID,
            ConversationMessage(author="SharedUser", text="Secret in Group A", message_id=200, sender_id=999, chat_id=GROUP_A_ID),
        )
        self.repo.add_context(
            GROUP_B_ID,
            ConversationMessage(author="SharedUser", text="Topic in Group B", message_id=200, sender_id=999, chat_id=GROUP_B_ID),
        )

        ctx_b = assemble_context(
            self.repo, GROUP_B_ID,
            ConversationMessage(author="SharedUser", text="Follow up", message_id=201, sender_id=999, chat_id=GROUP_B_ID),
            memory_enabled=True, max_messages=8, max_tokens=3000,
        )
        combined_b = " ".join(m.text for m in ctx_b)
        self.assertIn("Topic in Group B", combined_b)
        self.assertNotIn("Secret in Group A", combined_b)

    async def test_06_reply_chains_never_cross_groups_even_with_same_ids(self) -> None:
        # Chain in Group A: msg 300 -> 301
        self.repo.add_context(
            GROUP_A_ID,
            ConversationMessage(author="JD", text="A chain root", message_id=300, sender_id=101, chat_id=GROUP_A_ID),
        )
        self.repo.add_context(
            GROUP_A_ID,
            ConversationMessage(author="TelegramAI", text="A chain reply", message_id=301, reply_to_message_id=300, is_bot=True, chat_id=GROUP_A_ID),
        )
        # Chain in Group B with identical IDs 300 -> 301
        self.repo.add_context(
            GROUP_B_ID,
            ConversationMessage(author="Akash", text="B chain root", message_id=300, sender_id=202, chat_id=GROUP_B_ID),
        )
        self.repo.add_context(
            GROUP_B_ID,
            ConversationMessage(author="TelegramAI", text="B chain reply", message_id=301, reply_to_message_id=300, is_bot=True, chat_id=GROUP_B_ID),
        )

        ctx_a = assemble_context(
            self.repo, GROUP_A_ID,
            ConversationMessage(author="Dre", text="Continue", message_id=302, reply_to_message_id=301, chat_id=GROUP_A_ID),
            memory_enabled=True, max_messages=8, max_tokens=3000,
        )
        ctx_b = assemble_context(
            self.repo, GROUP_B_ID,
            ConversationMessage(author="Manish", text="Continue", message_id=302, reply_to_message_id=301, chat_id=GROUP_B_ID),
            memory_enabled=True, max_messages=8, max_tokens=3000,
        )

        self.assertIn("A chain root", " ".join(m.text for m in ctx_a))
        self.assertNotIn("B chain", " ".join(m.text for m in ctx_a))
        self.assertIn("B chain root", " ".join(m.text for m in ctx_b))
        self.assertNotIn("A chain", " ".join(m.text for m in ctx_b))

    async def test_07_group_b_asks_about_group_a_only_member_has_no_group_a_context(self) -> None:
        captured: list[list[dict]] = []

        async def _fake(messages, **kwargs):
            captured.append(messages)
            return {"content": "I haven't seen Hardik in this group."}

        upd = _make_update(
            chat_id=GROUP_B_ID, message_id=400, user_id=203, user_name="Manish",
            text="@AIChatBot_12_bot What did Hardik say earlier?",
        )
        with patch.object(self.client, "_chat_completion", AsyncMock(side_effect=_fake)):
            await self.handlers.respond(upd, self.tg_ctx)

        # Exclude the user's own question (last message) and verify Hardik's Group A message never leaked
        prior_context_text = " ".join(m["content"] for m in captured[0][:-1])
        self.assertNotIn("Group A gym plan", prior_context_text)
        self.assertNotIn("Hardik", prior_context_text)

    async def test_08_openrouter_client_rejects_injected_cross_chat_messages_and_summaries(self) -> None:
        poisoned_context = [
            ConversationMessage(author="ConversationSummary", text="Group A summary: JD, Nikhil", media_type="summary", chat_id=GROUP_A_ID),
            ConversationMessage(author="JD", text="Group A leaked message", message_id=1, chat_id=GROUP_A_ID),
            ConversationMessage(author="Pratik", text="Valid Group B message", message_id=1, chat_id=GROUP_B_ID),
        ]
        filtered = filter_messages_for_chat(poisoned_context, GROUP_B_ID)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].author, "Pratik")

        captured: list[list[dict]] = []

        async def _fake(messages, **kwargs):
            captured.append(messages)
            return {"content": "Clean Group B reply"}

        with patch.object(self.client, "_chat_completion", AsyncMock(side_effect=_fake)):
            await self.client.answer(
                "Rank people according to their humbleness here.",
                GroupSettings(chat_id=GROUP_B_ID),
                poisoned_context,
            )

        sent_text = " ".join(m["content"] for m in captured[0])
        self.assertIn("Valid Group B message", sent_text)
        self.assertNotIn("Group A leaked message", sent_text)
        self.assertNotIn("Group A summary", sent_text)
        self.assertNotIn("JD", sent_text)
        self.assertNotIn("Nikhil", sent_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)

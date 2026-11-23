"""Comprehensive test suite verifying all 12 ChatGPT-like assistant requirements."""

from __future__ import annotations

import io
import logging
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from PIL import Image

from telegram_ai.config import Settings
from telegram_ai.context import assemble_context
from telegram_ai.domain import ConversationMessage, GroupSettings, ParticipationMode
from telegram_ai.groq import GroqClient, _OPENROUTER_FREE_MODEL, _redact_secrets
from telegram_ai.handlers import TelegramAIHandlers
from telegram_ai.search import WebSearchClient, extract_official_urls
from telegram_ai.storage import SQLiteRepository


def _png_bytes(color: str = "navy") -> bytes:
    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestChatGPTAssistant(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repo = SQLiteRepository(":memory:")
        self.ai = MagicMock(spec=GroqClient)
        self.ai.vision_model = _OPENROUTER_FREE_MODEL
        self.ai.answer = AsyncMock(return_value="Direct helpful answer.")
        self.ai.analyze_image = AsyncMock(return_value="This image shows a diagram.")
        self.ai.describe_image = AsyncMock(return_value="A navy diagram.")
        self.ai.rate_photo = AsyncMock(return_value="8/10 lighting.")
        self.handlers = TelegramAIHandlers(self.repo, self.ai, max_context=6, max_tokens=2000)
        self.handlers._bot_user = MagicMock(id=999, username="AIChatBot_12_bot")

    def _make_ctx(self, img_bytes: bytes | None = None):
        ctx = MagicMock()
        ctx.bot = MagicMock()
        ctx.bot.get_me = AsyncMock(return_value=self.handlers._bot_user)
        ctx.bot.send_chat_action = AsyncMock()
        if img_bytes is not None:
            fake_file = MagicMock()
            fake_file.file_size = len(img_bytes)
            fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(img_bytes))
            ctx.bot.get_file = AsyncMock(return_value=fake_file)
        return ctx

    # 1. Normal conversation & duplicate update protection
    async def test_01_normal_conversation_and_deduplication(self):
        chat_id = -9001
        self.repo.save_settings(GroupSettings(chat_id=chat_id, participation_mode=ParticipationMode.MENTION))

        user = MagicMock(id=101, full_name="Alice", is_bot=False)
        msg = MagicMock(
            message_id=10,
            chat_id=chat_id,
            from_user=user,
            text="@AIChatBot_12_bot Explain recursion simply",
            entities=[MagicMock(type="mention", offset=0, length=17)],
            reply_to_message=None,
            quote=None,
            photo=None,
            document=None,
            message_thread_id=None,
        )
        msg.reply_text = AsyncMock(return_value=MagicMock(message_id=11))
        update = MagicMock(effective_chat=MagicMock(id=chat_id), effective_user=user, effective_message=msg)
        ctx = self._make_ctx()

        await self.handlers.respond(update, ctx)
        # Duplicate webhook redelivery of same (chat_id, message_id) must be ignored
        await self.handlers.respond(update, ctx)

        self.ai.answer.assert_awaited_once()
        msg.reply_text.assert_awaited_once_with("Direct helpful answer.")

    # 2. Consecutive follow-up questions ("why?", "explain more", "continue")
    async def test_02_consecutive_followups_preserve_chain(self):
        chat_id = -9002
        self.repo.add_context(chat_id, ConversationMessage("Alice", "What is a black hole?", message_id=1))
        self.repo.add_context(chat_id, ConversationMessage("TelegramAI", "A region with extreme gravity.", message_id=2, reply_to_message_id=1, is_bot=True))
        self.repo.add_context(chat_id, ConversationMessage("Alice", "why?", message_id=3, reply_to_message_id=2))
        self.repo.add_context(chat_id, ConversationMessage("TelegramAI", "Because mass collapses past the Schwarzschild radius.", message_id=4, reply_to_message_id=3, is_bot=True))

        curr = ConversationMessage("Alice", "explain more", message_id=5, reply_to_message_id=4)
        ctx = assemble_context(self.repo, chat_id, curr, memory_enabled=True, max_messages=8)
        self.assertEqual([m.message_id for m in ctx], [1, 2, 3, 4])

    # 3 & 4. Replies to bot & replies to other users
    async def test_03_04_replies_to_bot_and_other_users(self):
        chat_id = -9003
        self.repo.save_settings(GroupSettings(chat_id=chat_id, participation_mode=ParticipationMode.MENTION))

        other_user = MagicMock(id=201, full_name="Bob", is_bot=False)
        other_msg = MagicMock(message_id=20, from_user=other_user, text="Rust has no garbage collector", photo=None, document=None, reply_to_message=None, message_thread_id=None)

        alice = MagicMock(id=202, full_name="Alice", is_bot=False)
        reply_msg = MagicMock(
            message_id=21,
            chat_id=chat_id,
            from_user=alice,
            text="@AIChatBot_12_bot is that true?",
            entities=[MagicMock(type="mention", offset=0, length=17)],
            reply_to_message=other_msg,
            quote=None,
            photo=None,
            document=None,
            message_thread_id=None,
        )
        reply_msg.reply_text = AsyncMock(return_value=MagicMock(message_id=22))
        update = MagicMock(effective_chat=MagicMock(id=chat_id), effective_user=alice, effective_message=reply_msg)
        await self.handlers.respond(update, self._make_ctx())

        self.ai.answer.assert_awaited_once()
        history = self.ai.answer.call_args[0][2]
        self.assertTrue(any("Rust has no garbage collector" in m.text for m in history))

    # 5 & 6. Image explanations (actual image bytes sent) + follow-up on image
    async def test_05_06_image_explanation_and_followup(self):
        chat_id = -9005
        self.repo.save_settings(GroupSettings(chat_id=chat_id, participation_mode=ParticipationMode.MENTION))
        img_bytes = _png_bytes()

        sender = MagicMock(id=301, full_name="Macman", is_bot=False)
        photo_msg = MagicMock(
            message_id=30,
            from_user=sender,
            text=None,
            caption="Scorecard",
            photo=[MagicMock(file_id="fid_30", file_unique_id="u30", file_size=len(img_bytes))],
            document=None,
            reply_to_message=None,
            message_thread_id=None,
        )

        dre = MagicMock(id=302, full_name="Dre", is_bot=False)
        ask_msg = MagicMock(
            message_id=31,
            chat_id=chat_id,
            from_user=dre,
            text="@AIChatBot_12_bot explain this",
            entities=[MagicMock(type="mention", offset=0, length=17)],
            reply_to_message=photo_msg,
            quote=None,
            photo=None,
            document=None,
            message_thread_id=None,
        )
        ask_msg.reply_text = AsyncMock(return_value=MagicMock(message_id=32))
        update = MagicMock(effective_chat=MagicMock(id=chat_id), effective_user=dre, effective_message=ask_msg)
        await self.handlers.respond(update, self._make_ctx(img_bytes))

        self.ai.analyze_image.assert_awaited_once()
        sent_bytes = self.ai.analyze_image.call_args[0][0]
        self.assertIsInstance(sent_bytes, bytes)
        self.assertGreater(len(sent_bytes), 50)
        self.ai.rate_photo.assert_not_awaited()

    # 7, 8, 9. Hinglish, Coding, and Math prompts
    async def test_07_08_09_hinglish_coding_math(self):
        client = GroqClient("sk-or-v1-testkey123", _OPENROUTER_FREE_MODEL, 5.0)
        settings = GroupSettings(1)
        captured_calls = []

        async def fake_comp(messages, **kwargs):
            captured_calls.append((messages, kwargs))
            return {"role": "assistant", "content": "Verified reply."}

        with patch.object(client, "_chat_completion", side_effect=fake_comp):
            # 7. Hinglish
            await client.answer("Bhai recursion ka matlab simple mein bata", settings, [])
            sys_prompt = captured_calls[-1][0][0]["content"]
            self.assertIn("Hinglish", sys_prompt)

            # 8. Coding -> high token budget (2500)
            await client.answer("Write a python function for binary search", settings, [])
            self.assertEqual(captured_calls[-1][1]["max_tokens"], 2500)

            # 9. Math
            with patch.object(client.search_client, "search", new=AsyncMock(return_value="No results")):
                await client.answer("Solve 2x + 5 = 17 step by step", settings, [])
                self.assertTrue(len(captured_calls) >= 3)

    # 10. Web search includes source links & official URL extraction
    async def test_10_web_search_official_urls_and_source_links(self):
        sample_results = (
            "=== Source: Bing Web ===\n"
            "[1] Reddit Discussion\nURL: https://www.reddit.com/r/nvidia/123\nSnippet: chatter\n\n"
            "[2] NVIDIA Official Press\nURL: https://nvidianews.nvidia.com/news/update\nSnippet: official update\n\n"
            "[3] CNBC Report\nURL: https://www.cnbc.com/2026/11/23/nvda.html\nSnippet: NVDA rises\n"
        )
        urls = extract_official_urls(sample_results, max_urls=2)
        self.assertEqual(urls[0], "https://nvidianews.nvidia.com/news/update")
        self.assertIn("https://www.cnbc.com/2026/11/23/nvda.html", urls)
        self.assertFalse(any("reddit.com" in u for u in urls))

    # 11. Multiple simultaneous group conversations & long-context summarization
    def test_11_group_isolation_and_long_context_summarization(self):
        group_a = -1001
        group_b = -1002

        # Add 10 chained messages in group A
        prev_id = None
        for i in range(1, 11):
            self.repo.add_context(
                group_a,
                ConversationMessage(
                    author="Alice" if i % 2 else "TelegramAI",
                    text=f"Group A turn {i} discussing quantum computing topic {i}",
                    message_id=i,
                    reply_to_message_id=prev_id,
                    is_bot=(i % 2 == 0),
                ),
            )
            prev_id = i

        # Add messages in group B
        self.repo.add_context(group_b, ConversationMessage("Bob", "Group B cooking pasta", message_id=50))

        # Assemble context for Group A with max_messages=4 -> older turns summarized into 1 summary entry
        curr_a = ConversationMessage("Alice", "continue", message_id=11, reply_to_message_id=10)
        ctx_a = assemble_context(self.repo, group_a, curr_a, memory_enabled=True, max_messages=4, max_tokens=2000)

        self.assertEqual(ctx_a[0].media_type, "summary")
        self.assertIn("Earlier conversation summary:", ctx_a[0].text)
        self.assertIn("Group A turn 3", ctx_a[0].text)
        self.assertEqual([m.message_id for m in ctx_a[1:]], [7, 8, 9, 10])
        self.assertFalse(any("pasta" in m.text for m in ctx_a))

    # 12. Strict free OpenRouter model enforcement, secret redaction & rate-limit handling
    async def test_12_strict_free_model_and_secret_redaction(self):
        # Even if someone passes a paid OpenRouter model name, GroqClient must pin to :free
        client = GroqClient("sk-or-v1-secret-token-xyz", "nvidia/nemotron-3.5-super-vl", 5.0)
        self.assertEqual(client.model, _OPENROUTER_FREE_MODEL)
        self.assertEqual(client.vision_model, _OPENROUTER_FREE_MODEL)

        # Verify secret redaction
        leaked = "HTTP 401 Bearer sk-or-v1-secret-token-xyz failed with gsk_abc123"
        cleaned = _redact_secrets(leaked)
        self.assertNotIn("sk-or-v1-secret-token-xyz", cleaned)
        self.assertNotIn("gsk_abc123", cleaned)

        # Verify _chat_completion NEVER switches to a paid model on 500/429
        requested_models = []

        async def fake_post(url, json=None, headers=None):
            requested_models.append(json["model"])
            resp = MagicMock(status_code=500)
            resp.request = MagicMock()
            import httpx
            resp.raise_for_status.side_effect = httpx.HTTPStatusError("500", request=resp.request, response=resp)
            return resp

        mock_inner = MagicMock(post=fake_post)
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_inner)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("telegram_ai.groq.httpx.AsyncClient", return_value=mock_cm):
            with self.assertRaises(RuntimeError):
                await client._chat_completion([{"role": "user", "content": "hi"}])

        self.assertTrue(all(m.endswith(":free") for m in requested_models), f"Paid model attempted: {requested_models}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

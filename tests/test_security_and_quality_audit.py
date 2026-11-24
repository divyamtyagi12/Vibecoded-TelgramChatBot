"""Comprehensive security, search precision, formatting, chunking, and performance audit tests (mocked/offline)."""

from __future__ import annotations

import logging
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from telegram_ai.context import assemble_context
from telegram_ai.domain import ConversationMessage, GroupSettings
from telegram_ai.handlers import _reply_text_safe, _split_telegram_message
from telegram_ai.openrouter import (
    OpenRouterClient,
    _clean_chat_response,
    needs_web_search,
)
from telegram_ai.search import WebSearchClient, _is_safe_public_url
from telegram_ai.storage import SQLiteRepository
from telegram_ai.vision import download_telegram_image


class TestSecurityAudit(unittest.IsolatedAsyncioTestCase):
    """Offline security tests for SSRF, log redaction, webhook validation, and unsafe image rejection."""

    def test_01_ssrf_blocks_metadata_localhost_and_private_ips(self) -> None:
        unsafe_urls = [
            "http://169.254.169.254/latest/meta-data/",
            "http://127.0.0.1:8080/admin",
            "http://localhost/secret",
            "http://[::1]/",
            "http://10.0.0.1/internal",
            "http://192.168.1.1/router",
            "http://172.16.0.1/private",
            "http://metadata.google.internal/computeMetadata/v1/",
            "file:///etc/passwd",
            "ftp://example.com/file",
            "http://example.com:8443/nonstandard",
        ]
        for url in unsafe_urls:
            self.assertFalse(_is_safe_public_url(url), f"Expected SSRF block for {url}")

    async def test_02_fetch_page_blocks_ssrf_without_network_call(self) -> None:
        sc = WebSearchClient(timeout_seconds=3.0)
        res = await sc.fetch_page("http://169.254.169.254/latest/meta-data/")
        self.assertIn("Blocked unsafe or private URL", res)

    def test_03_webhook_log_filter_redacts_telegram_token_and_api_keys(self) -> None:
        import api.telegram as tg_api

        self.assertEqual(logging.getLogger("httpx").level, logging.WARNING)
        self.assertEqual(logging.getLogger("httpcore").level, logging.WARNING)

        filt = tg_api._SecretRedactingFilter()
        rec = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="POST https://api.telegram.org/bot123456:ABCdef_999/sendMessage key=gsk_secret123 Bearer tok_abc",
            args=(),
            exc_info=None,
        )
        filt.filter(rec)
        rendered = rec.getMessage()
        self.assertNotIn("123456:ABCdef_999", rendered)
        self.assertNotIn("gsk_secret123", rendered)
        self.assertNotIn("tok_abc", rendered)
        self.assertIn("bot[REDACTED_TOKEN]", rendered)

    async def test_04_webhook_rejects_malformed_json_and_non_dict_payloads(self) -> None:
        import api.telegram as tg_api

        secret = "audit-secret"
        bad_json_req = MagicMock()
        bad_json_req.json = AsyncMock(side_effect=ValueError("bad json"))
        list_req = MagicMock()
        list_req.json = AsyncMock(return_value=["not", "a", "dict"])

        with (
            patch.object(tg_api, "_ensure_initialized", AsyncMock()),
            patch.object(tg_api, "settings", MagicMock(telegram_webhook_secret=secret)),
        ):
            with self.assertRaises(HTTPException) as ctx1:
                await tg_api.receive_update(bad_json_req, telegram_secret=secret)
            self.assertEqual(ctx1.exception.status_code, 400)

            with self.assertRaises(HTTPException) as ctx2:
                await tg_api.receive_update(list_req, telegram_secret=secret)
            self.assertEqual(ctx2.exception.status_code, 400)

    async def test_05_vision_download_rejects_html_and_svg_payloads(self) -> None:
        for payload in (b"<!DOCTYPE html><html>xss</html>", b"<svg onload=alert(1)></svg>", b"%PDF-1.7"):
            fake_file = MagicMock()
            fake_file.file_size = len(payload)
            fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(payload))
            bot = MagicMock()
            bot.get_file = AsyncMock(return_value=fake_file)

            img_bytes, mime, err = await download_telegram_image(bot, "fid_bad", max_retries=1)
            self.assertIsNone(img_bytes)
            self.assertIsNone(mime)
            self.assertEqual(err, "invalid_format")


class TestQualityReliabilityAndPerformance(unittest.IsolatedAsyncioTestCase):
    """Offline quality, search-precision, tool-call sanitization, chunking, and latency tests."""

    def test_01_search_precision_blocks_false_positives_and_catches_live_queries(self) -> None:
        must_not_search = [
            "How does Python match-case work?",
            "What is an F1 score in machine learning?",
            "How do you train a neural network?",
            "Explain Taylor series in calculus",
            "Explain price elasticity of demand",
            "Write a Python script using ChatGPT API",
            "Hey, what's up?",
            "Explain recursion simply",
        ]
        for q in must_not_search:
            self.assertFalse(needs_web_search(q, []), f"False positive search on: {q}")

        must_search = [
            "Latest Gemini model ranking 2026",
            "India cricket match live score today",
            "nvidia stock price today",
            "Latest Solo Leveling episode release date",
            "F1 timing qualifying today",
            "live running status train 12004",
        ]
        for q in must_search:
            self.assertTrue(needs_web_search(q, []), f"Missed live search on: {q}")

    def test_02_latex_cleanup_preserves_currency_and_code_blocks(self) -> None:
        raw = "Ball = $0.05 and NVDA = $155.13. Solve $2x + 6 = 18$: $$x = \\frac{12}{2}$$ so $x = 6$."
        cleaned = _clean_chat_response(raw)
        self.assertIn("$0.05", cleaned)
        self.assertIn("$155.13", cleaned)
        self.assertNotIn("$$", cleaned)
        self.assertNotIn("\\frac", cleaned)
        self.assertIn("12/2", cleaned)
        self.assertIn("x = 6", cleaned)

        code_msg = "```bash\necho $HOME and $PATH\n```"
        self.assertEqual(_clean_chat_response(code_msg), code_msg)

    async def test_03_telegram_4096_char_chunking_splits_cleanly(self) -> None:
        long_text = ("Line of explanation about algorithms.\n" * 160).strip()
        self.assertGreater(len(long_text), 4096)
        chunks = _split_telegram_message(long_text, limit=4000)
        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 4000)

        msg = MagicMock()
        msg.reply_text = AsyncMock(side_effect=[MagicMock(message_id=1), MagicMock(message_id=2)])
        await _reply_text_safe(msg, long_text)
        self.assertEqual(msg.reply_text.await_count, len(chunks))

    async def test_04_tool_call_strips_provider_reasoning_field_and_finishes_in_two_calls(self) -> None:
        search_client = MagicMock(spec=WebSearchClient)
        search_client.search = AsyncMock(return_value="NVDA = 155.13 USD\nURL: https://finance.yahoo.com/quote/NVDA/")
        client = OpenRouterClient("gsk_test", "openai/gpt-oss-120b", 10.0, search_client=search_client)

        sent_payloads: list[list[dict]] = []

        async def _fake_completion(messages, **kwargs):
            sent_payloads.append(list(messages))
            if len(sent_payloads) == 1:
                return {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "internal groq reasoning that triggers 400 if echoed back",
                    "tool_calls": [
                        {
                            "id": "call_nvda",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": '{"query": "NVDA stock price"}'},
                        }
                    ],
                }
            # Second call must NOT receive tools schema again
            self.assertNotIn("tools", kwargs)
            return {"role": "assistant", "content": "NVDA is trading at $155.13 USD (https://finance.yahoo.com/quote/NVDA/)."}

        with patch.object(client, "_chat_completion", AsyncMock(side_effect=_fake_completion)):
            ans = await client.answer("nvidia stock price today", GroupSettings(chat_id=1), [])

        self.assertEqual(len(sent_payloads), 2)
        self.assertIn("$155.13", ans)
        assistant_tool_msg = next(m for m in sent_payloads[1] if m.get("role") == "assistant")
        self.assertNotIn("reasoning", assistant_tool_msg)

    async def test_05_coding_followup_gets_expanded_token_budget(self) -> None:
        client = OpenRouterClient("gsk_test", "openai/gpt-oss-120b", 10.0)
        captured_tokens: list[int] = []

        async def _fake(messages, **kwargs):
            captured_tokens.append(kwargs["max_tokens"])
            return {"content": "Time complexity is O(n), space complexity is O(1)."}

        code_history = [
            ConversationMessage("Dev", "Debug this Python function:\n```python\ndef f(x): return x\n```", message_id=1, chat_id=1),
            ConversationMessage("TelegramAI", "```python\ndef f(x): return x\n```", message_id=2, is_bot=True, chat_id=1),
        ]
        with patch.object(client, "_chat_completion", AsyncMock(side_effect=_fake)):
            await client.answer("What is its time and space complexity?", GroupSettings(chat_id=1), code_history)

        self.assertEqual(captured_tokens[-1], 600)

    def test_06_local_performance_budgets_context_and_search_detection(self) -> None:
        repo = SQLiteRepository(":memory:")
        for i in range(1, 30):
            repo.add_context(
                -1009,
                ConversationMessage(author=f"U{i % 4}", text=f"Chat turn {i}", message_id=i, chat_id=-1009),
            )
        cur = ConversationMessage(author="U1", text="What do you think?", message_id=30, chat_id=-1009)

        t0 = time.perf_counter()
        ctx = assemble_context(repo, -1009, cur, memory_enabled=True, max_messages=8, max_tokens=3000)
        ctx_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        for _ in range(100):
            needs_web_search("How does Python match-case work?", ctx)
        detect_ms_avg = ((time.perf_counter() - t1) * 1000) / 100

        self.assertLess(ctx_ms, 50.0, f"Context assembly too slow: {ctx_ms:.2f}ms")
        self.assertLess(detect_ms_avg, 1.0, f"Search detection too slow: {detect_ms_avg:.3f}ms")

    async def test_07_empty_content_reasoning_starvation_recovers_via_retry(self) -> None:
        client = OpenRouterClient("gsk_test", "openai/gpt-oss-120b", 10.0)
        calls: list[dict] = []

        async def _fake(messages, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return {"role": "assistant", "content": "", "reasoning": "ran out of tokens thinking"}
            return {"role": "assistant", "content": "Hardik Sinha is the impostor because 'Khata hu' is not a food."}

        with patch.object(client, "_chat_completion", AsyncMock(side_effect=_fake)):
            ans = await client.answer("All Clues Revealed! Which food is the impostor?", GroupSettings(chat_id=1), [])

        self.assertEqual(len(calls), 2)
        self.assertIn("Hardik Sinha", ans)
        self.assertGreaterEqual(calls[1]["max_tokens"], 800)


if __name__ == "__main__":
    unittest.main(verbosity=2)

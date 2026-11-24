"""Regression tests for contextual understanding, follow-ups, and user corrections."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from telegram_ai.domain import ConversationMessage, GroupSettings
from telegram_ai.openrouter import (
    OpenRouterClient,
    _build_contextual_user_prompt,
    needs_web_search,
)
from telegram_ai.search import WebSearchClient


class TestContextUnderstanding(unittest.IsolatedAsyncioTestCase):
    def _make_client(self):
        search = MagicMock(spec=WebSearchClient)
        search.search = AsyncMock(
            return_value=(
                "=== Source: Bing Web ===\n"
                "[1] Google Gemini Models\n"
                "URL: https://ai.google.dev/gemini-api/docs/models\n"
                "Snippet: Latest Gemini models and experimental codenames including Argon, Hydrogen, Neon."
            )
        )
        search.fetch_page = AsyncMock(return_value="Official docs excerpt.")
        return OpenRouterClient("gsk_test_key", "openai/gpt-oss-120b", 10.0, search_client=search), search

    # 1. Gemini ranking -> "What about Argon Hydrogen Neon?" -> "Model bhi h Gemini k."
    async def test_01_gemini_ranking_argon_hydrogen_neon_and_correction(self):
        client, search = self._make_client()
        settings = GroupSettings(1)
        history = [
            ConversationMessage("Dre", "Gemini ka best model ranking?", message_id=1),
            ConversationMessage("TelegramAI", "Gemini 2.5 Pro and Flash are currently top.", message_id=2, is_bot=True),
        ]

        # Step A: "What about Argon Hydrogen Neon?" must inherit Gemini context AND trigger web search
        self.assertTrue(needs_web_search("What about Argon Hydrogen Neon?", history))
        anchored = _build_contextual_user_prompt("What about Argon Hydrogen Neon?", history)
        self.assertIn("Gemini ka best model ranking?", anchored)
        self.assertIn("Interpret ambiguous terms strictly in this conversation context", anchored)

        # Verify proactive search query combines Gemini topic + Argon Hydrogen Neon
        with patch.object(
            client,
            "_chat_completion",
            side_effect=[
                {"role": "assistant", "content": "Argon, Hydrogen, and Neon are elements."},
                {"role": "assistant", "content": "Argon, Hydrogen, and Neon are internal/experimental Gemini codenames, not standard public releases. https://ai.google.dev/gemini-api/docs/models"},
            ],
        ):
            reply = await client.answer("What about Argon Hydrogen Neon?", settings, history)
            self.assertIn("codenames", reply)
            search.search.assert_awaited()
            query_used = search.search.call_args[0][0]
            self.assertIn("Gemini", query_used)
            self.assertIn("Argon Hydrogen Neon", query_used)

        # Step B: User correction "Model bhi h Gemini k." links back to the earlier Argon/Hydrogen/Neon question
        history_with_mistake = history + [
            ConversationMessage("Dre", "What about Argon Hydrogen Neon?", message_id=3),
            ConversationMessage("TelegramAI", "Argon, Hydrogen, and Neon are chemical elements.", message_id=4, is_bot=True),
        ]
        self.assertTrue(needs_web_search("Model bhi h Gemini k.", history_with_mistake))
        corrected = _build_contextual_user_prompt("Model bhi h Gemini k.", history_with_mistake)
        self.assertIn("User correction to your previous reply", corrected)
        self.assertIn("What about Argon Hydrogen Neon?", corrected)
        self.assertIn("do NOT repeat your old answer", corrected)

    # 2. AI models -> "What about Claude?"
    def test_02_ai_models_what_about_claude(self):
        history = [
            ConversationMessage("Alice", "Best AI models right now?", message_id=1),
            ConversationMessage("TelegramAI", "OpenAI and Google lead on several benchmarks.", message_id=2, is_bot=True),
        ]
        self.assertTrue(needs_web_search("What about Claude?", history))
        prompt = _build_contextual_user_prompt("What about Claude?", history)
        self.assertIn("Best AI models right now?", prompt)

    # 3. Cricket discussion -> "What about Kohli?"
    def test_03_cricket_what_about_kohli(self):
        history = [
            ConversationMessage("Bob", "How did Rohit play in today's cricket match?", message_id=10),
            ConversationMessage("TelegramAI", "Rohit scored 64 off 48 balls.", message_id=11, is_bot=True),
        ]
        self.assertTrue(needs_web_search("What about Kohli?", history))
        prompt = _build_contextual_user_prompt("What about Kohli?", history)
        self.assertIn("cricket match", prompt)

    # 4. Anime discussion -> "What about season 2?"
    def test_04_anime_what_about_season_2(self):
        history = [
            ConversationMessage("Alice", "Solo Leveling season 1 was amazing", message_id=20),
            ConversationMessage("TelegramAI", "Yeah, the animation and pacing were great.", message_id=21, is_bot=True),
        ]
        self.assertFalse(needs_web_search("What about season 2?", history))
        prompt = _build_contextual_user_prompt("What about season 2?", history)
        self.assertIn("Solo Leveling season 1", prompt)

    # 5. Coding discussion -> "Why this approach?"
    def test_05_coding_why_this_approach(self):
        history = [
            ConversationMessage("Dev", "Write binary search in Python", message_id=30),
            ConversationMessage("TelegramAI", "```python\n# iterative binary search\n```", message_id=31, is_bot=True),
        ]
        self.assertFalse(needs_web_search("Why this approach?", history))
        prompt = _build_contextual_user_prompt("Why this approach?", history)
        self.assertIn("Write binary search in Python", prompt)

    # 6. User correction -> "I meant the model."
    def test_06_user_correction_i_meant_the_model(self):
        history = [
            ConversationMessage("Dre", "Compare Gemini models", message_id=40),
            ConversationMessage("TelegramAI", "Here are the main Gemini tiers.", message_id=41, is_bot=True),
            ConversationMessage("Dre", "What about Neon?", message_id=42),
            ConversationMessage("TelegramAI", "Neon is a noble gas.", message_id=43, is_bot=True),
        ]
        prompt = _build_contextual_user_prompt("I meant the model.", history)
        self.assertIn("User correction", prompt)
        self.assertIn("What about Neon?", prompt)
        self.assertIn("Compare Gemini models", prompt)

    # 7. Ambiguous follow-up with no sufficient context
    def test_07_ambiguous_followup_without_context(self):
        self.assertFalse(needs_web_search("What about season 2?", []))
        prompt = _build_contextual_user_prompt("What about season 2?", [])
        self.assertEqual(prompt, "What about season 2?")

    # 8. Stock query with dates resolves ticker cleanly and cleans fake citation brackets
    def test_08_stock_date_query_and_citation_cleanup(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch
        from telegram_ai.openrouter import _clean_chat_response
        from telegram_ai.search import WebSearchClient

        cleaned = _clean_chat_response("NVDA is at $155.13【3†L1-L3】.")
        self.assertEqual(cleaned, "NVDA is at $155.13.")

        mock_search_resp = MagicMock()
        mock_search_resp.status_code = 200
        mock_search_resp.json.return_value = {
            "quotes": [{"symbol": "NVDA", "shortname": "NVIDIA Corporation"}],
            "news": [],
        }
        mock_chart_resp = MagicMock()
        mock_chart_resp.status_code = 200
        mock_chart_resp.json.return_value = {
            "chart": {
                "result": [
                    {
                        "meta": {
                            "regularMarketPrice": 155.13,
                            "chartPreviousClose": 165.05,
                            "currency": "USD",
                            "regularMarketTime": 1795531320,
                        }
                    }
                ]
            }
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[mock_search_resp, mock_chart_resp])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        async def _run():
            with patch("telegram_ai.search.httpx.AsyncClient", return_value=mock_client):
                sc = WebSearchClient(timeout_seconds=5.0)
                return await sc._search_yahoo_finance("NVIDIA stock price today November 24 2026")

        res = asyncio.run(_run())
        self.assertIsNotNone(res)
        self.assertIn("NVDA", res)
        self.assertIn("155.13 USD", res)
        self.assertIn("https://finance.yahoo.com/quote/NVDA/", res)
        # Verify the Yahoo search query was clean ("NVDA"), not polluted by "24" or "November"
        first_call_params = mock_client.get.call_args_list[0].kwargs.get("params", {})
        self.assertEqual(first_call_params.get("q"), "NVDA")


if __name__ == "__main__":
    unittest.main(verbosity=2)

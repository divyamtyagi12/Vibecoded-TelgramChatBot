"""Tests for real-time web search, live score, train tracking, and Groq tool calling."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from telegram_ai.domain import ConversationMessage, GroupSettings, ParticipationMode
from telegram_ai.groq import GroqClient, _LIVE_KEYWORDS_RE, _TOOLS
from telegram_ai.search import WebSearchClient


class TestWebSearchClient:
    @pytest.mark.asyncio
    async def test_empty_query_returns_message(self):
        client = WebSearchClient()
        res = await client.search("")
        assert "No search query provided" in res

    @pytest.mark.asyncio
    async def test_search_html_parsing(self):
        sample_html = """
        <html>
        <body>
            <table>
                <tr><td><a href="https://example.com/train12004" class="result-link">Train 12004 Live Running Status</a></td></tr>
                <tr><td class="result-snippet">Train 12004 is currently arriving at Kanpur Central on time.</td></tr>
                <tr><td><a href="https://example.com/score" class="result-link">Live Cricket Score India vs Aus</a></td></tr>
                <tr><td class="result-snippet">India 210/4 in 20 overs.</td></tr>
            </table>
        </body>
        </html>
        """
        client = WebSearchClient()
        with patch.object(client, "_post_search", new=AsyncMock(return_value=sample_html)):
            result = await client.search("train 12004")
            assert "Train 12004 Live Running Status" in result
            assert "Kanpur Central" in result
            assert "https://example.com/train12004" in result
            assert "[1]" in result
            assert "[2]" in result

    @pytest.mark.asyncio
    async def test_fetch_page_cleans_tags(self):
        client = WebSearchClient()
        html_page = "<html><head><script>alert(1)</script><style>body{}</style></head><body><nav>Menu</nav><main>Train is at Platform 2</main><footer>Footer</footer></body></html>"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html_page

        with patch("telegram_ai.search.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            mock_client.return_value.__aenter__.return_value = instance

            text = await client.fetch_page("https://example.com/status")
            assert "Train is at Platform 2" in text
            assert "alert" not in text
            assert "body{}" not in text
            assert "Menu" not in text


class TestGroqLiveTracking:
    def test_live_keywords_regex(self):
        assert _LIVE_KEYWORDS_RE.search("What is the live score of India match?")
        assert _LIVE_KEYWORDS_RE.search("live running status of train 12004")
        assert _LIVE_KEYWORDS_RE.search("What is the current time in Tokyo?")
        assert _LIVE_KEYWORDS_RE.search("today's cricket scorecard")
        assert _LIVE_KEYWORDS_RE.search("F1 timing")
        assert not _LIVE_KEYWORDS_RE.search("write a python hello world function")

    def test_tools_schema_defined(self):
        tool_names = [t["function"]["name"] for t in _TOOLS]
        assert "web_search" in tool_names
        assert "fetch_webpage" in tool_names

    @pytest.mark.asyncio
    async def test_answer_triggers_tool_and_synthesizes(self):
        search_client = MagicMock(spec=WebSearchClient)
        search_client.search = AsyncMock(
            return_value="Train 12004 Swarn Shatabdi: Currently at Kanpur Central (CNB), Platform 1, Running on time."
        )

        groq = GroqClient("FAKE_KEY", "openai/gpt-oss-120b", 10.0, search_client=search_client)
        settings = GroupSettings(123, "concise", ParticipationMode.MENTION, True)
        context = []

        step1_response = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query": "live running status train 12004"}',
                    },
                }
            ],
        }
        step2_response = {
            "role": "assistant",
            "content": "Train 12004 (Swarn Shatabdi) is currently at Kanpur Central on platform 1, running on time.",
        }

        with patch.object(groq, "_chat_completion", side_effect=[step1_response, step2_response]):
            answer = await groq.answer("live running status train 12004", settings, context)
            assert "Kanpur Central" in answer
            assert "running on time" in answer
            search_client.search.assert_awaited_once_with("live running status train 12004", max_results=3)

    @pytest.mark.asyncio
    async def test_answer_fallback_on_unaware_response(self):
        search_client = MagicMock(spec=WebSearchClient)
        search_client.search = AsyncMock(
            return_value="Live Score: India 195/3 in 18 overs vs Australia 190."
        )

        groq = GroqClient("FAKE_KEY", "openai/gpt-oss-120b", 10.0, search_client=search_client)
        settings = GroupSettings(123, "concise", ParticipationMode.MENTION, True)

        refusal_msg = {
            "role": "assistant",
            "content": "As an AI, I don't have access to real-time information or live match scores.",
        }
        final_msg = {
            "role": "assistant",
            "content": "India is currently 195/3 in 18 overs, needing 191 to win.",
        }

        with patch.object(groq, "_chat_completion", side_effect=[refusal_msg, final_msg]):
            answer = await groq.answer("live cricket score India", settings, [])
            assert "195/3" in answer
            search_client.search.assert_awaited()

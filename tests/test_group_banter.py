"""Tests for harmless group rankings, playful banter, titles, roasts, and refusal handling."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from telegram_ai.domain import ConversationMessage, GroupSettings
from telegram_ai.openrouter import OpenRouterClient, needs_web_search


class TestGroupBanterAndRankings(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.client = OpenRouterClient(
            api_key="gsk_test_key",
            model="openai/gpt-oss-120b",
            timeout_seconds=10.0,
        )
        self.settings = GroupSettings(chat_id=-100999)

    async def _capture_call(
        self,
        prompt: str,
        context: list[ConversationMessage] | None = None,
        mock_reply: str = "Playful reply!",
    ) -> tuple[str, list[dict]]:
        captured: list[list[dict]] = []

        async def _fake_completion(messages, **kwargs):
            captured.append(messages)
            return {"content": mock_reply}

        with patch.object(self.client, "_chat_completion", AsyncMock(side_effect=_fake_completion)):
            reply = await self.client.answer(prompt, self.settings, context or [])

        # Must make a single AI call (fast path, no extra classification call)
        self.assertEqual(len(captured), 1)
        return reply, captured[0]

    async def test_01_rank_these_members_uses_single_call_and_banter_prompt(self) -> None:
        prompt = "Rank these members: JD, Nikhil, Hardik, Dre."
        self.assertFalse(needs_web_search(prompt, []))
        reply, msgs = await self._capture_call(
            prompt,
            mock_reply="Ranking kis basis pe? Funny, most active, smartest ya random?",
        )
        self.assertIn("GROUP FUN & BANTER", msgs[0]["content"])
        self.assertIn("never refuse harmless requests", msgs[0]["content"].lower())
        self.assertNotIn("not able to rank", reply.lower())

    async def test_02_rank_them_randomly(self) -> None:
        history = [
            ConversationMessage("Dre", "Rank these members: JD, Nikhil, Hardik, Dre.", message_id=1),
            ConversationMessage("TelegramAI", "Ranking kis basis pe? Funny, most active, smartest ya random?", message_id=2, is_bot=True),
        ]
        reply, msgs = await self._capture_call(
            "Rank them randomly",
            context=history,
            mock_reply="Purely random order just for fun: 1. Hardik, 2. JD, 3. Dre, 4. Nikhil.",
        )
        self.assertIn("random", reply.lower())
        self.assertIn("purely random", msgs[0]["content"].lower())

    async def test_03_who_is_the_funniest(self) -> None:
        history = [
            ConversationMessage("JD", "Bhai aaj toh server ne bhi chutti le li 😂", message_id=10),
            ConversationMessage("Nikhil", "ok", message_id=11),
        ]
        reply, msgs = await self._capture_call(
            "Who is the funniest?",
            context=history,
            mock_reply="Recent chat ke hisaab se JD le gaya point.",
        )
        self.assertIn("JD", reply)
        self.assertIn("Speaker: JD", " ".join(m["content"] for m in msgs))

    async def test_04_give_everyone_a_funny_title(self) -> None:
        reply, msgs = await self._capture_call(
            "Give everyone a funny title: JD, Nikhil, Hardik, Dre",
            mock_reply="JD: Chief Meme Officer, Nikhil: Silent Lurker, Hardik: Late-Night Philosopher, Dre: Chaos Coordinator.",
        )
        self.assertIn("JD", reply)
        self.assertIn("funny awards/titles", msgs[0]["content"].lower())

    async def test_05_roast_the_group(self) -> None:
        reply, msgs = await self._capture_call(
            "Roast the group",
            mock_reply="Is group mein 90% plans bante hain aur 0% execute hote hain.",
        )
        self.assertIn("friendly roasts", msgs[0]["content"].lower())
        self.assertTrue(len(reply) > 10)

    async def test_06_rank_members_by_activity_using_chat_history(self) -> None:
        history = [
            ConversationMessage("Dre", "msg 1", message_id=1),
            ConversationMessage("Dre", "msg 2", message_id=2),
            ConversationMessage("Dre", "msg 3", message_id=3),
            ConversationMessage("JD", "msg 4", message_id=4),
            ConversationMessage("Nikhil", "msg 5", message_id=5),
        ]
        reply, msgs = await self._capture_call(
            "Rank members by activity",
            context=history,
            mock_reply="Recent chat ke hisaab se: 1. Dre (3 msgs), 2. JD & Nikhil (1 each).",
        )
        self.assertIn("Dre", reply)
        combined = " ".join(m["content"] for m in msgs[1:])
        self.assertEqual(combined.count("[Speaker: Dre]"), 3)

    async def test_07_rank_members_by_intelligence_and_refusal_detector(self) -> None:
        self.assertTrue(self.client._is_canned_refusal("I'm not able to rank individuals."))
        self.assertTrue(self.client._is_canned_refusal("Sorry, I cannot rank people."))
        reply, msgs = await self._capture_call(
            "Rank JD, Nikhil, Hardik, Dre by intelligence",
            mock_reply="Real IQ toh pata nahi, par random fun roll ke hisaab se: 1. Nikhil, 2. Dre, 3. JD, 4. Hardik.",
        )
        self.assertIn("without pretending to know anyone's real intelligence", msgs[0]["content"].lower())
        self.assertNotIn("not able to rank", reply.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)

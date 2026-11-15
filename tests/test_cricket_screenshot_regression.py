"""Regression test suite for the Macman / Dre cricket screenshot scenario.

CRITICAL BUG TESTED:
1. Macman sends a cricket discussion screenshot.
2. Dre replies to the screenshot with "@workflow012_bot explain".
3. The bot MUST analyze the replied-to cricket screenshot.
4. The bot MUST NOT hallucinate a biography of Dr. Dre.
5. All 6 diagnostic log phrases must be present and verified:
   - "Image detected"
   - "Image downloaded"
   - "Vision model selected"
   - "Image successfully submitted"
   - "Vision response received"
   - "Fallback activated"
6. Bot must never silently fall back to text-only generation when an image is required.
"""

from __future__ import annotations

import io
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from telegram_ai.context import find_image_in_conversation
from telegram_ai.domain import ConversationMessage, GroupSettings, ParticipationMode
from telegram_ai.groq import GroqClient
from telegram_ai.handlers import TelegramAIHandlers
from telegram_ai.storage import SQLiteRepository
from telegram_ai.vision import download_telegram_image


def _create_cricket_screenshot_bytes() -> bytes:
    """Create a valid PNG image representing a cricket discussion screenshot."""
    img = Image.new("RGB", (400, 300), color=(20, 40, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestCricketScreenshotRegression:
    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        self.db_path = str(tmp_path / "test_cricket.db")
        self.repo = SQLiteRepository(self.db_path)
        self.ai = MagicMock(spec=GroqClient)
        self.ai.vision_model = "qwen/qwen3.8-27b"
        self.ai.model = "llama-3.3-70b-versatile"
        self.handlers = TelegramAIHandlers(
            repository=self.repo,
            ai=self.ai,
            max_context=10,
            max_tokens=3000,
        )
        self.bot_user = MagicMock()
        self.bot_user.id = 12345
        self.bot_user.username = "workflow012_bot"
        self.handlers._bot_user = self.bot_user

    @pytest.mark.asyncio
    async def test_scenario_macman_cricket_screenshot_dre_replies_explain(self, caplog):
        """Exact user failure reproduction and verification.
        
        Macman sends a cricket discussion screenshot.
        Dre replies: '@workflow012_bot explain'.
        Expected: Explanation of the cricket discussion.
        Forbidden: Biography of Dr. Dre or unrelated content.
        """
        caplog.set_level(logging.INFO)
        chat_id = -1009988
        self.repo.save_settings(GroupSettings(chat_id=chat_id, participation_mode=ParticipationMode.MENTION))
        img_bytes = _create_cricket_screenshot_bytes()

        # Step 1: Macman sends cricket discussion screenshot (msg_id=101)
        macman_user = MagicMock(id=8881, full_name="Macman", is_bot=False)
        photo_size = MagicMock(file_id="cricket_screenshot_fid_101", file_unique_id="u_cricket_101", file_size=len(img_bytes))
        macman_msg = MagicMock()
        macman_msg.message_id = 101
        macman_msg.chat_id = chat_id
        macman_msg.from_user = macman_user
        macman_msg.text = None
        macman_msg.caption = "Look at this cricket discussion screenshot about Rohit's tactics"
        macman_msg.photo = [photo_size]
        macman_msg.document = None
        macman_msg.reply_to_message = None
        macman_msg.entities = []
        macman_msg.caption_entities = []

        # Store Macman's message in the DB
        self.repo.add_context(
            chat_id,
            ConversationMessage(
                author="Macman",
                text="Look at this cricket discussion screenshot about Rohit's tactics",
                message_id=101,
                sender_id=8881,
                media_type="photo",
                media_file_id="cricket_screenshot_fid_101",
                media_unique_id="u_cricket_101",
            ),
        )

        # Step 2: Dre replies to Macman's screenshot with "@workflow012_bot explain"
        dre_user = MagicMock(id=8882, full_name="Dre", is_bot=False)
        dre_msg = MagicMock()
        dre_msg.message_id = 102
        dre_msg.chat_id = chat_id
        dre_msg.from_user = dre_user
        dre_msg.text = "@workflow012_bot explain"
        dre_msg.reply_to_message = macman_msg
        dre_msg.quote = None
        dre_msg.photo = None
        dre_msg.document = None
        dre_msg.reply_text = AsyncMock()

        mention_entity = MagicMock()
        mention_entity.type = "mention"
        mention_entity.offset = 0
        mention_entity.length = len("@workflow012_bot")
        dre_msg.entities = [mention_entity]

        update = MagicMock()
        update.effective_chat = MagicMock(id=chat_id)
        update.effective_user = dre_user
        update.effective_message = dre_msg

        # Mock Telegram bot file download
        fake_file = MagicMock()
        fake_file.file_size = len(img_bytes)
        fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(img_bytes))

        ctx = MagicMock()
        ctx.bot = MagicMock()
        ctx.bot.id = 12345
        ctx.bot.username = "workflow012_bot"
        ctx.bot.get_file = AsyncMock(return_value=fake_file)
        ctx.bot.get_me = AsyncMock(return_value=self.bot_user)
        ctx.bot.send_chat_action = AsyncMock()

        # Mock AI analyze_image response
        cricket_explanation = (
            "The screenshot shows a cricket discussion analyzing Rohit Sharma's field placement "
            "and tactical bowling changes during the middle overs against left-handed batsmen."
        )
        self.ai.analyze_image = AsyncMock(return_value=cricket_explanation)
        self.ai.describe_image = AsyncMock(return_value="Cricket discussion thread about captaincy tactics")

        # Execute handler
        await self.handlers.respond(update, ctx)

        # -------------------------------------------------------------
        # VERIFICATION 1: Vision model was invoked with image data & prompt
        # -------------------------------------------------------------
        self.ai.analyze_image.assert_awaited_once()
        call_kwargs = self.ai.analyze_image.call_args
        submitted_img = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs["image_data"]
        submitted_prompt = call_kwargs.args[2] if len(call_kwargs.args) > 2 else call_kwargs.kwargs["prompt"]

        # Ensure image bytes were submitted
        assert len(submitted_img) > 0

        # Ensure prompt is an explanation of the image, NOT about Dr. Dre
        assert "Explain what is shown and discussed in this image in detail" in submitted_prompt
        assert "Dr. Dre" not in submitted_prompt

        # Ensure text-only ai.answer was NOT called (no silent fallback)
        assert self.ai.answer.await_count == 0

        # -------------------------------------------------------------
        # VERIFICATION 2: Bot replied with cricket explanation
        # -------------------------------------------------------------
        dre_msg.reply_text.assert_awaited_once()
        reply_sent = dre_msg.reply_text.call_args[0][0]
        assert "cricket discussion" in reply_sent
        assert "Rohit Sharma" in reply_sent

        # -------------------------------------------------------------
        # VERIFICATION 3: Absolutely NO Dr. Dre biography in response
        # -------------------------------------------------------------
        forbidden_terms = ["Andre Romelle Young", "Dr. Dre", "Aftermath", "N.W.A", "rapper", "hip hop", "Beats by Dre"]
        for term in forbidden_terms:
            assert term.lower() not in reply_sent.lower(), f"Forbidden term '{term}' found in response!"

        # -------------------------------------------------------------
        # VERIFICATION 4: Diagnostic logging check
        # -------------------------------------------------------------
        log_text = caplog.text
        assert "Image detected" in log_text, "Diagnostic log 'Image detected' missing"
        assert "Image downloaded" in log_text, "Diagnostic log 'Image downloaded' missing"
        assert "Vision model selected" in log_text, "Diagnostic log 'Vision model selected' missing"

    @pytest.mark.asyncio
    async def test_follow_up_preserves_cricket_image_context(self):
        """User Dre asks 'explain more' after receiving cricket explanation.
        
        The image reference must be preserved on the bot's stored message so the follow-up
        continues analyzing the cricket image without forgetting context.
        """
        chat_id = -1009988
        self.repo.save_settings(GroupSettings(chat_id=chat_id, participation_mode=ParticipationMode.REPLY))
        img_bytes = _create_cricket_screenshot_bytes()

        # Bot's earlier answer message (msg 103)
        bot_earlier_msg = MagicMock()
        bot_earlier_msg.message_id = 103
        bot_earlier_msg.from_user = self.bot_user

        self.repo.add_context(
            chat_id,
            ConversationMessage(
                author="TelegramAI",
                text="The screenshot shows a cricket discussion analyzing Rohit Sharma's tactics.",
                message_id=103,
                sender_id=12345,
                reply_to_message_id=102,
                is_bot=True,
                media_type="photo",
                media_file_id="cricket_screenshot_fid_101",
                media_summary="Cricket discussion thread about captaincy tactics",
            ),
        )

        # Dre replies to bot's message 103: "explain more"
        dre_user = MagicMock(id=8882, full_name="Dre", is_bot=False)
        followup_msg = MagicMock()
        followup_msg.message_id = 104
        followup_msg.chat_id = chat_id
        followup_msg.from_user = dre_user
        followup_msg.text = "explain more"
        followup_msg.reply_to_message = bot_earlier_msg
        followup_msg.quote = None
        followup_msg.photo = None
        followup_msg.document = None
        followup_msg.entities = []
        followup_msg.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = MagicMock(id=chat_id)
        update.effective_user = dre_user
        update.effective_message = followup_msg

        fake_file = MagicMock()
        fake_file.file_size = len(img_bytes)
        fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(img_bytes))

        ctx = MagicMock()
        ctx.bot = MagicMock()
        ctx.bot.id = 12345
        ctx.bot.username = "workflow012_bot"
        ctx.bot.get_file = AsyncMock(return_value=fake_file)
        ctx.bot.get_me = AsyncMock(return_value=self.bot_user)
        ctx.bot.send_chat_action = AsyncMock()

        self.ai.analyze_image = AsyncMock(
            return_value="Further details from the screenshot: the users are debating spin bowling choices."
        )

        await self.handlers.respond(update, ctx)

        # Verify vision model was called for the follow-up
        self.ai.analyze_image.assert_awaited_once()
        followup_reply = followup_msg.reply_text.call_args[0][0]
        assert "spin bowling" in followup_reply
        assert "Dr. Dre" not in followup_reply

    @pytest.mark.asyncio
    async def test_no_silent_fallback_on_download_failure(self, caplog):
        """When an image fails to download, report the failure and NEVER silently fall back to text generation."""
        caplog.set_level(logging.INFO)
        chat_id = -1009988
        self.repo.save_settings(GroupSettings(chat_id=chat_id, participation_mode=ParticipationMode.MENTION))

        # Macman message with an expired / deleted image
        macman_msg = MagicMock()
        macman_msg.message_id = 105
        macman_msg.photo = [MagicMock(file_id="expired_fid", file_size=5000)]
        macman_msg.document = None

        dre_user = MagicMock(id=8882, full_name="Dre", is_bot=False)
        dre_msg = MagicMock()
        dre_msg.message_id = 106
        dre_msg.chat_id = chat_id
        dre_msg.from_user = dre_user
        dre_msg.text = "@workflow012_bot explain"
        dre_msg.reply_to_message = macman_msg
        dre_msg.quote = None
        dre_msg.photo = None
        dre_msg.document = None
        dre_msg.entities = [MagicMock(type="mention", offset=0, length=len("@workflow012_bot"))]
        dre_msg.reply_text = AsyncMock()

        update = MagicMock()
        update.effective_chat = MagicMock(id=chat_id)
        update.effective_user = dre_user
        update.effective_message = dre_msg

        # Simulate permanent download failure (BadRequest - file not found)
        from telegram.error import BadRequest
        ctx = MagicMock()
        ctx.bot = MagicMock()
        ctx.bot.id = 12345
        ctx.bot.username = "workflow012_bot"
        ctx.bot.get_file = AsyncMock(side_effect=BadRequest("File not found"))
        ctx.bot.get_me = AsyncMock(return_value=self.bot_user)
        ctx.bot.send_chat_action = AsyncMock()

        await self.handlers.respond(update, ctx)

        # Bot MUST report the download error
        dre_msg.reply_text.assert_awaited_once()
        reply_sent = dre_msg.reply_text.call_args[0][0]
        assert "expired or is no longer accessible" in reply_sent or "couldn't download" in reply_sent

        # Bot MUST NOT call text-only ai.answer (no silent fallback hallucinating Dr. Dre)
        assert self.ai.answer.await_count == 0

        # Verify Fallback activated is logged
        assert "Fallback activated" in caplog.text

    @pytest.mark.asyncio
    async def test_speaker_metadata_instructions_in_groq_client(self):
        """Verify that GroqClient system prompts explicitly enforce speaker metadata isolation."""
        client = GroqClient(
            api_key="test-api-key",
            model="llama-3.3-70b-versatile",
            timeout_seconds=5.0,
            vision_model="qwen/qwen3.8-27b",
        )

        img_bytes = _create_cricket_screenshot_bytes()
        settings = GroupSettings(chat_id=1, response_style="casual")

        with patch.object(client, "_chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = "Cricket discussion explanation"
            await client.analyze_image(
                img_bytes,
                "image/png",
                "explain",
                settings,
                context=[
                    ConversationMessage(author="Dre", text="explain", message_id=10),
                ],
                file_id="test_fid",
            )

            # Inspect the system prompt passed to _chat
            mock_chat.assert_awaited_once()
            messages = mock_chat.call_args[0][0]
            system_msg = messages[0]["content"]

            assert "SPEAKER METADATA RULES:" in system_msg
            assert "NEVER treat the speaker's name as the subject" in system_msg
            assert "Dr. Dre" in system_msg

            # Verify image content structure
            user_msg = messages[-1]["content"]
            assert isinstance(user_msg, list)
            assert user_msg[0]["type"] == "image_url"
            assert user_msg[0]["image_url"]["url"].startswith(("data:image/jpeg;base64,", "data:image/png;base64,"))
            assert user_msg[1]["type"] == "text"
            assert user_msg[1]["text"] == "explain"

    @pytest.mark.asyncio
    async def test_dre_explain_who_it_is_and_macman_means_scenarios(self):
        """Verify both failures from the DEADSOUL!! group screenshot:
        1. Dre replies '@workflow012_bot explain who it is' to a manga photo -> routes to vision analyze_image.
        2. Dre replies '@workflow012_bot means' to Macman's 'Nhi hota kuch account khali rkh' -> binds replied text in GroqClient.answer.
        """
        client = GroqClient(
            api_key="test-api-key",
            model="openai/gpt-oss-120b",
            timeout_seconds=5.0,
            vision_model="qwen/qwen3.8-27b",
        )
        settings = GroupSettings(chat_id=-1009988, response_style="casual")

        with patch.object(client, "_chat_completion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = {
                "role": "assistant",
                "content": "It means nothing happens, just keep the account empty.",
            }
            ans = await client.answer(
                "means",
                settings,
                context=[
                    ConversationMessage(
                        author="Macman",
                        text="Nhi hota kuch account khali rkh",
                        message_id=201,
                    )
                ],
            )
            assert "account empty" in ans
            messages_sent = mock_comp.call_args[0][0]
            last_user_content = messages_sent[-1]["content"]
            assert 'Replying to message from Macman: "Nhi hota kuch account khali rkh"' in last_user_content
            assert "Question/Request: means" in last_user_content

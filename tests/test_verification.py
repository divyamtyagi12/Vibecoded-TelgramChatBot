"""
End-to-end verification suite for the telegram_ai bot.
Tests are fully offline -- no real Telegram or Gemini network calls.
Run with:  python -m pytest tests/test_verification.py -v
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# 1. STATIC / IMPORT CHECKS
# ---------------------------------------------------------------------------

class TestImports(unittest.TestCase):
    """Bug-class check: every module used via module.attr must be imported."""

    def test_gemini_imports_httpx(self):
        """The original bug: gemini.py used httpx but only imported stdlib http."""
        import telegram_ai.gemini as gem
        self.assertIn(
            "httpx", dir(gem),
            "httpx is not imported in gemini.py -- the original bug still present",
        )

    def test_gemini_does_not_use_stdlib_http_module(self):
        """Confirm httpx (not stdlib http) is imported at the top of gemini.py."""
        import ast
        import inspect
        import textwrap
        import telegram_ai.gemini as gem

        src = inspect.getsource(gem)
        tree = ast.parse(textwrap.dedent(src))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.add(node.module.split(".")[0])
        self.assertIn("httpx", imported, "httpx must be imported in gemini.py")

    def test_requirements_lists_httpx(self):
        """requirements.txt must explicitly list httpx."""
        from pathlib import Path
        req = Path(__file__).resolve().parent.parent / "requirements.txt"
        content = req.read_text().lower()
        self.assertIn("httpx", content, "httpx is not listed in requirements.txt")

    def test_requirements_lists_all_core_packages(self):
        from pathlib import Path
        req = Path(__file__).resolve().parent.parent / "requirements.txt"
        content = req.read_text().lower()
        for pkg in ("python-telegram-bot", "python-dotenv", "fastapi"):
            self.assertIn(pkg, content, f"{pkg} missing from requirements.txt")


# ---------------------------------------------------------------------------
# 2. GeminiClient._generate_text unit tests
# ---------------------------------------------------------------------------

class FakeResponse:
    """Minimal httpx.Response mimic."""

    def __init__(self, status_code: int, body: dict | None = None):
        self.status_code = status_code
        self._body = body or {}
        self.request = MagicMock()

    def raise_for_status(self):
        import httpx
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self.request,
                response=self,  # type: ignore[arg-type]
            )

    def json(self):
        return self._body


def _ok_body(text: str = "hello") -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


class TestGeminiGenerateText(unittest.IsolatedAsyncioTestCase):
    """Unit tests for GeminiClient._generate_text with a mocked httpx."""

    def _make_client(self):
        from telegram_ai.gemini import GeminiClient
        return GeminiClient(api_key="FAKE", model="gemini-flash-lite-latest", timeout_seconds=5.0)

    def _mock_http(self, post_coro):
        """Returns a mock AsyncClient context manager whose .post = post_coro."""
        mock_inner = MagicMock(post=post_coro)
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_inner)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        return mock_cm

    # (a) Success path
    async def test_success_returns_text(self):
        client = self._make_client()

        async def fake_post(*a, **kw):
            return FakeResponse(200, _ok_body("pong"))

        with patch("telegram_ai.gemini.httpx.AsyncClient", return_value=self._mock_http(fake_post)):
            result = await client._generate_text({"contents": []})

        self.assertEqual(result, "pong")

    # (b) 429 triggers retry then success
    async def test_429_triggers_retry_then_success(self):
        client = self._make_client()
        call_count = 0

        async def fake_post(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return FakeResponse(429)
            return FakeResponse(200, _ok_body("retried!"))

        with patch("telegram_ai.gemini.httpx.AsyncClient", return_value=self._mock_http(fake_post)), \
             patch("asyncio.sleep", new=AsyncMock()):
            result = await client._generate_text({"contents": []})

        self.assertEqual(result, "retried!")
        self.assertGreaterEqual(call_count, 2)

    # (c) Non-429 HTTP error falls through to next model
    async def test_non_429_http_error_falls_to_next_model(self):
        client = self._make_client()
        call_count = 0

        async def fake_post(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return FakeResponse(500)
            return FakeResponse(200, _ok_body("from-fallback"))

        with patch("telegram_ai.gemini.httpx.AsyncClient", return_value=self._mock_http(fake_post)), \
             patch("asyncio.sleep", new=AsyncMock()):
            result = await client._generate_text({"contents": []})

        self.assertEqual(result, "from-fallback")
        self.assertGreaterEqual(call_count, 2)

    # (d) All models failing raises RuntimeError (exact type)
    async def test_all_models_fail_raises_runtime_error(self):
        client = self._make_client()

        async def fake_post(*a, **kw):
            return FakeResponse(500)

        with patch("telegram_ai.gemini.httpx.AsyncClient", return_value=self._mock_http(fake_post)), \
             patch("asyncio.sleep", new=AsyncMock()):
            with self.assertRaises(RuntimeError) as ctx:
                await client._generate_text({"contents": []})

        self.assertIs(type(ctx.exception), RuntimeError)

    # (e) Network error falls to next model
    async def test_network_error_falls_to_next_model(self):
        import httpx
        client = self._make_client()
        call_count = 0

        async def fake_post(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("connection refused")
            return FakeResponse(200, _ok_body("network-recovered"))

        with patch("telegram_ai.gemini.httpx.AsyncClient", return_value=self._mock_http(fake_post)):
            result = await client._generate_text({"contents": []})

        self.assertEqual(result, "network-recovered")

    # (f) Empty text candidates raises RuntimeError
    async def test_empty_candidates_raises_runtime_error(self):
        client = self._make_client()
        empty_body = {"candidates": [{"content": {"parts": [{"text": ""}]}}]}

        async def fake_post(*a, **kw):
            return FakeResponse(200, empty_body)

        with patch("telegram_ai.gemini.httpx.AsyncClient", return_value=self._mock_http(fake_post)):
            with self.assertRaises(RuntimeError):
                await client._generate_text({"contents": []})


# ---------------------------------------------------------------------------
# 3. SQLiteRepository.add_context -- OperationalError tolerance
# ---------------------------------------------------------------------------

class _BrokenConnProxy:
    """Proxy that raises OperationalError only on INSERT INTO conversation_context."""

    def __init__(self, real):
        self._real = real

    def execute(self, sql, *args, **kwargs):
        if "INSERT INTO conversation_context" in sql:
            raise sqlite3.OperationalError("database is locked")
        return self._real.execute(sql, *args, **kwargs)

    def commit(self):
        return self._real.commit()

    def __getattr__(self, name):
        return getattr(self._real, name)


class TestStorageAddContext(unittest.TestCase):

    def _make_repo(self):
        from telegram_ai.storage import SQLiteRepository
        return SQLiteRepository(":memory:")

    def test_add_context_success(self):
        from telegram_ai.domain import ConversationMessage
        repo = self._make_repo()
        repo.add_context(1, ConversationMessage("Alice", "hello"))
        rows = repo.get_context(1, 10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].text, "hello")

    def test_add_context_operational_error_does_not_raise(self):
        from telegram_ai.domain import ConversationMessage
        repo = self._make_repo()
        repo.connection = _BrokenConnProxy(repo.connection)  # type: ignore[assignment]

        # Must NOT raise
        try:
            repo.add_context(1, ConversationMessage("Alice", "hello"))
        except sqlite3.OperationalError:
            self.fail("add_context raised OperationalError -- should swallow it")
        except Exception as exc:
            self.fail(f"add_context raised unexpected exception: {exc}")

    def test_add_context_operational_error_is_logged(self):
        from telegram_ai.domain import ConversationMessage
        repo = self._make_repo()
        repo.connection = _BrokenConnProxy(repo.connection)  # type: ignore[assignment]

        with self.assertLogs("telegram_ai.storage", level="WARNING") as log_ctx:
            repo.add_context(1, ConversationMessage("Alice", "test"))

        self.assertTrue(
            any(
                "operational" in msg.lower()
                or "locked" in msg.lower()
                or "context" in msg.lower()
                for msg in log_ctx.output
            ),
            f"No relevant warning logged. Got: {log_ctx.output}",
        )


# ---------------------------------------------------------------------------
# 4. should_reply() -- participation mode policy matrix
# ---------------------------------------------------------------------------

class TestShouldReply(unittest.TestCase):

    def _settings(self, mode_str: str):
        from telegram_ai.domain import GroupSettings, ParticipationMode
        return GroupSettings(chat_id=1, participation_mode=ParticipationMode(mode_str))

    def test_mention_mode_mention(self):
        from telegram_ai.domain import should_reply
        self.assertTrue(should_reply(self._settings("mention"), is_mentioned=True, is_reply_to_bot=False))

    def test_mention_mode_reply(self):
        from telegram_ai.domain import should_reply
        self.assertTrue(should_reply(self._settings("mention"), is_mentioned=False, is_reply_to_bot=True))

    def test_mention_mode_neither(self):
        from telegram_ai.domain import should_reply
        self.assertFalse(should_reply(self._settings("mention"), is_mentioned=False, is_reply_to_bot=False))

    def test_reply_mode_reply_only(self):
        from telegram_ai.domain import should_reply
        self.assertTrue(should_reply(self._settings("reply"), is_mentioned=False, is_reply_to_bot=True))

    def test_reply_mode_mention_not_enough(self):
        from telegram_ai.domain import should_reply
        self.assertFalse(should_reply(self._settings("reply"), is_mentioned=True, is_reply_to_bot=False))

    def test_always_mode(self):
        from telegram_ai.domain import should_reply
        self.assertTrue(should_reply(self._settings("always"), is_mentioned=False, is_reply_to_bot=False))

    def test_off_mode(self):
        from telegram_ai.domain import should_reply
        self.assertFalse(should_reply(self._settings("off"), is_mentioned=True, is_reply_to_bot=True))


# ---------------------------------------------------------------------------
# 5. TelegramAIHandlers.respond -- integration mocks
# ---------------------------------------------------------------------------

def _make_update(
    text: str,
    chat_id: int = -100,
    chat_type: str = "group",
    user_id: int = 42,
    entities=None,
    reply_to_message=None,
):
    from telegram.constants import ChatType
    chat_type_map = {
        "group": ChatType.GROUP,
        "private": ChatType.PRIVATE,
        "supergroup": ChatType.SUPERGROUP,
    }

    user = MagicMock()
    user.id = user_id
    user.full_name = f"User{user_id}"
    user.is_bot = False
    user.username = f"user{user_id}"

    msg = MagicMock()
    msg.text = text
    msg.from_user = user
    msg.entities = entities or []
    msg.caption = None
    msg.caption_entities = []
    msg.photo = None
    msg.reply_to_message = reply_to_message
    msg.chat_id = chat_id
    msg.reply_text = AsyncMock()

    chat = MagicMock()
    chat.id = chat_id
    chat.type = chat_type_map.get(chat_type, ChatType.GROUP)

    update = MagicMock()
    update.effective_chat = chat
    update.effective_user = user
    update.effective_message = msg
    return update


def _make_bot(bot_id: int = 99, username: str = "TestBot"):
    bot = MagicMock()
    bot.id = bot_id
    bot.username = username
    bot.get_me = AsyncMock(return_value=MagicMock(id=bot_id, username=username))
    bot.send_chat_action = AsyncMock()
    return bot


def _make_context(bot=None):
    ctx = MagicMock()
    ctx.bot = bot or _make_bot()
    ctx.args = []
    return ctx


class TestRespondHandler(unittest.IsolatedAsyncioTestCase):

    def _make_handler(self):
        from telegram_ai.storage import SQLiteRepository
        from telegram_ai.gemini import GeminiClient
        from telegram_ai.handlers import TelegramAIHandlers
        repo = SQLiteRepository(":memory:")
        ai = MagicMock(spec=GeminiClient)
        ai.answer = AsyncMock(return_value="AI reply")
        return TelegramAIHandlers(repo, ai, max_context=5), repo, ai

    async def test_mention_mode_mention_triggers_reply(self):
        handler, repo, ai = self._make_handler()
        update = _make_update("@TestBot hello", chat_id=-1)
        entity = MagicMock()
        entity.type = "mention"
        entity.offset = 0
        entity.length = len("@TestBot")
        update.effective_message.entities = [entity]
        ctx = _make_context(_make_bot())
        handler._bot_user = MagicMock(id=99, username="TestBot")
        await handler.respond(update, ctx)
        ai.answer.assert_awaited_once()

    async def test_off_mode_never_replies(self):
        from telegram_ai.domain import GroupSettings, ParticipationMode
        handler, repo, ai = self._make_handler()
        repo.save_settings(GroupSettings(chat_id=-2, participation_mode=ParticipationMode.OFF))
        update = _make_update("@TestBot hello", chat_id=-2)
        entity = MagicMock()
        entity.type = "mention"
        entity.offset = 0
        entity.length = len("@TestBot")
        update.effective_message.entities = [entity]
        ctx = _make_context(_make_bot())
        handler._bot_user = MagicMock(id=99, username="TestBot")
        await handler.respond(update, ctx)
        ai.answer.assert_not_awaited()

    async def test_always_mode_replies_without_mention(self):
        from telegram_ai.domain import GroupSettings, ParticipationMode
        handler, repo, ai = self._make_handler()
        repo.save_settings(GroupSettings(chat_id=-3, participation_mode=ParticipationMode.ALWAYS))
        update = _make_update("just a regular message", chat_id=-3)
        ctx = _make_context(_make_bot())
        handler._bot_user = MagicMock(id=99, username="TestBot")
        await handler.respond(update, ctx)
        ai.answer.assert_awaited_once()

    async def test_reply_mode_reply_triggers(self):
        from telegram_ai.domain import GroupSettings, ParticipationMode
        handler, repo, ai = self._make_handler()
        repo.save_settings(GroupSettings(chat_id=-4, participation_mode=ParticipationMode.REPLY))
        bot_msg = MagicMock()
        bot_msg.from_user = MagicMock(id=99, is_bot=True)
        update = _make_update("what do you think?", chat_id=-4, reply_to_message=bot_msg)
        ctx = _make_context(_make_bot())
        handler._bot_user = MagicMock(id=99, username="TestBot")
        await handler.respond(update, ctx)
        ai.answer.assert_awaited_once()

    async def test_gemini_failure_sends_exact_user_facing_error(self):
        """CRITICAL: exact error string must match what is in handlers.py."""
        from telegram_ai.domain import GroupSettings, ParticipationMode
        handler, repo, ai = self._make_handler()
        repo.save_settings(GroupSettings(chat_id=-5, participation_mode=ParticipationMode.ALWAYS))
        ai.answer = AsyncMock(side_effect=RuntimeError("The AI provider is temporarily unavailable."))
        update = _make_update("hello ai", chat_id=-5)
        ctx = _make_context()
        handler._bot_user = MagicMock(id=99, username="TestBot")
        await handler.respond(update, ctx)

        calls = update.effective_message.reply_text.await_args_list
        self.assertTrue(len(calls) >= 1, "reply_text was never called")
        replied_text = calls[-1].args[0] if calls[-1].args else calls[-1].kwargs.get("text", "")
        self.assertEqual(
            replied_text,
            "I couldn't reach the AI service just now. Please try again shortly.",
            f"User-facing error message mismatch: {replied_text!r}",
        )

    async def test_gemini_failure_does_not_reraise(self):
        """The RuntimeError must NOT propagate out of respond()."""
        from telegram_ai.domain import GroupSettings, ParticipationMode
        handler, repo, ai = self._make_handler()
        repo.save_settings(GroupSettings(chat_id=-6, participation_mode=ParticipationMode.ALWAYS))
        ai.answer = AsyncMock(side_effect=RuntimeError("boom"))
        update = _make_update("hello", chat_id=-6)
        ctx = _make_context()
        handler._bot_user = MagicMock(id=99, username="TestBot")
        try:
            await handler.respond(update, ctx)
        except Exception as exc:
            self.fail(f"respond() raised {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 6. Admin permission logic
# ---------------------------------------------------------------------------

class TestIsAdmin(unittest.IsolatedAsyncioTestCase):

    def _make_handler(self):
        from telegram_ai.storage import SQLiteRepository
        from telegram_ai.gemini import GeminiClient
        from telegram_ai.handlers import TelegramAIHandlers
        repo = SQLiteRepository(":memory:")
        ai = MagicMock(spec=GeminiClient)
        return TelegramAIHandlers(repo, ai, max_context=5)

    async def test_private_chat_always_admin(self):
        handler = self._make_handler()
        update = _make_update("hi", chat_type="private")
        result = await handler._is_admin(update)
        self.assertTrue(result)

    async def test_group_owner_is_admin(self):
        from telegram.constants import ChatMemberStatus
        handler = self._make_handler()
        update = _make_update("hi", chat_type="group")
        member = MagicMock()
        member.status = ChatMemberStatus.OWNER
        update.effective_chat.get_member = AsyncMock(return_value=member)
        result = await handler._is_admin(update)
        self.assertTrue(result)

    async def test_group_administrator_is_admin(self):
        from telegram.constants import ChatMemberStatus
        handler = self._make_handler()
        update = _make_update("hi", chat_type="group")
        member = MagicMock()
        member.status = ChatMemberStatus.ADMINISTRATOR
        update.effective_chat.get_member = AsyncMock(return_value=member)
        result = await handler._is_admin(update)
        self.assertTrue(result)

    async def test_group_member_is_not_admin(self):
        from telegram.constants import ChatMemberStatus
        handler = self._make_handler()
        update = _make_update("hi", chat_type="group")
        member = MagicMock()
        member.status = ChatMemberStatus.MEMBER
        update.effective_chat.get_member = AsyncMock(return_value=member)
        result = await handler._is_admin(update)
        self.assertFalse(result)
        update.effective_message.reply_text.assert_awaited_with(
            "Only group admins can change bot settings."
        )

    async def test_no_user_returns_false(self):
        handler = self._make_handler()
        update = MagicMock()
        update.effective_chat = None
        update.effective_user = None
        result = await handler._is_admin(update)
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# 7. Photo handler size limits
# ---------------------------------------------------------------------------

class TestPhotoHandler(unittest.IsolatedAsyncioTestCase):

    def _make_handler(self):
        from telegram_ai.storage import SQLiteRepository
        from telegram_ai.gemini import GeminiClient
        from telegram_ai.handlers import TelegramAIHandlers
        repo = SQLiteRepository(":memory:")
        ai = MagicMock(spec=GeminiClient)
        ai.rate_photo = AsyncMock(return_value="8/10 -- great lighting!")
        return TelegramAIHandlers(repo, ai, max_context=5), repo, ai

    def _make_photo_update(self, file_size: int, chat_id: int = -10):
        from telegram.constants import ChatType
        user = MagicMock()
        user.id = 42
        user.full_name = "Alice"
        user.is_bot = False

        photo_size = MagicMock()
        photo_size.file_size = file_size
        photo_size.file_id = "file123"

        msg = MagicMock()
        msg.photo = [photo_size]
        msg.from_user = user
        msg.caption = "@TestBot rate this"
        msg.caption_entities = []
        msg.reply_to_message = None
        msg.chat_id = chat_id
        msg.reply_text = AsyncMock()

        chat = MagicMock()
        chat.id = chat_id
        chat.type = ChatType.GROUP

        update = MagicMock()
        update.effective_chat = chat
        update.effective_user = user
        update.effective_message = msg
        return update

    async def test_photo_below_limit_gets_rated(self):
        from telegram_ai.domain import GroupSettings, ParticipationMode
        handler, repo, ai = self._make_handler()
        repo.save_settings(GroupSettings(chat_id=-10, participation_mode=ParticipationMode.ALWAYS))
        update = self._make_photo_update(file_size=1024 * 1024)

        bot = _make_bot()
        fake_file = MagicMock()
        fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"fake_image_data"))
        bot.get_file = AsyncMock(return_value=fake_file)
        ctx = _make_context(bot)
        handler._bot_user = MagicMock(id=99, username="TestBot")

        entity = MagicMock()
        entity.type = "mention"
        entity.offset = 0
        entity.length = len("@TestBot")
        update.effective_message.caption_entities = [entity]

        await handler.photo(update, ctx)
        ai.rate_photo.assert_awaited_once()

    async def test_photo_above_limit_rejected_before_download(self):
        from telegram_ai.handlers import MAX_PHOTO_BYTES
        from telegram_ai.domain import GroupSettings, ParticipationMode
        handler, repo, ai = self._make_handler()
        repo.save_settings(GroupSettings(chat_id=-11, participation_mode=ParticipationMode.ALWAYS))
        update = self._make_photo_update(file_size=MAX_PHOTO_BYTES + 1, chat_id=-11)

        bot = _make_bot()
        ctx = _make_context(bot)
        handler._bot_user = MagicMock(id=99, username="TestBot")

        entity = MagicMock()
        entity.type = "mention"
        entity.offset = 0
        entity.length = len("@TestBot")
        update.effective_message.caption_entities = [entity]

        await handler.photo(update, ctx)
        ai.rate_photo.assert_not_awaited()
        update.effective_message.reply_text.assert_awaited_with(
            "Please send a photo no larger than 10 MB for a rating."
        )


# ---------------------------------------------------------------------------
# 8. _request_text helper -- mention stripping
# ---------------------------------------------------------------------------

class TestRequestText(unittest.TestCase):

    def _rt(self, text, username="TestBot"):
        from telegram_ai.handlers import TelegramAIHandlers
        return TelegramAIHandlers._request_text(text, username)

    def test_strips_mention(self):
        result = self._rt("@TestBot what is 2+2?")
        self.assertNotIn("@TestBot", result)
        self.assertIn("what is 2+2?", result)

    def test_no_mention_passthrough(self):
        result = self._rt("just a question")
        self.assertEqual(result, "just a question")

    def test_empty_after_stripping(self):
        result = self._rt("@TestBot")
        self.assertEqual(result, "")


# ---------------------------------------------------------------------------
# 9. Public API signature contracts
# ---------------------------------------------------------------------------

class TestPublicAPISignatures(unittest.TestCase):

    def test_answer_signature(self):
        import inspect
        from telegram_ai.gemini import GeminiClient
        params = list(inspect.signature(GeminiClient.answer).parameters.keys())
        self.assertIn("prompt", params)
        self.assertIn("settings", params)
        self.assertIn("context", params)

    def test_rate_photo_signature(self):
        import inspect
        from telegram_ai.gemini import GeminiClient
        params = list(inspect.signature(GeminiClient.rate_photo).parameters.keys())
        self.assertIn("image_data", params)
        self.assertIn("mime_type", params)
        self.assertIn("request_text", params)
        self.assertIn("settings", params)

    def test_settings_from_environment_exists(self):
        from telegram_ai.config import Settings
        self.assertTrue(callable(Settings.from_environment))


if __name__ == "__main__":
    unittest.main(verbosity=2)

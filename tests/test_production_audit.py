"""Production-readiness audit test suite (offline/mocked tests clearly separated from live integration)."""

from __future__ import annotations

import asyncio
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from PIL import Image, ImageDraw
from telegram.error import BadRequest

from telegram_ai.context import assemble_context
from telegram_ai.domain import ConversationMessage, GroupSettings, ParticipationMode
from telegram_ai.handlers import TelegramAIHandlers
from telegram_ai.openrouter import (
    OpenRouterClient,
    QuotaExhaustedError,
    _OPENROUTER_FREE_MODEL,
)
from telegram_ai.storage import PostgresRepository, SQLiteRepository, create_repository
from telegram_ai.vision import download_telegram_image


def _render_test_screenshot(label: str = "IND 245/4 (42.3 ov)") -> bytes:
    img = Image.new("RGB", (320, 120), color=(15, 25, 45))
    draw = ImageDraw.Draw(img)
    draw.text((20, 45), label, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakePgCursor:
    """In-memory SQLite-backed cursor emulating psycopg dict_row behavior for offline PostgresRepository tests."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._cur = conn.cursor()
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._cur.close()

    def execute(self, sql: str, params=None):
        # Translate PostgreSQL DDL/DML syntax to SQLite for offline unit testing
        q = sql.replace("%s", "?")
        q = q.replace("BIGSERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
        q = q.replace("BIGINT", "INTEGER")
        q = q.replace("TIMESTAMPTZ NOT NULL DEFAULT NOW()", "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP")
        q = q.replace("BOOLEAN NOT NULL DEFAULT TRUE", "INTEGER NOT NULL DEFAULT 1")
        q = q.replace("BOOLEAN NOT NULL DEFAULT FALSE", "INTEGER NOT NULL DEFAULT 0")
        q = q.replace("EXCLUDED.", "excluded.")
        q = q.replace(
            "ON CONFLICT (chat_id, message_id) WHERE message_id IS NOT NULL DO UPDATE SET",
            "ON CONFLICT(chat_id, message_id) DO UPDATE SET",
        )
        if "WHERE message_id IS NOT NULL" in q and "CREATE UNIQUE INDEX" in q:
            q = q.replace(" WHERE message_id IS NOT NULL", "")
        if "NOW() - (? || ' days')::INTERVAL" in q:
            q = "DELETE FROM conversation_context WHERE datetime(created_at) < datetime('now', '-' || ? || ' days')"
        if params is not None:
            self._cur.execute(q, params)
        else:
            self._cur.execute(q)
        self.rowcount = self._cur.rowcount

    def fetchone(self):
        row = self._cur.fetchone()
        return dict(row) if row is not None else None

    def fetchall(self):
        return [dict(r) for r in self._cur.fetchall()]


class _FakePgConn:
    def __init__(self, sqlite_conn: sqlite3.Connection):
        self._conn = sqlite_conn

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return _FakePgCursor(self._conn)

    def commit(self):
        self._conn.commit()


class TestProductionAuditOffline(unittest.IsolatedAsyncioTestCase):
    """Offline/mocked production-readiness audit tests (1-12)."""

    # 1. Actual Telegram image-download workflow
    async def test_01_telegram_image_download_workflow(self):
        img_bytes = _render_test_screenshot()
        bot = MagicMock()
        fake_file = MagicMock(file_size=len(img_bytes))
        fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(img_bytes))
        bot.get_file = AsyncMock(return_value=fake_file)

        data, mime, err = await download_telegram_image(bot, "valid_file_id_123")
        self.assertIsNone(err)
        self.assertIsNotNone(data)
        self.assertIn(mime, ("image/jpeg", "image/png"))
        self.assertGreater(len(data or b""), 100)

    # 2. Image replies from other group members include caption context & real image bytes
    async def test_02_image_reply_includes_caption_and_bytes(self):
        repo = SQLiteRepository(":memory:")
        ai = MagicMock(spec=OpenRouterClient)
        ai.vision_model = _OPENROUTER_FREE_MODEL
        ai.analyze_image = AsyncMock(return_value="Scorecard shows India at 245/4.")
        ai.describe_image = AsyncMock(return_value="Cricket scorecard IND 245/4")
        handlers = TelegramAIHandlers(repo, ai, max_context=8)
        handlers._bot_user = MagicMock(id=999, username="AIChatBot_12_bot")

        chat_id = -4001
        repo.save_settings(GroupSettings(chat_id=chat_id, participation_mode=ParticipationMode.MENTION))
        img_bytes = _render_test_screenshot()

        macman = MagicMock(id=501, full_name="Macman", is_bot=False)
        photo_msg = MagicMock(
            message_id=101,
            from_user=macman,
            text=None,
            caption="Check India score",
            photo=[MagicMock(file_id="fid_score_101", file_unique_id="u101", file_size=len(img_bytes))],
            document=None,
            reply_to_message=None,
            message_thread_id=None,
        )

        dre = MagicMock(id=502, full_name="Dre", is_bot=False)
        reply_msg = MagicMock(
            message_id=102,
            chat_id=chat_id,
            from_user=dre,
            text="@AIChatBot_12_bot Explain this",
            entities=[MagicMock(type="mention", offset=0, length=17)],
            reply_to_message=photo_msg,
            quote=None,
            photo=None,
            document=None,
            message_thread_id=None,
        )
        reply_msg.reply_text = AsyncMock(return_value=MagicMock(message_id=103))

        fake_file = MagicMock(file_size=len(img_bytes))
        fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(img_bytes))
        ctx = MagicMock()
        ctx.bot = MagicMock(
            get_me=AsyncMock(return_value=handlers._bot_user),
            send_chat_action=AsyncMock(),
            get_file=AsyncMock(return_value=fake_file),
        )
        update = MagicMock(effective_chat=MagicMock(id=chat_id), effective_user=dre, effective_message=reply_msg)

        await handlers.respond(update, ctx)
        ai.analyze_image.assert_awaited_once()
        submitted_bytes = ai.analyze_image.call_args[0][0]
        submitted_prompt = ai.analyze_image.call_args[0][2]
        self.assertGreater(len(submitted_bytes), 100)
        self.assertIn("Check India score", submitted_prompt)
        self.assertIn("Explain what is shown and discussed in this image", submitted_prompt)

    # 3. Image follow-up questions
    async def test_03_image_followup_question(self):
        repo = SQLiteRepository(":memory:")
        ai = MagicMock(spec=OpenRouterClient)
        ai.vision_model = _OPENROUTER_FREE_MODEL
        ai.analyze_image = AsyncMock(return_value="They need 55 more runs from 45 balls.")
        ai.describe_image = AsyncMock(return_value="Cricket scorecard")
        handlers = TelegramAIHandlers(repo, ai, max_context=8)
        handlers._bot_user = MagicMock(id=999, username="AIChatBot_12_bot")

        chat_id = -4002
        repo.save_settings(GroupSettings(chat_id=chat_id, participation_mode=ParticipationMode.REPLY))
        repo.add_context(
            chat_id,
            ConversationMessage(
                author="TelegramAI",
                text="India is 245/4 after 42.3 overs.",
                message_id=201,
                is_bot=True,
                media_type="photo",
                media_file_id="fid_score_200",
                media_summary="Cricket scorecard IND 245/4",
            ),
        )
        img_bytes = _render_test_screenshot()
        bot_msg = MagicMock(message_id=201, from_user=handlers._bot_user, text="India is 245/4.", photo=None, document=None)
        dre = MagicMock(id=502, full_name="Dre", is_bot=False)
        followup = MagicMock(
            message_id=202,
            chat_id=chat_id,
            from_user=dre,
            text="Explain more details",
            entities=[],
            reply_to_message=bot_msg,
            quote=None,
            photo=None,
            document=None,
            message_thread_id=None,
        )
        followup.reply_text = AsyncMock(return_value=MagicMock(message_id=203))
        fake_file = MagicMock(file_size=len(img_bytes))
        fake_file.download_as_bytearray = AsyncMock(return_value=bytearray(img_bytes))
        ctx = MagicMock()
        ctx.bot = MagicMock(
            get_me=AsyncMock(return_value=handlers._bot_user),
            send_chat_action=AsyncMock(),
            get_file=AsyncMock(return_value=fake_file),
        )
        update = MagicMock(effective_chat=MagicMock(id=chat_id), effective_user=dre, effective_message=followup)

        await handlers.respond(update, ctx)
        ai.analyze_image.assert_awaited_once()
        followup.reply_text.assert_awaited_once_with("They need 55 more runs from 45 balls.")

    # 4. Invalid Telegram file references (never pretends to analyze)
    async def test_04_invalid_telegram_file_reference(self):
        bot = MagicMock()
        bot.get_file = AsyncMock(side_effect=BadRequest("File not found"))
        data, mime, err = await download_telegram_image(bot, "expired_file_id")
        self.assertIsNone(data)
        self.assertIsNone(mime)
        self.assertEqual(err, "not_found")

        data2, mime2, err2 = await download_telegram_image(bot, "")
        self.assertIsNone(data2)
        self.assertEqual(err2, "invalid_file_id")

    # 5 & 6. Supabase PostgreSQL persistence & simultaneous database writes
    async def test_05_06_postgres_repository_persistence_and_concurrency(self):
        shared_sqlite = sqlite3.connect(":memory:", check_same_thread=False)
        shared_sqlite.row_factory = sqlite3.Row
        pg_repo = PostgresRepository(
            "postgresql://user:pass@db.supabase.co:5432/postgres",
            connection_factory=lambda _dsn: _FakePgConn(shared_sqlite),
        )

        # Settings persistence
        pg_repo.save_settings(GroupSettings(-5001, "concise", ParticipationMode.ALWAYS, True))
        loaded = pg_repo.get_settings(-5001)
        self.assertEqual(loaded.participation_mode, ParticipationMode.ALWAYS)
        self.assertTrue(loaded.memory_enabled)

        # Concurrent writes across two groups
        async def writer(group_id: int, base: int):
            for i in range(10):
                pg_repo.add_context(
                    group_id,
                    ConversationMessage(
                        author=f"User{base}",
                        text=f"Message {i}",
                        message_id=base + i,
                        sender_id=base,
                        reply_to_message_id=(base + i - 1) if i > 0 else None,
                    ),
                )
                await asyncio.sleep(0.001)

        await asyncio.gather(writer(-5001, 100), writer(-5002, 200))

        chain_a = pg_repo.get_reply_chain(-5001, 109, max_depth=10)
        chain_b = pg_repo.get_reply_chain(-5002, 209, max_depth=10)
        self.assertEqual(len(chain_a), 10)
        self.assertEqual(len(chain_b), 10)
        self.assertTrue(all(m.author == "User100" for m in chain_a))
        self.assertTrue(all(m.author == "User200" for m in chain_b))

        # Factory returns SQLite when DATABASE_URL is empty
        local_repo = create_repository("", ":memory:")
        self.assertIsInstance(local_repo, SQLiteRepository)

    # 7. Cold-start memory persistence across fresh repository instances
    def test_07_cold_start_memory_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "cold_start.sqlite3"
            repo1 = SQLiteRepository(db_file)
            repo1.add_context(-6001, ConversationMessage("Alice", "What is Rust?", message_id=1))
            repo1.add_context(-6001, ConversationMessage("TelegramAI", "A systems language.", message_id=2, reply_to_message_id=1, is_bot=True))
            repo1.connection.close()

            # Simulate cold start with brand-new repository instance
            repo2 = SQLiteRepository(db_file)
            curr = ConversationMessage("Alice", "explain more", message_id=3, reply_to_message_id=2)
            ctx = assemble_context(repo2, -6001, curr, memory_enabled=True)
            self.assertEqual([m.message_id for m in ctx], [1, 2])
            repo2.connection.close()

    # 8 & 9. OpenRouter rate limits (QuotaExhaustedError) & strict free-model enforcement
    async def test_08_09_rate_limit_quota_error_and_free_model_only(self):
        client = OpenRouterClient("sk-or-v1-test-key", "nvidia/nemotron-3.5-super-vl:free", 5.0)
        attempted_models = []

        async def fake_post(url, json=None, headers=None):
            attempted_models.append(json["model"])
            resp = MagicMock(status_code=429, headers={"retry-after": "0.1"})
            resp.request = MagicMock()
            return resp

        mock_inner = MagicMock(post=fake_post)
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_inner)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("telegram_ai.openrouter.httpx.AsyncClient", return_value=mock_cm), \
             patch("asyncio.sleep", new=AsyncMock()):
            with self.assertRaises(QuotaExhaustedError):
                await client._chat_completion([{"role": "user", "content": "hi"}])

        # Bounded retries (3 attempts), strictly free model only
        self.assertEqual(len(attempted_models), 3)
        self.assertTrue(all(m == _OPENROUTER_FREE_MODEL for m in attempted_models))

        # Handler returns understandable quota message
        repo = SQLiteRepository(":memory:")
        ai = MagicMock(spec=OpenRouterClient)
        ai.answer = AsyncMock(side_effect=QuotaExhaustedError("429"))
        handlers = TelegramAIHandlers(repo, ai, max_context=5)
        handlers._bot_user = MagicMock(id=999, username="AIChatBot_12_bot")
        repo.save_settings(GroupSettings(-7001, participation_mode=ParticipationMode.ALWAYS))

        user = MagicMock(id=1, full_name="Alice", is_bot=False)
        msg = MagicMock(
            message_id=10, chat_id=-7001, from_user=user, text="hello",
            entities=[], reply_to_message=None, quote=None, photo=None, document=None, message_thread_id=None,
        )
        msg.reply_text = AsyncMock()
        update = MagicMock(effective_chat=MagicMock(id=-7001), effective_user=user, effective_message=msg)
        ctx = MagicMock(bot=MagicMock(get_me=AsyncMock(return_value=handlers._bot_user), send_chat_action=AsyncMock()))
        await handlers.respond(update, ctx)
        msg.reply_text.assert_awaited_once_with(
            "The free AI quota is temporarily rate-limited or exhausted. Please try again in a minute."
        )

    # 10 & 11. Duplicate webhook delivery & group isolation
    async def test_10_11_duplicate_delivery_and_group_isolation(self):
        repo = SQLiteRepository(":memory:")
        ai = MagicMock(spec=OpenRouterClient)
        ai.answer = AsyncMock(return_value="Reply")
        handlers = TelegramAIHandlers(repo, ai, max_context=5)
        handlers._bot_user = MagicMock(id=999, username="AIChatBot_12_bot")

        repo.save_settings(GroupSettings(-8001, participation_mode=ParticipationMode.ALWAYS))
        repo.save_settings(GroupSettings(-8002, participation_mode=ParticipationMode.ALWAYS))
        repo.add_context(-8001, ConversationMessage("Alice", "Secret group 1 topic", message_id=1))

        user = MagicMock(id=2, full_name="Bob", is_bot=False)
        msg = MagicMock(
            message_id=50, chat_id=-8002, from_user=user, text="Hello group 2",
            entities=[], reply_to_message=None, quote=None, photo=None, document=None, message_thread_id=None,
        )
        msg.reply_text = AsyncMock(return_value=MagicMock(message_id=51))
        update = MagicMock(effective_chat=MagicMock(id=-8002), effective_user=user, effective_message=msg)
        ctx = MagicMock(bot=MagicMock(get_me=AsyncMock(return_value=handlers._bot_user), send_chat_action=AsyncMock()))

        await handlers.respond(update, ctx)
        await handlers.respond(update, ctx)  # duplicate delivery

        ai.answer.assert_awaited_once()
        passed_history = ai.answer.call_args[0][2]
        self.assertFalse(any("Secret group 1 topic" in m.text for m in passed_history))

    # 12. SQLite migration compatibility (legacy table upgraded non-destructively)
    def test_12_sqlite_migration_compatibility(self):
        raw = sqlite3.connect(":memory:")
        raw.executescript(
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
            INSERT INTO conversation_context (chat_id, author, text) VALUES (-99, 'OldUser', 'Preserved row');
            """
        )
        raw.commit()
        repo = SQLiteRepository.__new__(SQLiteRepository)
        repo.connection = raw
        repo.connection.row_factory = sqlite3.Row
        repo._create_schema()

        rows = repo.get_context(-99, 5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].text, "Preserved row")


if __name__ == "__main__":
    unittest.main(verbosity=2)

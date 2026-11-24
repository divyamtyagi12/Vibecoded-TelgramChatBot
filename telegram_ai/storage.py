"""Local persistence with safe migrations, reply-chain tracking, and multimodal support."""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from os import PathLike
from typing import Any

from telegram_ai.domain import ConversationMessage, GroupSettings, ParticipationMode

logger = logging.getLogger(__name__)


class SQLiteRepository:
    def __init__(self, path: str | PathLike[str]) -> None:
        self._lock = threading.RLock()
        resolved = self._resolve_path(path)
        self.connection = sqlite3.connect(resolved, check_same_thread=False, timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    @staticmethod
    def _resolve_path(path: str | PathLike[str]) -> str:
        """Return a writable path, falling back to /tmp on read-only filesystems."""
        if str(path) == ":memory:":
            return ":memory:"
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            # Test that we can actually write here
            test_conn = sqlite3.connect(str(path))
            test_conn.close()
            return str(path)
        except (OSError, sqlite3.OperationalError):
            fallback = "/tmp/telegram_ai.sqlite3"
            logger.warning("DB path %s not writable, falling back to %s", path, fallback)
            return fallback

    def _create_schema(self) -> None:
        """Ensure initial tables exist and apply non-destructive schema migrations."""
        if not hasattr(self, "_lock"):
            self._lock = threading.RLock()
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS group_settings (
                chat_id INTEGER PRIMARY KEY,
                response_style TEXT NOT NULL,
                participation_mode TEXT NOT NULL CHECK (participation_mode IN ('mention', 'reply', 'always', 'off')),
                memory_enabled INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS conversation_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                author TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                comment TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        # Inspect existing columns in conversation_context to migrate safely
        cursor = self.connection.execute("PRAGMA table_info(conversation_context)")
        existing_cols = {row["name"] for row in cursor.fetchall()}

        new_columns = [
            ("message_id", "INTEGER"),
            ("sender_id", "INTEGER"),
            ("reply_to_message_id", "INTEGER"),
            ("thread_id", "INTEGER"),
            ("is_bot", "INTEGER NOT NULL DEFAULT 0"),
            ("media_type", "TEXT"),
            ("media_file_id", "TEXT"),
            ("media_unique_id", "TEXT"),
            ("media_summary", "TEXT"),
            ("quote_text", "TEXT"),
            ("conversation_id", "TEXT"),
            ("failed_operation", "TEXT"),
        ]

        for col_name, col_type in new_columns:
            if col_name not in existing_cols:
                try:
                    self.connection.execute(
                        f"ALTER TABLE conversation_context ADD COLUMN {col_name} {col_type}"
                    )
                except sqlite3.OperationalError:
                    logger.debug("Column %s already added or migration skipped", col_name)

        # Create indexes for fast lookup and chain reconstruction
        self.connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS context_by_chat ON conversation_context(chat_id, id DESC);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_msg ON conversation_context(chat_id, message_id);
            CREATE INDEX IF NOT EXISTS idx_chat_reply ON conversation_context(chat_id, reply_to_message_id);
            CREATE INDEX IF NOT EXISTS idx_chat_thread ON conversation_context(chat_id, thread_id);
            CREATE INDEX IF NOT EXISTS idx_chat_created ON conversation_context(chat_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_chat_conv ON conversation_context(chat_id, conversation_id);
            """
        )
        self.connection.commit()

    def get_settings(self, chat_id: int) -> GroupSettings:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM group_settings WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            if row is None:
                settings = GroupSettings(chat_id=chat_id)
                self.save_settings(settings)
                return settings
            return GroupSettings(
                chat_id=row["chat_id"],
                response_style=row["response_style"],
                participation_mode=ParticipationMode(row["participation_mode"]),
                memory_enabled=bool(row["memory_enabled"]),
            )

    def save_settings(self, settings: GroupSettings) -> None:
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO group_settings (chat_id, response_style, participation_mode, memory_enabled)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    response_style = excluded.response_style,
                    participation_mode = excluded.participation_mode,
                    memory_enabled = excluded.memory_enabled
                """,
                (
                    settings.chat_id,
                    settings.response_style,
                    settings.participation_mode.value,
                    settings.memory_enabled,
                ),
            )
            self.connection.commit()

    def add_context(self, chat_id: int, message: ConversationMessage) -> None:
        """Store or update a message in the conversation repository safely."""
        with self._lock:
            try:
                c_id = int(chat_id) if isinstance(chat_id, (int, float)) else 0
                m_id = int(message.message_id) if isinstance(message.message_id, (int, float)) else None
                s_id = int(message.sender_id) if isinstance(message.sender_id, (int, float)) else None
                r_id = int(message.reply_to_message_id) if isinstance(message.reply_to_message_id, (int, float)) else None
                t_id = int(message.thread_id) if isinstance(message.thread_id, (int, float)) else None

                author = (str(message.author or ""))[:128]
                text = (str(message.text or ""))[:4000]
                quote_text = (str(message.quote_text))[:1000] if isinstance(message.quote_text, str) else None
                media_summary = (str(message.media_summary))[:2000] if isinstance(message.media_summary, str) else None
                media_type = str(message.media_type) if isinstance(message.media_type, str) else None
                media_file_id = str(message.media_file_id) if isinstance(message.media_file_id, str) else None
                media_unique_id = str(message.media_unique_id) if isinstance(message.media_unique_id, str) else None
                conversation_id = str(message.conversation_id)[:128] if isinstance(message.conversation_id, str) else None
                failed_operation = str(message.failed_operation)[:2000] if isinstance(message.failed_operation, str) else None

                if m_id is not None:
                    self.connection.execute(
                        """
                        INSERT INTO conversation_context (
                            chat_id, message_id, sender_id, author, text,
                            reply_to_message_id, thread_id, is_bot,
                            media_type, media_file_id, media_unique_id,
                            media_summary, quote_text, conversation_id, failed_operation
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(chat_id, message_id) DO UPDATE SET
                            text = excluded.text,
                            media_summary = COALESCE(excluded.media_summary, conversation_context.media_summary),
                            media_file_id = COALESCE(excluded.media_file_id, conversation_context.media_file_id),
                            media_type = COALESCE(excluded.media_type, conversation_context.media_type),
                            quote_text = COALESCE(excluded.quote_text, conversation_context.quote_text),
                            conversation_id = COALESCE(excluded.conversation_id, conversation_context.conversation_id),
                            failed_operation = COALESCE(excluded.failed_operation, conversation_context.failed_operation)
                        """,
                        (
                            c_id,
                            m_id,
                            s_id,
                            author,
                            text,
                            r_id,
                            t_id,
                            1 if message.is_bot else 0,
                            media_type,
                            media_file_id,
                            media_unique_id,
                            media_summary,
                            quote_text,
                            conversation_id,
                            failed_operation,
                        ),
                    )
                else:
                    self.connection.execute(
                        """
                        INSERT INTO conversation_context (
                            chat_id, author, text, thread_id, is_bot,
                            media_type, media_file_id, media_unique_id,
                            media_summary, quote_text, conversation_id, failed_operation
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            c_id,
                            author,
                            text,
                            t_id,
                            1 if message.is_bot else 0,
                            media_type,
                            media_file_id,
                            media_unique_id,
                            media_summary,
                            quote_text,
                            conversation_id,
                            failed_operation,
                        ),
                    )
                self.connection.commit()
            except sqlite3.OperationalError:
                logger.warning(
                    "add_context: could not persist context for chat %s (OperationalError)",
                    chat_id,
                    exc_info=True,
                )

    def get_message(self, chat_id: int, message_id: int) -> ConversationMessage | None:
        """Retrieve a specific message by its Telegram message_id."""
        if not isinstance(message_id, (int, float)) or not isinstance(chat_id, (int, float)):
            return None
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM conversation_context WHERE chat_id = ? AND message_id = ? ORDER BY id DESC LIMIT 1",
                (int(chat_id), int(message_id)),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_message(row)

    def update_media_summary(self, chat_id: int, message_id: int, summary: str) -> None:
        """Update or cache the visual summary for a stored image message."""
        if not isinstance(message_id, (int, float)) or not isinstance(chat_id, (int, float)):
            return
        try:
            self.connection.execute(
                "UPDATE conversation_context SET media_summary = ? WHERE chat_id = ? AND message_id = ?",
                (str(summary)[:2000], int(chat_id), int(message_id)),
            )
            self.connection.commit()
        except sqlite3.OperationalError:
            logger.warning(
                "update_media_summary: failed for chat %s message %s",
                chat_id,
                message_id,
                exc_info=True,
            )

    def update_failed_operation(
        self, chat_id: int, message_id: int, failed_operation: str | None
    ) -> None:
        """Store or clear the failed operation for a message to allow contextual retry."""
        if not isinstance(message_id, (int, float)) or not isinstance(chat_id, (int, float)):
            return
        try:
            val = str(failed_operation)[:2000] if failed_operation is not None else None
            self.connection.execute(
                "UPDATE conversation_context SET failed_operation = ? WHERE chat_id = ? AND message_id = ?",
                (val, int(chat_id), int(message_id)),
            )
            self.connection.commit()
        except sqlite3.OperationalError:
            logger.warning(
                "update_failed_operation: failed for chat %s message %s",
                chat_id,
                message_id,
                exc_info=True,
            )

    def get_reply_chain(
        self, chat_id: int, message_id: int, max_depth: int = 10
    ) -> list[ConversationMessage]:
        """Reconstruct the entire conversation ancestor chain by traversing reply_to_message_id upwards.
        
        Returns the messages in chronological order (oldest ancestor first, replied-to message last).
        """
        if not isinstance(message_id, (int, float)) or not isinstance(chat_id, (int, float)):
            return []
        chain: list[ConversationMessage] = []
        curr_id: int | None = int(message_id)
        visited: set[int] = set()

        for _ in range(max_depth):
            if curr_id is None or curr_id in visited:
                break
            visited.add(curr_id)
            msg = self.get_message(chat_id, curr_id)
            if msg is None:
                break
            chain.append(msg)
            curr_id = msg.reply_to_message_id

        # Reverse so chronological order: root ancestor -> direct parent
        chain.reverse()
        return chain

    def get_thread_context(
        self,
        chat_id: int,
        thread_id: int | None,
        limit: int = 10,
        exclude_message_ids: set[int] | None = None,
    ) -> list[ConversationMessage]:
        """Retrieve recent messages belonging to a specific topic/thread or general chat."""
        c_id = int(chat_id) if isinstance(chat_id, (int, float)) else 0
        t_id = int(thread_id) if isinstance(thread_id, (int, float)) else None
        exclude = set(exclude_message_ids) if exclude_message_ids else set()

        with self._lock:
            if t_id is not None:
                rows = self.connection.execute(
                    """
                    SELECT * FROM conversation_context
                    WHERE chat_id = ? AND thread_id = ?
                    ORDER BY id DESC LIMIT ?
                    """,
                    (c_id, t_id, limit * 2),
                ).fetchall()
            else:
                rows = self.connection.execute(
                    """
                    SELECT * FROM conversation_context
                    WHERE chat_id = ?
                    ORDER BY id DESC LIMIT ?
                    """,
                    (c_id, limit * 2),
                ).fetchall()

        filtered: list[ConversationMessage] = []
        for row in rows:
            msg = self._row_to_message(row)
            if msg.message_id and msg.message_id in exclude:
                continue
            filtered.append(msg)
            if len(filtered) >= limit:
                break

        return list(reversed(filtered))

    def get_context(self, chat_id: int, limit: int) -> list[ConversationMessage]:
        """Retrieve the last `limit` messages chronologically (standard context lookup)."""
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM conversation_context WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
                (chat_id, limit),
            ).fetchall()
        return [self._row_to_message(row) for row in reversed(rows)]

    def cleanup_old_messages(self, retention_days: int = 14) -> int:
        """Purge messages older than retention_days to maintain storage health."""
        try:
            cursor = self.connection.execute(
                "DELETE FROM conversation_context WHERE datetime(created_at) < datetime('now', ?)",
                (f"-{retention_days} days",),
            )
            deleted = cursor.rowcount
            self.connection.commit()
            return deleted
        except sqlite3.OperationalError:
            logger.warning("cleanup_old_messages: OperationalError during cleanup", exc_info=True)
            return 0

    def add_feedback(self, chat_id: int, user_id: int, rating: int, comment: str) -> None:
        self.connection.execute(
            "INSERT INTO feedback (chat_id, user_id, rating, comment) VALUES (?, ?, ?, ?)",
            (chat_id, user_id, rating, comment[:2000]),
        )
        self.connection.commit()

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> ConversationMessage:
        keys = set(row.keys())
        return ConversationMessage(
            author=row["author"],
            text=row["text"],
            message_id=row["message_id"] if "message_id" in keys else None,
            sender_id=row["sender_id"] if "sender_id" in keys else None,
            reply_to_message_id=row["reply_to_message_id"] if "reply_to_message_id" in keys else None,
            thread_id=row["thread_id"] if "thread_id" in keys else None,
            is_bot=bool(row["is_bot"]) if "is_bot" in keys else False,
            media_type=row["media_type"] if "media_type" in keys else None,
            media_file_id=row["media_file_id"] if "media_file_id" in keys else None,
            media_unique_id=row["media_unique_id"] if "media_unique_id" in keys else None,
            media_summary=row["media_summary"] if "media_summary" in keys else None,
            quote_text=row["quote_text"] if "quote_text" in keys else None,
            conversation_id=row["conversation_id"] if "conversation_id" in keys else None,
            failed_operation=row["failed_operation"] if "failed_operation" in keys else None,
            created_at=row["created_at"] if "created_at" in keys else None,
        )


class PostgresRepository:
    """Persistent PostgreSQL repository designed for Supabase on serverless Vercel."""

    def __init__(self, dsn: str, connection_factory: Any = None) -> None:
        self._dsn = dsn
        self._connection_factory = connection_factory
        self._create_schema()

    def _connect(self) -> Any:
        if self._connection_factory is not None:
            return self._connection_factory(self._dsn)
        import psycopg  # pyrefly: ignore[missing-import]
        from psycopg.rows import dict_row  # pyrefly: ignore[missing-import]

        return psycopg.connect(
            self._dsn,
            row_factory=dict_row,
            autocommit=False,
            prepare_threshold=None,  # Compatible with Supabase PgBouncer transaction pooler
            connect_timeout=10,
        )

    def _create_schema(self) -> None:
        """Create tables and indexes idempotently on Supabase PostgreSQL."""
        ddl_statements = (
            """
            CREATE TABLE IF NOT EXISTS group_settings (
                chat_id BIGINT PRIMARY KEY,
                response_style TEXT NOT NULL,
                participation_mode TEXT NOT NULL CHECK (participation_mode IN ('mention', 'reply', 'always', 'off')),
                memory_enabled BOOLEAN NOT NULL DEFAULT TRUE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS conversation_context (
                id BIGSERIAL PRIMARY KEY,
                chat_id BIGINT NOT NULL,
                message_id BIGINT,
                sender_id BIGINT,
                author TEXT NOT NULL,
                text TEXT NOT NULL,
                reply_to_message_id BIGINT,
                thread_id BIGINT,
                is_bot BOOLEAN NOT NULL DEFAULT FALSE,
                media_type TEXT,
                media_file_id TEXT,
                media_unique_id TEXT,
                media_summary TEXT,
                quote_text TEXT,
                conversation_id TEXT,
                failed_operation TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id BIGSERIAL PRIMARY KEY,
                chat_id BIGINT NOT NULL,
                user_id BIGINT NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                comment TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS context_by_chat ON conversation_context(chat_id, id DESC)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_msg ON conversation_context(chat_id, message_id) WHERE message_id IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_chat_reply ON conversation_context(chat_id, reply_to_message_id)",
            "CREATE INDEX IF NOT EXISTS idx_chat_thread ON conversation_context(chat_id, thread_id)",
            "CREATE INDEX IF NOT EXISTS idx_chat_created ON conversation_context(chat_id, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_chat_conv ON conversation_context(chat_id, conversation_id)",
        )
        with self._connect() as conn:
            with conn.cursor() as cur:
                for stmt in ddl_statements:
                    cur.execute(stmt)
            conn.commit()

    def get_settings(self, chat_id: int) -> GroupSettings:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM group_settings WHERE chat_id = %s", (int(chat_id),))
                row = cur.fetchone()
        if row is None:
            settings = GroupSettings(chat_id=int(chat_id))
            self.save_settings(settings)
            return settings
        return GroupSettings(
            chat_id=int(row["chat_id"]),
            response_style=str(row["response_style"]),
            participation_mode=ParticipationMode(row["participation_mode"]),
            memory_enabled=bool(row["memory_enabled"]),
        )

    def save_settings(self, settings: GroupSettings) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO group_settings (chat_id, response_style, participation_mode, memory_enabled)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (chat_id) DO UPDATE SET
                        response_style = EXCLUDED.response_style,
                        participation_mode = EXCLUDED.participation_mode,
                        memory_enabled = EXCLUDED.memory_enabled
                    """,
                    (
                        int(settings.chat_id),
                        settings.response_style,
                        settings.participation_mode.value,
                        bool(settings.memory_enabled),
                    ),
                )
            conn.commit()

    def add_context(self, chat_id: int, message: ConversationMessage) -> None:
        """Persist or upsert a conversation message safely under concurrent webhook load."""
        try:
            c_id = int(chat_id) if isinstance(chat_id, (int, float)) else 0
            m_id = int(message.message_id) if isinstance(message.message_id, (int, float)) else None
            s_id = int(message.sender_id) if isinstance(message.sender_id, (int, float)) else None
            r_id = int(message.reply_to_message_id) if isinstance(message.reply_to_message_id, (int, float)) else None
            t_id = int(message.thread_id) if isinstance(message.thread_id, (int, float)) else None

            author = (str(message.author or ""))[:128]
            text = (str(message.text or ""))[:4000]
            quote_text = (str(message.quote_text))[:1000] if isinstance(message.quote_text, str) else None
            media_summary = (str(message.media_summary))[:2000] if isinstance(message.media_summary, str) else None
            media_type = str(message.media_type) if isinstance(message.media_type, str) else None
            media_file_id = str(message.media_file_id) if isinstance(message.media_file_id, str) else None
            media_unique_id = str(message.media_unique_id) if isinstance(message.media_unique_id, str) else None
            conversation_id = str(message.conversation_id)[:128] if isinstance(message.conversation_id, str) else None
            failed_operation = str(message.failed_operation)[:2000] if isinstance(message.failed_operation, str) else None

            with self._connect() as conn:
                with conn.cursor() as cur:
                    if m_id is not None:
                        cur.execute(
                            """
                            INSERT INTO conversation_context (
                                chat_id, message_id, sender_id, author, text,
                                reply_to_message_id, thread_id, is_bot,
                                media_type, media_file_id, media_unique_id,
                                media_summary, quote_text, conversation_id, failed_operation
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (chat_id, message_id) WHERE message_id IS NOT NULL DO UPDATE SET
                                text = EXCLUDED.text,
                                media_summary = COALESCE(EXCLUDED.media_summary, conversation_context.media_summary),
                                media_file_id = COALESCE(EXCLUDED.media_file_id, conversation_context.media_file_id),
                                media_type = COALESCE(EXCLUDED.media_type, conversation_context.media_type),
                                quote_text = COALESCE(EXCLUDED.quote_text, conversation_context.quote_text),
                                conversation_id = COALESCE(EXCLUDED.conversation_id, conversation_context.conversation_id),
                                failed_operation = COALESCE(EXCLUDED.failed_operation, conversation_context.failed_operation)
                            """,
                            (
                                c_id, m_id, s_id, author, text,
                                r_id, t_id, bool(message.is_bot),
                                media_type, media_file_id, media_unique_id,
                                media_summary, quote_text, conversation_id, failed_operation,
                            ),
                        )
                    else:
                        cur.execute(
                            """
                            INSERT INTO conversation_context (
                                chat_id, author, text, thread_id, is_bot,
                                media_type, media_file_id, media_unique_id,
                                media_summary, quote_text, conversation_id, failed_operation
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                c_id, author, text, t_id, bool(message.is_bot),
                                media_type, media_file_id, media_unique_id,
                                media_summary, quote_text, conversation_id, failed_operation,
                            ),
                        )
                conn.commit()
        except Exception:
            logger.warning("Postgres add_context failed for chat %s", chat_id, exc_info=True)

    def get_message(self, chat_id: int, message_id: int) -> ConversationMessage | None:
        if not isinstance(message_id, (int, float)) or not isinstance(chat_id, (int, float)):
            return None
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM conversation_context WHERE chat_id = %s AND message_id = %s ORDER BY id DESC LIMIT 1",
                    (int(chat_id), int(message_id)),
                )
                row = cur.fetchone()
        return self._row_to_message(row) if row is not None else None

    def update_media_summary(self, chat_id: int, message_id: int, summary: str) -> None:
        if not isinstance(message_id, (int, float)) or not isinstance(chat_id, (int, float)):
            return
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE conversation_context SET media_summary = %s WHERE chat_id = %s AND message_id = %s",
                        (str(summary)[:2000], int(chat_id), int(message_id)),
                    )
                conn.commit()
        except Exception:
            logger.warning("Postgres update_media_summary failed for chat %s msg %s", chat_id, message_id, exc_info=True)

    def update_failed_operation(
        self, chat_id: int, message_id: int, failed_operation: str | None
    ) -> None:
        if not isinstance(message_id, (int, float)) or not isinstance(chat_id, (int, float)):
            return
        try:
            val = str(failed_operation)[:2000] if failed_operation is not None else None
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE conversation_context SET failed_operation = %s WHERE chat_id = %s AND message_id = %s",
                        (val, int(chat_id), int(message_id)),
                    )
                conn.commit()
        except Exception:
            logger.warning("Postgres update_failed_operation failed for chat %s msg %s", chat_id, message_id, exc_info=True)

    def get_reply_chain(
        self, chat_id: int, message_id: int, max_depth: int = 10
    ) -> list[ConversationMessage]:
        if not isinstance(message_id, (int, float)) or not isinstance(chat_id, (int, float)):
            return []
        chain: list[ConversationMessage] = []
        curr_id: int | None = int(message_id)
        visited: set[int] = set()
        for _ in range(max_depth):
            if curr_id is None or curr_id in visited:
                break
            visited.add(curr_id)
            msg = self.get_message(chat_id, curr_id)
            if msg is None:
                break
            chain.append(msg)
            curr_id = msg.reply_to_message_id
        chain.reverse()
        return chain

    def get_thread_context(
        self,
        chat_id: int,
        thread_id: int | None,
        limit: int = 10,
        exclude_message_ids: set[int] | None = None,
    ) -> list[ConversationMessage]:
        c_id = int(chat_id) if isinstance(chat_id, (int, float)) else 0
        t_id = int(thread_id) if isinstance(thread_id, (int, float)) else None
        exclude = set(exclude_message_ids) if exclude_message_ids else set()

        with self._connect() as conn:
            with conn.cursor() as cur:
                if t_id is not None:
                    cur.execute(
                        "SELECT * FROM conversation_context WHERE chat_id = %s AND thread_id = %s ORDER BY id DESC LIMIT %s",
                        (c_id, t_id, limit * 2),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM conversation_context WHERE chat_id = %s ORDER BY id DESC LIMIT %s",
                        (c_id, limit * 2),
                    )
                rows = cur.fetchall()

        filtered: list[ConversationMessage] = []
        for row in rows:
            msg = self._row_to_message(row)
            if msg.message_id and msg.message_id in exclude:
                continue
            filtered.append(msg)
            if len(filtered) >= limit:
                break
        return list(reversed(filtered))

    def get_context(self, chat_id: int, limit: int) -> list[ConversationMessage]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM conversation_context WHERE chat_id = %s ORDER BY id DESC LIMIT %s",
                    (int(chat_id), int(limit)),
                )
                rows = cur.fetchall()
        return [self._row_to_message(row) for row in reversed(rows)]

    def cleanup_old_messages(self, retention_days: int = 14) -> int:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM conversation_context WHERE created_at < NOW() - (%s || ' days')::INTERVAL",
                        (str(int(retention_days)),),
                    )
                    deleted = cur.rowcount or 0
                conn.commit()
            return deleted
        except Exception:
            logger.warning("Postgres cleanup_old_messages failed", exc_info=True)
            return 0

    def add_feedback(self, chat_id: int, user_id: int, rating: int, comment: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO feedback (chat_id, user_id, rating, comment) VALUES (%s, %s, %s, %s)",
                    (int(chat_id), int(user_id), int(rating), comment[:2000]),
                )
            conn.commit()

    @staticmethod
    def _row_to_message(row: dict[str, Any]) -> ConversationMessage:
        created = row.get("created_at")
        return ConversationMessage(
            author=str(row.get("author") or ""),
            text=str(row.get("text") or ""),
            message_id=row.get("message_id"),
            sender_id=row.get("sender_id"),
            reply_to_message_id=row.get("reply_to_message_id"),
            thread_id=row.get("thread_id"),
            is_bot=bool(row.get("is_bot", False)),
            media_type=row.get("media_type"),
            media_file_id=row.get("media_file_id"),
            media_unique_id=row.get("media_unique_id"),
            media_summary=row.get("media_summary"),
            quote_text=row.get("quote_text"),
            conversation_id=row.get("conversation_id"),
            failed_operation=row.get("failed_operation"),
            created_at=str(created) if created is not None else None,
        )


def create_repository(
    database_url: str | None,
    sqlite_path: str | PathLike[str],
) -> SQLiteRepository | PostgresRepository:
    """Create a Supabase PostgreSQL repository when DATABASE_URL/SUPABASE_DB_URL is set, else SQLite."""
    clean_url = (database_url or "").strip()
    if clean_url.startswith(("postgres://", "postgresql://")):
        logger.info("Initializing persistent Supabase PostgreSQL repository")
        return PostgresRepository(clean_url)
    logger.info("Initializing local SQLite repository at %s", sqlite_path)
    return SQLiteRepository(sqlite_path)

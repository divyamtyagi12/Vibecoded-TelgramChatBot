"""Local persistence with safe migrations, reply-chain tracking, and multimodal support."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from os import PathLike
from typing import Any

from telegram_ai.domain import ConversationMessage, GroupSettings, ParticipationMode

logger = logging.getLogger(__name__)


class SQLiteRepository:
    def __init__(self, path: str | PathLike[str]) -> None:
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
            CREATE TABLE IF NOT EXISTS known_users (
                chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                username TEXT,
                full_name TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (chat_id, user_id)
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

    def upsert_known_user(
        self,
        chat_id: int,
        user_id: int,
        username: str | None = None,
        full_name: str | None = None,
    ) -> None:
        """Record or update a user's @username and display name for moderation lookups."""
        if not isinstance(chat_id, (int, float)) or not isinstance(user_id, (int, float)):
            return
        try:
            clean_uname = username.strip().lstrip("@").lower() if isinstance(username, str) and username.strip() else None
            clean_fname = full_name.strip()[:128] if isinstance(full_name, str) and full_name.strip() else None
            self.connection.execute(
                """
                INSERT INTO known_users (chat_id, user_id, username, full_name, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(chat_id, user_id) DO UPDATE SET
                    username = COALESCE(excluded.username, known_users.username),
                    full_name = COALESCE(excluded.full_name, known_users.full_name),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (int(chat_id), int(user_id), clean_uname, clean_fname),
            )
            self.connection.commit()
        except sqlite3.OperationalError:
            logger.debug("upsert_known_user failed for chat %s user %s", chat_id, user_id)

    def find_user_by_name(self, chat_id: int, name_query: str) -> tuple[int, str] | None:
        """Find a sender_id and author/username in known_users or recent chat history matching name_query."""
        clean = name_query.strip().lstrip("@").lower()
        if not clean:
            return None
        c_id = int(chat_id)
        try:
            known_rows = self.connection.execute(
                """
                SELECT user_id, username, full_name FROM known_users
                WHERE chat_id = ?
                ORDER BY updated_at DESC LIMIT 500
                """,
                (c_id,),
            ).fetchall()
            # 1. Exact @username or full_name match in known_users
            for row in known_rows:
                uname = str(row["username"] or "").lower()
                fname = str(row["full_name"] or "")
                if uname and uname == clean:
                    return int(row["user_id"]), f"@{row['username']}"
                if fname and fname.lower() == clean:
                    return int(row["user_id"]), fname
            # 2. Partial match in known_users
            for row in known_rows:
                uname = str(row["username"] or "").lower()
                fname = str(row["full_name"] or "")
                if (uname and clean in uname) or (fname and clean in fname.lower()):
                    display = f"@{row['username']}" if uname == clean else (fname or f"@{row['username']}")
                    return int(row["user_id"]), display
        except sqlite3.OperationalError:
            pass

        rows = self.connection.execute(
            """
            SELECT sender_id, author FROM conversation_context
            WHERE chat_id = ? AND sender_id IS NOT NULL AND is_bot = 0
            ORDER BY id DESC LIMIT 200
            """,
            (c_id,),
        ).fetchall()
        # Exact match first
        for row in rows:
            author = str(row["author"] or "")
            if author.lower() == clean:
                return int(row["sender_id"]), author
        # Partial / prefix match next
        for row in rows:
            author = str(row["author"] or "")
            if clean in author.lower():
                return int(row["sender_id"]), author
        return None

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

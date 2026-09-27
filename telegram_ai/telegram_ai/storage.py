"""Local persistence. Replace with a managed database repository in production."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from os import PathLike

from telegram_ai.domain import ConversationMessage, GroupSettings, ParticipationMode

logger = logging.getLogger(__name__)


class SQLiteRepository:
    def __init__(self, path: str | PathLike[str]) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def _create_schema(self) -> None:
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
            CREATE INDEX IF NOT EXISTS context_by_chat ON conversation_context(chat_id, id DESC);
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
        # Saving short-term context is a best-effort side effect of every
        # message. A locked/readonly database file should not take down the
        # whole reply pipeline (it previously crashed before the AI was
        # even called) — log and continue instead.
        try:
            self.connection.execute(
                "INSERT INTO conversation_context (chat_id, author, text) VALUES (?, ?, ?)",
                (chat_id, message.author[:128], message.text[:4000]),
            )
            self.connection.commit()
        except sqlite3.Error:
            logger.exception("Failed to save conversation context for chat %s", chat_id)

    def get_context(self, chat_id: int, limit: int) -> list[ConversationMessage]:
        rows = self.connection.execute(
            "SELECT author, text FROM conversation_context WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
        return [ConversationMessage(author=row["author"], text=row["text"]) for row in reversed(rows)]

    def add_feedback(self, chat_id: int, user_id: int, rating: int, comment: str) -> None:
        self.connection.execute(
            "INSERT INTO feedback (chat_id, user_id, rating, comment) VALUES (?, ?, ?, ?)",
            (chat_id, user_id, rating, comment[:2000]),
        )
        self.connection.commit()

"""Application configuration loaded from the environment."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    gemini_api_key: str
    gemini_model: str
    gemini_image_model: str
    telegram_webhook_secret: str
    database_path: Path
    max_context_messages: int
    request_timeout_seconds: float

    @classmethod
    def from_environment(cls) -> "Settings":
        project_directory = Path(__file__).resolve().parent.parent
        load_dotenv(project_directory / ".env")
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        missing = [
            name
            for name, value in (
                ("TELEGRAM_BOT_TOKEN", token),
                ("GEMINI_API_KEY", api_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")

        max_context = int(os.getenv("MAX_CONTEXT_MESSAGES", "8"))
        if not 0 <= max_context <= 30:
            raise RuntimeError("MAX_CONTEXT_MESSAGES must be between 0 and 30")

        timeout = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "12"))
        if not 1 <= timeout <= 60:
            raise RuntimeError("REQUEST_TIMEOUT_SECONDS must be between 1 and 60")

        default_database_path = "/tmp/telegram_ai.sqlite3" if os.getenv("VERCEL") else "data/telegram_ai.sqlite3"
        database_path = Path(os.getenv("DATABASE_PATH", default_database_path))
        if not database_path.is_absolute():
            database_path = project_directory / database_path

        return cls(
            telegram_bot_token=token,
            gemini_api_key=api_key,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
            gemini_image_model=os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image").strip(),
            telegram_webhook_secret=os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip(),
            database_path=database_path,
            max_context_messages=max_context,
            request_timeout_seconds=timeout,
        )

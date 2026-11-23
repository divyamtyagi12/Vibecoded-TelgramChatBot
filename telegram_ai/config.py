"""Application configuration loaded from the environment."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_OPENROUTER_FREE_MODEL = "nvidia/nemotron-3.5-super-vl:free"


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    openrouter_api_key: str
    openrouter_model: str
    telegram_webhook_secret: str
    database_path: Path
    max_context_messages: int
    request_timeout_seconds: float
    openrouter_vision_model: str = DEFAULT_OPENROUTER_FREE_MODEL
    database_url: str = ""
    history_retention_days: int = 14
    max_context_tokens: int = 3000

    # Backward-compatible aliases during migration from Groq naming
    @property
    def groq_api_key(self) -> str:
        return self.openrouter_api_key

    @property
    def groq_model(self) -> str:
        return self.openrouter_model

    @property
    def groq_vision_model(self) -> str:
        return self.openrouter_vision_model

    @classmethod
    def from_environment(cls) -> "Settings":
        project_directory = Path(__file__).resolve().parent.parent
        load_dotenv(project_directory / ".env")
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        api_key = (
            os.getenv("OPENROUTER_API_KEY", "").strip()
            or os.getenv("GROQ_API_KEY", "").strip()
        )
        missing = [
            name
            for name, value in (
                ("TELEGRAM_BOT_TOKEN", token),
                ("OPENROUTER_API_KEY", api_key),
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

        retention = int(os.getenv("HISTORY_RETENTION_DAYS", "14"))
        if not 1 <= retention <= 365:
            retention = 14

        tokens_limit = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))
        if not 500 <= tokens_limit <= 16000:
            tokens_limit = 3000

        database_url = (
            os.getenv("SUPABASE_DB_URL", "").strip()
            or os.getenv("DATABASE_URL", "").strip()
        )

        if os.getenv("VERCEL"):
            database_path = Path("/tmp/telegram_ai.sqlite3")
        else:
            database_path = Path(os.getenv("DATABASE_PATH", "data/telegram_ai.sqlite3"))
            if not database_path.is_absolute():
                database_path = project_directory / database_path

        default_model = (
            DEFAULT_OPENROUTER_FREE_MODEL
            if api_key.startswith("sk-or-")
            else "openai/gpt-oss-120b"
        )
        raw_model = (
            os.getenv("OPENROUTER_MODEL", "").strip()
            or os.getenv("GROQ_MODEL", "").strip()
            or default_model
        )
        if api_key.startswith("sk-or-") and raw_model != DEFAULT_OPENROUTER_FREE_MODEL:
            raw_model = DEFAULT_OPENROUTER_FREE_MODEL

        raw_vision_model = (
            os.getenv("OPENROUTER_VISION_MODEL", "").strip()
            or os.getenv("GROQ_VISION_MODEL", "").strip()
            or (DEFAULT_OPENROUTER_FREE_MODEL if api_key.startswith("sk-or-") else "qwen/qwen3.8-27b")
        )
        if api_key.startswith("sk-or-") and raw_vision_model != DEFAULT_OPENROUTER_FREE_MODEL:
            raw_vision_model = DEFAULT_OPENROUTER_FREE_MODEL

        return cls(
            telegram_bot_token=token,
            openrouter_api_key=api_key,
            openrouter_model=raw_model,
            telegram_webhook_secret=os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip(),
            database_path=database_path,
            max_context_messages=max_context,
            request_timeout_seconds=timeout,
            openrouter_vision_model=raw_vision_model,
            database_url=database_url,
            history_retention_days=retention,
            max_context_tokens=tokens_limit,
        )

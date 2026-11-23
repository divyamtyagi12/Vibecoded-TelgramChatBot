"""Application configuration loaded from the environment."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    groq_api_key: str
    groq_model: str
    telegram_webhook_secret: str
    database_path: Path
    max_context_messages: int
    request_timeout_seconds: float
    groq_vision_model: str = "nvidia/nemotron-3.5-super-vl"
    history_retention_days: int = 14
    max_context_tokens: int = 3000

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
                ("GROQ_API_KEY", api_key),
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

        if os.getenv("VERCEL"):
            database_path = Path("/tmp/telegram_ai.sqlite3")
        else:
            database_path = Path(os.getenv("DATABASE_PATH", "data/telegram_ai.sqlite3"))
            if not database_path.is_absolute():
                database_path = project_directory / database_path

        default_model = "nvidia/nemotron-3.5-super-vl" if api_key.startswith("sk-or-") else "openai/gpt-oss-120b"
        raw_model = (
            os.getenv("OPENROUTER_MODEL", "").strip()
            or os.getenv("GROQ_MODEL", "").strip()
            or default_model
        )
        if api_key.startswith("sk-or-") and raw_model in (
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
            "qwen/qwen3.8-27b",
        ):
            raw_model = "nvidia/nemotron-3.5-super-vl"

        raw_vision_model = (
            os.getenv("OPENROUTER_VISION_MODEL", "").strip()
            or os.getenv("GROQ_VISION_MODEL", "").strip()
            or ("nvidia/nemotron-3.5-super-vl" if api_key.startswith("sk-or-") else "qwen/qwen3.8-27b")
        )
        if api_key.startswith("sk-or-") and raw_vision_model == "qwen/qwen3.8-27b":
            raw_vision_model = "nvidia/nemotron-3.5-super-vl"

        return cls(
            telegram_bot_token=token,
            groq_api_key=api_key,
            groq_model=raw_model,
            telegram_webhook_secret=os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip(),
            database_path=database_path,
            max_context_messages=max_context,
            request_timeout_seconds=timeout,
            groq_vision_model=raw_vision_model,
            history_retention_days=retention,
            max_context_tokens=tokens_limit,
        )

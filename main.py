"""Entrypoint for the long-polling Telegram bot."""

from __future__ import annotations

import logging

from telegram.ext import Application

from telegram_ai.config import Settings
from telegram_ai.gemini import GeminiClient
from telegram_ai.handlers import TelegramAIHandlers, register_handlers
from telegram_ai.storage import SQLiteRepository


def run() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    settings = Settings.from_environment()
    repository = SQLiteRepository(settings.database_path)
    ai = GeminiClient(settings.gemini_api_key, settings.gemini_model, settings.request_timeout_seconds)
    handlers = TelegramAIHandlers(repository, ai, settings.max_context_messages)
    application = Application.builder().token(settings.telegram_bot_token).post_init(handlers.configure).build()
    register_handlers(application, handlers)
    application.run_polling(allowed_updates=["message"])

"""Entrypoint for the long-polling Telegram bot."""

from __future__ import annotations

import logging

from telegram.ext import Application

from telegram_ai.config import Settings
from telegram_ai.handlers import TelegramAIHandlers, register_handlers
from telegram_ai.openrouter import OpenRouterClient
from telegram_ai.storage import create_repository


def run() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    settings = Settings.from_environment()
    repository = create_repository(settings.database_url, settings.database_path)
    repository.cleanup_old_messages(settings.history_retention_days)
    ai = OpenRouterClient(
        settings.openrouter_api_key,
        settings.openrouter_model,
        settings.request_timeout_seconds,
        vision_model=settings.openrouter_vision_model,
    )
    handlers = TelegramAIHandlers(
        repository,
        ai,
        settings.max_context_messages,
        max_tokens=settings.max_context_tokens,
    )
    application = Application.builder().token(settings.telegram_bot_token).post_init(handlers.configure).build()
    register_handlers(application, handlers)
    application.run_polling(allowed_updates=["message"])

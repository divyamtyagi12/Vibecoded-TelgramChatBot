"""Telegram webhook endpoint for Vercel."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import secrets

from fastapi import FastAPI, Header, HTTPException, Request
from telegram import Update
from telegram.ext import Application, ContextTypes

from telegram_ai.config import Settings
from telegram_ai.gemini import GeminiClient
from telegram_ai.handlers import TelegramAIHandlers, register_handlers
from telegram_ai.storage import SQLiteRepository

logger = logging.getLogger(__name__)


async def _log_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception while processing update %s", update, exc_info=context.error)


settings = Settings.from_environment()
repository = SQLiteRepository(settings.database_path)
ai = GeminiClient(
    settings.gemini_api_key,
    settings.gemini_model,
    settings.request_timeout_seconds,
)
handlers = TelegramAIHandlers(repository, ai, settings.max_context_messages)
telegram_application = Application.builder().token(settings.telegram_bot_token).build()
register_handlers(telegram_application, handlers)
telegram_application.add_error_handler(_log_error)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await telegram_application.initialize()
    await handlers.configure(telegram_application)
    await telegram_application.start()
    try:
        yield
    finally:
        await telegram_application.stop()
        await telegram_application.shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def health_check() -> dict[str, bool]:
    return {"ok": True}


@app.post("/")
async def receive_update(
    request: Request,
    telegram_secret: str | None = Header(
        default=None, alias="X-Telegram-Bot-Api-Secret-Token"
    ),
) -> dict[str, bool]:
    if not settings.telegram_webhook_secret:
        raise HTTPException(status_code=500, detail="TELEGRAM_WEBHOOK_SECRET is not configured")
    if not telegram_secret or not secrets.compare_digest(
        telegram_secret, settings.telegram_webhook_secret
    ):
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    update = Update.de_json(await request.json(), telegram_application.bot)
    await telegram_application.process_update(update)
    return {"ok": True}

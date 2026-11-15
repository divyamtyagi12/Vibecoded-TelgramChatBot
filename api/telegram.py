"""Telegram webhook endpoint for Vercel."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import secrets

# pyrefly: ignore[missing-import]
from fastapi import FastAPI, Header, HTTPException, Request
from telegram import Update
from telegram.ext import Application

from telegram_ai.config import Settings
from telegram_ai.groq import GroqClient
from telegram_ai.handlers import TelegramAIHandlers, register_handlers
from telegram_ai.storage import SQLiteRepository

logger = logging.getLogger(__name__)

settings = Settings.from_environment()
repository = SQLiteRepository(settings.database_path)
ai = GroqClient(
    settings.groq_api_key,
    settings.groq_model,
    settings.request_timeout_seconds,
    vision_model=settings.groq_vision_model,
)
handlers = TelegramAIHandlers(
    repository,
    ai,
    settings.max_context_messages,
    max_tokens=settings.max_context_tokens,
)
telegram_application = Application.builder().token(settings.telegram_bot_token).build()
register_handlers(telegram_application, handlers)


async def _ensure_initialized() -> None:
    if not getattr(telegram_application, "_initialized", False):
        await telegram_application.initialize()
        try:
            await handlers.configure(telegram_application)
        except Exception:
            logger.warning("Could not configure bot commands on cold start", exc_info=True)
        await telegram_application.start()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await _ensure_initialized()
    try:
        yield
    finally:
        if getattr(telegram_application, "running", False):
            await telegram_application.stop()
        if getattr(telegram_application, "_initialized", False):
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

    await _ensure_initialized()
    update = Update.de_json(await request.json(), telegram_application.bot)
    try:
        await telegram_application.process_update(update)
    except Exception:
        logger.exception("Unhandled error while processing Telegram update")
    return {"ok": True}

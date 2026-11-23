"""Telegram webhook endpoint for Vercel."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import asynccontextmanager
import logging
import secrets

# pyrefly: ignore[missing-import]
from fastapi import FastAPI, Header, HTTPException, Request
from telegram import Update
from telegram.ext import Application

from telegram_ai.config import Settings
from telegram_ai.handlers import TelegramAIHandlers, register_handlers
from telegram_ai.openrouter import OpenRouterClient
from telegram_ai.storage import create_repository

logger = logging.getLogger(__name__)

settings = Settings.from_environment()
repository = create_repository(settings.database_url, settings.database_path)
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
telegram_application = Application.builder().token(settings.telegram_bot_token).build()
register_handlers(telegram_application, handlers)
_seen_update_ids: OrderedDict[int, None] = OrderedDict()


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
    payload = await request.json()
    update_id = payload.get("update_id") if isinstance(payload, dict) else None
    if isinstance(update_id, int):
        if update_id in _seen_update_ids:
            logger.info("Ignoring duplicate Telegram webhook update_id=%s", update_id)
            return {"ok": True}
        _seen_update_ids[update_id] = None
        if len(_seen_update_ids) > 1000:
            _seen_update_ids.popitem(last=False)

    update = Update.de_json(payload, telegram_application.bot)
    try:
        await telegram_application.process_update(update)
    except Exception:
        logger.exception("Unhandled error while processing Telegram update")
    return {"ok": True}

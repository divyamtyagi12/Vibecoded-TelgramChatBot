# TelegramAI Group Assistant

ChatGPT-like Telegram group assistant powered by **OpenRouter** (`nvidia/nemotron-3.5-super-vl:free`) with multimodal vision, real-time multi-source web search, and persistent conversation memory.

## Features
- **OpenRouter Free Model:** Strictly uses `nvidia/nemotron-3.5-super-vl:free` for both text and vision (never falls back to paid models).
- **Multimodal Vision:** Understands photos, screenshots, memes, code screenshots, math problems, cricket scorecards, anime images, and image documents—both direct uploads and replies to other users' images.
- **Persistent Memory:** Supports **Supabase PostgreSQL** (`SUPABASE_DB_URL` / `DATABASE_URL`) for production on Vercel, with automatic fallback to local **SQLite** for development.
- **Multi-Source Web Search:** Parallel search across DuckDuckGo, Bing RSS, Google News RSS, Yahoo Finance, and official webpages with source citations.

## Environment Variables
| Variable | Required | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Yes | Telegram bot token from `@BotFather` |
| `TELEGRAM_WEBHOOK_SECRET` | Vercel | Secret token verified via `X-Telegram-Bot-Api-Secret-Token` |
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key (`sk-or-v1-...`) |
| `OPENROUTER_MODEL` | No | Defaults to `nvidia/nemotron-3.5-super-vl:free` |
| `OPENROUTER_VISION_MODEL` | No | Defaults to `nvidia/nemotron-3.5-super-vl:free` |
| `SUPABASE_DB_URL` / `DATABASE_URL` | Production | Supabase PostgreSQL connection string (`postgresql://...`) |
| `DATABASE_PATH` | Local | Local SQLite path (default: `data/telegram_ai.sqlite3`) |
| `MAX_CONTEXT_MESSAGES` | No | Conversation turns kept verbatim before summarization (default: `8`) |
| `REQUEST_TIMEOUT_SECONDS` | No | HTTP request timeout in seconds (default: `12`) |

"""Minimal Gemini REST client, kept separate from the Telegram adapter."""

from __future__ import annotations

import asyncio
import base64
import logging

import httpx

from telegram_ai.domain import ConversationMessage, GroupSettings

logger = logging.getLogger(__name__)

# Models to try in order — fastest/lightest first for group chat speed
_FALLBACK_MODELS = (
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-pro-latest",
)

# Generation config tuned for fast casual group chat replies
_CHAT_GEN_CONFIG = {
    "temperature": 0.8,
    "maxOutputTokens": 250,
    "thinkingConfig": {"thinkingBudget": 0},  # disable thinking for speed
}


class GeminiClient:
    def __init__(
        self, api_key: str, model: str, timeout_seconds: float
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def answer(
        self,
        prompt: str,
        settings: GroupSettings,
        context: list[ConversationMessage],
    ) -> str:
        context_text = "\n".join(f"{item.author}: {item.text}" for item in context)
        system_instruction = (
            "You are TelegramAI, a friendly member of this Telegram group. "
            "Reply in a casual, playful tone matching the user's language (English, Hindi, or Hinglish). "
            "Use slang, short replies, jokes, teasing, and witty clapbacks when appropriate. "
            "Never be overly formal, apologize unnecessarily, or give generic help prompts. "
            f"The group admin's requested communication style is: {settings.response_style}"
        )
        request = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                f"Recent opt-in group context:\n{context_text or '(none)'}\n\n"
                                f"Current user question:\n{prompt}"
                            )
                        }
                    ],
                }
            ],
            "generationConfig": _CHAT_GEN_CONFIG,
        }
        return await self._generate_text(request)

    async def rate_photo(
        self,
        image_data: bytes,
        mime_type: str,
        request_text: str,
        settings: GroupSettings,
    ) -> str:
        system_instruction = (
            "You give concise, supportive feedback on a user-supplied photo for a Telegram group. "
            "Start with a visual-presentation score out of 10, then give at most three useful, "
            "specific suggestions. Rate composition, lighting, clarity, framing, and profile-photo "
            "suitability when relevant. Do not rate attractiveness, guess identity, age, ethnicity, "
            "health, or other sensitive personal traits. "
            f"The group admin's requested communication style is: {settings.response_style}"
        )
        request = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": base64.b64encode(image_data).decode("ascii"),
                            }
                        },
                        {
                            "text": (
                                "Rate this photo. The user's requested focus is: "
                                f"{request_text or 'general visual presentation'}"
                            )
                        },
                    ],
                }
            ],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 350},
        }
        return await self._generate_text(request)

    async def _generate_text(self, request: dict) -> str:
        candidate_models = [self.model]
        for fallback in _FALLBACK_MODELS:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = None
        for model_name in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
            # Retry up to 2 times on rate-limit (429) before trying next model
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                        response = await client.post(url, json=request)

                    if response.status_code == 429:
                        wait = 2 ** attempt  # 1s, 2s, 4s
                        logger.warning("Rate limited on %s, retrying in %ss", model_name, wait)
                        await asyncio.sleep(wait)
                        last_error = httpx.HTTPStatusError(
                            "429 Rate Limited", request=response.request, response=response
                        )
                        continue  # retry same model

                    response.raise_for_status()
                    payload = response.json()
                    parts = payload["candidates"][0]["content"]["parts"]
                    text = "".join(part.get("text", "") for part in parts).strip()
                    if not text:
                        raise ValueError("Gemini returned no text")
                    return text

                except (KeyError, IndexError, TypeError, ValueError) as error:
                    last_error = error
                    break  # bad response structure → skip to next model
                except httpx.HTTPStatusError as error:
                    if error.response.status_code == 429:
                        last_error = error
                        continue  # already handled above but safety net
                    last_error = error
                    break  # non-429 HTTP error → skip to next model
                except httpx.HTTPError as error:
                    last_error = error
                    break  # network error → skip to next model
                except Exception as error:  # noqa: BLE001 - last-resort safety net
                    # Any unexpected error (bad SDK usage, missing import, etc.)
                    # should degrade to the friendly RuntimeError below instead
                    # of crashing the update handler with an unhandled traceback.
                    logger.exception("Unexpected error calling %s", model_name)
                    last_error = error
                    break
            else:
                # All retries exhausted for this model, try next
                continue

        # Do not include provider bodies or API keys in user-facing errors or logs.
        raise RuntimeError("The AI provider is temporarily unavailable.") from last_error

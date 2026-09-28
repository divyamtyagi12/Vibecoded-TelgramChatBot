"""Minimal Groq REST client using the OpenAI-compatible chat completions API."""

from __future__ import annotations

import asyncio
import base64
import logging

import httpx

from telegram_ai.domain import ConversationMessage, GroupSettings

logger = logging.getLogger(__name__)

_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Models to try in order — fastest first.
# All valid as of September 2026 — check console.groq.com for updates.
_FALLBACK_MODELS = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
)

_VISION_MODEL = "openai/gpt-oss-120b"  # best available with potential vision support


class GroqClient:
    def __init__(
        self, api_key: str, model: str, timeout_seconds: float
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    async def answer(
        self,
        prompt: str,
        settings: GroupSettings,
        context: list[ConversationMessage],
    ) -> str:
        context_text = "\n".join(f"{item.author}: {item.text}" for item in context)
        system_content = (
            "You are TelegramAI, a friendly member of this Telegram group. "
            "Reply in a casual, playful tone matching the user's language (English, Hindi, or Hinglish). "
            "Use slang, short replies, jokes, teasing, and witty clapbacks when appropriate. "
            "Never be overly formal, apologize unnecessarily, or give generic help prompts. "
            f"The group admin's requested communication style is: {settings.response_style}"
        )
        user_content = (
            f"Recent opt-in group context:\n{context_text or '(none)'}\n\n"
            f"Current user question:\n{prompt}"
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        return await self._chat(messages, max_tokens=250, temperature=0.8)

    async def rate_photo(
        self,
        image_data: bytes,
        mime_type: str,
        request_text: str,
        settings: GroupSettings,
    ) -> str:
        """Rate a photo using the Groq vision model (llama-4-scout)."""
        system_content = (
            "You give concise, supportive feedback on a user-supplied photo for a Telegram group. "
            "Start with a visual-presentation score out of 10, then give at most three useful, "
            "specific suggestions. Rate composition, lighting, clarity, framing, and profile-photo "
            "suitability when relevant. Do not rate attractiveness, guess identity, age, ethnicity, "
            "health, or other sensitive personal traits. "
            f"The group admin's requested communication style is: {settings.response_style}"
        )
        b64_image = base64.b64encode(image_data).decode("ascii")
        user_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{b64_image}",
                },
            },
            {
                "type": "text",
                "text": (
                    "Rate this photo. The user's requested focus is: "
                    f"{request_text or 'general visual presentation'}"
                ),
            },
        ]
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        # Use the vision model specifically; fall back to text-only error message
        return await self._chat(
            messages, max_tokens=350, temperature=0.4, force_model=_VISION_MODEL
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    async def _chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 250,
        temperature: float = 0.7,
        force_model: str | None = None,
    ) -> str:
        if force_model:
            candidate_models = [force_model]
        else:
            candidate_models = [self.model]
            for fallback in _FALLBACK_MODELS:
                if fallback not in candidate_models:
                    candidate_models.append(fallback)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error = None
        for model_name in candidate_models:
            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                        response = await client.post(
                            _GROQ_API_URL, json=payload, headers=headers
                        )

                    if response.status_code == 429:
                        wait = 2**attempt  # 1s, 2s, 4s
                        logger.warning(
                            "Rate limited on %s, retrying in %ss", model_name, wait
                        )
                        await asyncio.sleep(wait)
                        last_error = httpx.HTTPStatusError(
                            "429 Rate Limited",
                            request=response.request,
                            response=response,
                        )
                        continue  # retry same model

                    response.raise_for_status()
                    data = response.json()
                    text = data["choices"][0]["message"]["content"].strip()
                    if not text:
                        raise ValueError("Groq returned no text")
                    return text

                except (KeyError, IndexError, TypeError, ValueError) as error:
                    last_error = error
                    break  # bad response structure → skip to next model
                except httpx.HTTPStatusError as error:
                    if error.response.status_code == 429:
                        last_error = error
                        continue  # already handled above
                    last_error = error
                    break  # non-429 HTTP error → skip to next model
                except httpx.HTTPError as error:
                    last_error = error
                    break  # network error → skip to next model
            else:
                continue  # all retries exhausted, try next model

        raise RuntimeError("The AI provider is temporarily unavailable.") from last_error

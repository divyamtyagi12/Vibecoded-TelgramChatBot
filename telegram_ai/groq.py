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
        system_content = (
            "You are TelegramAI, an intelligent, helpful, and highly capable AI bot in this Telegram chat.\n\n"
            "Core Guidelines:\n"
            "1. Answer any question accurately, clearly, and directly. When asked for information on a subject, KEEP IT SIMPLE and concise—do not give long, unnecessary explanations unless the user explicitly asks for an explanation or more detail.\n"
            "2. For programming and technical questions, provide correct, practical code with brief explanations.\n"
            "3. Seamlessly match the language and tone of the user (e.g., English, Hindi, or Hinglish).\n"
            "4. Handling vulgar, sensitive, or provocative questions: Handle them carefully, tactfully, and maturely. Stay composed, defuse hostility, use smart wit or neutral framing, and redirect constructively without being preachy, lecturing, or generating explicit vulgarity.\n"
            "5. STRICT RULE - NEVER USE CANNED REFUSALS: Under no circumstances output robotic refusal boilerplate such as 'I cannot help with this', 'I can't assist with that', 'As an AI...', or 'Sorry, I am unable to...'. Always give a direct answer, a thoughtful perspective, or a graceful conversational deflection instead.\n"
            f"6. The group admin's requested communication style is: {settings.response_style}"
        )
        # Build proper multi-turn message history so the model has real conversational memory
        messages: list[dict] = [{"role": "system", "content": system_content}]
        for msg in context:
            if msg.author == "TelegramAI":
                messages.append({"role": "assistant", "content": msg.text})
            else:
                messages.append({"role": "user", "content": f"{msg.author}: {msg.text}"})
        # Add the current prompt as the latest user turn
        messages.append({"role": "user", "content": prompt})
        return await self._chat(messages, max_tokens=1000, temperature=0.6)

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

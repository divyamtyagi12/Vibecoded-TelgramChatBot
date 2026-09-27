"""Minimal Gemini REST client, kept separate from the Telegram adapter."""

from __future__ import annotations

import base64

import httpx

from telegram_ai.domain import ConversationMessage, GroupSettings


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
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 700},
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
        for fallback in ("gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-3.5-flash", "gemini-flash-latest"):
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = None
        for model_name in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(
                        url, headers={"x-goog-api-key": self.api_key}, json=request
                    )
                    response.raise_for_status()
                    payload = response.json()
                parts = payload["candidates"][0]["content"]["parts"]
                text = "".join(part.get("text", "") for part in parts).strip()
                if not text:
                    raise ValueError("Gemini returned no text")
                return text
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
                last_error = error
                continue

        # Do not include provider bodies or API keys in user-facing errors or logs.
        raise RuntimeError("The AI provider is temporarily unavailable.") from last_error

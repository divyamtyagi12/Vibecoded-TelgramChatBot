"""Minimal Gemini REST client, kept separate from the Telegram adapter."""

from __future__ import annotations

import base64
import logging

import httpx

from telegram_ai.domain import ConversationMessage, GroupSettings

logger = logging.getLogger(__name__)


class ImageGenerationError(RuntimeError):
    """A safe, user-facing category for a failed image-generation request."""

    def __init__(self, kind: str) -> None:
        super().__init__("The image generator is temporarily unavailable.")
        self.kind = kind


class GeminiClient:
    def __init__(
        self, api_key: str, model: str, image_model: str, timeout_seconds: float
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.image_model = image_model
        self.timeout_seconds = timeout_seconds

    async def answer(
        self,
        prompt: str,
        settings: GroupSettings,
        context: list[ConversationMessage],
    ) -> str:
        context_text = "\n".join(f"{item.author}: {item.text}" for item in context)
        system_instruction = (
            "You are TelegramAI, a trusted but fallible assistant in a Telegram group. "
            "Answer naturally and concisely. Do not fabricate facts, citations, links, "
            "or certainty. Say when you do not know. Avoid requesting sensitive personal data. "
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

    async def generate_image(self, prompt: str) -> tuple[bytes, str]:
        request = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["IMAGE"]},
        }
        url = f"https://generativelanguage.googleapis.com/v1/models/{self.image_model}:generateContent"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds * 4) as client:
                response = await client.post(
                    url, headers={"x-goog-api-key": self.api_key}, json=request
                )
                response.raise_for_status()
                parts = response.json()["candidates"][0]["content"]["parts"]
            image_part = next(part["inlineData"] for part in parts if "inlineData" in part)
            image_data = base64.b64decode(image_part["data"], validate=True)
            mime_type = image_part.get("mimeType", "image/png")
            if not image_data or not mime_type.startswith("image/"):
                raise ValueError("Gemini returned an invalid image")
            return image_data, mime_type
        except httpx.HTTPStatusError as error:
            status_code = error.response.status_code
            kind = {
                401: "authentication",
                403: "access",
                404: "model",
                429: "rate_limited",
            }.get(status_code, "provider")
            logger.warning(
                "Gemini image generation failed: status=%s category=%s",
                status_code,
                kind,
            )
            raise ImageGenerationError(kind) from error
        except httpx.TimeoutException as error:
            logger.warning("Gemini image generation timed out")
            raise ImageGenerationError("timeout") from error
        except httpx.HTTPError as error:
            logger.warning("Gemini image request failed: error_type=%s", type(error).__name__)
            raise ImageGenerationError("network") from error
        except (KeyError, IndexError, TypeError, ValueError) as error:
            logger.warning("Gemini image response was invalid: error_type=%s", type(error).__name__)
            raise ImageGenerationError("invalid_response") from error

"""Groq REST client with real-time web search and live tracking capabilities."""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timedelta, timezone
import json
import logging
import re
from typing import Any

import httpx

from telegram_ai.domain import ConversationMessage, GroupSettings
from telegram_ai.search import WebSearchClient

logger = logging.getLogger(__name__)

_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Groq production models in priority order
_FALLBACK_MODELS = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
)

_VISION_MODEL = "openai/gpt-oss-120b"

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live web in real time for live sports scores (cricket, football, F1, etc.), "
                "live train running status, live timing, flight status, current weather, latest news, "
                "or real-time updates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific search query (e.g. 'live cricket score India', 'live train running status 12004', 'current time in London').",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": "Fetch and extract text content from a specific webpage URL for deeper details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full HTTP/HTTPS URL of the page to inspect.",
                    }
                },
                "required": ["url"],
            },
        },
    },
]

_LIVE_KEYWORDS_RE = re.compile(
    r"\b(live|score|scorecard|train|pnr|running\s+status|timing|match|who\s+won|cricket|football|ipl|f1|weather|today|current\s+time)\b",
    re.IGNORECASE,
)

_CODE_KEYWORDS_RE = re.compile(
    r"\b("
    r"code|script|program|function|algorithm|class|regex|sql\s+query|"
    r"implement|implementation|snippet|syntax|debug|bug\s+fix|"
    r"write\s+(a\s+|an\s+)?(python|javascript|js|typescript|ts|html|css|bash|shell|c\+\+|c#|java|go|rust|sql|php|ruby)\b|"
    r"how\s+to\s+(code|program|implement)\b|"
    r"write\s+(a\s+|an\s+)?(script|program|function|crawler|scraper|bot|api|endpoint)\b"
    r")\b",
    re.IGNORECASE,
)

_DETAILED_KEYWORDS_RE = re.compile(
    r"\b("
    r"in[- ]detail|detailed|in[- ]depth|deep[- ]dive|step[- ]by[- ]step|step\s+by\s+step|"
    r"comprehensively|thoroughly|elaborate|elaborated|full\s+explanation|complete\s+guide|"
    r"long\s+explanation|essay|breakdown\s+in\s+detail|explain\s+in\s+detail|"
    r"detailed\s+explanation|detailed\s+guide|detailed\s+breakdown|in[- ]depth\s+explanation"
    r")\b",
    re.IGNORECASE,
)


def _clean_chat_response(text: str) -> str:
    """Clean robotic intro/outro filler while preserving all factual content, markdown, and code."""
    if not text:
        return text

    # Remove leading robotic pleasantries if present at the very beginning
    text = re.sub(
        r"^(?:certainly|sure thing|sure|of course|here you go|here is the answer)[!:,.\s]+\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove trailing robotic sign-offs if present on their own line at the end
    trailing_patterns = (
        r"\n+(?:hope this helps!?|let me know if you (?:have|need) (?:any|anything)(?: other)? questions!?[.!]?|feel free to ask.*|have a great day!?)\s*$",
    )
    for pat in trailing_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    return text.strip()


class GroqClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float,
        search_client: WebSearchClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.search_client = search_client or WebSearchClient(
            timeout_seconds=min(timeout_seconds, 6.0)
        )

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #

    async def answer(
        self,
        prompt: str,
        settings: GroupSettings,
        context: list[ConversationMessage],
    ) -> str:
        # Calculate current real-time timestamp (IST and UTC)
        now_utc = datetime.now(timezone.utc)
        now_ist = now_utc + timedelta(hours=5, minutes=30)
        current_time_str = (
            f"Current Real-Time Clock: {now_ist.strftime('%A, %B %d, %Y, %I:%M %p')} IST "
            f"({now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC)"
        )

        system_content = (
            "You are a natural, intelligent person chatting in a Telegram group. "
            "You are direct, casual, and conversational—never sound like a textbook, formal assistant, or ChatGPT article.\n\n"
            "CONVERSATION & TONE:\n"
            "- Respond like a real human in a Telegram chat: natural, authentic, relaxed, and direct.\n"
            "- Avoid textbook explanations, academic lectures, or corporate customer support speak.\n"
            "- Match the user's language, tone, and vibe seamlessly (English, Hindi, Hinglish, casual slang).\n"
            "- Understand Hinglish and casual slang effortlessly.\n"
            "- Use emojis occasionally and naturally like in normal chat, never spam or force them.\n"
            "- NEVER use greetings or introductory filler (no 'Hello!', 'Hey there!', 'Certainly!', 'Sure thing!', 'Great question!').\n"
            "- NEVER use sign-offs, disclaimers, or conclusion fluff (no 'Hope this helps!', 'Feel free to ask!', 'Let me know!').\n"
            "- NEVER use robotic AI boilerplate or canned refusals (no 'As an AI...', 'I cannot assist with...'). Always give the direct answer.\n\n"
            "RESPONSE LENGTH RULES:\n"
            "- Simple questions (facts, definitions, quick questions): 1–2 sentences.\n"
            "- Normal questions (general queries, advice, comparisons, opinions): 2–4 sentences.\n"
            "- Complex questions: 4–6 sentences maximum by default.\n"
            "- Detailed explanation requests: If the user explicitly asks for detail (e.g. 'in detail', 'elaborate', 'step-by-step guide', 'deep dive'), provide a properly structured, comprehensive, and in-depth explanation.\n"
            "- Code requests: If the user asks for code, provide complete, clean, working code. Keep explanations around the code minimal (at most 1–2 brief lines). Never cut off code.\n\n"
            "FORMATTING GUIDELINES:\n"
            "- Absolutely NO Markdown tables unless the user explicitly requested a table.\n"
            "- NO unnecessary headings (avoid #, ##, ### for everyday answers).\n"
            "- NO 10-step guides or long checklists for simple or normal questions.\n"
            "- Avoid excessive bold formatting.\n"
            "- Use short, readable paragraphs instead of structured articles. Prefer direct answers over structured articles.\n"
            "- Never sacrifice accuracy for brevity. Include essential facts.\n\n"
            "EXAMPLES OF DESIRED CHAT RESPONSES:\n"
            "User: What is Python?\n"
            "You: Python is a programming language used for AI, automation, web development, and more.\n\n"
            "User: How can I become an esports player?\n"
            "You: Pick one competitive game, grind ranked, join scrims, and start playing tournaments. Build your skills and get noticed by teams.\n\n"
            "User: Who is smarter, Light or L?\n"
            "You: They're extremely close, but L arguably has better deduction skills, while Light is better at manipulation and long-term planning.\n\n"
            "User: Bhai weekend pe kya karu bore ho raha hu?\n"
            "You: Koi badhiya movie ya anime binge kar le, ya doston ke saath gaming session plan kar le. Mood chill ho jayega.\n\n"
            "User: Explain machine learning in detail.\n"
            "You: [Provide a comprehensive, properly structured, detailed explanation because the user explicitly asked]\n\n"
            "LIVE & REAL-TIME SEARCH:\n"
            "- You have access to real-time tools: `web_search` and `fetch_webpage`.\n"
            "- ALWAYS call `web_search` when the user asks for live sports scores, live train status, timings, weather, latest news, or current events.\n"
            "- Train running status format (2-3 lines max): train name/number, current location/delay, next station.\n"
            "- Live score format (1-2 lines max): teams, score, match state.\n"
            "- Live timing format (1 line): city/timezone local time & date.\n\n"
            f"Group custom style guideline: {settings.response_style}\n"
            f"{current_time_str}"
        )

        is_code = bool(_CODE_KEYWORDS_RE.search(prompt))
        is_detailed = bool(_DETAILED_KEYWORDS_RE.search(prompt))

        if is_code:
            answer_max_tokens = 2500
        elif is_detailed:
            answer_max_tokens = 2000
        else:
            answer_max_tokens = 500

        # Build multi-turn context
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]
        for msg in context:
            if msg.author == "TelegramAI":
                messages.append({"role": "assistant", "content": msg.text})
            else:
                messages.append({"role": "user", "content": f"{msg.author}: {msg.text}"})
        messages.append({"role": "user", "content": prompt})

        # Run tool-execution loop (up to 2 tool rounds)
        tool_used = False
        for step in range(2):
            msg = await self._chat_completion(
                messages,
                tools=_TOOLS,
                tool_choice="auto",
                max_tokens=answer_max_tokens,
                temperature=0.6,
            )
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                answer_text = (msg.get("content") or "").strip()
                if answer_text:
                    # Check if the model gave a false negative / refusal despite being asked a live query
                    if self._is_refusal_or_unaware(answer_text) and _LIVE_KEYWORDS_RE.search(prompt):
                        logger.info("Model gave refusal on live query; executing fallback search")
                        break
                    return _clean_chat_response(answer_text)
                break

            tool_used = True
            messages.append(msg)
            for tc in tool_calls:
                fname = tc.get("function", {}).get("name")
                call_id = tc.get("id", "")
                try:
                    args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                except Exception:
                    args = {}

                if fname == "web_search":
                    query = args.get("query", prompt)
                    tool_result = await self.search_client.search(query, max_results=3)
                elif fname == "fetch_webpage":
                    url = args.get("url", "")
                    tool_result = await self.search_client.fetch_page(url, max_chars=1500)
                else:
                    tool_result = f"Unsupported tool: {fname}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_result,
                })

        # If a live query was asked and tools were not triggered or model refused, do a proactive fallback search
        if not tool_used and _LIVE_KEYWORDS_RE.search(prompt):
            logger.info("Executing proactive live search for: %r", prompt)
            search_data = await self.search_client.search(prompt, max_results=3)
            messages.append({
                "role": "user",
                "content": (
                    f"Here is real-time search information retrieved for this query:\n\n{search_data}\n\n"
                    "Provide an ultra-concise, direct answer (1-3 sentences or short bullet points max). "
                    "Do not include any pleasantries or filler."
                ),
            })

        if tool_used:
            messages.append({
                "role": "user",
                "content": (
                    "Provide an ultra-concise final answer (1-3 short bullet points or sentences max). "
                    "Include only essential facts. Do not include pleasantries, disclaimers, or lengthy explanations."
                ),
            })

        # Final synthesis call
        final_msg = await self._chat_completion(
            messages,
            tools=_TOOLS,
            tool_choice="auto",
            max_tokens=answer_max_tokens,
            temperature=0.6,
        )
        final_text = (final_msg.get("content") or "").strip()
        if not final_text and final_msg.get("tool_calls"):
            tc = final_msg["tool_calls"][0]
            fname = tc.get("function", {}).get("name")
            call_id = tc.get("id", "")
            try:
                args = json.loads(tc.get("function", {}).get("arguments", "{}"))
            except Exception:
                args = {}
            if fname == "web_search":
                res = await self.search_client.search(args.get("query", prompt), max_results=3)
            elif fname == "fetch_webpage":
                res = await self.search_client.fetch_page(args.get("url", ""), max_chars=1200)
            else:
                res = "Data received."
            messages.append(final_msg)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": res})
            extra_msg = await self._chat_completion(
                messages,
                tools=_TOOLS,
                tool_choice="auto",
                max_tokens=answer_max_tokens,
                temperature=0.6,
            )
            final_text = (extra_msg.get("content") or "").strip()

        return _clean_chat_response(final_text) or "I was unable to retrieve that information right now. Please try again."

    async def rate_photo(
        self,
        image_data: bytes,
        mime_type: str,
        request_text: str,
        settings: GroupSettings,
    ) -> str:
        """Rate a photo using the Groq vision model."""
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
        return await self._chat(
            messages,
            max_tokens=350,
            temperature=0.4,
            force_model=_VISION_MODEL,
        )

    async def _chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 250,
        temperature: float = 0.7,
        force_model: str | None = None,
    ) -> str:
        msg = await self._chat_completion(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            force_model=force_model,
        )
        text = (msg.get("content") or "").strip()
        if not text:
            raise RuntimeError("Groq returned no text")
        return text

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    async def _chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.5,
        force_model: str | None = None,
    ) -> dict[str, Any]:
        """Low-level chat completion caller with model fallback and 429 rate limit backoff."""
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
            payload: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if "gpt-oss" in model_name:
                payload["reasoning_effort"] = "low"
            if tools:
                payload["tools"] = tools
                if tool_choice:
                    payload["tool_choice"] = tool_choice

            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                        response = await client.post(
                            _GROQ_API_URL, json=payload, headers=headers
                        )

                    if response.status_code == 429:
                        wait = 2**attempt + 0.5
                        try:
                            resp_headers = getattr(response, "headers", {})
                            retry_after = float(resp_headers.get("retry-after", wait))
                            wait = min(max(retry_after, wait), 6.0)
                        except (ValueError, TypeError):
                            pass
                        logger.warning(
                            "Rate limited on %s, retrying in %.1fs", model_name, wait
                        )
                        await asyncio.sleep(wait)
                        last_error = httpx.HTTPStatusError(
                            "429 Rate Limited",
                            request=response.request,
                            response=response,
                        )
                        continue

                    response.raise_for_status()
                    data = response.json()
                    msg = data["choices"][0]["message"]
                    return msg

                except (KeyError, IndexError, TypeError, ValueError) as error:
                    last_error = error
                    break  # invalid response structure -> next model
                except httpx.HTTPStatusError as error:
                    if error.response.status_code == 429:
                        last_error = error
                        continue
                    last_error = error
                    break  # other HTTP error -> next model
                except httpx.HTTPError as error:
                    last_error = error
                    break  # network error -> next model
            else:
                continue

        raise RuntimeError("The AI provider is temporarily unavailable.") from last_error

    @staticmethod
    def _is_refusal_or_unaware(text: str) -> bool:
        low = text.lower()
        phrases = (
            "don't have access to real-time",
            "do not have access to real-time",
            "don't have access to live",
            "do not have access to live",
            "cannot provide real-time",
            "cannot track live",
            "as an ai",
            "my knowledge cutoff",
            "i am unable to provide live",
        )
        return any(p in low for p in phrases)

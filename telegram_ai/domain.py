"""Domain types and policy decisions independent of Telegram or Gemini."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


import re


class ParticipationMode(StrEnum):
    MENTION = "mention"
    REPLY = "reply"
    ALWAYS = "always"
    OFF = "off"


@dataclass(frozen=True)
class GroupSettings:
    chat_id: int
    response_style: str = (
        "Chat naturally and concisely like WhatsApp/Telegram (1-3 sentences by default), "
        "in English or Hinglish unless another language is requested."
    )
    participation_mode: ParticipationMode = ParticipationMode.MENTION
    memory_enabled: bool = True


@dataclass(frozen=True)
class ConversationMessage:
    author: str
    text: str
    message_id: int | None = None
    sender_id: int | None = None
    reply_to_message_id: int | None = None
    thread_id: int | None = None
    is_bot: bool = False
    media_type: str | None = None
    media_file_id: str | None = None
    media_unique_id: str | None = None
    media_summary: str | None = None
    quote_text: str | None = None
    conversation_id: str | None = None
    failed_operation: str | None = None
    created_at: str | None = None
    chat_id: int | None = None


_RATING_PATTERNS = re.compile(
    r"\b("
    r"rate\s+(?:this|my|the)?\s*(?:photo|pic|picture|image|outfit|fit|face|dp|pfp|shot)?|"
    r"rate\s+me|photo\s+rating|pic\s+rating|give\s+(?:a\s+)?rating|"
    r"score\s+(?:this|my)?\s*(?:photo|pic|picture|image)?|"
    r"how\s+do\s+i\s+look"
    r")\b",
    re.IGNORECASE,
)

_RETRY_PATTERNS = re.compile(
    r"^(try|try\s+again|retry|again|please\s+try|pls\s+try|try\s+now|re-try)[!.]*$",
    re.IGNORECASE,
)

_EXPLAIN_CONTEXT_PATTERNS = re.compile(
    r"^(explain|what\s+is|give|tell\s+me|summarize)?\s*(the\s+)?context[?!.]*$",
    re.IGNORECASE,
)


def is_photo_rating_intent(text: str) -> bool:
    """Determine whether the user specifically requests a photo rating vs general visual analysis."""
    if not text:
        return False
    # If the user specifically asks to find an error, solve, explain, or code, it's not a photo rating
    if re.search(r"\b(code|error|bug|solve|math|equation|meme|joke|explain|what is)\b", text, re.IGNORECASE):
        # Unless 'rate' is explicitly the primary command
        if not re.search(r"\brate\s+(?:this|my)?\s*(?:photo|pic|picture)\b", text, re.IGNORECASE):
            return False
    return bool(_RATING_PATTERNS.search(text.strip()))


def is_retry_intent(text: str) -> bool:
    """Check if the user is asking to retry a previous failed action."""
    if not text:
        return False
    return bool(_RETRY_PATTERNS.search(text.strip()))


def is_explain_context_intent(text: str) -> bool:
    """Check if the user is asking to explain the context of the ongoing discussion."""
    if not text:
        return False
    return bool(_EXPLAIN_CONTEXT_PATTERNS.search(text.strip()))


def should_reply(
    settings: GroupSettings,
    *,
    is_mentioned: bool,
    is_reply_to_bot: bool,
) -> bool:
    """Apply the policy selected by the group admin."""
    return {
        ParticipationMode.MENTION: is_mentioned or is_reply_to_bot,
        ParticipationMode.REPLY: is_reply_to_bot,
        ParticipationMode.ALWAYS: True,
        ParticipationMode.OFF: False,
    }[settings.participation_mode]

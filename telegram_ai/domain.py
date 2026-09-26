"""Domain types and policy decisions independent of Telegram or Gemini."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ParticipationMode(StrEnum):
    MENTION = "mention"
    REPLY = "reply"
    ALWAYS = "always"
    OFF = "off"


@dataclass(frozen=True)
class GroupSettings:
    chat_id: int
    response_style: str = "Be helpful, concise, and welcoming."
    participation_mode: ParticipationMode = ParticipationMode.MENTION
    memory_enabled: bool = False


@dataclass(frozen=True)
class ConversationMessage:
    author: str
    text: str


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

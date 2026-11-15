"""Intelligent conversation context construction and isolation."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from telegram_ai.domain import ConversationMessage
from telegram_ai.storage import SQLiteRepository

logger = logging.getLogger(__name__)


def estimate_message_tokens(msg: ConversationMessage) -> int:
    """Rough token estimation (approx 4 chars per token plus overhead)."""
    text_len = len(msg.text or "")
    author_len = len(msg.author or "")
    quote_len = len(msg.quote_text or "")
    media_len = len(msg.media_summary or "")
    return max(1, (text_len + author_len + quote_len + media_len) // 4 + 4)


def find_image_in_conversation(
    message: Any,
    chat_id: int,
    repository: SQLiteRepository,
) -> tuple[str | None, str | None, ConversationMessage | None]:
    """Find any referenced image in current message, replied message, or ancestor chain.
    
    Returns:
        (file_id, media_type, source_message)
    """
    # 1. Direct photo or document on current message
    photos = getattr(message, "photo", None)
    if isinstance(photos, (list, tuple)) and len(photos) > 0:
        fid = getattr(photos[-1], "file_id", None)
        if isinstance(fid, str):
            return fid, "photo", None

    doc = getattr(message, "document", None)
    if doc and isinstance(getattr(doc, "mime_type", None), str) and doc.mime_type.startswith("image/"):
        fid = getattr(doc, "file_id", None)
        if isinstance(fid, str):
            return fid, "document_image", None

    # 2. Directly replied-to message object
    reply_msg = getattr(message, "reply_to_message", None)
    if reply_msg:
        r_photos = getattr(reply_msg, "photo", None)
        if isinstance(r_photos, (list, tuple)) and len(r_photos) > 0:
            fid = getattr(r_photos[-1], "file_id", None)
            if isinstance(fid, str):
                return fid, "photo", None

        r_doc = getattr(reply_msg, "document", None)
        if r_doc and isinstance(getattr(r_doc, "mime_type", None), str) and r_doc.mime_type.startswith("image/"):
            fid = getattr(r_doc, "file_id", None)
            if isinstance(fid, str):
                return fid, "document_image", None

        r_id = getattr(reply_msg, "message_id", None)
        if isinstance(r_id, int):
            # Check DB for direct parent
            stored_parent = repository.get_message(chat_id, r_id)
            if (
                stored_parent
                and stored_parent.media_file_id
                and stored_parent.media_type in ("photo", "document_image")
            ):
                return stored_parent.media_file_id, stored_parent.media_type, stored_parent

            # 3. Check ancestor chain (walking up from parent to root)
            chain = repository.get_reply_chain(chat_id, r_id, max_depth=10)
            for ancestor in reversed(chain):
                if (
                    ancestor.media_file_id
                    and ancestor.media_type in ("photo", "document_image")
                ):
                    return ancestor.media_file_id, ancestor.media_type, ancestor

    return None, None, None


def assemble_context(
    repository: SQLiteRepository,
    chat_id: int,
    current_message: ConversationMessage,
    *,
    memory_enabled: bool,
    max_messages: int = 8,
    max_tokens: int = 3000,
) -> list[ConversationMessage]:
    """Build conversation context following the priority hierarchy:
    
    1. Explicit Telegram quote & directly replied-to message
    2. Relevant reply-chain history (traversed backwards up to root)
    3. Relevant image context (media_summary)
    4. Recent messages from the same discussion/thread (when not in an isolated reply chain)
    5. Token budget enforcement
    
    Guarantees:
    - Never includes current_message in the returned context (prevents duplicate prompt).
    - Preserves chronological order (oldest to newest).
    - Isolates simultaneous group discussions by keeping reply chains pure.
    - Honors memory_enabled setting.
    """
    if not memory_enabled:
        # When memory is disabled, only include the directly replied-to message if any
        if current_message.reply_to_message_id is not None:
            parent = repository.get_message(chat_id, current_message.reply_to_message_id)
            if parent:
                return [parent]
        return []

    context: list[ConversationMessage] = []
    used_ids: set[int] = set()
    if current_message.message_id is not None:
        used_ids.add(current_message.message_id)

    # 1 & 2. Reply Chain Reconstruction
    if current_message.reply_to_message_id is not None:
        chain = repository.get_reply_chain(
            chat_id,
            current_message.reply_to_message_id,
            max_depth=max_messages,
        )
        for msg in chain:
            if msg.message_id and msg.message_id in used_ids:
                continue
            context.append(msg)
            if msg.message_id:
                used_ids.add(msg.message_id)

    # 3. If NOT an explicit reply (e.g. standalone question or topic mention),
    # pull recent messages from the same thread/chat
    if current_message.reply_to_message_id is None:
        recent = repository.get_thread_context(
            chat_id,
            thread_id=current_message.thread_id,
            limit=max_messages,
            exclude_message_ids=used_ids,
        )
        for msg in recent:
            if msg.message_id and msg.message_id in used_ids:
                continue
            context.append(msg)
            if msg.message_id:
                used_ids.add(msg.message_id)

    # Enforce token limit (trim oldest ancestors first, always keep most recent)
    total_tokens = sum(estimate_message_tokens(m) for m in context)
    while context and total_tokens > max_tokens:
        removed = context.pop(0)
        total_tokens -= estimate_message_tokens(removed)

    return context


def format_message_content(msg: ConversationMessage) -> str:
    """Format a stored message's text including quote and media summary for the AI."""
    parts = []
    if msg.quote_text:
        parts.append(f'[Quoting: "{msg.quote_text}"]')
    if msg.media_summary:
        parts.append(f"[Image Context: {msg.media_summary}]")
    if msg.text:
        parts.append(msg.text)
    return "\n".join(parts) if parts else "[No text content]"

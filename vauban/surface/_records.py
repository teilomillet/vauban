"""Typed parsing for surface prompt records."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban.types import SurfacePrompt

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_SURFACE_STYLE = "unspecified"
DEFAULT_SURFACE_LANGUAGE = "unspecified"
DEFAULT_SURFACE_FRAMING = "unspecified"
DEFAULT_SURFACE_TURN_DEPTH = 1
_ALLOWED_SURFACE_ROLES: frozenset[str] = frozenset(
    {"system", "user", "assistant"},
)


class SurfacePromptRecordError(ValueError):
    """Raised when one JSONL surface prompt record is invalid."""


def parse_surface_prompt_record(
    obj: dict[str, object],
    line_no: int,
    path: Path,
) -> SurfacePrompt:
    """Parse one surface prompt JSON object into a SurfacePrompt."""
    label = _require_non_empty_text(obj, "label", line_no, path)
    category = _require_non_empty_text(obj, "category", line_no, path)
    messages = _optional_messages(obj, "messages", line_no, path)
    prompt_raw = obj.get("prompt")
    if prompt_raw is None:
        if messages is None:
            msg = (
                "surface prompts line"
                f" {line_no} must include key 'prompt' or 'messages'"
                f" in {path}"
            )
            raise SurfacePromptRecordError(msg)
        prompt = _derive_prompt_from_messages(messages)
    elif not isinstance(prompt_raw, str) or not prompt_raw.strip():
        msg = (
            f"surface prompts line {line_no} has invalid key 'prompt'"
            f" in {path}; expected a non-empty string"
        )
        raise SurfacePromptRecordError(msg)
    else:
        prompt = prompt_raw

    style = _optional_non_empty_text(
        obj,
        "style",
        DEFAULT_SURFACE_STYLE,
        line_no,
        path,
    )
    language = _optional_non_empty_text(
        obj,
        "language",
        DEFAULT_SURFACE_LANGUAGE,
        line_no,
        path,
    )
    framing = _optional_non_empty_text(
        obj,
        "framing",
        DEFAULT_SURFACE_FRAMING,
        line_no,
        path,
    )
    turn_depth = _optional_turn_depth(
        obj,
        "turn_depth",
        _infer_turn_depth(messages),
        line_no,
        path,
    )

    return SurfacePrompt(
        prompt=prompt,
        label=label,
        category=category,
        style=style,
        language=language,
        turn_depth=turn_depth,
        framing=framing,
        messages=messages,
    )


def _require_non_empty_text(
    obj: dict[str, object],
    key: str,
    line_no: int,
    path: Path,
) -> str:
    """Read a required non-empty string key from one parsed JSON object."""
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = (
            f"surface prompts line {line_no} must include non-empty"
            f" string key {key!r} in {path}"
        )
        raise SurfacePromptRecordError(msg)
    return value


def _optional_non_empty_text(
    obj: dict[str, object],
    key: str,
    default: str,
    line_no: int,
    path: Path,
) -> str:
    """Read an optional non-empty string key, defaulting when absent."""
    value = obj.get(key)
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        msg = (
            f"surface prompts line {line_no} has invalid optional key"
            f" {key!r} in {path}; expected a non-empty string"
        )
        raise SurfacePromptRecordError(msg)
    return value


def _optional_turn_depth(
    obj: dict[str, object],
    key: str,
    default: int,
    line_no: int,
    path: Path,
) -> int:
    """Read optional integer turn depth (>= 1), defaulting when absent."""
    value = obj.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = (
            f"surface prompts line {line_no} has invalid optional key"
            f" {key!r} in {path}; expected an integer >= 1"
        )
        raise SurfacePromptRecordError(msg)
    return value


def _optional_messages(
    obj: dict[str, object],
    key: str,
    line_no: int,
    path: Path,
) -> list[dict[str, str]] | None:
    """Read optional chat messages from one surface prompt record."""
    value = obj.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        msg = (
            f"surface prompts line {line_no} has invalid key {key!r}"
            f" in {path}; expected a non-empty list"
        )
        raise SurfacePromptRecordError(msg)

    messages: list[dict[str, str]] = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            msg = (
                f"surface prompts line {line_no} has invalid {key}[{i}]"
                f" in {path}; expected an object"
            )
            raise SurfacePromptRecordError(msg)
        role_raw = item.get("role")
        content_raw = item.get("content")
        if (
            not isinstance(role_raw, str)
            or role_raw not in _ALLOWED_SURFACE_ROLES
            or not isinstance(content_raw, str)
            or not content_raw.strip()
        ):
            msg = (
                f"surface prompts line {line_no} has invalid {key}[{i}]"
                f" in {path}; expected role/content strings with role in"
                f" {sorted(_ALLOWED_SURFACE_ROLES)}"
            )
            raise SurfacePromptRecordError(msg)
        messages.append({"role": role_raw, "content": content_raw})

    return messages


def _derive_prompt_from_messages(messages: list[dict[str, str]]) -> str:
    """Derive a display prompt from messages, preferring the last user turn."""
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    return messages[-1]["content"]


def _infer_turn_depth(messages: list[dict[str, str]] | None) -> int:
    """Infer turn depth from message history."""
    if messages is None:
        return DEFAULT_SURFACE_TURN_DEPTH
    user_turns = sum(1 for message in messages if message["role"] == "user")
    return user_turns if user_turns > 0 else DEFAULT_SURFACE_TURN_DEPTH

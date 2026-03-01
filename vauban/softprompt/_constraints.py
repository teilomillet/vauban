"""Token-constraint helpers for soft prompt attacks."""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING

from vauban import _ops as ops
from vauban._forward import force_eval

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import Tokenizer


def _is_invisible_char(c: str) -> bool:
    """Check if a character is invisible."""
    cat = unicodedata.category(c)
    if cat in {"Cf", "Zl", "Zp"}:
        return True
    if cat == "Zs" and c != " ":
        return True
    return bool(cat == "Cc" and ord(c) > 127)


def _is_emoji_char(c: str) -> bool:
    """Check if a character is an emoji or miscellaneous symbol."""
    cat = unicodedata.category(c)
    if cat == "So":
        return True
    cp = ord(c)
    if 0x1F600 <= cp <= 0x1F64F:
        return True
    if 0x1F300 <= cp <= 0x1F5FF:
        return True
    if 0x1F680 <= cp <= 0x1F6FF:
        return True
    if 0x1F900 <= cp <= 0x1F9FF:
        return True
    if 0x2600 <= cp <= 0x26FF:
        return True
    return bool(0x2700 <= cp <= 0x27BF)


def _matches_constraint(text: str, constraint: str) -> bool:
    """Check if decoded token text matches a single constraint."""
    if not text:
        return False
    if constraint == "ascii":
        return all(32 <= ord(c) < 127 for c in text)
    if constraint == "alpha":
        return text.isalpha()
    if constraint == "alphanumeric":
        return all(c.isalnum() or c == " " for c in text)
    if constraint == "non_latin":
        return all(ord(c) > 127 for c in text)
    if constraint == "chinese":
        return all(0x4E00 <= ord(c) <= 0x9FFF for c in text)
    if constraint == "non_alphabetic":
        return all(not c.isalpha() for c in text)
    if constraint == "invisible":
        return all(_is_invisible_char(c) for c in text)
    if constraint == "zalgo":
        return (
            all(0x0300 <= ord(c) <= 0x036F or c.isalpha() for c in text)
            and any(0x0300 <= ord(c) <= 0x036F for c in text)
        )
    if constraint == "emoji":
        return all(_is_emoji_char(c) for c in text)
    msg = f"Unknown token constraint: {constraint!r}"
    raise ValueError(msg)


def _build_vocab_mask(
    tokenizer: Tokenizer,
    vocab_size: int,
    constraint: str | list[str] | None,
) -> Array | None:
    """Build a boolean mask of allowed token IDs for constrained search."""
    if constraint is None:
        return None
    constraints = [constraint] if isinstance(constraint, str) else constraint
    allowed = ops.zeros((vocab_size,), dtype=ops.bool_)
    for tid in range(vocab_size):
        text = tokenizer.decode([tid])
        if all(_matches_constraint(text, item) for item in constraints):
            allowed[tid] = True
    force_eval(allowed)
    return allowed

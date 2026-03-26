"""Token-constraint helpers for soft prompt attacks.

Includes embedding-based glitch token detection per "Fishing for Magikarp"
(arxiv 2405.05417): under-trained tokens with extreme embedding norms cause
model collapse and should be excluded from adversarial search.
"""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING

import numpy as np

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


# ---------------------------------------------------------------------------
# Glitch token detection (embedding-based)
# ---------------------------------------------------------------------------

# Constraints that exclude tokens (negative) rather than select them (positive).
_NEGATIVE_CONSTRAINTS: frozenset[str] = frozenset({"exclude_glitch"})


def _detect_glitch_token_ids(
    embed_matrix: Array,
    sigma_threshold: float = 3.0,
) -> set[int]:
    """Detect under-trained tokens via embedding norm outlier analysis.

    Tokens with L2 norm more than *sigma_threshold* standard deviations
    below the mean are flagged as under-trained (likely glitch tokens).
    These cause model collapse when encountered during generation.

    The method follows "Fishing for Magikarp" (arxiv 2405.05417) which
    identifies under-trained tokens by their anomalously low embedding
    norms — the result of receiving few gradient updates during training.

    Returns a set of token IDs to exclude.
    """
    force_eval(embed_matrix)
    # Convert to float32 numpy for stable computation
    emb_f32 = embed_matrix.astype(ops.float32)
    force_eval(emb_f32)
    emb_np = np.array(emb_f32)

    norms = np.linalg.norm(emb_np, axis=1)
    mean_norm = float(norms.mean())
    std_norm = float(norms.std())

    low_threshold = mean_norm - sigma_threshold * std_norm
    high_threshold = mean_norm + sigma_threshold * std_norm

    glitch_ids: set[int] = set()
    for tid in range(len(norms)):
        if norms[tid] < low_threshold or norms[tid] > high_threshold:
            glitch_ids.add(tid)

    return glitch_ids


def _build_vocab_mask(
    tokenizer: Tokenizer,
    vocab_size: int,
    constraint: str | list[str] | None,
    *,
    embed_matrix: Array | None = None,
    glitch_token_ids: set[int] | None = None,
) -> Array | None:
    """Build a boolean mask of allowed token IDs for constrained search.

    Supports both positive constraints (include only matching tokens) and
    negative constraints like ``"exclude_glitch"`` (exclude detected tokens).

    Args:
        tokenizer: Tokenizer for decoding token IDs to text.
        vocab_size: Total vocabulary size.
        constraint: Constraint name(s) or ``None`` for unrestricted search.
        embed_matrix: Embedding weight matrix, required for ``exclude_glitch``.
        glitch_token_ids: Pre-computed set of glitch token IDs to exclude.
            If provided with ``exclude_glitch``, these are used directly
            instead of running detection (useful for caching or user-supplied
            lists from the full behavioral entropy scan).
    """
    if constraint is None:
        return None
    constraints = [constraint] if isinstance(constraint, str) else constraint

    positive = [c for c in constraints if c not in _NEGATIVE_CONSTRAINTS]
    negative = [c for c in constraints if c in _NEGATIVE_CONSTRAINTS]

    # --- Positive constraints (include matching tokens) ---
    if positive:
        allowed = ops.zeros((vocab_size,), dtype=ops.bool_)
        for tid in range(vocab_size):
            text = tokenizer.decode([tid])
            if all(_matches_constraint(text, item) for item in positive):
                allowed[tid] = True
    else:
        # No positive constraints → start with all allowed
        allowed = ops.ones((vocab_size,), dtype=ops.bool_)

    # --- Negative constraints (exclude matching tokens) ---
    if "exclude_glitch" in negative:
        if glitch_token_ids is not None:
            excluded = glitch_token_ids
        elif embed_matrix is not None:
            excluded = _detect_glitch_token_ids(embed_matrix)
        else:
            msg = (
                "exclude_glitch constraint requires either embed_matrix"
                " or pre-computed glitch_token_ids"
            )
            raise ValueError(msg)
        for tid in excluded:
            if tid < vocab_size:
                allowed[tid] = False

    force_eval(allowed)
    return allowed

# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer/control-plane robustness inventory with redacted reporting."""

from __future__ import annotations

from collections import Counter

from vauban._forward import get_transformer
from vauban.softprompt._constraints import _is_emoji_char, _is_invisible_char
from vauban.types import CausalLM, TokenAuditConfig, TokenAuditResult, Tokenizer

_CATEGORY_ORDER: tuple[str, ...] = (
    "decode_error",
    "empty_decode",
    "replacement_char",
    "contains_non_printing",
    "invisible_only",
    "emoji_only",
    "non_latin_only",
    "template_like",
    "leading_space",
    "trailing_space",
)

_TEMPLATE_MARKERS: tuple[str, ...] = (
    "<|",
    "|>",
    "[inst]",
    "[/inst]",
    "<bos",
    "<eos",
    "<pad",
    "<unk",
    "<mask",
    "<s>",
    "</s>",
    "<start_of_turn>",
    "<end_of_turn>",
    "<start_of_image>",
    "<end_of_image>",
    "<user>",
    "</user>",
    "<assistant>",
    "</assistant>",
    "<system>",
    "</system>",
    "<tool>",
    "</tool>",
    "<<",
    ">>",
)


def run_token_audit(
    model: CausalLM,
    tokenizer: Tokenizer,
    config: TokenAuditConfig,
    *,
    model_path: str,
) -> TokenAuditResult:
    """Run a redacted tokenizer/control-plane audit.

    The audit intentionally emits only aggregate counts and qualitative
    findings. It does not expose raw token strings, token IDs, or candidate
    bypass sequences.
    """
    vocab_size = _vocab_size(model)
    scanned_token_count = _scan_limit(vocab_size, config.max_token_id)
    category_counts = {name: 0 for name in _CATEGORY_ORDER}
    surface_counts: Counter[str] | None = (
        Counter() if config.include_duplicate_surface_scan else None
    )

    for token_id in range(scanned_token_count):
        decoded = _decode_single_token(tokenizer, token_id)
        if decoded is None:
            category_counts["decode_error"] += 1
            continue
        if surface_counts is not None:
            surface_counts[decoded] += 1
        if decoded == "":
            category_counts["empty_decode"] += 1
        if "\ufffd" in decoded:
            category_counts["replacement_char"] += 1
        if any(_is_invisible_char(char) for char in decoded):
            category_counts["contains_non_printing"] += 1
        if _is_invisible_only(decoded):
            category_counts["invisible_only"] += 1
        if _is_emoji_only(decoded):
            category_counts["emoji_only"] += 1
        if _is_non_latin_only(decoded):
            category_counts["non_latin_only"] += 1
        if _looks_template_like(decoded):
            category_counts["template_like"] += 1
        if decoded.startswith(" "):
            category_counts["leading_space"] += 1
        if decoded.endswith(" "):
            category_counts["trailing_space"] += 1

    duplicate_surface_count, duplicate_token_count = _duplicate_metrics(surface_counts)
    declared_special_token_count = _declared_special_token_count(tokenizer)
    chat_template_declared = _chat_template_declared(tokenizer)

    return TokenAuditResult(
        model_path=model_path,
        tokenizer_class=type(tokenizer).__name__,
        vocab_size=vocab_size,
        scanned_token_count=scanned_token_count,
        declared_special_token_count=declared_special_token_count,
        chat_template_declared=chat_template_declared,
        category_counts=category_counts,
        duplicate_surface_count=duplicate_surface_count,
        duplicate_token_count=duplicate_token_count,
        findings=_build_findings(
            category_counts=category_counts,
            duplicate_surface_count=duplicate_surface_count,
            declared_special_token_count=declared_special_token_count,
            chat_template_declared=chat_template_declared,
        ),
        notes=_build_notes(
            vocab_size=vocab_size,
            scanned_token_count=scanned_token_count,
            duplicate_scan_enabled=config.include_duplicate_surface_scan,
            declared_special_token_count=declared_special_token_count,
        ),
    )


def _vocab_size(model: CausalLM) -> int:
    """Read the vocabulary size from the model input embedding matrix."""
    transformer = get_transformer(model)
    embed_tokens = transformer.embed_tokens
    weight = getattr(embed_tokens, "weight", None)
    shape = getattr(weight, "shape", None)
    if not isinstance(shape, tuple) or not shape:
        msg = "embed_tokens.weight.shape must be a non-empty tuple"
        raise TypeError(msg)
    vocab_size = int(shape[0])
    if vocab_size <= 0:
        msg = f"model vocabulary size must be > 0, got {vocab_size}"
        raise ValueError(msg)
    return vocab_size


def _scan_limit(vocab_size: int, max_token_id: int | None) -> int:
    """Return how many token IDs to scan."""
    if max_token_id is None:
        return vocab_size
    return min(vocab_size, max_token_id + 1)


def _decode_single_token(tokenizer: Tokenizer, token_id: int) -> str | None:
    """Decode one token ID, returning ``None`` on tokenizer errors."""
    try:
        decoded = tokenizer.decode([token_id])
    except Exception:
        return None
    if not isinstance(decoded, str):
        msg = f"tokenizer.decode([{token_id}]) must return str"
        raise TypeError(msg)
    return decoded


def _is_invisible_only(text: str) -> bool:
    """Return whether all decoded characters are invisible/non-printing."""
    return bool(text) and all(_is_invisible_char(char) for char in text)


def _is_emoji_only(text: str) -> bool:
    """Return whether all decoded characters are emoji/symbol surfaces."""
    if not text or "\ufffd" in text:
        return False
    return all(_is_emoji_char(char) for char in text)


def _is_non_latin_only(text: str) -> bool:
    """Return whether the decoded surface is visible non-Latin text only."""
    if not text:
        return False
    visible_chars = [char for char in text if not _is_invisible_char(char)]
    if not visible_chars:
        return False
    if any(_is_emoji_char(char) for char in visible_chars):
        return False
    return all(ord(char) > 127 for char in visible_chars)


def _looks_template_like(text: str) -> bool:
    """Return whether the decoded surface resembles prompt-control markup."""
    stripped = text.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if any(marker in lowered for marker in _TEMPLATE_MARKERS):
        return True
    return len(lowered) >= 3 and (
        (lowered.startswith("<") and lowered.endswith(">"))
        or (lowered.startswith("[") and lowered.endswith("]"))
    )


def _duplicate_metrics(
    surface_counts: Counter[str] | None,
) -> tuple[int | None, int | None]:
    """Compute duplicate-surface metrics from decoded token counts."""
    if surface_counts is None:
        return None, None
    duplicate_surface_count = 0
    duplicate_token_count = 0
    for count in surface_counts.values():
        if count <= 1:
            continue
        duplicate_surface_count += 1
        duplicate_token_count += count
    return duplicate_surface_count, duplicate_token_count


def _declared_special_token_count(tokenizer: Tokenizer) -> int | None:
    """Return the tokenizer's declared special-token count when available."""
    all_special_ids = getattr(tokenizer, "all_special_ids", None)
    special_ids = _int_values(all_special_ids)
    if special_ids is not None:
        return len(special_ids)

    collected_ids: set[int] = set()
    for attr in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "mask_token_id",
        "cls_token_id",
        "sep_token_id",
    ):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            collected_ids.add(value)
    additional_special_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
    extra_ids = _int_values(additional_special_ids)
    if extra_ids is not None:
        collected_ids.update(extra_ids)
    if collected_ids:
        return len(collected_ids)

    special_tokens_map = getattr(tokenizer, "special_tokens_map", None)
    if isinstance(special_tokens_map, dict):
        declared = _count_special_token_values(special_tokens_map)
        return declared if declared > 0 else None
    return None


def _int_values(raw: object) -> set[int] | None:
    """Convert one raw sequence of ints into a set."""
    if not isinstance(raw, list):
        return None
    result: set[int] = set()
    for value in raw:
        if isinstance(value, int) and not isinstance(value, bool):
            result.add(value)
    return result


def _count_special_token_values(mapping: dict[object, object]) -> int:
    """Count unique declared special-token values from a tokenizer mapping."""
    flattened: set[str] = set()
    for value in mapping.values():
        flattened.update(_flatten_special_token_value(value))
    return len(flattened)


def _flatten_special_token_value(value: object) -> set[str]:
    """Flatten one special-token mapping value into comparable strings."""
    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, (list, tuple)):
        flattened: set[str] = set()
        for item in value:
            flattened.update(_flatten_special_token_value(item))
        return flattened
    rendered = str(value).strip()
    return {rendered} if rendered else set()


def _chat_template_declared(tokenizer: Tokenizer) -> bool:
    """Return whether the tokenizer exposes a non-empty chat template."""
    chat_template = getattr(tokenizer, "chat_template", None)
    return isinstance(chat_template, str) and bool(chat_template.strip())


def _build_findings(
    *,
    category_counts: dict[str, int],
    duplicate_surface_count: int | None,
    declared_special_token_count: int | None,
    chat_template_declared: bool,
) -> list[str]:
    """Build high-level redacted findings from aggregate counts."""
    findings: list[str] = []
    if (
        category_counts["contains_non_printing"] > 0
        or category_counts["invisible_only"] > 0
    ):
        findings.append(
            "Vocabulary includes invisible or non-printing decoded surfaces;"
            " logging, normalization, and gateway filters should not rely on"
            " visible text alone."
        )
    if category_counts["template_like"] > 0:
        findings.append(
            "Vocabulary includes template-like control-plane surfaces; prompt"
            " wrappers and reserved markers should be version-pinned and"
            " treated as privileged syntax."
        )
    if duplicate_surface_count is not None and duplicate_surface_count > 0:
        findings.append(
            "Multiple token IDs decode to the same visible surface; text-only"
            " audits can lose token-level distinctions."
        )
    if (
        category_counts["empty_decode"] > 0
        or category_counts["replacement_char"] > 0
        or category_counts["decode_error"] > 0
    ):
        findings.append(
            "Some token IDs are not faithfully represented by ordinary decoded"
            " text; tokenizer-space audits should not depend on surface text"
            " alone."
        )
    if declared_special_token_count is not None and declared_special_token_count > 0:
        findings.append(
            "Tokenizer declares reserved special tokens; downstream tooling"
            " should distinguish ordinary content from control symbols."
        )
    if chat_template_declared:
        findings.append(
            "Tokenizer exposes a chat template; application-layer wrappers"
            " should be audited together with the model snapshot."
        )
    return findings


def _build_notes(
    *,
    vocab_size: int,
    scanned_token_count: int,
    duplicate_scan_enabled: bool,
    declared_special_token_count: int | None,
) -> list[str]:
    """Build explanatory notes for the token audit report."""
    notes = [
        (
            "This report is intentionally redacted: it omits raw token"
            " strings, token IDs, and candidate bypass sequences."
        ),
        (
            "Category counts describe tokenizer/control-plane surface classes"
            " only; they are not proof of a working jailbreak or guardrail"
            " failure."
        ),
    ]
    if scanned_token_count < vocab_size:
        notes.append(
            f"Scan truncated to the first {scanned_token_count} token IDs out of"
            f" {vocab_size} total."
        )
    else:
        notes.append(f"Scanned the full tokenizer space ({vocab_size} token IDs).")
    if not duplicate_scan_enabled:
        notes.append(
            "Duplicate decoded-surface analysis was disabled by configuration."
        )
    if declared_special_token_count is None:
        notes.append(
            "Tokenizer did not expose a stable declared special-token inventory."
        )
    return notes

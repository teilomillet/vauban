"""Character-level text perturbation for defense robustness testing.

Six techniques obfuscate text at the character level to test whether
defenses (SIC, CAST, SCAN) remain effective against input evasion:

* **leetspeak** — substitutes characters with visually similar digits/symbols.
* **homoglyph** — replaces ASCII with Cyrillic/Greek/fullwidth lookalikes.
* **zero_width** — inserts invisible Unicode characters between letters.
* **mixed_case** — randomises letter casing.
* **phonetic** — applies common phonetic substitutions.
* **random** — picks a random technique per word.

Three intensity levels control how much of the text is perturbed:

* **1** — trigger words only.
* **2** — trigger words + surrounding context.
* **3** — entire text.
"""

from __future__ import annotations

import random as _random
import re
from typing import Literal

# ── Trigger word set ──────────────────────────────────────────────────

DEFAULT_TRIGGER_WORDS: frozenset[str] = frozenset({
    # Action words
    "hack", "exploit", "bypass", "crack", "attack", "penetrate",
    "inject", "manipulate", "override", "disable", "circumvent", "evade",
    # Security terms
    "malware", "virus", "trojan", "payload", "shellcode", "rootkit",
    "keylogger", "backdoor", "vulnerability",
    # Sensitive topics
    "weapon", "bomb", "explosive", "poison", "drug", "synthesize",
    # System terms
    "jailbreak", "unlock", "root", "sudo", "admin", "privilege",
    # Social engineering
    "phishing", "scam", "impersonate", "deceive", "fraud",
    # AI-specific
    "ignore", "disregard", "forget", "pretend", "roleplay",
})

# ── Character mappings ────────────────────────────────────────────────

_LEET_MAP: dict[str, tuple[str, ...]] = {
    "a": ("4", "@"),
    "b": ("8",),
    "c": ("(",),
    "e": ("3",),
    "g": ("6", "9"),
    "h": ("#",),
    "i": ("1", "!"),
    "l": ("1", "|"),
    "o": ("0",),
    "s": ("5", "$"),
    "t": ("7", "+"),
    "z": ("2",),
}

_HOMOGLYPH_MAP: dict[str, tuple[str, ...]] = {
    # Cyrillic lookalikes
    "a": ("\u0430",),  # Cyrillic small a
    "c": ("\u0441",),  # Cyrillic small es
    "e": ("\u0435",),  # Cyrillic small ie
    "o": ("\u043e",),  # Cyrillic small o
    "p": ("\u0440",),  # Cyrillic small er
    "x": ("\u0445",),  # Cyrillic small ha
    "y": ("\u0443",),  # Cyrillic small u
    # Greek lookalikes
    "i": ("\u03b9",),  # Greek iota
    "n": ("\u03b7",),  # Greek eta
    "v": ("\u03bd",),  # Greek nu
    "w": ("\u03c9",),  # Greek omega
    # Fullwidth
    "b": ("\uff42",),  # fullwidth b
    "d": ("\uff44",),  # fullwidth d
    "f": ("\uff46",),  # fullwidth f
    "g": ("\uff47",),  # fullwidth g
    "h": ("\uff48",),  # fullwidth h
    "k": ("\uff4b",),  # fullwidth k
    "l": ("\uff4c",),  # fullwidth l
    "m": ("\uff4d",),  # fullwidth m
    "r": ("\uff52",),  # fullwidth r
    "s": ("\uff53",),  # fullwidth s
    "t": ("\uff54",),  # fullwidth t
    "u": ("\uff55",),  # fullwidth u
    "z": ("\uff5a",),  # fullwidth z
}

_ZERO_WIDTH_CHARS: tuple[str, ...] = (
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # byte order mark
)

_PHONETIC_SUBS: tuple[tuple[str, str], ...] = (
    ("ph", "f"),
    ("ck", "k"),
    ("ight", "ite"),
    ("qu", "kw"),
    ("x", "ks"),
    ("wh", "w"),
)

# ── Technique type ────────────────────────────────────────────────────

type PerturbTechnique = Literal[
    "leetspeak", "homoglyph", "zero_width", "mixed_case", "phonetic", "random",
]
type PerturbIntensity = Literal[1, 2, 3]

_ALL_TECHNIQUES: tuple[PerturbTechnique, ...] = (
    "leetspeak", "homoglyph", "zero_width", "mixed_case", "phonetic",
)

# ── Core technique functions ──────────────────────────────────────────


def _apply_leetspeak(word: str, rng: _random.Random) -> str:
    """Replace eligible characters with leet substitutions."""
    chars = list(word)
    for i, ch in enumerate(chars):
        lower = ch.lower()
        if lower in _LEET_MAP:
            chars[i] = rng.choice(_LEET_MAP[lower])
    return "".join(chars)


def _apply_homoglyph(word: str, rng: _random.Random) -> str:
    """Replace eligible characters with Unicode homoglyphs."""
    chars = list(word)
    for i, ch in enumerate(chars):
        lower = ch.lower()
        if lower in _HOMOGLYPH_MAP:
            chars[i] = rng.choice(_HOMOGLYPH_MAP[lower])
    return "".join(chars)


def _apply_zero_width(word: str, rng: _random.Random) -> str:
    """Insert zero-width characters between letters."""
    if len(word) <= 1:
        return word
    result: list[str] = [word[0]]
    for ch in word[1:]:
        result.append(rng.choice(_ZERO_WIDTH_CHARS))
        result.append(ch)
    return "".join(result)


def _apply_mixed_case(word: str, rng: _random.Random) -> str:
    """Randomise casing of alphabetic characters."""
    return "".join(
        ch.upper() if rng.random() > 0.5 else ch.lower()
        for ch in word
    )


def _apply_phonetic(word: str, rng: _random.Random) -> str:
    """Apply phonetic substitutions."""
    _ = rng  # deterministic — rng unused but kept for uniform API
    result = word.lower()
    for old, new in _PHONETIC_SUBS:
        result = result.replace(old, new)
    return result


_TECHNIQUE_FN: dict[str, type[None] | object] = {
    "leetspeak": _apply_leetspeak,
    "homoglyph": _apply_homoglyph,
    "zero_width": _apply_zero_width,
    "mixed_case": _apply_mixed_case,
    "phonetic": _apply_phonetic,
}


def _apply_technique(
    word: str,
    technique: PerturbTechnique,
    rng: _random.Random,
) -> str:
    """Apply a single technique to a word."""
    if technique == "random":
        technique = rng.choice(_ALL_TECHNIQUES)
    fn = _TECHNIQUE_FN[technique]
    return fn(word, rng)  # type: ignore[operator]


# ── Trigger detection ─────────────────────────────────────────────────

_WORD_RE = re.compile(r"\b\w+\b")


def _is_trigger(word: str, triggers: frozenset[str]) -> bool:
    """Check if a word matches any trigger (case-insensitive)."""
    return word.lower() in triggers


def _context_indices(
    words: list[tuple[int, int, str]],
    trigger_positions: set[int],
) -> set[int]:
    """Expand trigger positions to include ±1 neighbouring words."""
    expanded = set(trigger_positions)
    for pos in trigger_positions:
        if pos > 0:
            expanded.add(pos - 1)
        if pos < len(words) - 1:
            expanded.add(pos + 1)
    return expanded


# ── Public API ────────────────────────────────────────────────────────


def perturb(
    text: str,
    technique: PerturbTechnique = "random",
    intensity: PerturbIntensity = 2,
    seed: int | None = None,
    trigger_words: frozenset[str] | None = None,
) -> str:
    """Perturb text using character-level obfuscation.

    Args:
        text: Input text to perturb.
        technique: One of ``"leetspeak"``, ``"homoglyph"``,
            ``"zero_width"``, ``"mixed_case"``, ``"phonetic"``,
            or ``"random"`` (picks per-word).
        intensity: 1 = trigger words only, 2 = triggers + context,
            3 = entire text.
        seed: Random seed for reproducibility.
        trigger_words: Custom trigger word set; uses defaults if None.

    Returns:
        The perturbed text string.
    """
    rng = _random.Random(seed)
    triggers = trigger_words if trigger_words is not None else DEFAULT_TRIGGER_WORDS

    # Find all word spans
    words: list[tuple[int, int, str]] = [
        (m.start(), m.end(), m.group()) for m in _WORD_RE.finditer(text)
    ]

    if not words:
        return text

    # Determine which word positions to perturb
    if intensity == 3:
        targets = set(range(len(words)))
    else:
        trigger_positions = {
            i for i, (_, _, w) in enumerate(words) if _is_trigger(w, triggers)
        }
        if intensity == 2:
            targets = _context_indices(words, trigger_positions)
        else:
            targets = trigger_positions

    # Build result by replacing targeted words in-place
    parts: list[str] = []
    prev_end = 0
    for i, (start, end, word) in enumerate(words):
        parts.append(text[prev_end:start])  # preserve non-word chars
        if i in targets:
            parts.append(_apply_technique(word, technique, rng))
        else:
            parts.append(word)
        prev_end = end
    parts.append(text[prev_end:])  # trailing text

    return "".join(parts)


def perturb_batch(
    texts: list[str],
    technique: PerturbTechnique = "random",
    intensity: PerturbIntensity = 2,
    seed: int | None = None,
    trigger_words: frozenset[str] | None = None,
) -> list[str]:
    """Perturb multiple texts.

    Each text gets the same technique and intensity, but with an
    incrementing seed for per-text variation (if seed is given).
    """
    return [
        perturb(
            t,
            technique=technique,
            intensity=intensity,
            seed=seed + i if seed is not None else None,
            trigger_words=trigger_words,
        )
        for i, t in enumerate(texts)
    ]


__all__ = [
    "DEFAULT_TRIGGER_WORDS",
    "perturb",
    "perturb_batch",
]

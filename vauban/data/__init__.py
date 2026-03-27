# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Dataset registry for bundled JSONL prompt files.

Static metadata — no filesystem scanning at import time.  Every bundled
JSONL file has a ``BundledDataset`` entry so users can discover datasets
programmatically (``list_datasets()``) or via the CLI (``vauban man datasets``).
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BundledDataset:
    """Metadata for one bundled JSONL file."""

    name: str
    filename: str
    description: str
    count: int
    has_categories: bool
    categories: frozenset[str] = frozenset()


_DATA_DIR = Path(__file__).resolve().parent


# ── Registry ──────────────────────────────────────────────────────────

BUNDLED_DATASETS: tuple[BundledDataset, ...] = (
    BundledDataset(
        name="harmful",
        filename="harmful.jsonl",
        description="128 harmful prompts (uncategorized). Used by [data].harmful.",
        count=128,
        has_categories=False,
    ),
    BundledDataset(
        name="harmless",
        filename="harmless.jsonl",
        description="128 harmless prompts (uncategorized). Used by [data].harmless.",
        count=128,
        has_categories=False,
    ),
    BundledDataset(
        name="harmful_100",
        filename="harmful_100.jsonl",
        description="100 categorized harmful prompts across 10 harm categories.",
        count=100,
        has_categories=True,
        categories=frozenset({
            "cyber_attack", "disinformation", "drugs", "exploitation",
            "financial_crime", "fraud", "malware", "surveillance",
            "violence", "weapons",
        }),
    ),
    BundledDataset(
        name="harmless_100",
        filename="harmless_100.jsonl",
        description="100 categorized harmless prompts across 10 benign topics.",
        count=100,
        has_categories=True,
        categories=frozenset({
            "arts", "creative", "education", "geography", "health",
            "history", "math", "practical", "science", "technology",
        }),
    ),
    BundledDataset(
        name="harmful_infix",
        filename="harmful_infix.jsonl",
        description="18 harmful prompts with {suffix} infix placeholders.",
        count=18,
        has_categories=False,
    ),
    BundledDataset(
        name="harmful_infix_100",
        filename="harmful_infix_100.jsonl",
        description="100 categorized harmful prompts with infix placeholders.",
        count=100,
        has_categories=True,
        categories=frozenset({
            "cyber_attack", "disinformation", "drugs", "exploitation",
            "financial_crime", "fraud", "malware", "surveillance",
            "violence", "weapons",
        }),
    ),
    BundledDataset(
        name="eval",
        filename="eval.jsonl",
        description="32 mixed harmful/harmless prompts for quick evaluation.",
        count=32,
        has_categories=False,
    ),
    BundledDataset(
        name="surface",
        filename="surface.jsonl",
        description=(
            "64 categorized surface prompts (label + category). "
            "Default for [surface].prompts_path = 'default'."
        ),
        count=64,
        has_categories=True,
        categories=frozenset({
            "creative", "drugs", "education", "financial_crime", "fraud",
            "hacking", "malware", "science", "trivia", "violence", "weapons",
        }),
    ),
    BundledDataset(
        name="surface_multilingual",
        filename="surface_multilingual.jsonl",
        description=(
            "69 multilingual surface prompts (en/fr/es/de/zh/ar/ja). "
            "Used with prompts_path = 'default_multilingual'."
        ),
        count=69,
        has_categories=True,
        categories=frozenset({
            "cooking", "crime", "drugs", "fraud", "hacking",
            "history", "science", "trivia", "weapons",
        }),
    ),
    BundledDataset(
        name="surface_full",
        filename="surface_full.jsonl",
        description=(
            "132 prompts covering all taxonomy categories with varied styles. "
            "Used with prompts_path = 'default_full'."
        ),
        count=132,
        has_categories=True,
        categories=frozenset({
            "weapons", "violence", "terrorism",
            "cyber_attack", "malware",
            "fraud", "financial_crime", "disinformation", "social_engineering",
            "exploitation", "sexual_content", "child_safety",
            "self_harm",
            "hate_speech", "discrimination",
            "surveillance", "privacy_pii", "doxxing",
            "bioweapons", "chemical_weapons", "radiological_nuclear",
            "drugs", "professional_malpractice", "radicalization",
            "trivia", "science", "creative", "education",
            "history", "health", "practical", "technology", "arts",
        }),
    ),
    BundledDataset(
        name="arena_harmful",
        filename="arena_harmful.jsonl",
        description="6 arena benchmark harmful prompts.",
        count=6,
        has_categories=False,
    ),
    BundledDataset(
        name="arena_injection_payloads",
        filename="arena_injection_payloads.jsonl",
        description="12 prompt injection payloads for arena evaluation.",
        count=12,
        has_categories=False,
    ),
    BundledDataset(
        name="arena_safeguards_harmful",
        filename="arena_safeguards_harmful.jsonl",
        description="12 safeguards-focused harmful prompts for arena evaluation.",
        count=12,
        has_categories=False,
    ),
    BundledDataset(
        name="clean_documents",
        filename="clean_documents.jsonl",
        description="49 clean document excerpts for SIC calibration.",
        count=49,
        has_categories=False,
    ),
    BundledDataset(
        name="jailbreak_templates",
        filename="jailbreak_templates.jsonl",
        description=(
            "30 jailbreak prompt templates across 5 strategies "
            "(identity dissolution, boundary exploit, semantic inversion, "
            "dual response, competitive pressure). Each has a {payload} placeholder."
        ),
        count=30,
        has_categories=False,
    ),
)

_DATASET_BY_NAME: dict[str, BundledDataset] = {
    ds.name: ds for ds in BUNDLED_DATASETS
}


# ── Public API ────────────────────────────────────────────────────────


def list_datasets() -> tuple[BundledDataset, ...]:
    """Return all bundled datasets."""
    return BUNDLED_DATASETS


def get_dataset(name: str) -> BundledDataset:
    """Look up a bundled dataset by name.

    Raises ``KeyError`` if *name* is not in the registry.
    """
    try:
        return _DATASET_BY_NAME[name]
    except KeyError:
        available = ", ".join(sorted(_DATASET_BY_NAME))
        msg = f"Unknown dataset {name!r}. Available: {available}"
        raise KeyError(msg) from None


def dataset_path(name: str) -> Path:
    """Return the absolute path to a bundled dataset's JSONL file."""
    ds = get_dataset(name)
    return _DATA_DIR / ds.filename


__all__ = [
    "BUNDLED_DATASETS",
    "BundledDataset",
    "dataset_path",
    "get_dataset",
    "list_datasets",
]

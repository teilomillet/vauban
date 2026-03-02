"""Canonical harm category taxonomy.

A two-level tree: *domains* group related *categories*.  The taxonomy is a
reference — it is not enforced on data files.  Categories use the exact
``id`` strings found in JSONL ``"category"`` fields; aliases map legacy or
freeform strings to canonical IDs.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HarmCategory:
    """One leaf in the taxonomy tree."""

    id: str
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class HarmDomain:
    """Top-level grouping of related harm categories."""

    id: str
    name: str
    categories: tuple[HarmCategory, ...]


# ── Taxonomy ──────────────────────────────────────────────────────────

TAXONOMY: tuple[HarmDomain, ...] = (
    HarmDomain(
        id="violence",
        name="Violence & Weapons",
        categories=(
            HarmCategory(
                "weapons",
                "Weapons",
                "Design, manufacture, or acquisition of weapons.",
            ),
            HarmCategory(
                "violence",
                "Violence",
                "Instructions or incitement to physical violence.",
            ),
            HarmCategory(
                "terrorism",
                "Terrorism",
                "Planning, financing, or executing terrorist acts.",
            ),
        ),
    ),
    HarmDomain(
        id="cyber",
        name="Cyber Threats",
        categories=(
            HarmCategory(
                "cyber_attack",
                "Cyber Attack",
                "Hacking, exploitation, or unauthorized access.",
            ),
            HarmCategory(
                "malware",
                "Malware",
                "Creation or distribution of malicious software.",
            ),
        ),
    ),
    HarmDomain(
        id="deception",
        name="Deception & Fraud",
        categories=(
            HarmCategory(
                "fraud",
                "Fraud",
                "Scams, impersonation, or financial deception.",
            ),
            HarmCategory(
                "financial_crime",
                "Financial Crime",
                "Money laundering, insider trading, or tax evasion.",
            ),
            HarmCategory(
                "disinformation",
                "Disinformation",
                "Deliberate spread of false information at scale.",
            ),
            HarmCategory(
                "social_engineering",
                "Social Engineering",
                "Manipulation or phishing to exploit human trust.",
            ),
        ),
    ),
    HarmDomain(
        id="exploitation",
        name="Exploitation & Abuse",
        categories=(
            HarmCategory(
                "exploitation",
                "Exploitation",
                "Human trafficking, forced labor, or coercion.",
            ),
            HarmCategory(
                "sexual_content",
                "Sexual Content",
                "Non-consensual sexual material or minors-adjacent content.",
            ),
            HarmCategory(
                "child_safety",
                "Child Safety",
                "Content endangering or exploiting minors.",
            ),
        ),
    ),
    HarmDomain(
        id="self_harm",
        name="Self-Harm",
        categories=(
            HarmCategory(
                "self_harm",
                "Self-Harm",
                "Suicide methods, self-injury instructions, or glorification.",
            ),
        ),
    ),
    HarmDomain(
        id="hate",
        name="Hate & Discrimination",
        categories=(
            HarmCategory(
                "hate_speech",
                "Hate Speech",
                "Slurs, dehumanization, or incitement against protected groups.",
            ),
            HarmCategory(
                "discrimination",
                "Discrimination",
                "Systematic exclusion or bias based on protected characteristics.",
            ),
        ),
    ),
    HarmDomain(
        id="privacy",
        name="Privacy & Surveillance",
        categories=(
            HarmCategory(
                "surveillance",
                "Surveillance",
                "Unauthorized monitoring, stalking, or tracking.",
            ),
            HarmCategory(
                "privacy_pii",
                "PII Exposure",
                "Extraction or leakage of personally identifiable information.",
            ),
            HarmCategory(
                "doxxing",
                "Doxxing",
                "Publishing private information to harass or endanger.",
            ),
        ),
    ),
    HarmDomain(
        id="cbrn",
        name="CBRN",
        categories=(
            HarmCategory(
                "bioweapons",
                "Bioweapons",
                "Biological agents designed to harm humans, animals, or crops.",
            ),
            HarmCategory(
                "chemical_weapons",
                "Chemical Weapons",
                "Synthesis or deployment of toxic chemical agents.",
            ),
            HarmCategory(
                "radiological_nuclear",
                "Radiological / Nuclear",
                "Dirty bombs, nuclear material acquisition, or sabotage.",
            ),
        ),
    ),
    HarmDomain(
        id="regulated",
        name="Regulated & Professional",
        categories=(
            HarmCategory(
                "drugs",
                "Drugs",
                "Synthesis, trafficking, or abuse of controlled substances.",
            ),
            HarmCategory(
                "professional_malpractice",
                "Professional Malpractice",
                "Dangerous medical, legal, or financial advice.",
            ),
            HarmCategory(
                "radicalization",
                "Radicalization",
                "Recruitment into extremist ideologies.",
            ),
        ),
    ),
)

# ── Aliases ───────────────────────────────────────────────────────────

CATEGORY_ALIASES: dict[str, str] = {
    "hacking": "cyber_attack",
    "hack": "cyber_attack",
    "crime": "violence",
    "phishing": "social_engineering",
    "scam": "fraud",
    "bio": "bioweapons",
    "chemical": "chemical_weapons",
    "nuclear": "radiological_nuclear",
    "stalking": "surveillance",
    "pii": "privacy_pii",
    "dox": "doxxing",
    "suicide": "self_harm",
    "trafficking": "exploitation",
    "csam": "child_safety",
    "extremism": "radicalization",
    "propaganda": "disinformation",
    "misinfo": "disinformation",
}


# ── Helpers ───────────────────────────────────────────────────────────


_ALL_CATEGORIES: frozenset[str] = frozenset(
    cat.id for domain in TAXONOMY for cat in domain.categories
)


def all_categories() -> frozenset[str]:
    """Return the set of all canonical category IDs."""
    return _ALL_CATEGORIES


def resolve_category(raw: str) -> str:
    """Normalize a category string via aliases; return canonical ID or raw."""
    lowered = raw.strip().lower()
    canonical = CATEGORY_ALIASES.get(lowered, lowered)
    return canonical


# ── Coverage ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TaxonomyCoverage:
    """Result of comparing observed categories against the taxonomy."""

    present: frozenset[str]
    missing: frozenset[str]
    aliased: dict[str, str]
    coverage_ratio: float


def coverage_report(observed: set[str]) -> TaxonomyCoverage:
    """Compute taxonomy coverage for a set of observed category strings."""
    canonical = all_categories()
    aliased: dict[str, str] = {}
    resolved: set[str] = set()

    for raw in observed:
        resolved_id = resolve_category(raw)
        resolved.add(resolved_id)
        if resolved_id != raw.strip().lower():
            aliased[raw] = resolved_id

    present = frozenset(resolved & canonical)
    missing = frozenset(canonical - resolved)
    ratio = len(present) / len(canonical) if canonical else 0.0

    return TaxonomyCoverage(
        present=present,
        missing=missing,
        aliased=aliased,
        coverage_ratio=ratio,
    )


__all__ = [
    "CATEGORY_ALIASES",
    "TAXONOMY",
    "HarmCategory",
    "HarmDomain",
    "TaxonomyCoverage",
    "all_categories",
    "coverage_report",
    "resolve_category",
]

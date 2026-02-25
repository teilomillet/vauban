"""Typo detection and help hints for TOML config validation."""

import difflib

from vauban.config._types import TomlDict

_KNOWN_SECTIONS: frozenset[str] = frozenset({
    "model",
    "data",
    "measure",
    "cut",
    "eval",
    "surface",
    "detect",
    "optimize",
    "softprompt",
    "sic",
    "depth",
    "probe",
    "steer",
    "output",
    "verbose",
})


def check_unknown_sections(raw: TomlDict) -> list[str]:
    """Check for unknown top-level TOML sections and suggest corrections.

    Returns a list of warning strings for any section not in _KNOWN_SECTIONS.
    Uses difflib.get_close_matches to suggest typo fixes.
    """
    warnings: list[str] = []
    for key in raw:
        if key in _KNOWN_SECTIONS:
            continue
        matches = difflib.get_close_matches(
            key, list(_KNOWN_SECTIONS), n=1, cutoff=0.6,
        )
        if matches:
            warnings.append(
                f"Unknown section [{key}]"
                f" — did you mean [{matches[0]}]?",
            )
        else:
            warnings.append(
                f"Unknown section [{key}]"
                f" — not a recognized vauban config section",
            )
    return warnings

"""Defense-related field parsing for the [softprompt] config section."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vauban.config._parse_helpers import SectionReader

if TYPE_CHECKING:
    from vauban.config._types import TomlDict


@dataclass(frozen=True, slots=True)
class _SoftPromptDefenseSection:
    """Parsed defense-evaluation fields for softprompt."""

    defense_eval: str | None
    defense_eval_layer: int | None
    defense_eval_alpha: float
    defense_eval_threshold: float
    defense_eval_sic_threshold: float | None
    defense_eval_sic_mode: str
    defense_eval_sic_max_iterations: int
    defense_eval_cast_layers: list[int] | None
    defense_eval_alpha_tiers: list[tuple[float, float]] | None


def _parse_softprompt_defense(sec: TomlDict) -> _SoftPromptDefenseSection:
    """Parse the defense-related [softprompt] fields."""
    reader = SectionReader("[softprompt]", sec)

    defense_eval = reader.optional_literal(
        "defense_eval",
        ("sic", "cast", "both"),
    )
    defense_eval_layer = reader.optional_integer("defense_eval_layer")
    defense_eval_alpha = reader.number("defense_eval_alpha", default=1.0)
    defense_eval_threshold = reader.number("defense_eval_threshold", default=0.0)
    defense_eval_sic_threshold = reader.optional_number("defense_eval_sic_threshold")
    defense_eval_sic_mode = reader.literal(
        "defense_eval_sic_mode",
        ("direction", "generation", "svf"),
        default="direction",
    )

    defense_eval_sic_max_iterations = reader.integer(
        "defense_eval_sic_max_iterations",
        default=3,
    )
    if defense_eval_sic_max_iterations < 1:
        msg = (
            "[softprompt].defense_eval_sic_max_iterations must be >= 1, got"
            f" {defense_eval_sic_max_iterations}"
        )
        raise ValueError(msg)

    defense_eval_cast_layers = reader.optional_int_list(
        "defense_eval_cast_layers",
    )

    defense_eval_alpha_tiers = reader.number_pairs("defense_eval_alpha_tiers")

    return _SoftPromptDefenseSection(
        defense_eval=defense_eval,
        defense_eval_layer=defense_eval_layer,
        defense_eval_alpha=defense_eval_alpha,
        defense_eval_threshold=defense_eval_threshold,
        defense_eval_sic_threshold=defense_eval_sic_threshold,
        defense_eval_sic_mode=defense_eval_sic_mode,
        defense_eval_sic_max_iterations=defense_eval_sic_max_iterations,
        defense_eval_cast_layers=defense_eval_cast_layers,
        defense_eval_alpha_tiers=defense_eval_alpha_tiers,
    )

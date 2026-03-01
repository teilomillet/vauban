"""GAN-related field parsing for the [softprompt] config section."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vauban.config._parse_helpers import SectionReader

if TYPE_CHECKING:
    from vauban.config._types import TomlDict


@dataclass(frozen=True, slots=True)
class _SoftPromptGanSection:
    """Parsed GAN and search fields for softprompt."""

    gan_rounds: int
    gan_step_multiplier: float
    gan_direction_escalation: float
    gan_token_escalation: int
    gan_defense_escalation: bool
    gan_defense_alpha_multiplier: float
    gan_defense_threshold_escalation: float
    gan_defense_sic_iteration_escalation: int
    gan_multiturn: bool
    gan_multiturn_max_turns: int
    prompt_pool_size: int | None
    beam_width: int


def _parse_softprompt_gan(sec: TomlDict) -> _SoftPromptGanSection:
    """Parse GAN loop and beam-search fields."""
    reader = SectionReader("[softprompt]", sec)

    gan_rounds = reader.integer("gan_rounds", default=0)
    if gan_rounds < 0:
        msg = f"[softprompt].gan_rounds must be >= 0, got {gan_rounds}"
        raise ValueError(msg)

    gan_step_multiplier = reader.number("gan_step_multiplier", default=1.5)
    if gan_step_multiplier <= 0:
        msg = (
            "[softprompt].gan_step_multiplier must be > 0,"
            f" got {gan_step_multiplier}"
        )
        raise ValueError(msg)

    gan_direction_escalation = reader.number(
        "gan_direction_escalation",
        default=0.25,
    )

    gan_token_escalation = reader.integer("gan_token_escalation", default=4)
    if gan_token_escalation < 0:
        msg = (
            "[softprompt].gan_token_escalation must be >= 0,"
            f" got {gan_token_escalation}"
        )
        raise ValueError(msg)

    gan_defense_escalation = reader.boolean(
        "gan_defense_escalation",
        default=False,
    )

    gan_defense_alpha_multiplier = reader.number(
        "gan_defense_alpha_multiplier",
        default=1.5,
    )
    if gan_defense_alpha_multiplier <= 0:
        msg = (
            "[softprompt].gan_defense_alpha_multiplier must be > 0, got"
            f" {gan_defense_alpha_multiplier}"
        )
        raise ValueError(msg)

    gan_defense_threshold_escalation = reader.number(
        "gan_defense_threshold_escalation",
        default=0.5,
    )
    if gan_defense_threshold_escalation < 0:
        msg = (
            "[softprompt].gan_defense_threshold_escalation must be >= 0,"
            f" got {gan_defense_threshold_escalation}"
        )
        raise ValueError(msg)

    gan_defense_sic_iteration_escalation = reader.integer(
        "gan_defense_sic_iteration_escalation",
        default=1,
    )
    if gan_defense_sic_iteration_escalation < 0:
        msg = (
            "[softprompt].gan_defense_sic_iteration_escalation must be >= 0,"
            f" got {gan_defense_sic_iteration_escalation}"
        )
        raise ValueError(msg)

    gan_multiturn = reader.boolean("gan_multiturn", default=False)
    gan_multiturn_max_turns = reader.integer("gan_multiturn_max_turns", default=10)
    if gan_multiturn_max_turns < 1:
        msg = (
            "[softprompt].gan_multiturn_max_turns must be >= 1,"
            f" got {gan_multiturn_max_turns}"
        )
        raise ValueError(msg)

    prompt_pool_size = reader.optional_integer("prompt_pool_size")
    if prompt_pool_size is not None and prompt_pool_size < 1:
        msg = (
            "[softprompt].prompt_pool_size must be >= 1,"
            f" got {prompt_pool_size}"
        )
        raise ValueError(msg)

    beam_width = reader.integer("beam_width", default=1)
    if beam_width < 1:
        msg = f"[softprompt].beam_width must be >= 1, got {beam_width}"
        raise ValueError(msg)

    return _SoftPromptGanSection(
        gan_rounds=gan_rounds,
        gan_step_multiplier=gan_step_multiplier,
        gan_direction_escalation=gan_direction_escalation,
        gan_token_escalation=gan_token_escalation,
        gan_defense_escalation=gan_defense_escalation,
        gan_defense_alpha_multiplier=gan_defense_alpha_multiplier,
        gan_defense_threshold_escalation=gan_defense_threshold_escalation,
        gan_defense_sic_iteration_escalation=gan_defense_sic_iteration_escalation,
        gan_multiturn=gan_multiturn,
        gan_multiturn_max_turns=gan_multiturn_max_turns,
        prompt_pool_size=prompt_pool_size,
        beam_width=beam_width,
    )

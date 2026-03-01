"""Shared early-return mode metadata for run + validation flows."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from vauban.types import PipelineConfig

type EarlyModePhase = Literal["before_prompts", "after_measure"]
type EarlyModePredicate = Callable[[PipelineConfig], bool]


@dataclass(frozen=True, slots=True)
class EarlyModeSpec:
    """Typed registration for an early-return pipeline mode."""

    section: str
    mode: str
    phase: EarlyModePhase
    requires_direction: bool
    enabled: EarlyModePredicate


def _has_depth(config: PipelineConfig) -> bool:
    """Return whether [depth] mode is active."""
    return config.depth is not None


def _has_probe(config: PipelineConfig) -> bool:
    """Return whether [probe] mode is active."""
    return config.probe is not None


def _has_steer(config: PipelineConfig) -> bool:
    """Return whether [steer] mode is active."""
    return config.steer is not None


def _has_cast(config: PipelineConfig) -> bool:
    """Return whether [cast] mode is active."""
    return config.cast is not None


def _has_sic(config: PipelineConfig) -> bool:
    """Return whether [sic] mode is active."""
    return config.sic is not None


def _has_optimize(config: PipelineConfig) -> bool:
    """Return whether [optimize] mode is active."""
    return config.optimize is not None


def _has_compose_optimize(config: PipelineConfig) -> bool:
    """Return whether [compose_optimize] mode is active."""
    return config.compose_optimize is not None


def _has_softprompt(config: PipelineConfig) -> bool:
    """Return whether [softprompt] mode is active."""
    return config.softprompt is not None


def _has_svf(config: PipelineConfig) -> bool:
    """Return whether [svf] mode is active."""
    return config.svf is not None


EARLY_MODE_SPECS: tuple[EarlyModeSpec, ...] = (
    EarlyModeSpec("[depth]", "depth", "before_prompts", False, _has_depth),
    EarlyModeSpec("[svf]", "svf", "before_prompts", False, _has_svf),
    EarlyModeSpec("[probe]", "probe", "after_measure", True, _has_probe),
    EarlyModeSpec("[steer]", "steer", "after_measure", True, _has_steer),
    EarlyModeSpec("[cast]", "cast", "after_measure", True, _has_cast),
    EarlyModeSpec("[sic]", "sic", "after_measure", False, _has_sic),
    EarlyModeSpec("[optimize]", "optimize", "after_measure", True, _has_optimize),
    EarlyModeSpec(
        "[compose_optimize]",
        "compose_optimize",
        "after_measure",
        False,
        _has_compose_optimize,
    ),
    EarlyModeSpec(
        "[softprompt]",
        "softprompt",
        "after_measure",
        False,
        _has_softprompt,
    ),
)

EARLY_RETURN_PRECEDENCE: tuple[str, ...] = tuple(
    spec.section for spec in EARLY_MODE_SPECS
)


def active_early_modes(config: PipelineConfig) -> list[str]:
    """Return active early-return sections in runtime precedence order."""
    return [spec.section for spec in EARLY_MODE_SPECS if spec.enabled(config)]


def active_early_mode_for_phase(
    config: PipelineConfig,
    phase: EarlyModePhase,
) -> EarlyModeSpec | None:
    """Return the first active early mode for a given phase."""
    for spec in EARLY_MODE_SPECS:
        if spec.phase == phase and spec.enabled(config):
            return spec
    return None

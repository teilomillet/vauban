"""Shared early-return mode metadata for run + validation flows."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from vauban.types import PipelineConfig

type EarlyModePhase = Literal["standalone", "before_prompts", "after_measure"]
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


def _has_sss(config: PipelineConfig) -> bool:
    """Return whether [sss] mode is active."""
    return config.sss is not None


def _has_awareness(config: PipelineConfig) -> bool:
    """Return whether [awareness] mode is active."""
    return config.awareness is not None


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


def _has_defend(config: PipelineConfig) -> bool:
    """Return whether [defend] mode is active."""
    return config.defend is not None


def _has_circuit(config: PipelineConfig) -> bool:
    """Return whether [circuit] mode is active."""
    return config.circuit is not None


def _has_features(config: PipelineConfig) -> bool:
    """Return whether [features] mode is active."""
    return config.features is not None


def _has_linear_probe(config: PipelineConfig) -> bool:
    """Return whether [linear_probe] mode is active."""
    return config.linear_probe is not None


def _has_fusion(config: PipelineConfig) -> bool:
    """Return whether [fusion] mode is active."""
    return config.fusion is not None


def _has_repbend(config: PipelineConfig) -> bool:
    """Return whether [repbend] mode is active."""
    return config.repbend is not None


def _has_lora_export(config: PipelineConfig) -> bool:
    """Return whether [lora_export] mode is active."""
    return config.lora_export is not None


def _has_lora_analysis(config: PipelineConfig) -> bool:
    """Return whether [lora_analysis] mode is active."""
    return config.lora_analysis is not None


def _has_standalone_api_eval(config: PipelineConfig) -> bool:
    """Return whether standalone [api_eval] mode is active.

    True when [api_eval] is present AND token_text is set, meaning the
    user wants to evaluate pre-optimized tokens without a local model.
    """
    return config.api_eval is not None and config.api_eval.token_text is not None


EARLY_MODE_SPECS: tuple[EarlyModeSpec, ...] = (
    EarlyModeSpec(
        "[api_eval]",
        "api_eval",
        "standalone",
        False,
        _has_standalone_api_eval,
    ),
    EarlyModeSpec("[depth]", "depth", "before_prompts", False, _has_depth),
    EarlyModeSpec("[svf]", "svf", "before_prompts", False, _has_svf),
    EarlyModeSpec("[features]", "features", "before_prompts", False, _has_features),
    EarlyModeSpec("[probe]", "probe", "after_measure", True, _has_probe),
    EarlyModeSpec("[steer]", "steer", "after_measure", True, _has_steer),
    EarlyModeSpec("[sss]", "sss", "after_measure", True, _has_sss),
    EarlyModeSpec("[awareness]", "awareness", "after_measure", True, _has_awareness),
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
    EarlyModeSpec(
        "[defend]",
        "defend",
        "after_measure",
        False,
        _has_defend,
    ),
    EarlyModeSpec(
        "[circuit]",
        "circuit",
        "after_measure",
        False,
        _has_circuit,
    ),
    EarlyModeSpec(
        "[linear_probe]",
        "linear_probe",
        "after_measure",
        False,
        _has_linear_probe,
    ),
    EarlyModeSpec(
        "[fusion]",
        "fusion",
        "before_prompts",
        False,
        _has_fusion,
    ),
    EarlyModeSpec(
        "[repbend]",
        "repbend",
        "after_measure",
        True,
        _has_repbend,
    ),
    EarlyModeSpec(
        "[lora_export]",
        "lora_export",
        "after_measure",
        True,
        _has_lora_export,
    ),
    EarlyModeSpec(
        "[lora_analysis]",
        "lora_analysis",
        "after_measure",
        False,
        _has_lora_analysis,
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

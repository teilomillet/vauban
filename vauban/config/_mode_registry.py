"""Shared early-return mode metadata for run, docs, and validation flows."""

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
    description: str
    validation_label: str
    manual_trigger: str
    manual_output: str


def _field_is_set(field: str) -> EarlyModePredicate:
    """Build a predicate that checks ``getattr(config, field) is not None``."""

    def _check(config: PipelineConfig) -> bool:
        return getattr(config, field) is not None

    _check.__name__ = f"_has_{field}"
    _check.__qualname__ = f"_field_is_set.<locals>._has_{field}"
    return _check


def _has_standalone_api_eval(config: PipelineConfig) -> bool:
    """Return whether standalone [api_eval] mode is active.

    True when [api_eval] is present AND token_text is set, meaning the
    user wants to evaluate pre-optimized tokens without a local model.
    """
    return config.api_eval is not None and config.api_eval.token_text is not None


EARLY_MODE_SPECS: tuple[EarlyModeSpec, ...] = (
    EarlyModeSpec(
        "[remote]",
        "remote",
        "standalone",
        False,
        _field_is_set("remote"),
        "Probe remote models via batch inference API.",
        "remote probe",
        "[remote] section present.",
        "remote_report.json (+ optional activation .npy files).",
    ),
    EarlyModeSpec(
        "[api_eval]",
        "api_eval",
        "standalone",
        False,
        _has_standalone_api_eval,
        "Test optimized tokens against remote API endpoints.",
        "standalone API eval",
        "[api_eval] with token_text set (standalone).",
        "api_eval_report.json.",
    ),
    EarlyModeSpec(
        "[depth]",
        "depth",
        "before_prompts",
        False,
        _field_is_set("depth"),
        "Deep-thinking token analysis via JSD profiles.",
        "depth analysis",
        "[depth] section present.",
        "depth_report.json (+ optional depth_direction.npy).",
    ),
    EarlyModeSpec(
        "[svf]",
        "svf",
        "before_prompts",
        False,
        _field_is_set("svf"),
        "Steering Vector Field boundary MLP training.",
        "SVF training",
        "[svf] section present.",
        "svf_report.json.",
    ),
    EarlyModeSpec(
        "[features]",
        "features",
        "before_prompts",
        False,
        _field_is_set("features"),
        "Sparse autoencoder training for feature decomposition.",
        "SAE feature decomposition",
        "[features] section present.",
        "features_report.json + sae_layer_*.safetensors.",
    ),
    EarlyModeSpec(
        "[probe]",
        "probe",
        "after_measure",
        True,
        _field_is_set("probe"),
        "Per-layer projection inspection for prompts.",
        "probe inspection",
        "[probe] section present.",
        "probe_report.json.",
    ),
    EarlyModeSpec(
        "[steer]",
        "steer",
        "after_measure",
        True,
        _field_is_set("steer"),
        "Runtime activation steering for text generation.",
        "steer generation",
        "[steer] section present.",
        "steer_report.json.",
    ),
    EarlyModeSpec(
        "[sss]",
        "sss",
        "after_measure",
        True,
        _field_is_set("sss"),
        "Sensitivity-scaled steering via Jacobian analysis.",
        "sensitivity-scaled steering",
        "[sss] section present.",
        "sss_report.json.",
    ),
    EarlyModeSpec(
        "[awareness]",
        "awareness",
        "after_measure",
        True,
        _field_is_set("awareness"),
        "Steering awareness detection via sensitivity comparison.",
        "steering awareness detection",
        "[awareness] section present.",
        "awareness_report.json.",
    ),
    EarlyModeSpec(
        "[cast]",
        "cast",
        "after_measure",
        True,
        _field_is_set("cast"),
        "Conditional activation steering with threshold gating.",
        "CAST steering",
        "[cast] section present.",
        "cast_report.json.",
    ),
    EarlyModeSpec(
        "[sic]",
        "sic",
        "after_measure",
        False,
        _field_is_set("sic"),
        "Iterative input sanitization defense.",
        "SIC sanitization",
        "[sic] section present.",
        "sic_report.json.",
    ),
    EarlyModeSpec(
        "[optimize]",
        "optimize",
        "after_measure",
        True,
        _field_is_set("optimize"),
        "Optuna multi-objective hyperparameter search.",
        "Optuna optimization",
        "[optimize] section present.",
        "optimize_report.json.",
    ),
    EarlyModeSpec(
        "[compose_optimize]",
        "compose_optimize",
        "after_measure",
        False,
        _field_is_set("compose_optimize"),
        "Bayesian optimization of composition weights.",
        "composition optimization",
        "[compose_optimize] section present.",
        "compose_optimize_report.json.",
    ),
    EarlyModeSpec(
        "[softprompt]",
        "softprompt",
        "after_measure",
        False,
        _field_is_set("softprompt"),
        "Continuous/discrete soft prompt attack optimization.",
        "soft prompt attack",
        "[softprompt] section present.",
        "softprompt_report.json.",
    ),
    EarlyModeSpec(
        "[jailbreak]",
        "jailbreak",
        "after_measure",
        False,
        _field_is_set("jailbreak"),
        "Evaluate defenses against known jailbreak prompt strategies.",
        "jailbreak template evaluation",
        "[jailbreak] section present.",
        "jailbreak_report.json.",
    ),
    EarlyModeSpec(
        "[defend]",
        "defend",
        "after_measure",
        False,
        _field_is_set("defend"),
        "Composed defense stack (scan + SIC + policy + intent).",
        "defense stack",
        "[defend] section present.",
        "defend_report.json.",
    ),
    EarlyModeSpec(
        "[circuit]",
        "circuit",
        "after_measure",
        False,
        _field_is_set("circuit"),
        "Causal circuit tracing via activation patching.",
        "circuit tracing",
        "[circuit] section present.",
        "circuit_report.json.",
    ),
    EarlyModeSpec(
        "[linear_probe]",
        "linear_probe",
        "after_measure",
        False,
        _field_is_set("linear_probe"),
        "Train linear probes to measure refusal encoding.",
        "linear probe training",
        "[linear_probe] section present.",
        "linear_probe_report.json.",
    ),
    EarlyModeSpec(
        "[fusion]",
        "fusion",
        "before_prompts",
        False,
        _field_is_set("fusion"),
        "Latent fusion jailbreak via hidden state blending.",
        "fusion training",
        "[fusion] section present.",
        "fusion_report.json.",
    ),
    EarlyModeSpec(
        "[repbend]",
        "repbend",
        "after_measure",
        True,
        _field_is_set("repbend"),
        "RepBend contrastive fine-tuning for safety hardening.",
        "RepBend fine-tuning",
        "[repbend] section present.",
        "repbend_report.json + modified weights.",
    ),
    EarlyModeSpec(
        "[lora_export]",
        "lora_export",
        "after_measure",
        True,
        _field_is_set("lora_export"),
        "Export measured direction as a LoRA adapter.",
        "LoRA export",
        "[lora_export] section present.",
        "lora_export_report.json + lora_adapter/.",
    ),
    EarlyModeSpec(
        "[lora_analysis]",
        "lora_analysis",
        "after_measure",
        False,
        _field_is_set("lora_analysis"),
        "Decompose LoRA adapters via SVD for structural analysis.",
        "LoRA analysis",
        "[lora_analysis] section present.",
        "lora_analysis_report.json.",
    ),
    EarlyModeSpec(
        "[flywheel]",
        "flywheel",
        "after_measure",
        False,
        _field_is_set("flywheel"),
        "Closed-loop attack-defense co-evolution flywheel.",
        "flywheel co-evolution",
        "[flywheel] section present.",
        "flywheel_report.json + flywheel_*.jsonl.",
    ),
)

EARLY_RETURN_PRECEDENCE: tuple[str, ...] = tuple(
    spec.section for spec in EARLY_MODE_SPECS
)

EARLY_MODE_DESCRIPTION_BY_MODE: dict[str, str] = {
    spec.mode: spec.description for spec in EARLY_MODE_SPECS
}

EARLY_MODE_LABEL_BY_SECTION: dict[str, str] = {
    spec.section: spec.validation_label for spec in EARLY_MODE_SPECS
}


def active_early_modes(config: PipelineConfig) -> list[str]:
    """Return active early-return sections in runtime precedence order."""
    return [spec.section for spec in EARLY_MODE_SPECS if spec.enabled(config)]


def early_mode_label(section: str) -> str | None:
    """Return the user-facing validation label for an early-return section."""
    return EARLY_MODE_LABEL_BY_SECTION.get(section)


def active_early_mode_for_phase(
    config: PipelineConfig,
    phase: EarlyModePhase,
) -> EarlyModeSpec | None:
    """Return the first active early mode for a given phase."""
    for spec in EARLY_MODE_SPECS:
        if spec.phase == phase and spec.enabled(config):
            return spec
    return None

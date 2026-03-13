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
        _has_depth,
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
        _has_svf,
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
        _has_features,
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
        _has_probe,
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
        _has_steer,
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
        _has_sss,
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
        _has_awareness,
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
        _has_cast,
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
        _has_sic,
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
        _has_optimize,
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
        _has_compose_optimize,
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
        _has_softprompt,
        "Continuous/discrete soft prompt attack optimization.",
        "soft prompt attack",
        "[softprompt] section present.",
        "softprompt_report.json.",
    ),
    EarlyModeSpec(
        "[defend]",
        "defend",
        "after_measure",
        False,
        _has_defend,
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
        _has_circuit,
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
        _has_linear_probe,
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
        _has_fusion,
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
        _has_repbend,
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
        _has_lora_export,
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
        _has_lora_analysis,
        "Decompose LoRA adapters via SVD for structural analysis.",
        "LoRA analysis",
        "[lora_analysis] section present.",
        "lora_analysis_report.json.",
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

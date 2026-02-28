"""Typo detection and help hints for TOML config validation."""

import difflib
from typing import cast

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
    "cast",
    "api_eval",
    "output",
    "verbose",
})

_KNOWN_KEYS: dict[str, frozenset[str]] = {
    "model": frozenset({"path"}),
    "data": frozenset({"harmful", "harmless", "borderline"}),
    "measure": frozenset({
        "mode", "top_k", "clip_quantile", "transfer_models", "diff_model",
        "measure_only",
    }),
    "cut": frozenset({
        "alpha", "layers", "norm_preserve", "biprojected",
        "layer_strategy", "layer_top_k", "layer_weights", "sparsity",
        "dbdi_target", "false_refusal_ortho", "layer_type_filter",
    }),
    "eval": frozenset({
        "prompts", "max_tokens", "num_prompts", "refusal_phrases",
        "refusal_mode",
    }),
    "surface": frozenset({
        "prompts",
        "generate",
        "max_tokens",
        "progress",
        "max_worst_cell_refusal_after",
        "max_worst_cell_refusal_delta",
        "min_coverage_score",
    }),
    "detect": frozenset({
        "mode", "top_k", "clip_quantile", "alpha", "max_tokens",
    }),
    "optimize": frozenset({
        "n_trials", "alpha_min", "alpha_max", "sparsity_min", "sparsity_max",
        "search_norm_preserve", "search_strategies", "layer_top_k_min",
        "layer_top_k_max", "max_tokens", "seed", "timeout",
    }),
    "softprompt": frozenset({
        "mode", "n_tokens", "n_steps", "learning_rate", "init_scale",
        "batch_size", "top_k", "direction_weight", "target_prefixes",
        "max_gen_tokens", "seed", "embed_reg_weight", "patience",
        "lr_schedule", "n_restarts", "prompt_strategy", "direction_mode",
        "direction_layers", "loss_mode", "egd_temperature",
        "token_constraint", "eos_loss_mode", "eos_loss_weight",
        "kl_ref_weight", "ref_model", "worst_k", "grad_accum_steps",
        "transfer_models", "target_repeat_count", "system_prompt",
        "beam_width", "init_tokens",
        # Defense evaluation
        "defense_eval", "defense_eval_layer", "defense_eval_alpha",
        "defense_eval_threshold", "defense_eval_sic_mode",
        "defense_eval_sic_max_iterations", "defense_eval_cast_layers",
        "defense_eval_alpha_tiers",
        # Defense-aware attack
        "defense_aware_weight",
        # Multi-model transfer scoring
        "transfer_loss_weight", "transfer_rerank_count",
        # GAN loop
        "gan_rounds", "gan_step_multiplier", "gan_direction_escalation",
        "gan_token_escalation", "gan_defense_escalation",
        "gan_defense_alpha_multiplier", "gan_defense_threshold_escalation",
        "gan_defense_sic_iteration_escalation",
        "gan_multiturn", "gan_multiturn_max_turns",
        # Prompt pool
        "prompt_pool_size",
    }),
    "sic": frozenset({
        "mode", "threshold", "max_iterations", "max_tokens", "target_layer",
        "sanitize_system_prompt", "max_sanitize_tokens", "block_on_failure",
        "calibrate", "calibrate_prompts",
    }),
    "depth": frozenset({
        "prompts", "settling_threshold", "deep_fraction", "top_k_logits",
        "max_tokens", "extract_direction", "direction_prompts",
        "clip_quantile",
    }),
    "probe": frozenset({"prompts"}),
    "steer": frozenset({"prompts", "layers", "alpha", "max_tokens"}),
    "cast": frozenset({"prompts", "layers", "alpha", "threshold", "max_tokens"}),
    "api_eval": frozenset({
        "endpoints", "max_tokens", "timeout", "system_prompt",
        "multiturn", "multiturn_max_turns", "follow_up_prompts",
    }),
    "output": frozenset({"dir"}),
}


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


def check_unknown_keys(raw: TomlDict) -> list[str]:
    """Check for unknown keys within known TOML sections and suggest corrections.

    Returns a list of warning strings for any key not recognized in its section.
    Skips sub-table dicts in [data] (HF dataset tables like [data.harmful]).
    """
    warnings: list[str] = []
    for section, known_keys in _KNOWN_KEYS.items():
        section_data = raw.get(section)
        if not isinstance(section_data, dict):
            continue
        section_dict = cast("dict[str, object]", section_data)
        for key in section_dict:
            if key in known_keys:
                continue
            # Skip sub-table dicts in [data] (e.g. [data.harmful] HF tables)
            if section == "data" and isinstance(section_dict[key], dict):
                continue
            matches = difflib.get_close_matches(
                str(key), list(known_keys), n=1, cutoff=0.6,
            )
            if matches:
                warnings.append(
                    f"Unknown key [{section}].{key}"
                    f" — did you mean [{section}].{matches[0]}?",
                )
            else:
                warnings.append(
                    f"Unknown key [{section}].{key}"
                    f" — not a recognized key in [{section}]",
                )
    return warnings


# ---------------------------------------------------------------------------
# Value-level validation: enum values and numeric ranges
# ---------------------------------------------------------------------------

_KNOWN_VALUES: dict[tuple[str, str], frozenset[str]] = {
    ("measure", "mode"): frozenset({"direction", "subspace", "dbdi", "diff"}),
    ("cut", "layer_strategy"): frozenset({"all", "above_median", "top_k"}),
    ("cut", "dbdi_target"): frozenset({"red", "hdd", "both"}),
    ("cut", "layer_type_filter"): frozenset({"global", "sliding"}),
    ("softprompt", "mode"): frozenset({"continuous", "gcg", "egd"}),
    ("softprompt", "prompt_strategy"): frozenset({
        "all", "cycle", "first", "worst_k", "sample",
    }),
    ("softprompt", "direction_mode"): frozenset({"last", "raid", "all_positions"}),
    ("softprompt", "loss_mode"): frozenset({"targeted", "untargeted", "defensive"}),
    ("softprompt", "lr_schedule"): frozenset({"constant", "cosine"}),
    ("softprompt", "token_constraint"): frozenset({
        "ascii", "alpha", "alphanumeric", "non_latin", "chinese",
        "non_alphabetic", "invisible", "zalgo", "emoji",
    }),
    ("softprompt", "eos_loss_mode"): frozenset({"none", "force", "suppress"}),
    ("softprompt", "defense_eval"): frozenset({"sic", "cast", "both"}),
    ("softprompt", "defense_eval_sic_mode"): frozenset({"direction", "generation"}),
    ("eval", "refusal_mode"): frozenset({"phrases", "judge"}),
    ("sic", "mode"): frozenset({"direction", "generation"}),
    ("sic", "calibrate_prompts"): frozenset({"harmless", "harmful"}),
    ("detect", "mode"): frozenset({"fast", "probe", "full"}),
}

# (min_inclusive, max_exclusive) — None = unbounded
_NUMERIC_RANGES: dict[tuple[str, str], tuple[float | None, float | None]] = {
    ("measure", "clip_quantile"): (0.0, 0.5),
    ("cut", "sparsity"): (0.0, 1.0),
    ("depth", "clip_quantile"): (0.0, 0.5),
    ("softprompt", "n_tokens"): (1, None),
    ("softprompt", "n_steps"): (1, None),
    ("eval", "max_tokens"): (1, None),
    ("eval", "num_prompts"): (1, None),
}


def check_value_constraints(raw: TomlDict) -> list[str]:
    """Check enum values and numeric ranges in TOML config.

    Returns a list of warning strings for invalid enum values or
    out-of-range numeric values.
    """
    warnings: list[str] = []

    for (section, key), allowed in _KNOWN_VALUES.items():
        section_data = raw.get(section)
        if not isinstance(section_data, dict):
            continue
        section_dict = cast("dict[str, object]", section_data)
        value = section_dict.get(key)
        if value is None or not isinstance(value, str):
            continue
        if value not in allowed:
            matches = difflib.get_close_matches(
                value, sorted(allowed), n=1, cutoff=0.5,
            )
            hint = f" — did you mean {matches[0]!r}?" if matches else ""
            warnings.append(
                f"Invalid value [{section}].{key} = {value!r}"
                f" — expected one of: {', '.join(sorted(allowed))}{hint}",
            )

    for (section, key), (lo, hi) in _NUMERIC_RANGES.items():
        section_data = raw.get(section)
        if not isinstance(section_data, dict):
            continue
        num_dict = cast("dict[str, object]", section_data)
        value = num_dict.get(key)
        if value is None or not isinstance(value, (int, float)):
            continue
        if lo is not None and value < lo:
            warnings.append(
                f"Out-of-range [{section}].{key} = {value}"
                f" — must be >= {lo}",
            )
        if hi is not None and value >= hi:
            warnings.append(
                f"Out-of-range [{section}].{key} = {value}"
                f" — must be < {hi}",
            )

    return warnings

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

_KNOWN_KEYS: dict[str, frozenset[str]] = {
    "model": frozenset({"path"}),
    "data": frozenset({"harmful", "harmless", "borderline"}),
    "measure": frozenset({"mode", "top_k", "clip_quantile"}),
    "cut": frozenset({
        "alpha", "layers", "norm_preserve", "biprojected",
        "layer_strategy", "layer_top_k", "layer_weights", "sparsity",
        "dbdi_target", "false_refusal_ortho", "layer_type_filter",
    }),
    "eval": frozenset({
        "prompts", "max_tokens", "num_prompts", "refusal_phrases",
    }),
    "surface": frozenset({"prompts", "generate", "max_tokens", "progress"}),
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
        "transfer_models",
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
        section_dict: dict[str, object] = section_data  # type: ignore[assignment]
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

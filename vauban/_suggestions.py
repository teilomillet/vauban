"""Typo detection and help hints for TOML config validation."""

import difflib
from typing import cast

from vauban.config._schema import KNOWN_SECTION_KEYS, KNOWN_TOP_LEVEL_KEYS
from vauban.config._types import TomlDict

_KNOWN_SECTIONS: frozenset[str] = KNOWN_TOP_LEVEL_KEYS

_KNOWN_KEYS: dict[str, frozenset[str]] = KNOWN_SECTION_KEYS


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
    ("softprompt", "mode"): frozenset({"continuous", "gcg", "egd", "cold"}),
    ("softprompt", "prompt_strategy"): frozenset({
        "all", "cycle", "first", "worst_k", "sample",
    }),
    ("softprompt", "direction_mode"): frozenset({"last", "raid", "all_positions"}),
    ("softprompt", "loss_mode"): frozenset({
        "targeted", "untargeted", "defensive", "externality",
    }),
    ("softprompt", "lr_schedule"): frozenset({"constant", "cosine"}),
    ("softprompt", "token_constraint"): frozenset({
        "ascii", "alpha", "alphanumeric", "non_latin", "chinese",
        "non_alphabetic", "invisible", "zalgo", "emoji",
    }),
    ("softprompt", "eos_loss_mode"): frozenset({"none", "force", "suppress"}),
    ("softprompt", "defense_eval"): frozenset({"sic", "cast", "both"}),
    ("softprompt", "defense_eval_sic_mode"): frozenset({"direction", "generation"}),
    ("softprompt", "injection_context"): frozenset({
        "web_page", "tool_output", "code_file",
    }),
    ("softprompt", "token_position"): frozenset({
        "prefix", "suffix", "infix",
    }),
    ("api_eval", "token_position"): frozenset({
        "prefix", "suffix", "infix",
    }),
    ("eval", "refusal_mode"): frozenset({"phrases", "judge"}),
    ("sic", "mode"): frozenset({"direction", "generation", "svf"}),
    ("sic", "calibrate_prompts"): frozenset({"harmless", "harmful"}),
    ("intent", "mode"): frozenset({"embedding", "judge"}),
    ("policy", "default_action"): frozenset({"allow", "block"}),
    ("circuit", "metric"): frozenset({"kl", "logit_diff"}),
    ("circuit", "granularity"): frozenset({"layer", "component"}),
    ("awareness", "mode"): frozenset({"fast", "full"}),
    ("detect", "mode"): frozenset({"fast", "probe", "full", "margin"}),
    ("meta", "status"): frozenset({
        "wip", "promising", "dead_end", "baseline", "superseded", "archived",
    }),
    ("lora_export", "format"): frozenset({"mlx", "peft"}),
    ("lora_export", "polarity"): frozenset({"remove", "add"}),
}

# (min_inclusive, max_exclusive) — None = unbounded
_NUMERIC_RANGES: dict[tuple[str, str], tuple[float | None, float | None]] = {
    ("measure", "clip_quantile"): (0.0, 0.5),
    ("cut", "sparsity"): (0.0, 1.0),
    ("depth", "clip_quantile"): (0.0, 0.5),
    ("softprompt", "n_tokens"): (1, None),
    ("softprompt", "n_steps"): (1, None),
    ("softprompt", "perplexity_weight"): (0.0, None),
    ("eval", "max_tokens"): (1, None),
    ("eval", "num_prompts"): (1, None),
    ("features", "d_sae"): (1, None),
    ("features", "n_epochs"): (1, None),
    ("features", "batch_size"): (1, None),
    ("features", "l1_coeff"): (0.0, None),
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

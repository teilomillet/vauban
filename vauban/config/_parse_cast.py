# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Parse the [cast] section of a TOML config."""

from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.config._types import TomlDict
from vauban.types import AlphaTier, CastConfig


def _parse_cast(raw: TomlDict) -> CastConfig | None:
    """Parse the optional [cast] section into a CastConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("cast")
    if sec is None:
        return None
    reader = SectionReader("[cast]", require_toml_table("[cast]", sec))

    # -- prompts (required) --
    prompts = reader.string_list("prompts")
    if not prompts:
        msg = "[cast].prompts must be non-empty"
        raise ValueError(msg)

    # -- layers (optional) --
    layers = reader.optional_int_list("layers")

    # -- alpha (optional, default 1.0) --
    alpha = reader.number("alpha", default=1.0)

    # -- threshold (optional, default 0.0) --
    threshold = reader.number("threshold", default=0.0)

    # -- max_tokens (optional, default 100) --
    max_tokens = reader.integer("max_tokens", default=100)
    if max_tokens < 1:
        msg = f"[cast].max_tokens must be >= 1, got {max_tokens}"
        raise ValueError(msg)

    # -- condition_direction (optional) --
    condition_direction = reader.optional_string("condition_direction")

    # -- alpha_tiers (optional) --
    alpha_tiers = _parse_alpha_tiers(reader)

    # -- direction_source (optional, default "linear") --
    direction_source = reader.literal(
        "direction_source", ("linear", "svf"), default="linear",
    )

    # -- svf_boundary_path (optional, required when direction_source="svf") --
    svf_boundary_path = reader.optional_string("svf_boundary_path")

    if direction_source == "svf" and svf_boundary_path is None:
        msg = (
            "[cast].svf_boundary_path is required"
            " when direction_source = 'svf'"
        )
        raise ValueError(msg)

    # -- bank_path (optional) --
    bank_path = reader.optional_string("bank_path")

    # -- composition (optional, dict of name -> float) --
    composition = reader.str_float_table("composition")

    # -- externality_monitor (optional, bool) --
    externality_monitor = reader.boolean("externality_monitor", default=False)

    # -- displacement_threshold (optional, float) --
    displacement_threshold = reader.number(
        "displacement_threshold", default=0.0,
    )

    # -- baseline_activations_path (optional, string) --
    baseline_activations_path = reader.optional_string(
        "baseline_activations_path",
    )

    return CastConfig(
        prompts=prompts,
        layers=layers,
        alpha=alpha,
        threshold=threshold,
        max_tokens=max_tokens,
        condition_direction_path=condition_direction,
        alpha_tiers=alpha_tiers,
        direction_source=direction_source,
        svf_boundary_path=svf_boundary_path,
        bank_path=bank_path,
        composition=composition,
        externality_monitor=externality_monitor,
        displacement_threshold=displacement_threshold,
        baseline_activations_path=baseline_activations_path,
    )


def _parse_alpha_tiers(reader: SectionReader) -> list[AlphaTier] | None:
    """Parse the optional alpha_tiers array of tables."""
    alpha_tiers_raw = reader.data.get("alpha_tiers")
    if alpha_tiers_raw is None:
        return None
    if not isinstance(alpha_tiers_raw, list):
        msg = (
            f"[cast].alpha_tiers must be a list of tables,"
            f" got {type(alpha_tiers_raw).__name__}"
        )
        raise TypeError(msg)
    alpha_tiers: list[AlphaTier] = []
    for i, tier_raw in enumerate(alpha_tiers_raw):
        tier = require_toml_table(f"[cast].alpha_tiers[{i}]", tier_raw)
        t_threshold = tier.get("threshold")
        t_alpha = tier.get("alpha")
        if t_threshold is None or t_alpha is None:
            msg = (
                f"[cast].alpha_tiers[{i}] must have 'threshold'"
                f" and 'alpha' keys"
            )
            raise ValueError(msg)
        if not isinstance(t_threshold, int | float):
            msg = (
                f"[cast].alpha_tiers[{i}].threshold must be a number,"
                f" got {type(t_threshold).__name__}"
            )
            raise TypeError(msg)
        if not isinstance(t_alpha, int | float):
            msg = (
                f"[cast].alpha_tiers[{i}].alpha must be a number,"
                f" got {type(t_alpha).__name__}"
            )
            raise TypeError(msg)
        alpha_tiers.append(
            AlphaTier(threshold=float(t_threshold), alpha=float(t_alpha)),
        )

    # Validate tiers are sorted by ascending threshold
    for i in range(1, len(alpha_tiers)):
        if alpha_tiers[i].threshold < alpha_tiers[i - 1].threshold:
            msg = (
                "[cast].alpha_tiers must be sorted by ascending threshold"
            )
            raise ValueError(msg)

    return alpha_tiers

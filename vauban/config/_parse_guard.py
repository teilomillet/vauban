# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Parse the [guard] section of a TOML config."""

from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.config._types import TomlDict
from vauban.types import GuardConfig, GuardTierSpec, GuardZone

_VALID_ZONES: set[str] = {"green", "yellow", "orange", "red"}


def _parse_guard(raw: TomlDict) -> GuardConfig | None:
    """Parse the optional [guard] section into a GuardConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("guard")
    if sec is None:
        return None
    reader = SectionReader("[guard]", require_toml_table("[guard]", sec))

    # -- prompts (required) --
    prompts = reader.string_list("prompts")
    if not prompts:
        msg = "[guard].prompts must be non-empty"
        raise ValueError(msg)

    # -- layers (optional) --
    layers = reader.optional_int_list("layers")

    # -- max_tokens (optional, default 100) --
    max_tokens = reader.integer("max_tokens", default=100)
    if max_tokens < 1:
        msg = f"[guard].max_tokens must be >= 1, got {max_tokens}"
        raise ValueError(msg)

    # -- max_rewinds (optional, default 3) --
    max_rewinds = reader.integer("max_rewinds", default=3)
    if max_rewinds < 0:
        msg = f"[guard].max_rewinds must be >= 0, got {max_rewinds}"
        raise ValueError(msg)

    # -- checkpoint_interval (optional, default 1) --
    checkpoint_interval = reader.integer("checkpoint_interval", default=1)
    if checkpoint_interval < 1:
        msg = (
            f"[guard].checkpoint_interval must be >= 1,"
            f" got {checkpoint_interval}"
        )
        raise ValueError(msg)

    # -- calibrate (optional, default false) --
    calibrate = reader.boolean("calibrate", default=False)

    # -- calibrate_prompts (optional, default "harmless") --
    calibrate_prompts = reader.literal(
        "calibrate_prompts", ("harmless", "harmful"), default="harmless",
    )

    # -- defensive_prompt (optional) --
    defensive_prompt = reader.optional_string("defensive_prompt")

    # -- defensive_embeddings_path (optional) --
    defensive_embeddings_path = reader.optional_string(
        "defensive_embeddings_path",
    )

    # -- condition_direction (optional) --
    condition_direction = reader.optional_string("condition_direction")

    # -- tiers (optional, array of tables) --
    tiers = _parse_guard_tiers(reader)

    return GuardConfig(
        prompts=prompts,
        layers=layers,
        max_tokens=max_tokens,
        tiers=tiers,
        max_rewinds=max_rewinds,
        checkpoint_interval=checkpoint_interval,
        defensive_prompt=defensive_prompt,
        defensive_embeddings_path=defensive_embeddings_path,
        calibrate=calibrate,
        calibrate_prompts=calibrate_prompts,
        condition_direction_path=condition_direction,
    )


def _parse_guard_tiers(
    reader: SectionReader,
) -> list[GuardTierSpec]:
    """Parse the optional tiers array of tables.

    Returns the default 4-tier set when the key is absent.
    """
    tiers_raw = reader.data.get("tiers")
    if tiers_raw is None:
        from vauban.types import _DEFAULT_GUARD_TIERS
        return list(_DEFAULT_GUARD_TIERS)

    if not isinstance(tiers_raw, list):
        msg = (
            f"[guard].tiers must be a list of tables,"
            f" got {type(tiers_raw).__name__}"
        )
        raise TypeError(msg)

    tiers: list[GuardTierSpec] = []
    for i, tier_raw in enumerate(tiers_raw):
        tier = require_toml_table(f"[guard].tiers[{i}]", tier_raw)

        t_threshold = tier.get("threshold")
        t_zone = tier.get("zone")
        t_alpha = tier.get("alpha")

        if t_threshold is None or t_zone is None or t_alpha is None:
            msg = (
                f"[guard].tiers[{i}] must have 'threshold',"
                f" 'zone', and 'alpha' keys"
            )
            raise ValueError(msg)

        if not isinstance(t_threshold, int | float):
            msg = (
                f"[guard].tiers[{i}].threshold must be a number,"
                f" got {type(t_threshold).__name__}"
            )
            raise TypeError(msg)

        if not isinstance(t_zone, str) or t_zone not in _VALID_ZONES:
            msg = (
                f"[guard].tiers[{i}].zone must be one of"
                f" {_VALID_ZONES}, got {t_zone!r}"
            )
            raise ValueError(msg)

        if not isinstance(t_alpha, int | float):
            msg = (
                f"[guard].tiers[{i}].alpha must be a number,"
                f" got {type(t_alpha).__name__}"
            )
            raise TypeError(msg)

        zone: GuardZone = t_zone  # type: ignore[assignment]
        tiers.append(
            GuardTierSpec(
                threshold=float(t_threshold),
                zone=zone,
                alpha=float(t_alpha),
            ),
        )

    # Validate ascending threshold order
    for i in range(1, len(tiers)):
        if tiers[i].threshold < tiers[i - 1].threshold:
            msg = "[guard].tiers must be sorted by ascending threshold"
            raise ValueError(msg)

    if not tiers:
        msg = "[guard].tiers must not be empty"
        raise ValueError(msg)

    return tiers

"""Parse the [steer] section of a TOML config."""

from vauban.config._parse_helpers import SectionReader
from vauban.config._types import TomlDict
from vauban.types import SteerConfig


def _parse_steer(raw: TomlDict) -> SteerConfig | None:
    """Parse the optional [steer] section into a SteerConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("steer")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[steer] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    reader = SectionReader("[steer]", sec)

    # -- prompts (required) --
    prompts = reader.string_list("prompts")
    if not prompts:
        msg = "[steer].prompts must be non-empty"
        raise ValueError(msg)

    # -- layers (optional) --
    layers = reader.optional_int_list("layers")

    # -- alpha (optional, default 1.0) --
    alpha = reader.number("alpha", default=1.0)

    # -- max_tokens (optional, default 100) --
    max_tokens = reader.integer("max_tokens", default=100)
    if max_tokens < 1:
        msg = f"[steer].max_tokens must be >= 1, got {max_tokens}"
        raise ValueError(msg)

    # -- direction_source (optional, default "linear") --
    direction_source = reader.literal(
        "direction_source", ("linear", "svf"), default="linear",
    )

    # -- svf_boundary_path (optional, required when direction_source="svf") --
    svf_boundary_path = reader.optional_string("svf_boundary_path")

    if direction_source == "svf" and svf_boundary_path is None:
        msg = (
            "[steer].svf_boundary_path is required"
            " when direction_source = 'svf'"
        )
        raise ValueError(msg)

    # -- bank_path (optional) --
    bank_path = reader.optional_string("bank_path")

    # -- composition (optional, dict of name -> float) --
    composition = reader.str_float_table("composition")

    return SteerConfig(
        prompts=prompts,
        layers=layers,
        alpha=alpha,
        max_tokens=max_tokens,
        direction_source=direction_source,
        svf_boundary_path=svf_boundary_path,
        bank_path=bank_path,
        composition=composition,
    )

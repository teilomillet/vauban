"""Parse the [depth] section of a TOML config."""

from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.config._types import TomlDict
from vauban.types import DepthConfig


def _parse_depth(raw: TomlDict) -> DepthConfig | None:
    """Parse the optional [depth] section into a DepthConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("depth")
    if sec is None:
        return None
    reader = SectionReader("[depth]", require_toml_table("[depth]", sec))

    # -- prompts (required) --
    prompts = reader.string_list("prompts")
    if not prompts:
        msg = "[depth].prompts must be non-empty"
        raise ValueError(msg)

    # -- settling_threshold --
    settling_threshold = reader.number("settling_threshold", default=0.5)
    if settling_threshold <= 0.0 or settling_threshold > 1.0:
        msg = (
            "[depth].settling_threshold must be in (0.0, 1.0],"
            f" got {settling_threshold}"
        )
        raise ValueError(msg)

    # -- deep_fraction --
    deep_fraction = reader.number("deep_fraction", default=0.85)
    if deep_fraction <= 0.0 or deep_fraction > 1.0:
        msg = (
            "[depth].deep_fraction must be in (0.0, 1.0],"
            f" got {deep_fraction}"
        )
        raise ValueError(msg)

    # -- top_k_logits --
    top_k_logits = reader.integer("top_k_logits", default=1000)
    if top_k_logits < 1:
        msg = (
            "[depth].top_k_logits must be >= 1,"
            f" got {top_k_logits}"
        )
        raise ValueError(msg)

    # -- max_tokens --
    max_tokens = reader.integer("max_tokens", default=0)
    if max_tokens < 0:
        msg = (
            "[depth].max_tokens must be >= 0,"
            f" got {max_tokens}"
        )
        raise ValueError(msg)

    # -- extract_direction --
    extract_direction = reader.boolean("extract_direction", default=False)

    # -- clip_quantile --
    clip_quantile = reader.number("clip_quantile", default=0.0)
    if clip_quantile < 0.0 or clip_quantile >= 0.5:
        msg = (
            "[depth].clip_quantile must be in [0.0, 0.5),"
            f" got {clip_quantile}"
        )
        raise ValueError(msg)

    # -- direction_prompts --
    direction_prompts = reader.optional_string_list("direction_prompts")

    # Cross-field validation: extract_direction needs >= 2 effective prompts
    if extract_direction:
        effective = direction_prompts if direction_prompts is not None else prompts
        if len(effective) < 2:
            src = "direction_prompts" if direction_prompts is not None else "prompts"
            msg = (
                f"[depth].extract_direction = true requires >= 2"
                f" {src}, got {len(effective)}"
            )
            raise ValueError(msg)

    return DepthConfig(
        prompts=prompts,
        settling_threshold=settling_threshold,
        deep_fraction=deep_fraction,
        top_k_logits=top_k_logits,
        max_tokens=max_tokens,
        extract_direction=extract_direction,
        direction_prompts=direction_prompts,
        clip_quantile=clip_quantile,
    )

"""Parse the [detect] section of a TOML config."""

from vauban.config._parse_helpers import SectionReader
from vauban.config._types import TomlDict
from vauban.types import DetectConfig


def _parse_detect(raw: TomlDict) -> DetectConfig | None:
    """Parse the optional [detect] section into a DetectConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("detect")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[detect] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    reader = SectionReader("[detect]", sec)

    mode = reader.literal(
        "mode", ("fast", "probe", "full", "margin"), default="full",
    )
    top_k = reader.integer("top_k", default=5)
    clip_quantile = reader.number("clip_quantile", default=0.0)
    alpha = reader.number("alpha", default=1.0)
    max_tokens = reader.integer("max_tokens", default=100)

    # -- margin_directions (list of direction file paths, for margin mode) --
    margin_directions = reader.string_list("margin_directions", default=[])

    if mode == "margin" and not margin_directions:
        msg = "[detect].margin_directions is required when mode = 'margin'"
        raise ValueError(msg)

    # -- margin_alphas (list of alpha values to sweep) --
    margin_alphas = reader.number_list(
        "margin_alphas", default=[0.5, 1.0, 2.0],
    )

    # -- svf_compare (bool) --
    svf_compare = reader.boolean("svf_compare", default=False)

    return DetectConfig(
        mode=mode,
        top_k=top_k,
        clip_quantile=clip_quantile,
        alpha=alpha,
        max_tokens=max_tokens,
        margin_directions=margin_directions,
        margin_alphas=margin_alphas,
        svf_compare=svf_compare,
    )

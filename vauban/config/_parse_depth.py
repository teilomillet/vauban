"""Parse the [depth] section of a TOML config."""

from vauban.config._types import TomlDict
from vauban.types import DepthConfig


def _parse_depth(raw: TomlDict) -> DepthConfig | None:
    """Parse the optional [depth] section into a DepthConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("depth")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[depth] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    # -- prompts (required) --
    prompts_raw = sec.get("prompts")  # type: ignore[arg-type]
    if prompts_raw is None:
        msg = "[depth].prompts is required"
        raise ValueError(msg)
    if not isinstance(prompts_raw, list):
        msg = (
            f"[depth].prompts must be a list of strings,"
            f" got {type(prompts_raw).__name__}"
        )
        raise TypeError(msg)
    if len(prompts_raw) == 0:
        msg = "[depth].prompts must be non-empty"
        raise ValueError(msg)
    for i, item in enumerate(prompts_raw):
        if not isinstance(item, str):
            msg = (
                f"[depth].prompts[{i}] must be a string,"
                f" got {type(item).__name__}"
            )
            raise TypeError(msg)
    prompts: list[str] = list(prompts_raw)

    # -- settling_threshold --
    settling_threshold = 0.5
    st_raw = sec.get("settling_threshold")  # type: ignore[arg-type]
    if st_raw is not None:
        if not isinstance(st_raw, int | float):
            msg = (
                f"[depth].settling_threshold must be a number,"
                f" got {type(st_raw).__name__}"
            )
            raise TypeError(msg)
        settling_threshold = float(st_raw)
        if settling_threshold <= 0.0 or settling_threshold > 1.0:
            msg = (
                "[depth].settling_threshold must be in (0.0, 1.0],"
                f" got {settling_threshold}"
            )
            raise ValueError(msg)

    # -- deep_fraction --
    deep_fraction = 0.85
    df_raw = sec.get("deep_fraction")  # type: ignore[arg-type]
    if df_raw is not None:
        if not isinstance(df_raw, int | float):
            msg = (
                f"[depth].deep_fraction must be a number,"
                f" got {type(df_raw).__name__}"
            )
            raise TypeError(msg)
        deep_fraction = float(df_raw)
        if deep_fraction <= 0.0 or deep_fraction > 1.0:
            msg = (
                "[depth].deep_fraction must be in (0.0, 1.0],"
                f" got {deep_fraction}"
            )
            raise ValueError(msg)

    # -- top_k_logits --
    top_k_logits = 1000
    tk_raw = sec.get("top_k_logits")  # type: ignore[arg-type]
    if tk_raw is not None:
        if not isinstance(tk_raw, int):
            msg = (
                f"[depth].top_k_logits must be an integer,"
                f" got {type(tk_raw).__name__}"
            )
            raise TypeError(msg)
        top_k_logits = int(tk_raw)
        if top_k_logits < 1:
            msg = (
                "[depth].top_k_logits must be >= 1,"
                f" got {top_k_logits}"
            )
            raise ValueError(msg)

    # -- max_tokens --
    max_tokens = 0
    mt_raw = sec.get("max_tokens")  # type: ignore[arg-type]
    if mt_raw is not None:
        if not isinstance(mt_raw, int):
            msg = (
                f"[depth].max_tokens must be an integer,"
                f" got {type(mt_raw).__name__}"
            )
            raise TypeError(msg)
        max_tokens = int(mt_raw)
        if max_tokens < 0:
            msg = (
                "[depth].max_tokens must be >= 0,"
                f" got {max_tokens}"
            )
            raise ValueError(msg)

    # -- extract_direction --
    extract_direction = False
    ed_raw = sec.get("extract_direction")  # type: ignore[arg-type]
    if ed_raw is not None:
        if not isinstance(ed_raw, bool):
            msg = (
                f"[depth].extract_direction must be a boolean,"
                f" got {type(ed_raw).__name__}"
            )
            raise TypeError(msg)
        extract_direction = ed_raw

    # -- clip_quantile --
    clip_quantile = 0.0
    cq_raw = sec.get("clip_quantile")  # type: ignore[arg-type]
    if cq_raw is not None:
        if not isinstance(cq_raw, int | float):
            msg = (
                f"[depth].clip_quantile must be a number,"
                f" got {type(cq_raw).__name__}"
            )
            raise TypeError(msg)
        clip_quantile = float(cq_raw)
        if clip_quantile < 0.0 or clip_quantile >= 0.5:
            msg = (
                "[depth].clip_quantile must be in [0.0, 0.5),"
                f" got {clip_quantile}"
            )
            raise ValueError(msg)

    # -- direction_prompts --
    direction_prompts: list[str] | None = None
    dp_raw = sec.get("direction_prompts")  # type: ignore[arg-type]
    if dp_raw is not None:
        if not isinstance(dp_raw, list):
            msg = (
                f"[depth].direction_prompts must be a list of strings,"
                f" got {type(dp_raw).__name__}"
            )
            raise TypeError(msg)
        for i, item in enumerate(dp_raw):
            if not isinstance(item, str):
                msg = (
                    f"[depth].direction_prompts[{i}] must be a string,"
                    f" got {type(item).__name__}"
                )
                raise TypeError(msg)
        direction_prompts = list(dp_raw)

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

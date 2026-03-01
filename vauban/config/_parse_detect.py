"""Parse the [detect] section of a TOML config."""

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

    mode_raw = sec.get("mode", "full")  # type: ignore[arg-type]
    if not isinstance(mode_raw, str):
        msg = f"[detect].mode must be a string, got {type(mode_raw).__name__}"
        raise TypeError(msg)
    valid_modes = ("fast", "probe", "full", "margin")
    if mode_raw not in valid_modes:
        msg = (
            f"[detect].mode must be one of {valid_modes!r},"
            f" got {mode_raw!r}"
        )
        raise ValueError(msg)

    top_k_raw = sec.get("top_k", 5)  # type: ignore[arg-type]
    if not isinstance(top_k_raw, int):
        msg = (
            f"[detect].top_k must be an integer,"
            f" got {type(top_k_raw).__name__}"
        )
        raise TypeError(msg)

    clip_quantile_raw = sec.get("clip_quantile", 0.0)  # type: ignore[arg-type]
    if not isinstance(clip_quantile_raw, int | float):
        msg = (
            f"[detect].clip_quantile must be a number,"
            f" got {type(clip_quantile_raw).__name__}"
        )
        raise TypeError(msg)

    alpha_raw = sec.get("alpha", 1.0)  # type: ignore[arg-type]
    if not isinstance(alpha_raw, int | float):
        msg = f"[detect].alpha must be a number, got {type(alpha_raw).__name__}"
        raise TypeError(msg)

    max_tokens_raw = sec.get("max_tokens", 100)  # type: ignore[arg-type]
    if not isinstance(max_tokens_raw, int):
        msg = (
            f"[detect].max_tokens must be an integer,"
            f" got {type(max_tokens_raw).__name__}"
        )
        raise TypeError(msg)

    # -- margin_directions (list of direction file paths, for margin mode) --
    margin_dirs_raw = sec.get("margin_directions", [])  # type: ignore[arg-type]
    if not isinstance(margin_dirs_raw, list):
        msg = (
            f"[detect].margin_directions must be a list,"
            f" got {type(margin_dirs_raw).__name__}"
        )
        raise TypeError(msg)
    margin_directions: list[str] = []
    for i, item in enumerate(margin_dirs_raw):
        if not isinstance(item, str):
            msg = (
                f"[detect].margin_directions[{i}] must be a string,"
                f" got {type(item).__name__}"
            )
            raise TypeError(msg)
        margin_directions.append(item)

    if mode_raw == "margin" and not margin_directions:
        msg = "[detect].margin_directions is required when mode = 'margin'"
        raise ValueError(msg)

    # -- margin_alphas (list of alpha values to sweep) --
    margin_alphas_raw = sec.get("margin_alphas", [0.5, 1.0, 2.0])  # type: ignore[arg-type]
    if not isinstance(margin_alphas_raw, list):
        msg = (
            f"[detect].margin_alphas must be a list,"
            f" got {type(margin_alphas_raw).__name__}"
        )
        raise TypeError(msg)
    margin_alphas: list[float] = []
    for i, item in enumerate(margin_alphas_raw):
        if not isinstance(item, int | float):
            msg = (
                f"[detect].margin_alphas[{i}] must be a number,"
                f" got {type(item).__name__}"
            )
            raise TypeError(msg)
        margin_alphas.append(float(item))

    # -- svf_compare (bool) --
    svf_compare_raw = sec.get("svf_compare", False)  # type: ignore[arg-type]
    if not isinstance(svf_compare_raw, bool):
        msg = (
            f"[detect].svf_compare must be a boolean,"
            f" got {type(svf_compare_raw).__name__}"
        )
        raise TypeError(msg)

    return DetectConfig(
        mode=mode_raw,
        top_k=int(top_k_raw),
        clip_quantile=float(clip_quantile_raw),
        alpha=float(alpha_raw),
        max_tokens=max_tokens_raw,
        margin_directions=margin_directions,
        margin_alphas=margin_alphas,
        svf_compare=svf_compare_raw,
    )

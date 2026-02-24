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
    valid_modes = ("fast", "probe", "full")
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

    return DetectConfig(
        mode=mode_raw,
        top_k=int(top_k_raw),
        clip_quantile=float(clip_quantile_raw),
        alpha=float(alpha_raw),
        max_tokens=max_tokens_raw,
    )

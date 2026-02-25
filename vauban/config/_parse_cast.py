"""Parse the [cast] section of a TOML config."""

from vauban.config._types import TomlDict
from vauban.types import CastConfig


def _parse_cast(raw: TomlDict) -> CastConfig | None:
    """Parse the optional [cast] section into a CastConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("cast")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[cast] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    # -- prompts (required) --
    prompts_raw = sec.get("prompts")  # type: ignore[arg-type]
    if prompts_raw is None:
        msg = "[cast].prompts is required"
        raise ValueError(msg)
    if not isinstance(prompts_raw, list):
        msg = (
            f"[cast].prompts must be a list of strings,"
            f" got {type(prompts_raw).__name__}"
        )
        raise TypeError(msg)
    if len(prompts_raw) == 0:
        msg = "[cast].prompts must be non-empty"
        raise ValueError(msg)
    for i, item in enumerate(prompts_raw):
        if not isinstance(item, str):
            msg = (
                f"[cast].prompts[{i}] must be a string,"
                f" got {type(item).__name__}"
            )
            raise TypeError(msg)

    # -- layers (optional) --
    layers_raw = sec.get("layers")  # type: ignore[arg-type]
    layers: list[int] | None = None
    if layers_raw is not None:
        if not isinstance(layers_raw, list):
            msg = (
                f"[cast].layers must be a list of integers,"
                f" got {type(layers_raw).__name__}"
            )
            raise TypeError(msg)
        for i, item in enumerate(layers_raw):
            if not isinstance(item, int):
                msg = (
                    f"[cast].layers[{i}] must be an integer,"
                    f" got {type(item).__name__}"
                )
                raise TypeError(msg)
        layers = list(layers_raw)

    # -- alpha (optional, default 1.0) --
    alpha_raw = sec.get("alpha", 1.0)  # type: ignore[arg-type]
    if not isinstance(alpha_raw, int | float):
        msg = (
            f"[cast].alpha must be a number,"
            f" got {type(alpha_raw).__name__}"
        )
        raise TypeError(msg)

    # -- threshold (optional, default 0.0) --
    threshold_raw = sec.get("threshold", 0.0)  # type: ignore[arg-type]
    if not isinstance(threshold_raw, int | float):
        msg = (
            f"[cast].threshold must be a number,"
            f" got {type(threshold_raw).__name__}"
        )
        raise TypeError(msg)

    # -- max_tokens (optional, default 100) --
    max_tokens_raw = sec.get("max_tokens", 100)  # type: ignore[arg-type]
    if not isinstance(max_tokens_raw, int):
        msg = (
            f"[cast].max_tokens must be an integer,"
            f" got {type(max_tokens_raw).__name__}"
        )
        raise TypeError(msg)
    if max_tokens_raw < 1:
        msg = f"[cast].max_tokens must be >= 1, got {max_tokens_raw}"
        raise ValueError(msg)

    return CastConfig(
        prompts=list(prompts_raw),
        layers=layers,
        alpha=float(alpha_raw),
        threshold=float(threshold_raw),
        max_tokens=max_tokens_raw,
    )


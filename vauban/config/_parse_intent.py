"""Parse the [intent] section of a TOML config."""

from vauban.config._types import TomlDict
from vauban.types import IntentConfig


def _parse_intent(raw: TomlDict) -> IntentConfig | None:
    """Parse the optional [intent] section into an IntentConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("intent")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[intent] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    # -- mode --
    mode_raw = sec.get("mode", "embedding")  # type: ignore[arg-type]
    if not isinstance(mode_raw, str):
        msg = (
            f"[intent].mode must be a string,"
            f" got {type(mode_raw).__name__}"
        )
        raise TypeError(msg)
    valid_modes = ("embedding", "judge")
    if mode_raw not in valid_modes:
        msg = (
            f"[intent].mode must be one of {valid_modes!r},"
            f" got {mode_raw!r}"
        )
        raise ValueError(msg)

    # -- target_layer --
    target_layer_raw = sec.get("target_layer")  # type: ignore[arg-type]
    target_layer: int | None = None
    if target_layer_raw is not None:
        if not isinstance(target_layer_raw, int):
            msg = (
                f"[intent].target_layer must be an integer,"
                f" got {type(target_layer_raw).__name__}"
            )
            raise TypeError(msg)
        target_layer = target_layer_raw

    # -- similarity_threshold --
    sim_raw = sec.get("similarity_threshold", 0.7)  # type: ignore[arg-type]
    if not isinstance(sim_raw, int | float):
        msg = (
            f"[intent].similarity_threshold must be a number,"
            f" got {type(sim_raw).__name__}"
        )
        raise TypeError(msg)

    # -- max_tokens --
    max_tokens_raw = sec.get("max_tokens", 10)  # type: ignore[arg-type]
    if not isinstance(max_tokens_raw, int):
        msg = (
            f"[intent].max_tokens must be an integer,"
            f" got {type(max_tokens_raw).__name__}"
        )
        raise TypeError(msg)

    return IntentConfig(
        mode=mode_raw,
        target_layer=target_layer,
        similarity_threshold=float(sim_raw),
        max_tokens=max_tokens_raw,
    )

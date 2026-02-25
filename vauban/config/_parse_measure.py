"""Parse the [measure] section of a TOML config."""

from vauban.config._types import TomlDict
from vauban.types import MeasureConfig


def _parse_measure(raw: TomlDict) -> MeasureConfig:
    """Parse the optional [measure] section into a MeasureConfig."""
    mode_raw = raw.get("mode", "direction")
    if not isinstance(mode_raw, str):
        msg = f"[measure].mode must be a string, got {type(mode_raw).__name__}"
        raise TypeError(msg)
    if mode_raw not in ("direction", "subspace", "dbdi"):
        msg = (
            f"[measure].mode must be 'direction', 'subspace', or 'dbdi',"
            f" got {mode_raw!r}"
        )
        raise ValueError(msg)

    top_k_raw = raw.get("top_k", 5)
    if not isinstance(top_k_raw, int):
        msg = (
            f"[measure].top_k must be an integer,"
            f" got {type(top_k_raw).__name__}"
        )
        raise TypeError(msg)

    clip_quantile_raw = raw.get("clip_quantile", 0.0)
    if not isinstance(clip_quantile_raw, int | float):
        msg = (
            f"[measure].clip_quantile must be a number,"
            f" got {type(clip_quantile_raw).__name__}"
        )
        raise TypeError(msg)
    clip_quantile = float(clip_quantile_raw)
    if not 0.0 <= clip_quantile < 0.5:
        msg = (
            f"[measure].clip_quantile must be in [0.0, 0.5),"
            f" got {clip_quantile}"
        )
        raise ValueError(msg)

    transfer_models_raw = raw.get("transfer_models", [])
    if not isinstance(transfer_models_raw, list):
        msg = (
            f"[measure].transfer_models must be a list,"
            f" got {type(transfer_models_raw).__name__}"
        )
        raise TypeError(msg)
    transfer_models: list[str] = []
    for i, item in enumerate(transfer_models_raw):
        if not isinstance(item, str):
            msg = (
                f"[measure].transfer_models[{i}] must be a string,"
                f" got {type(item).__name__}"
            )
            raise TypeError(msg)
        transfer_models.append(item)

    return MeasureConfig(
        mode=mode_raw,
        top_k=int(top_k_raw),
        clip_quantile=clip_quantile,
        transfer_models=transfer_models,
    )

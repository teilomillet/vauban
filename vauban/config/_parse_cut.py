"""Parse the [cut] section of a TOML config."""

from vauban.config._types import TomlDict
from vauban.types import CutConfig


def _parse_cut(raw: TomlDict) -> CutConfig:
    """Parse the [cut] section into a CutConfig."""
    layers_raw = raw.get("layers")
    layers: list[int] | None
    if layers_raw is None or layers_raw == "auto":
        layers = None
    elif isinstance(layers_raw, list):
        layers = [int(x) for x in layers_raw if isinstance(x, int | float)]
    else:
        msg = f"[cut].layers must be 'auto' or a list of ints, got {layers_raw!r}"
        raise TypeError(msg)

    alpha_raw = raw.get("alpha", 1.0)
    if not isinstance(alpha_raw, int | float):
        msg = f"[cut].alpha must be a number, got {type(alpha_raw).__name__}"
        raise TypeError(msg)

    layer_strategy_raw = raw.get("layer_strategy", "all")
    if not isinstance(layer_strategy_raw, str):
        msg = (
            f"[cut].layer_strategy must be a string,"
            f" got {type(layer_strategy_raw).__name__}"
        )
        raise TypeError(msg)
    valid_strategies = ("all", "above_median", "top_k")
    if layer_strategy_raw not in valid_strategies:
        msg = (
            f"[cut].layer_strategy must be one of {valid_strategies!r},"
            f" got {layer_strategy_raw!r}"
        )
        raise ValueError(msg)

    layer_top_k_raw = raw.get("layer_top_k", 10)
    if not isinstance(layer_top_k_raw, int | float):
        msg = (
            f"[cut].layer_top_k must be an integer,"
            f" got {type(layer_top_k_raw).__name__}"
        )
        raise TypeError(msg)

    layer_weights_raw = raw.get("layer_weights")
    layer_weights: list[float] | None = None
    if layer_weights_raw is not None:
        if not isinstance(layer_weights_raw, list):
            msg = (
                f"[cut].layer_weights must be a list of numbers,"
                f" got {type(layer_weights_raw).__name__}"
            )
            raise TypeError(msg)
        layer_weights = [
            float(x)  # type: ignore[arg-type]
            for x in layer_weights_raw
        ]

    sparsity_raw = raw.get("sparsity", 0.0)
    if not isinstance(sparsity_raw, int | float):
        msg = (
            f"[cut].sparsity must be a number,"
            f" got {type(sparsity_raw).__name__}"
        )
        raise TypeError(msg)
    sparsity = float(sparsity_raw)
    if not 0.0 <= sparsity < 1.0:
        msg = f"[cut].sparsity must be in [0.0, 1.0), got {sparsity}"
        raise ValueError(msg)

    dbdi_target_raw = raw.get("dbdi_target", "red")
    if not isinstance(dbdi_target_raw, str):
        msg = (
            f"[cut].dbdi_target must be a string,"
            f" got {type(dbdi_target_raw).__name__}"
        )
        raise TypeError(msg)
    valid_dbdi_targets = ("red", "hdd", "both")
    if dbdi_target_raw not in valid_dbdi_targets:
        msg = (
            f"[cut].dbdi_target must be one of {valid_dbdi_targets!r},"
            f" got {dbdi_target_raw!r}"
        )
        raise ValueError(msg)

    false_refusal_ortho_raw = raw.get("false_refusal_ortho", False)
    if not isinstance(false_refusal_ortho_raw, bool):
        msg = (
            f"[cut].false_refusal_ortho must be a boolean,"
            f" got {type(false_refusal_ortho_raw).__name__}"
        )
        raise TypeError(msg)

    layer_type_filter_raw = raw.get("layer_type_filter")
    layer_type_filter: str | None = None
    if layer_type_filter_raw is not None:
        if not isinstance(layer_type_filter_raw, str):
            msg = (
                f"[cut].layer_type_filter must be a string,"
                f" got {type(layer_type_filter_raw).__name__}"
            )
            raise TypeError(msg)
        valid_type_filters = ("global", "sliding")
        if layer_type_filter_raw not in valid_type_filters:
            msg = (
                f"[cut].layer_type_filter must be one of"
                f" {valid_type_filters!r}, got {layer_type_filter_raw!r}"
            )
            raise ValueError(msg)
        layer_type_filter = layer_type_filter_raw

    return CutConfig(
        alpha=float(alpha_raw),
        layers=layers,
        norm_preserve=bool(raw.get("norm_preserve", False)),
        biprojected=bool(raw.get("biprojected", False)),
        layer_strategy=layer_strategy_raw,
        layer_top_k=int(layer_top_k_raw),
        layer_weights=layer_weights,
        sparsity=sparsity,
        dbdi_target=dbdi_target_raw,
        false_refusal_ortho=false_refusal_ortho_raw,
        layer_type_filter=layer_type_filter,
    )

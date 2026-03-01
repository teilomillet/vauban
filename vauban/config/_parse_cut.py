"""Parse the [cut] section of a TOML config."""

from vauban.config._parse_helpers import SectionReader
from vauban.config._types import TomlDict
from vauban.types import CutConfig


def _parse_cut(raw: TomlDict) -> CutConfig:
    """Parse the [cut] section into a CutConfig."""
    reader = SectionReader("[cut]", raw)

    # layers accepts "auto" (string) or a list of ints — special case
    layers_raw = raw.get("layers")
    layers: list[int] | None
    if layers_raw is None or layers_raw == "auto":
        layers = None
    elif isinstance(layers_raw, list):
        layers = [int(x) for x in layers_raw if isinstance(x, int | float)]
    else:
        msg = f"[cut].layers must be 'auto' or a list of ints, got {layers_raw!r}"
        raise TypeError(msg)

    alpha = reader.number("alpha", default=1.0)
    layer_strategy = reader.literal(
        "layer_strategy", ("all", "above_median", "top_k"), default="all",
    )
    layer_top_k = reader.integer("layer_top_k", default=10)
    layer_weights = reader.optional_number_list("layer_weights")

    sparsity = reader.number("sparsity", default=0.0)
    if not 0.0 <= sparsity < 1.0:
        msg = f"[cut].sparsity must be in [0.0, 1.0), got {sparsity}"
        raise ValueError(msg)

    dbdi_target = reader.literal(
        "dbdi_target", ("red", "hdd", "both"), default="red",
    )
    false_refusal_ortho = reader.boolean("false_refusal_ortho", default=False)
    norm_preserve = reader.boolean("norm_preserve", default=False)
    biprojected = reader.boolean("biprojected", default=False)
    layer_type_filter = reader.optional_literal(
        "layer_type_filter", ("global", "sliding"),
    )

    return CutConfig(
        alpha=alpha,
        layers=layers,
        norm_preserve=norm_preserve,
        biprojected=biprojected,
        layer_strategy=layer_strategy,
        layer_top_k=layer_top_k,
        layer_weights=layer_weights,
        sparsity=sparsity,
        dbdi_target=dbdi_target,
        false_refusal_ortho=false_refusal_ortho,
        layer_type_filter=layer_type_filter,
    )

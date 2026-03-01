"""Parse the [features] section of a TOML config."""

from pathlib import Path

from vauban.config._parse_helpers import SectionReader
from vauban.config._types import TomlDict
from vauban.types import FeaturesConfig


def _parse_features(base_dir: Path, raw: TomlDict) -> FeaturesConfig | None:
    """Parse the optional [features] section into a FeaturesConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("features")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[features] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    reader = SectionReader("[features]", sec)

    # -- prompts_path (required) --
    prompts_path_str = reader.string("prompts_path")
    prompts_path = Path(prompts_path_str)
    if not prompts_path.is_absolute():
        prompts_path = base_dir / prompts_path

    # -- layers (required) --
    layers = reader.int_list("layers")
    if not layers:
        msg = "[features].layers must not be empty"
        raise ValueError(msg)
    for i, layer_val in enumerate(layers):
        if layer_val < 0:
            msg = f"[features].layers[{i}] must be >= 0, got {layer_val}"
            raise ValueError(msg)

    # -- d_sae (optional, default 2048) --
    d_sae = reader.integer("d_sae", default=2048)
    if d_sae < 1:
        msg = f"[features].d_sae must be >= 1, got {d_sae}"
        raise ValueError(msg)

    # -- l1_coeff (optional, default 1e-3) --
    l1_coeff = reader.number("l1_coeff", default=1e-3)
    if l1_coeff < 0:
        msg = f"[features].l1_coeff must be >= 0, got {l1_coeff}"
        raise ValueError(msg)

    # -- n_epochs (optional, default 5) --
    n_epochs = reader.integer("n_epochs", default=5)
    if n_epochs < 1:
        msg = f"[features].n_epochs must be >= 1, got {n_epochs}"
        raise ValueError(msg)

    # -- learning_rate (optional, default 1e-3) --
    lr = reader.number("learning_rate", default=1e-3)
    if lr <= 0:
        msg = f"[features].learning_rate must be > 0, got {lr}"
        raise ValueError(msg)

    # -- batch_size (optional, default 32) --
    batch_size = reader.integer("batch_size", default=32)
    if batch_size < 1:
        msg = f"[features].batch_size must be >= 1, got {batch_size}"
        raise ValueError(msg)

    # -- token_position (optional, default -1) --
    token_position = reader.integer("token_position", default=-1)

    # -- dead_feature_threshold (optional, default 1e-6) --
    dead_threshold = reader.number("dead_feature_threshold", default=1e-6)
    if dead_threshold < 0:
        msg = (
            f"[features].dead_feature_threshold must be >= 0,"
            f" got {dead_threshold}"
        )
        raise ValueError(msg)

    return FeaturesConfig(
        prompts_path=prompts_path,
        layers=layers,
        d_sae=d_sae,
        l1_coeff=l1_coeff,
        n_epochs=n_epochs,
        learning_rate=lr,
        batch_size=batch_size,
        token_position=token_position,
        dead_feature_threshold=dead_threshold,
    )

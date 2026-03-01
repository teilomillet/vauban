"""Parse the [features] section of a TOML config."""

from pathlib import Path

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

    # -- prompts_path (required) --
    prompts_raw = sec.get("prompts_path")  # type: ignore[arg-type]
    if prompts_raw is None:
        msg = "[features].prompts_path is required"
        raise ValueError(msg)
    if not isinstance(prompts_raw, str):
        msg = (
            f"[features].prompts_path must be a string,"
            f" got {type(prompts_raw).__name__}"
        )
        raise TypeError(msg)
    prompts_path = Path(prompts_raw)
    if not prompts_path.is_absolute():
        prompts_path = base_dir / prompts_path

    # -- layers (required) --
    layers_raw = sec.get("layers")  # type: ignore[arg-type]
    if layers_raw is None:
        msg = "[features].layers is required"
        raise ValueError(msg)
    if not isinstance(layers_raw, list):
        msg = (
            f"[features].layers must be a list of integers,"
            f" got {type(layers_raw).__name__}"
        )
        raise TypeError(msg)
    for i, item in enumerate(layers_raw):
        if not isinstance(item, int):
            msg = (
                f"[features].layers[{i}] must be an integer,"
                f" got {type(item).__name__}"
            )
            raise TypeError(msg)
    layers: list[int] = list(layers_raw)

    # -- d_sae (optional, default 2048) --
    d_sae_raw = sec.get("d_sae", 2048)  # type: ignore[arg-type]
    if not isinstance(d_sae_raw, int):
        msg = (
            f"[features].d_sae must be an integer,"
            f" got {type(d_sae_raw).__name__}"
        )
        raise TypeError(msg)
    if d_sae_raw < 1:
        msg = f"[features].d_sae must be >= 1, got {d_sae_raw}"
        raise ValueError(msg)

    # -- l1_coeff (optional, default 1e-3) --
    l1_raw = sec.get("l1_coeff", 1e-3)  # type: ignore[arg-type]
    if not isinstance(l1_raw, int | float):
        msg = (
            f"[features].l1_coeff must be a number,"
            f" got {type(l1_raw).__name__}"
        )
        raise TypeError(msg)

    # -- n_epochs (optional, default 5) --
    n_epochs_raw = sec.get("n_epochs", 5)  # type: ignore[arg-type]
    if not isinstance(n_epochs_raw, int):
        msg = (
            f"[features].n_epochs must be an integer,"
            f" got {type(n_epochs_raw).__name__}"
        )
        raise TypeError(msg)
    if n_epochs_raw < 1:
        msg = f"[features].n_epochs must be >= 1, got {n_epochs_raw}"
        raise ValueError(msg)

    # -- learning_rate (optional, default 1e-3) --
    lr_raw = sec.get("learning_rate", 1e-3)  # type: ignore[arg-type]
    if not isinstance(lr_raw, int | float):
        msg = (
            f"[features].learning_rate must be a number,"
            f" got {type(lr_raw).__name__}"
        )
        raise TypeError(msg)

    # -- batch_size (optional, default 32) --
    batch_raw = sec.get("batch_size", 32)  # type: ignore[arg-type]
    if not isinstance(batch_raw, int):
        msg = (
            f"[features].batch_size must be an integer,"
            f" got {type(batch_raw).__name__}"
        )
        raise TypeError(msg)
    if batch_raw < 1:
        msg = f"[features].batch_size must be >= 1, got {batch_raw}"
        raise ValueError(msg)

    # -- token_position (optional, default -1) --
    tok_pos_raw = sec.get("token_position", -1)  # type: ignore[arg-type]
    if not isinstance(tok_pos_raw, int):
        msg = (
            f"[features].token_position must be an integer,"
            f" got {type(tok_pos_raw).__name__}"
        )
        raise TypeError(msg)

    # -- dead_feature_threshold (optional, default 1e-6) --
    dead_raw = sec.get("dead_feature_threshold", 1e-6)  # type: ignore[arg-type]
    if not isinstance(dead_raw, int | float):
        msg = (
            f"[features].dead_feature_threshold must be a number,"
            f" got {type(dead_raw).__name__}"
        )
        raise TypeError(msg)

    return FeaturesConfig(
        prompts_path=prompts_path,
        layers=layers,
        d_sae=d_sae_raw,
        l1_coeff=float(l1_raw),
        n_epochs=n_epochs_raw,
        learning_rate=float(lr_raw),
        batch_size=batch_raw,
        token_position=tok_pos_raw,
        dead_feature_threshold=float(dead_raw),
    )

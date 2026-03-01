"""Parse the [svf] section of a TOML config."""

from pathlib import Path

from vauban.config._types import TomlDict
from vauban.types import SVFConfig


def _parse_svf(base_dir: Path, raw: TomlDict) -> SVFConfig | None:
    """Parse the optional [svf] section into an SVFConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("svf")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[svf] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    # -- prompts_target (required) --
    target_raw = sec.get("prompts_target")  # type: ignore[arg-type]
    if target_raw is None:
        msg = "[svf].prompts_target is required"
        raise ValueError(msg)
    if not isinstance(target_raw, str):
        msg = (
            f"[svf].prompts_target must be a string,"
            f" got {type(target_raw).__name__}"
        )
        raise TypeError(msg)
    target_path = Path(target_raw)
    if not target_path.is_absolute():
        target_path = base_dir / target_path

    # -- prompts_opposite (required) --
    opposite_raw = sec.get("prompts_opposite")  # type: ignore[arg-type]
    if opposite_raw is None:
        msg = "[svf].prompts_opposite is required"
        raise ValueError(msg)
    if not isinstance(opposite_raw, str):
        msg = (
            f"[svf].prompts_opposite must be a string,"
            f" got {type(opposite_raw).__name__}"
        )
        raise TypeError(msg)
    opposite_path = Path(opposite_raw)
    if not opposite_path.is_absolute():
        opposite_path = base_dir / opposite_path

    # -- projection_dim (optional, default 16) --
    proj_dim_raw = sec.get("projection_dim", 16)  # type: ignore[arg-type]
    if not isinstance(proj_dim_raw, int):
        msg = (
            f"[svf].projection_dim must be an integer,"
            f" got {type(proj_dim_raw).__name__}"
        )
        raise TypeError(msg)
    if proj_dim_raw < 1:
        msg = f"[svf].projection_dim must be >= 1, got {proj_dim_raw}"
        raise ValueError(msg)

    # -- hidden_dim (optional, default 64) --
    hidden_dim_raw = sec.get("hidden_dim", 64)  # type: ignore[arg-type]
    if not isinstance(hidden_dim_raw, int):
        msg = (
            f"[svf].hidden_dim must be an integer,"
            f" got {type(hidden_dim_raw).__name__}"
        )
        raise TypeError(msg)
    if hidden_dim_raw < 1:
        msg = f"[svf].hidden_dim must be >= 1, got {hidden_dim_raw}"
        raise ValueError(msg)

    # -- n_epochs (optional, default 10) --
    n_epochs_raw = sec.get("n_epochs", 10)  # type: ignore[arg-type]
    if not isinstance(n_epochs_raw, int):
        msg = (
            f"[svf].n_epochs must be an integer,"
            f" got {type(n_epochs_raw).__name__}"
        )
        raise TypeError(msg)
    if n_epochs_raw < 1:
        msg = f"[svf].n_epochs must be >= 1, got {n_epochs_raw}"
        raise ValueError(msg)

    # -- learning_rate (optional, default 1e-3) --
    lr_raw = sec.get("learning_rate", 1e-3)  # type: ignore[arg-type]
    if not isinstance(lr_raw, int | float):
        msg = (
            f"[svf].learning_rate must be a number,"
            f" got {type(lr_raw).__name__}"
        )
        raise TypeError(msg)

    # -- layers (optional) --
    layers_raw = sec.get("layers")  # type: ignore[arg-type]
    layers: list[int] | None = None
    if layers_raw is not None:
        if not isinstance(layers_raw, list):
            msg = (
                f"[svf].layers must be a list of integers,"
                f" got {type(layers_raw).__name__}"
            )
            raise TypeError(msg)
        for i, item in enumerate(layers_raw):
            if not isinstance(item, int):
                msg = (
                    f"[svf].layers[{i}] must be an integer,"
                    f" got {type(item).__name__}"
                )
                raise TypeError(msg)
        layers = list(layers_raw)

    return SVFConfig(
        prompts_target=target_path,
        prompts_opposite=opposite_path,
        projection_dim=proj_dim_raw,
        hidden_dim=hidden_dim_raw,
        n_epochs=n_epochs_raw,
        learning_rate=float(lr_raw),
        layers=layers,
    )

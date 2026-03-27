"""Parse the [svf] section of a TOML config."""

from pathlib import Path

from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.config._types import TomlDict
from vauban.types import SVFConfig


def _parse_svf(base_dir: Path, raw: TomlDict) -> SVFConfig | None:
    """Parse the optional [svf] section into an SVFConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("svf")
    if sec is None:
        return None
    reader = SectionReader("[svf]", require_toml_table("[svf]", sec))

    # -- prompts_target (required) --
    target_str = reader.string("prompts_target")
    target_path = Path(target_str)
    if not target_path.is_absolute():
        target_path = base_dir / target_path

    # -- prompts_opposite (required) --
    opposite_str = reader.string("prompts_opposite")
    opposite_path = Path(opposite_str)
    if not opposite_path.is_absolute():
        opposite_path = base_dir / opposite_path

    # -- projection_dim (optional, default 16) --
    proj_dim = reader.integer("projection_dim", default=16)
    if proj_dim < 1:
        msg = f"[svf].projection_dim must be >= 1, got {proj_dim}"
        raise ValueError(msg)

    # -- hidden_dim (optional, default 64) --
    hidden_dim = reader.integer("hidden_dim", default=64)
    if hidden_dim < 1:
        msg = f"[svf].hidden_dim must be >= 1, got {hidden_dim}"
        raise ValueError(msg)

    # -- n_epochs (optional, default 10) --
    n_epochs = reader.integer("n_epochs", default=10)
    if n_epochs < 1:
        msg = f"[svf].n_epochs must be >= 1, got {n_epochs}"
        raise ValueError(msg)

    # -- learning_rate (optional, default 1e-3) --
    lr = reader.number("learning_rate", default=1e-3)

    # -- layers (optional) --
    layers = reader.optional_int_list("layers")

    return SVFConfig(
        prompts_target=target_path,
        prompts_opposite=opposite_path,
        projection_dim=proj_dim,
        hidden_dim=hidden_dim,
        n_epochs=n_epochs,
        learning_rate=lr,
        layers=layers,
    )

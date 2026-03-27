"""Parse the [linear_probe] section of a TOML config."""

from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.config._types import TomlDict
from vauban.types import LinearProbeConfig


def _parse_linear_probe(raw: TomlDict) -> LinearProbeConfig | None:
    """Parse the optional [linear_probe] section into a LinearProbeConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("linear_probe")
    if sec is None:
        return None
    reader = SectionReader(
        "[linear_probe]",
        require_toml_table("[linear_probe]", sec),
    )

    # -- layers (required) --
    layers = reader.int_list("layers")
    if not layers:
        msg = "[linear_probe].layers must be non-empty"
        raise ValueError(msg)
    for i, v in enumerate(layers):
        if v < 0:
            msg = f"[linear_probe].layers[{i}] must be >= 0, got {v}"
            raise ValueError(msg)

    # -- n_epochs --
    n_epochs = reader.integer("n_epochs", default=20)
    if n_epochs < 1:
        msg = f"[linear_probe].n_epochs must be >= 1, got {n_epochs}"
        raise ValueError(msg)

    # -- learning_rate --
    learning_rate = reader.number("learning_rate", default=1e-2)
    if learning_rate <= 0:
        msg = f"[linear_probe].learning_rate must be > 0, got {learning_rate}"
        raise ValueError(msg)

    # -- batch_size --
    batch_size = reader.integer("batch_size", default=32)
    if batch_size < 1:
        msg = f"[linear_probe].batch_size must be >= 1, got {batch_size}"
        raise ValueError(msg)

    # -- token_position --
    token_position = reader.integer("token_position", default=-1)

    # -- regularization --
    regularization = reader.number("regularization", default=1e-4)
    if regularization < 0:
        msg = (
            f"[linear_probe].regularization must be >= 0, got {regularization}"
        )
        raise ValueError(msg)

    return LinearProbeConfig(
        layers=layers,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        token_position=token_position,
        regularization=regularization,
    )

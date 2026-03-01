"""Parse the [repbend] section of a TOML config."""

from vauban.config._parse_helpers import SectionReader
from vauban.config._types import TomlDict
from vauban.types import RepBendConfig


def _parse_repbend(raw: TomlDict) -> RepBendConfig | None:
    """Parse the optional [repbend] section into a RepBendConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("repbend")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[repbend] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    reader = SectionReader("[repbend]", sec)

    # -- layers (required) --
    layers = reader.int_list("layers")
    if not layers:
        msg = "[repbend].layers must be non-empty"
        raise ValueError(msg)
    for i, v in enumerate(layers):
        if v < 0:
            msg = f"[repbend].layers[{i}] must be >= 0, got {v}"
            raise ValueError(msg)

    # -- n_epochs --
    n_epochs = reader.integer("n_epochs", default=3)
    if n_epochs < 1:
        msg = f"[repbend].n_epochs must be >= 1, got {n_epochs}"
        raise ValueError(msg)

    # -- learning_rate --
    learning_rate = reader.number("learning_rate", default=1e-5)
    if learning_rate <= 0:
        msg = f"[repbend].learning_rate must be > 0, got {learning_rate}"
        raise ValueError(msg)

    # -- batch_size --
    batch_size = reader.integer("batch_size", default=8)
    if batch_size < 1:
        msg = f"[repbend].batch_size must be >= 1, got {batch_size}"
        raise ValueError(msg)

    # -- separation_coeff --
    separation_coeff = reader.number("separation_coeff", default=1.0)
    if separation_coeff <= 0:
        msg = (
            f"[repbend].separation_coeff must be > 0, got {separation_coeff}"
        )
        raise ValueError(msg)

    # -- token_position --
    token_position = reader.integer("token_position", default=-1)

    return RepBendConfig(
        layers=layers,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        separation_coeff=separation_coeff,
        token_position=token_position,
    )

"""Parse the [compose_optimize] section of a TOML config."""

from pathlib import Path

from vauban.config._types import TomlDict
from vauban.types import ComposeOptimizeConfig


def _parse_compose_optimize(
    base_dir: Path, raw: TomlDict,
) -> ComposeOptimizeConfig | None:
    """Parse the optional [compose_optimize] section.

    Returns None if the section is absent.
    """
    sec = raw.get("compose_optimize")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = (
            f"[compose_optimize] must be a table,"
            f" got {type(sec).__name__}"
        )
        raise TypeError(msg)

    # -- bank_path (required) --
    bank_path_raw = sec.get("bank_path")  # type: ignore[arg-type]
    if bank_path_raw is None:
        msg = "[compose_optimize].bank_path is required"
        raise ValueError(msg)
    if not isinstance(bank_path_raw, str):
        msg = (
            "[compose_optimize].bank_path must be a string,"
            f" got {type(bank_path_raw).__name__}"
        )
        raise TypeError(msg)
    bank_path = str((base_dir / bank_path_raw).resolve())

    # -- n_trials --
    n_trials_raw = sec.get("n_trials", 50)  # type: ignore[arg-type]
    if not isinstance(n_trials_raw, int):
        msg = (
            "[compose_optimize].n_trials must be an integer,"
            f" got {type(n_trials_raw).__name__}"
        )
        raise TypeError(msg)
    if n_trials_raw < 1:
        msg = (
            f"[compose_optimize].n_trials must be >= 1,"
            f" got {n_trials_raw}"
        )
        raise ValueError(msg)

    # -- max_tokens --
    max_tokens_raw = sec.get("max_tokens", 100)  # type: ignore[arg-type]
    if not isinstance(max_tokens_raw, int):
        msg = (
            "[compose_optimize].max_tokens must be an integer,"
            f" got {type(max_tokens_raw).__name__}"
        )
        raise TypeError(msg)

    # -- timeout --
    timeout_raw = sec.get("timeout")  # type: ignore[arg-type]
    timeout: float | None = None
    if timeout_raw is not None:
        if not isinstance(timeout_raw, int | float):
            msg = (
                "[compose_optimize].timeout must be a number,"
                f" got {type(timeout_raw).__name__}"
            )
            raise TypeError(msg)
        timeout = float(timeout_raw)

    # -- seed --
    seed_raw = sec.get("seed")  # type: ignore[arg-type]
    seed: int | None = None
    if seed_raw is not None:
        if not isinstance(seed_raw, int):
            msg = (
                "[compose_optimize].seed must be an integer,"
                f" got {type(seed_raw).__name__}"
            )
            raise TypeError(msg)
        seed = seed_raw

    return ComposeOptimizeConfig(
        bank_path=bank_path,
        n_trials=n_trials_raw,
        max_tokens=max_tokens_raw,
        timeout=timeout,
        seed=seed,
    )

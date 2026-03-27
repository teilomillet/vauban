# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Parse the [compose_optimize] section of a TOML config."""

from pathlib import Path

from vauban.config._parse_helpers import SectionReader, require_toml_table
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
    reader = SectionReader(
        "[compose_optimize]",
        require_toml_table("[compose_optimize]", sec),
    )

    # -- bank_path (required) --
    bank_path_str = reader.string("bank_path")
    bank_path = str((base_dir / bank_path_str).resolve())

    # -- n_trials --
    n_trials = reader.integer("n_trials", default=50)
    if n_trials < 1:
        msg = (
            f"[compose_optimize].n_trials must be >= 1,"
            f" got {n_trials}"
        )
        raise ValueError(msg)

    # -- max_tokens --
    max_tokens = reader.integer("max_tokens", default=100)

    # -- timeout --
    timeout = reader.optional_number("timeout")

    # -- seed --
    seed = reader.optional_integer("seed")

    return ComposeOptimizeConfig(
        bank_path=bank_path,
        n_trials=n_trials,
        max_tokens=max_tokens,
        timeout=timeout,
        seed=seed,
    )

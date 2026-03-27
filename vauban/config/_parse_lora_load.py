# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Parse the [lora] section of a TOML config."""

from typing import cast

from vauban.config._parse_helpers import SectionReader
from vauban.config._types import TomlDict
from vauban.types import LoraLoadConfig


def _parse_lora_load(raw: TomlDict) -> LoraLoadConfig | None:
    """Parse the optional [lora] section into a LoraLoadConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("lora")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[lora] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    reader = SectionReader("[lora]", cast("TomlDict", sec))

    adapter_path = reader.optional_string("adapter_path")
    adapter_paths = reader.optional_string_list("adapter_paths")
    weights = reader.optional_number_list("weights")

    # Validation: mutually exclusive
    if adapter_path is not None and adapter_paths is not None:
        msg = "[lora] adapter_path and adapter_paths are mutually exclusive"
        raise ValueError(msg)

    # At least one required
    if adapter_path is None and adapter_paths is None:
        msg = "[lora] requires either adapter_path or adapter_paths"
        raise ValueError(msg)

    # Weights length must match adapter_paths
    if weights is not None:
        if adapter_paths is None:
            msg = "[lora].weights requires adapter_paths (not adapter_path)"
            raise ValueError(msg)
        if len(weights) != len(adapter_paths):
            msg = (
                f"[lora].weights length ({len(weights)}) must match"
                f" adapter_paths length ({len(adapter_paths)})"
            )
            raise ValueError(msg)

    return LoraLoadConfig(
        adapter_path=adapter_path,
        adapter_paths=adapter_paths,
        weights=weights,
    )

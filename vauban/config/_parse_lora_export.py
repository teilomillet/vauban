"""Parse the [lora_export] section of a TOML config."""

from typing import cast

from vauban.config._parse_helpers import SectionReader
from vauban.config._types import TomlDict
from vauban.types import LoraExportConfig


def _parse_lora_export(raw: TomlDict) -> LoraExportConfig | None:
    """Parse the optional [lora_export] section into a LoraExportConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("lora_export")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[lora_export] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    reader = SectionReader("[lora_export]", cast("TomlDict", sec))

    fmt = reader.literal(
        "format",
        ("mlx", "peft"),
        default="mlx",
    )

    polarity = reader.literal(
        "polarity",
        ("remove", "add"),
        default="remove",
    )

    return LoraExportConfig(format=fmt, polarity=polarity)

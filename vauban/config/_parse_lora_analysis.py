"""Parse the [lora_analysis] section of a TOML config."""

from typing import cast

from vauban.config._parse_helpers import SectionReader
from vauban.config._types import TomlDict
from vauban.types import LoraAnalysisConfig


def _parse_lora_analysis(raw: TomlDict) -> LoraAnalysisConfig | None:
    """Parse the optional [lora_analysis] section into a LoraAnalysisConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("lora_analysis")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[lora_analysis] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    reader = SectionReader("[lora_analysis]", cast("TomlDict", sec))

    adapter_path = reader.optional_string("adapter_path")
    adapter_paths = reader.optional_string_list("adapter_paths")
    variance_threshold = reader.number("variance_threshold", default=0.99)
    align_with_direction = reader.boolean("align_with_direction", default=True)

    # Validation: mutually exclusive
    if adapter_path is not None and adapter_paths is not None:
        msg = (
            "[lora_analysis] adapter_path and adapter_paths"
            " are mutually exclusive"
        )
        raise ValueError(msg)

    # At least one required
    if adapter_path is None and adapter_paths is None:
        msg = (
            "[lora_analysis] requires either adapter_path or adapter_paths"
        )
        raise ValueError(msg)

    return LoraAnalysisConfig(
        adapter_path=adapter_path,
        adapter_paths=adapter_paths,
        variance_threshold=variance_threshold,
        align_with_direction=align_with_direction,
    )

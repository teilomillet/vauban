"""Parse the [probe] section of a TOML config."""

from vauban.config._parse_helpers import SectionReader
from vauban.config._types import TomlDict
from vauban.types import ProbeConfig


def _parse_probe(raw: TomlDict) -> ProbeConfig | None:
    """Parse the optional [probe] section into a ProbeConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("probe")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[probe] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    reader = SectionReader("[probe]", sec)

    prompts = reader.string_list("prompts")
    if not prompts:
        msg = "[probe].prompts must be non-empty"
        raise ValueError(msg)

    return ProbeConfig(prompts=prompts)

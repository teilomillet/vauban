"""Parse the [probe] section of a TOML config."""

from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.config._types import TomlDict
from vauban.types import ProbeConfig


def _parse_probe(raw: TomlDict) -> ProbeConfig | None:
    """Parse the optional [probe] section into a ProbeConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("probe")
    if sec is None:
        return None
    reader = SectionReader("[probe]", require_toml_table("[probe]", sec))

    prompts = reader.string_list("prompts")
    if not prompts:
        msg = "[probe].prompts must be non-empty"
        raise ValueError(msg)

    return ProbeConfig(prompts=prompts)

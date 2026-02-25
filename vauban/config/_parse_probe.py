"""Parse the [probe] section of a TOML config."""

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

    # -- prompts (required) --
    prompts_raw = sec.get("prompts")  # type: ignore[arg-type]
    if prompts_raw is None:
        msg = "[probe].prompts is required"
        raise ValueError(msg)
    if not isinstance(prompts_raw, list):
        msg = (
            f"[probe].prompts must be a list of strings,"
            f" got {type(prompts_raw).__name__}"
        )
        raise TypeError(msg)
    if len(prompts_raw) == 0:
        msg = "[probe].prompts must be non-empty"
        raise ValueError(msg)
    for i, item in enumerate(prompts_raw):
        if not isinstance(item, str):
            msg = (
                f"[probe].prompts[{i}] must be a string,"
                f" got {type(item).__name__}"
            )
            raise TypeError(msg)

    return ProbeConfig(prompts=list(prompts_raw))

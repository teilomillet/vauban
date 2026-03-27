"""Parse the [jailbreak] section of a TOML config."""

from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.config._types import TomlDict
from vauban.jailbreak import ALL_STRATEGIES
from vauban.types import JailbreakConfig


def _parse_jailbreak(raw: TomlDict) -> JailbreakConfig | None:
    """Parse the optional [jailbreak] section into a JailbreakConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("jailbreak")
    if sec is None:
        return None
    reader = SectionReader(
        "[jailbreak]", require_toml_table("[jailbreak]", sec),
    )

    # -- strategies (optional list, empty = all) --
    strategies = reader.string_list("strategies", default=[])
    for s in strategies:
        if s not in ALL_STRATEGIES:
            msg = (
                f"[jailbreak].strategies: unknown strategy {s!r}."
                f" Valid: {sorted(ALL_STRATEGIES)}"
            )
            raise ValueError(msg)

    # -- custom_templates_path (optional string) --
    custom_templates_path = reader.optional_string("custom_templates_path")

    # -- payloads_from (string, default "harmful") --
    payloads_from = reader.string("payloads_from", default="harmful")

    return JailbreakConfig(
        strategies=strategies,
        custom_templates_path=custom_templates_path,
        payloads_from=payloads_from,
    )

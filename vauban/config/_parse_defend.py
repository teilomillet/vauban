"""Parse the [defend] section of a TOML config."""

from typing import cast

from vauban.config._parse_intent import _parse_intent
from vauban.config._parse_policy import _parse_policy
from vauban.config._parse_scan import _parse_scan
from vauban.config._parse_sic import _parse_sic
from vauban.config._types import TomlDict
from vauban.types import DefenseStackConfig


def _parse_defend(raw: TomlDict) -> DefenseStackConfig | None:
    """Parse the optional [defend] section into a DefenseStackConfig.

    The [defend] section acts as a composition layer. It references
    other defense sections ([scan], [sic], [policy], [intent]) and
    adds a fail_fast toggle.

    Returns None if the section is absent.
    """
    sec = raw.get("defend")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[defend] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    defend_dict = cast("dict[str, object]", sec)

    # -- fail_fast --
    fail_fast_raw = defend_dict.get("fail_fast", True)
    if not isinstance(fail_fast_raw, bool):
        msg = (
            f"[defend].fail_fast must be a boolean,"
            f" got {type(fail_fast_raw).__name__}"
        )
        raise TypeError(msg)

    # Reuse existing section parsers — the [defend] section composes them
    scan_config = _parse_scan(raw)
    sic_config = _parse_sic(raw)
    policy_config = _parse_policy(raw)
    intent_config = _parse_intent(raw)

    return DefenseStackConfig(
        scan=scan_config,
        sic=sic_config,
        policy=policy_config,
        intent=intent_config,
        fail_fast=fail_fast_raw,
    )

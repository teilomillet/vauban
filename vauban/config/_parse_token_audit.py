# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Parse the [token_audit] section of a TOML config."""

from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.config._types import TomlDict
from vauban.types import TokenAuditConfig


def _parse_token_audit(raw: TomlDict) -> TokenAuditConfig | None:
    """Parse the optional [token_audit] section into a TokenAuditConfig."""
    sec = raw.get("token_audit")
    if sec is None:
        return None

    reader = SectionReader(
        "[token_audit]",
        require_toml_table("[token_audit]", sec),
    )

    max_token_id = reader.optional_integer("max_token_id")
    if max_token_id is not None and max_token_id < 0:
        msg = "[token_audit].max_token_id must be >= 0"
        raise ValueError(msg)

    include_duplicate_surface_scan = reader.boolean(
        "include_duplicate_surface_scan",
        default=True,
    )

    return TokenAuditConfig(
        max_token_id=max_token_id,
        include_duplicate_surface_scan=include_duplicate_surface_scan,
    )

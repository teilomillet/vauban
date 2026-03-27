# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Parse the [meta] section of a TOML config."""

import datetime
from pathlib import Path

from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.config._types import TomlDict
from vauban.types import MetaConfig, MetaDocRef

_VALID_STATUSES: tuple[str, ...] = (
    "archived", "baseline", "dead_end", "promising", "superseded", "wip",
)


def parse_meta(raw: TomlDict, config_path: Path) -> MetaConfig | None:
    """Parse the optional [meta] section into a MetaConfig.

    Returns None if the section is absent.  The *config_path* is used
    to derive a default ``id`` from the filename stem.
    """
    sec = raw.get("meta")
    if sec is None:
        return None
    reader = SectionReader("[meta]", require_toml_table("[meta]", sec))

    # -- id (optional, defaults to filename stem) --
    id_raw = reader.optional_string("id")
    if id_raw is not None:
        if not id_raw:
            msg = "[meta].id must be non-empty"
            raise ValueError(msg)
        meta_id: str = id_raw
    else:
        meta_id = config_path.stem

    # -- title (optional) --
    title = reader.string("title", default="")

    # -- status (optional, default "wip") --
    status = reader.literal("status", _VALID_STATUSES, default="wip")

    # -- parents (optional) --
    parents = reader.optional_string_list("parents")
    if parents is not None:
        for i, item in enumerate(parents):
            if not item:
                msg = f"[meta].parents[{i}] must be non-empty"
                raise ValueError(msg)
    else:
        parents = []

    # -- tags (optional) --
    tags = reader.optional_string_list("tags")
    if tags is not None:
        for i, item in enumerate(tags):
            if not item:
                msg = f"[meta].tags[{i}] must be non-empty"
                raise ValueError(msg)
    else:
        tags = []

    # -- notes (optional) --
    notes = reader.string("notes", default="")

    # -- date (optional, TOML native date or string) --
    date_raw = reader.data.get("date")
    date_str = ""
    if date_raw is not None:
        if isinstance(date_raw, datetime.date):
            date_str = date_raw.isoformat()
        elif isinstance(date_raw, str):
            date_str = date_raw
        else:
            msg = (
                f"[meta].date must be a date or string,"
                f" got {type(date_raw).__name__}"
            )
            raise TypeError(msg)

    # -- docs (optional, array of tables) --
    docs_raw = reader.data.get("docs")
    docs: list[MetaDocRef] = []
    if docs_raw is not None:
        if not isinstance(docs_raw, list):
            msg = (
                f"[meta].docs must be an array of tables,"
                f" got {type(docs_raw).__name__}"
            )
            raise TypeError(msg)
        for i, doc in enumerate(docs_raw):
            doc_table = require_toml_table(f"[[meta.docs]][{i}]", doc)
            path_val = doc_table.get("path")
            if path_val is None:
                msg = f"[[meta.docs]][{i}] requires 'path'"
                raise ValueError(msg)
            if not isinstance(path_val, str):
                msg = (
                    f"[[meta.docs]][{i}].path must be a string,"
                    f" got {type(path_val).__name__}"
                )
                raise TypeError(msg)
            label_val = doc_table.get("label", "")
            if not isinstance(label_val, str):
                msg = (
                    f"[[meta.docs]][{i}].label must be a string,"
                    f" got {type(label_val).__name__}"
                )
                raise TypeError(msg)
            docs.append(MetaDocRef(path=path_val, label=label_val))

    return MetaConfig(
        id=meta_id,
        title=title,
        status=status,
        parents=parents,
        tags=tags,
        notes=notes,
        docs=docs,
        date=date_str,
    )

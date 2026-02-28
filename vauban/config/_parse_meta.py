"""Parse the [meta] section of a TOML config."""

import datetime
from pathlib import Path

from vauban.config._types import TomlDict
from vauban.types import MetaConfig, MetaDocRef

_VALID_STATUSES: frozenset[str] = frozenset({
    "wip", "promising", "dead_end", "baseline", "superseded", "archived",
})


def parse_meta(raw: TomlDict, config_path: Path) -> MetaConfig | None:
    """Parse the optional [meta] section into a MetaConfig.

    Returns None if the section is absent.  The *config_path* is used
    to derive a default ``id`` from the filename stem.
    """
    sec = raw.get("meta")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[meta] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    # -- id (optional, defaults to filename stem) --
    id_raw = sec.get("id")  # type: ignore[arg-type]
    if id_raw is not None:
        if not isinstance(id_raw, str):
            msg = (
                f"[meta].id must be a string,"
                f" got {type(id_raw).__name__}"
            )
            raise TypeError(msg)
        if not id_raw:
            msg = "[meta].id must be non-empty"
            raise ValueError(msg)
        meta_id: str = id_raw
    else:
        meta_id = config_path.stem

    # -- title (optional) --
    title_raw = sec.get("title", "")  # type: ignore[arg-type]
    if not isinstance(title_raw, str):
        msg = (
            f"[meta].title must be a string,"
            f" got {type(title_raw).__name__}"
        )
        raise TypeError(msg)

    # -- status (optional, default "wip") --
    status_raw = sec.get("status", "wip")  # type: ignore[arg-type]
    if not isinstance(status_raw, str):
        msg = (
            f"[meta].status must be a string,"
            f" got {type(status_raw).__name__}"
        )
        raise TypeError(msg)
    if status_raw not in _VALID_STATUSES:
        msg = (
            f"[meta].status must be one of"
            f" {', '.join(sorted(_VALID_STATUSES))},"
            f" got {status_raw!r}"
        )
        raise ValueError(msg)

    # -- parents (optional) --
    parents_raw = sec.get("parents")  # type: ignore[arg-type]
    parents: list[str] = []
    if parents_raw is not None:
        if not isinstance(parents_raw, list):
            msg = (
                f"[meta].parents must be a list of strings,"
                f" got {type(parents_raw).__name__}"
            )
            raise TypeError(msg)
        for i, item in enumerate(parents_raw):
            if not isinstance(item, str):
                msg = (
                    f"[meta].parents[{i}] must be a string,"
                    f" got {type(item).__name__}"
                )
                raise TypeError(msg)
            if not item:
                msg = f"[meta].parents[{i}] must be non-empty"
                raise ValueError(msg)
        parents = list(parents_raw)

    # -- tags (optional) --
    tags_raw = sec.get("tags")  # type: ignore[arg-type]
    tags: list[str] = []
    if tags_raw is not None:
        if not isinstance(tags_raw, list):
            msg = (
                f"[meta].tags must be a list of strings,"
                f" got {type(tags_raw).__name__}"
            )
            raise TypeError(msg)
        for i, item in enumerate(tags_raw):
            if not isinstance(item, str):
                msg = (
                    f"[meta].tags[{i}] must be a string,"
                    f" got {type(item).__name__}"
                )
                raise TypeError(msg)
            if not item:
                msg = f"[meta].tags[{i}] must be non-empty"
                raise ValueError(msg)
        tags = list(tags_raw)

    # -- notes (optional) --
    notes_raw = sec.get("notes", "")  # type: ignore[arg-type]
    if not isinstance(notes_raw, str):
        msg = (
            f"[meta].notes must be a string,"
            f" got {type(notes_raw).__name__}"
        )
        raise TypeError(msg)

    # -- date (optional, TOML native date or string) --
    date_raw = sec.get("date")  # type: ignore[arg-type]
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
    docs_raw = sec.get("docs")  # type: ignore[arg-type]
    docs: list[MetaDocRef] = []
    if docs_raw is not None:
        if not isinstance(docs_raw, list):
            msg = (
                f"[meta].docs must be an array of tables,"
                f" got {type(docs_raw).__name__}"
            )
            raise TypeError(msg)
        for i, doc in enumerate(docs_raw):
            if not isinstance(doc, dict):
                msg = (
                    f"[[meta.docs]][{i}] must be a table,"
                    f" got {type(doc).__name__}"
                )
                raise TypeError(msg)
            path_val = doc.get("path")
            if path_val is None:
                msg = f"[[meta.docs]][{i}] requires 'path'"
                raise ValueError(msg)
            if not isinstance(path_val, str):
                msg = (
                    f"[[meta.docs]][{i}].path must be a string,"
                    f" got {type(path_val).__name__}"
                )
                raise TypeError(msg)
            label_val = doc.get("label", "")
            if not isinstance(label_val, str):
                msg = (
                    f"[[meta.docs]][{i}].label must be a string,"
                    f" got {type(label_val).__name__}"
                )
                raise TypeError(msg)
            docs.append(MetaDocRef(path=path_val, label=label_val))

    return MetaConfig(
        id=meta_id,
        title=title_raw,
        status=status_raw,
        parents=parents,
        tags=tags,
        notes=notes_raw,
        docs=docs,
        date=date_str,
    )

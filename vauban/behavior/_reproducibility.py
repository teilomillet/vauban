# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Reusable reproducibility metadata helpers for behavior reports."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from vauban._version import __version__

if TYPE_CHECKING:
    from vauban.behavior._primitives import JsonValue


def vauban_version() -> str:
    """Return the Vauban package version used to produce an artifact."""
    return __version__


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest for a file."""
    path_obj = Path(path)
    digest = hashlib.sha256()
    with path_obj.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def maybe_sha256_file(path: str | Path | None) -> str | None:
    """Return a SHA-256 digest when the path exists, otherwise None."""
    if path is None:
        return None
    path_obj = Path(path)
    if not path_obj.exists() or not path_obj.is_file():
        return None
    return sha256_file(path_obj)


def artifact_hashes(
    artifacts: dict[str, str | Path | None],
) -> dict[str, str]:
    """Hash existing artifact paths using stable human-readable labels."""
    hashes: dict[str, str] = {}
    for label, path in artifacts.items():
        digest = maybe_sha256_file(path)
        if digest is not None:
            hashes[label] = digest
    return hashes


def reproducibility_payload(
    *,
    command: str,
    config_path: str | Path | None,
    output_dir: str | Path | None,
    data_refs: tuple[str, ...],
    artifact_hashes_value: dict[str, str],
    scorers: tuple[str, ...],
    generation: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Build a compact JSON-compatible reproducibility payload."""
    payload: dict[str, JsonValue] = {
        "tool_version": vauban_version(),
        "command": command,
        "config_path": str(config_path) if config_path is not None else None,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "data_refs": list(data_refs),
        "artifact_hashes": dict(artifact_hashes_value),
        "scorers": list(scorers),
        "generation": dict(generation),
    }
    return {key: value for key, value in payload.items() if value is not None}

# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""File-level helpers for config validation."""

from __future__ import annotations

import json
import tomllib
from typing import TYPE_CHECKING

from vauban.surface._records import (
    SurfacePromptRecordError,
    parse_surface_prompt_record,
)
from vauban.types import DatasetRef

if TYPE_CHECKING:
    from pathlib import Path

    from vauban.config._types import TomlDict
    from vauban.config._validation_models import ValidationCollector

_SURFACE_FIX_EXAMPLE = (
    'use JSONL lines like {"prompt": "...",'
    ' "label": "harmful", "category": "weapons",'
    ' "messages": [{"role": "user", "content": "..."}],'
    ' "style": "direct", "language": "en",'
    ' "turn_depth": 1, "framing": "explicit"}'
)


def _load_raw_toml(path: Path) -> TomlDict:
    """Load raw TOML mapping for intent-level validation checks."""
    with path.open("rb") as f:
        raw: TomlDict = tomllib.load(f)
    return raw


def _load_refusal_phrases(path: Path) -> list[str]:
    """Load refusal phrases from a text file (one per line)."""
    phrases: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            phrases.append(stripped)
    if not phrases:
        msg = f"Refusal phrases file is empty: {path}"
        raise ValueError(msg)
    return phrases


def _validate_prompt_source(
    source: Path | DatasetRef,
    key: str,
    collector: ValidationCollector,
    *,
    min_recommended: int,
    missing_fix: str,
) -> int | None:
    """Validate a prompt source (local JSONL or HF dataset reference)."""
    if isinstance(source, DatasetRef):
        if source.limit is not None and source.limit < 2:
            collector.add(
                "MEDIUM",
                (
                    f"{key} uses HF dataset limit={source.limit};"
                    " this is likely too small for stable estimation"
                ),
                fix="increase [data.*].limit or remove it",
            )
        return None
    return _validate_prompt_jsonl_file(
        source,
        key,
        collector,
        min_recommended=min_recommended,
        missing_fix=missing_fix,
    )


def _validate_prompt_jsonl_file(
    path: Path,
    key: str,
    collector: ValidationCollector,
    *,
    min_recommended: int,
    missing_fix: str,
) -> int | None:
    """Validate JSONL prompt schema for files using {'prompt': str} lines."""
    if not path.exists():
        collector.add(
            "HIGH",
            f"{key} file not found: {path}",
            fix=missing_fix,
        )
        return None

    valid_count = 0
    seen_non_empty = 0
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            seen_non_empty += 1
            try:
                obj_raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                collector.add(
                    "HIGH",
                    (
                        f"{key} line {line_no} is not valid JSON"
                        f" ({exc.msg}) in {path}"
                    ),
                    fix='use JSONL lines like {"prompt": "your text"}',
                )
                return None
            if not isinstance(obj_raw, dict):
                collector.add(
                    "HIGH",
                    f"{key} line {line_no} must be a JSON object in {path}",
                    fix='use JSONL lines like {"prompt": "your text"}',
                )
                return None
            prompt_raw = obj_raw.get("prompt")
            if not isinstance(prompt_raw, str) or not prompt_raw.strip():
                collector.add(
                    "HIGH",
                    (
                        f"{key} line {line_no} must contain a non-empty"
                        f" string key 'prompt' in {path}"
                    ),
                    fix='ensure each line has {"prompt": "..."}',
                )
                return None
            valid_count += 1

    if seen_non_empty == 0:
        collector.add(
            "HIGH",
            f"{key} is empty: {path}",
            fix='add JSONL records like {"prompt": "..."}',
        )
        return 0

    if valid_count < min_recommended:
        collector.add(
            "LOW",
            (
                f"{key} has only {valid_count} prompt(s);"
                " results may be noisy on small sets"
            ),
            fix=f"use at least {min_recommended} prompts",
        )
    return valid_count


def _validate_surface_jsonl_file(
    path: Path,
    key: str,
    collector: ValidationCollector,
    *,
    missing_fix: str,
) -> int | None:
    """Validate JSONL surface schema for prompt/label/category records."""
    if not path.exists():
        collector.add(
            "HIGH",
            f"{key} file not found: {path}",
            fix=missing_fix,
        )
        return None

    valid_count = 0
    seen_non_empty = 0
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            seen_non_empty += 1
            try:
                obj_raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                collector.add(
                    "HIGH",
                    (
                        f"{key} line {line_no} is not valid JSON"
                        f" ({exc.msg}) in {path}"
                    ),
                    fix=_SURFACE_FIX_EXAMPLE,
                )
                return None
            if not isinstance(obj_raw, dict):
                collector.add(
                    "HIGH",
                    f"{key} line {line_no} must be a JSON object in {path}",
                    fix=_SURFACE_FIX_EXAMPLE,
                )
                return None
            obj: dict[str, object] = {}
            for raw_key, raw_value in obj_raw.items():
                if not isinstance(raw_key, str):
                    collector.add(
                        "HIGH",
                        (
                            "surface prompt keys must be strings on line"
                            f" {line_no} in {path}"
                        ),
                        fix=_SURFACE_FIX_EXAMPLE,
                    )
                    return None
                obj[raw_key] = raw_value
            try:
                parse_surface_prompt_record(obj, line_no, path)
            except SurfacePromptRecordError as exc:
                collector.add(
                    "HIGH",
                    str(exc),
                    fix=_SURFACE_FIX_EXAMPLE,
                )
                return None
            valid_count += 1

    if seen_non_empty == 0:
        collector.add(
            "HIGH",
            f"{key} is empty: {path}",
            fix='add JSONL records with prompt/label/category keys',
        )
        return 0

    return valid_count

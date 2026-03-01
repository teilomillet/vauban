"""Loading helpers for refusal surface prompt datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from vauban.surface._records import parse_surface_prompt_record

if TYPE_CHECKING:
    from vauban.types import SurfacePrompt


def default_surface_path() -> Path:
    """Return path to the bundled categorized surface prompt file."""
    return Path(__file__).resolve().parent.parent / "data" / "surface.jsonl"


def default_multilingual_surface_path() -> Path:
    """Return path to the bundled multilingual surface prompt file."""
    return (
        Path(__file__).resolve().parent.parent
        / "data"
        / "surface_multilingual.jsonl"
    )


def load_surface_prompts(path: str | Path) -> list[SurfacePrompt]:
    """Load surface prompts from a JSONL file."""
    path_obj = Path(path)
    prompts: list[SurfacePrompt] = []
    with path_obj.open() as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            obj_raw = json.loads(stripped)
            if not isinstance(obj_raw, dict):
                msg = (
                    f"surface prompts line {line_no} must be a JSON object"
                    f" in {path_obj}"
                )
                raise ValueError(msg)
            obj: dict[str, object] = {}
            for raw_key, raw_value in obj_raw.items():
                if not isinstance(raw_key, str):
                    msg = (
                        "surface prompt keys must be strings on line"
                        f" {line_no} in {path_obj}"
                    )
                    raise ValueError(msg)
                obj[raw_key] = raw_value
            prompts.append(parse_surface_prompt_record(obj, line_no, path_obj))
    return prompts

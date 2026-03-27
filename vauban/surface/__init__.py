# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Refusal surface mapping APIs."""

from vauban.surface._aggregate import (
    SurfaceGroups,
    aggregate,
    compare_surfaces,
    find_threshold,
    map_surface,
)
from vauban.surface._load import (
    default_full_surface_path,
    default_multilingual_surface_path,
    default_surface_path,
    load_surface_prompts,
)
from vauban.surface._records import (
    DEFAULT_SURFACE_FRAMING,
    DEFAULT_SURFACE_LANGUAGE,
    DEFAULT_SURFACE_STYLE,
    DEFAULT_SURFACE_TURN_DEPTH,
    SurfacePromptRecordError,
    parse_surface_prompt_record,
)
from vauban.surface._scan import scan

__all__ = [
    "DEFAULT_SURFACE_FRAMING",
    "DEFAULT_SURFACE_LANGUAGE",
    "DEFAULT_SURFACE_STYLE",
    "DEFAULT_SURFACE_TURN_DEPTH",
    "SurfaceGroups",
    "SurfacePromptRecordError",
    "aggregate",
    "compare_surfaces",
    "default_full_surface_path",
    "default_multilingual_surface_path",
    "default_surface_path",
    "find_threshold",
    "load_surface_prompts",
    "map_surface",
    "parse_surface_prompt_record",
    "scan",
]

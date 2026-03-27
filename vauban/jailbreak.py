# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Jailbreak prompt template bank for defense evaluation.

Loads bundled (or custom) jailbreak prompt templates and applies them to
payload prompts.  Each template has a ``{payload}`` placeholder that is
substituted with the actual harmful prompt.

Five strategies are bundled:

* **identity_dissolution** — override the model's identity and constraints.
* **boundary_exploit** — inject fake system/instruction boundaries.
* **semantic_inversion** — request refusal then its semantic opposite.
* **dual_response** — request both safe and unsafe responses.
* **competitive_pressure** — pressure via capability comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path

from vauban.data import dataset_path
from vauban.types import JailbreakTemplate

ALL_STRATEGIES: frozenset[str] = frozenset({
    "identity_dissolution",
    "boundary_exploit",
    "semantic_inversion",
    "dual_response",
    "competitive_pressure",
})


def default_templates_path() -> Path:
    """Return the path to the bundled jailbreak templates JSONL."""
    return dataset_path("jailbreak_templates")


def load_templates(path: str | Path | None = None) -> list[JailbreakTemplate]:
    """Load jailbreak templates from a JSONL file.

    Args:
        path: Path to a JSONL file.  Each line must have
            ``strategy``, ``name``, and ``template`` keys.
            If None, loads the bundled default templates.

    Returns:
        List of :class:`JailbreakTemplate` instances.
    """
    if path is None:
        path = default_templates_path()
    path = Path(path)

    templates: list[JailbreakTemplate] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            templates.append(JailbreakTemplate(
                strategy=obj["strategy"],
                name=obj["name"],
                template=obj["template"],
            ))
    return templates


def filter_by_strategy(
    templates: list[JailbreakTemplate],
    strategies: list[str],
) -> list[JailbreakTemplate]:
    """Filter templates to only include the given strategies.

    Args:
        templates: Full template list.
        strategies: Strategy names to keep.  If empty, returns all.

    Returns:
        Filtered list.
    """
    if not strategies:
        return templates
    allowed = set(strategies)
    return [t for t in templates if t.strategy in allowed]


def apply_templates(
    templates: list[JailbreakTemplate],
    payloads: list[str],
) -> list[tuple[JailbreakTemplate, str]]:
    """Substitute payloads into templates (cross-product).

    Args:
        templates: Jailbreak templates with ``{payload}`` placeholders.
        payloads: Harmful prompts to inject.

    Returns:
        List of ``(template, expanded_prompt)`` tuples.
    """
    results: list[tuple[JailbreakTemplate, str]] = []
    for template in templates:
        for payload in payloads:
            expanded = template.template.replace("{payload}", payload)
            results.append((template, expanded))
    return results


__all__ = [
    "ALL_STRATEGIES",
    "apply_templates",
    "default_templates_path",
    "filter_by_strategy",
    "load_templates",
]

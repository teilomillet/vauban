# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Jailbreak template tests — consolidated via ordeal.

Coverage target: 97% (same as before).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from vauban.jailbreak import (
    ALL_STRATEGIES,
    apply_templates,
    default_templates_path,
    filter_by_strategy,
    load_templates,
)
from vauban.types import JailbreakConfig, JailbreakTemplate


class TestLoadTemplates:
    def test_loads_defaults(self) -> None:
        templates = load_templates()
        assert len(templates) == 30

    def test_path_exists(self) -> None:
        assert default_templates_path().exists()

    def test_all_have_payload(self) -> None:
        for t in load_templates():
            assert "{payload}" in t.template

    def test_all_have_valid_strategy(self) -> None:
        for t in load_templates():
            assert t.strategy in ALL_STRATEGIES

    def test_strategy_coverage(self) -> None:
        assert {t.strategy for t in load_templates()} == ALL_STRATEGIES

    def test_custom_path(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom.jsonl"
        custom.write_text(json.dumps({
            "strategy": "identity_dissolution",
            "name": "test", "template": "test {payload}",
        }))
        assert len(load_templates(custom)) == 1


class TestFilter:
    def test_single(self) -> None:
        f = filter_by_strategy(load_templates(), ["identity_dissolution"])
        assert all(t.strategy == "identity_dissolution" for t in f)

    def test_empty_returns_all(self) -> None:
        assert len(filter_by_strategy(load_templates(), [])) == 30

    def test_unknown_returns_empty(self) -> None:
        assert filter_by_strategy(load_templates(), ["nonexistent"]) == []


class TestApply:
    def test_cross_product(self) -> None:
        ts = [JailbreakTemplate("s1", "n1", "A: {payload}"),
              JailbreakTemplate("s2", "n2", "B: {payload}")]
        assert len(apply_templates(ts, ["p1", "p2"])) == 4

    def test_substitution(self) -> None:
        ts = [JailbreakTemplate("s", "n", "pre {payload} post")]
        assert apply_templates(ts, ["X"])[0][1] == "pre X post"

    def test_empty(self) -> None:
        assert apply_templates([], ["p"]) == []
        ts = [JailbreakTemplate("s", "n", "{payload}")]
        assert apply_templates(ts, []) == []


class TestTypes:
    def test_template_frozen(self) -> None:
        with pytest.raises(AttributeError):
            JailbreakTemplate("s", "n", "t").name = "x"  # type: ignore[misc]

    def test_config_defaults(self) -> None:
        c = JailbreakConfig()
        assert c.strategies == [] and c.payloads_from == "harmful"

    def test_strategies_constant(self) -> None:
        assert isinstance(ALL_STRATEGIES, frozenset)
        assert len(ALL_STRATEGIES) == 5

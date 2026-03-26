"""Tests for jailbreak prompt template bank."""

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
    """Loading templates from JSONL."""

    def test_loads_default_templates(self) -> None:
        templates = load_templates()
        assert len(templates) > 0

    def test_default_path_exists(self) -> None:
        path = default_templates_path()
        assert path.exists()

    def test_all_templates_have_payload_placeholder(self) -> None:
        templates = load_templates()
        for t in templates:
            assert "{payload}" in t.template, (
                f"template {t.name} missing {{payload}} placeholder"
            )

    def test_all_templates_have_strategy(self) -> None:
        templates = load_templates()
        for t in templates:
            assert t.strategy in ALL_STRATEGIES, (
                f"template {t.name} has unknown strategy {t.strategy!r}"
            )

    def test_all_templates_have_name(self) -> None:
        templates = load_templates()
        for t in templates:
            assert t.name, "template has empty name"

    def test_template_count(self) -> None:
        templates = load_templates()
        assert len(templates) == 30

    def test_strategy_coverage(self) -> None:
        templates = load_templates()
        strategies = {t.strategy for t in templates}
        assert strategies == ALL_STRATEGIES

    def test_custom_path(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom.jsonl"
        custom.write_text(
            json.dumps({
                "strategy": "identity_dissolution",
                "name": "test",
                "template": "test {payload}",
            })
        )
        templates = load_templates(custom)
        assert len(templates) == 1
        assert templates[0].name == "test"


class TestFilterByStrategy:
    """Filtering templates by strategy."""

    def test_filter_single_strategy(self) -> None:
        templates = load_templates()
        filtered = filter_by_strategy(templates, ["identity_dissolution"])
        assert all(t.strategy == "identity_dissolution" for t in filtered)
        assert len(filtered) > 0

    def test_filter_multiple_strategies(self) -> None:
        templates = load_templates()
        filtered = filter_by_strategy(
            templates, ["identity_dissolution", "dual_response"],
        )
        strategies = {t.strategy for t in filtered}
        assert strategies == {"identity_dissolution", "dual_response"}

    def test_empty_filter_returns_all(self) -> None:
        templates = load_templates()
        filtered = filter_by_strategy(templates, [])
        assert len(filtered) == len(templates)

    def test_unknown_strategy_returns_empty(self) -> None:
        templates = load_templates()
        filtered = filter_by_strategy(templates, ["nonexistent"])
        assert len(filtered) == 0


class TestApplyTemplates:
    """Cross-product of templates x payloads."""

    def test_cross_product_size(self) -> None:
        templates = [
            JailbreakTemplate("s1", "n1", "A: {payload}"),
            JailbreakTemplate("s2", "n2", "B: {payload}"),
        ]
        payloads = ["p1", "p2", "p3"]
        result = apply_templates(templates, payloads)
        assert len(result) == 6  # 2 templates x 3 payloads

    def test_payload_substituted(self) -> None:
        templates = [JailbreakTemplate("s", "n", "Prefix {payload} suffix")]
        payloads = ["test_content"]
        result = apply_templates(templates, payloads)
        assert result[0][1] == "Prefix test_content suffix"

    def test_template_preserved_in_tuple(self) -> None:
        t = JailbreakTemplate("s", "n", "{payload}")
        result = apply_templates([t], ["x"])
        assert result[0][0] is t

    def test_empty_templates(self) -> None:
        result = apply_templates([], ["payload"])
        assert result == []

    def test_empty_payloads(self) -> None:
        templates = [JailbreakTemplate("s", "n", "{payload}")]
        result = apply_templates(templates, [])
        assert result == []


class TestJailbreakTypes:
    """Type correctness of config and template."""

    def test_template_is_frozen(self) -> None:
        t = JailbreakTemplate("s", "n", "t")
        with pytest.raises(AttributeError):
            t.name = "other"  # type: ignore[misc]

    def test_config_defaults(self) -> None:
        c = JailbreakConfig()
        assert c.strategies == []
        assert c.custom_templates_path is None
        assert c.payloads_from == "harmful"

    def test_config_custom(self) -> None:
        c = JailbreakConfig(
            strategies=["identity_dissolution"],
            payloads_from="custom.jsonl",
        )
        assert c.strategies == ["identity_dissolution"]
        assert c.payloads_from == "custom.jsonl"


class TestAllStrategies:
    """ALL_STRATEGIES constant."""

    def test_is_frozenset(self) -> None:
        assert isinstance(ALL_STRATEGIES, frozenset)

    def test_contains_five_strategies(self) -> None:
        assert len(ALL_STRATEGIES) == 5

    def test_known_strategies(self) -> None:
        expected = {
            "identity_dissolution",
            "boundary_exploit",
            "semantic_inversion",
            "dual_response",
            "competitive_pressure",
        }
        assert expected == ALL_STRATEGIES

# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for [cut] and [policy] config parsers."""

import pytest

from vauban.config._parse_cut import _parse_cut
from vauban.config._parse_policy import _parse_policy

# ── [cut] parser ─────────────────────────────────────────────────────


class TestParseCut:
    """Tests for _parse_cut parser."""

    def test_defaults(self) -> None:
        cfg = _parse_cut({})
        assert cfg.alpha == 1.0
        assert cfg.layers is None
        assert cfg.norm_preserve is False
        assert cfg.biprojected is False
        assert cfg.layer_strategy == "all"
        assert cfg.layer_top_k == 10
        assert cfg.layer_weights is None
        assert cfg.sparsity == 0.0
        assert cfg.dbdi_target == "red"
        assert cfg.false_refusal_ortho is False
        assert cfg.layer_type_filter is None

    def test_layers_auto(self) -> None:
        cfg = _parse_cut({"layers": "auto"})
        assert cfg.layers is None

    def test_layers_list(self) -> None:
        cfg = _parse_cut({"layers": [1, 5, 10]})
        assert cfg.layers == [1, 5, 10]

    def test_layers_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="layers"):
            _parse_cut({"layers": 42})

    def test_alpha_numeric(self) -> None:
        cfg = _parse_cut({"alpha": 2.5})
        assert cfg.alpha == 2.5

    def test_alpha_int(self) -> None:
        cfg = _parse_cut({"alpha": 3})
        assert cfg.alpha == 3.0

    def test_alpha_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="alpha"):
            _parse_cut({"alpha": "bad"})

    def test_layer_strategy_valid_values(self) -> None:
        for strategy in ("all", "above_median", "top_k"):
            cfg = _parse_cut({"layer_strategy": strategy})
            assert cfg.layer_strategy == strategy

    def test_layer_strategy_invalid(self) -> None:
        with pytest.raises(ValueError, match="layer_strategy"):
            _parse_cut({"layer_strategy": "invalid"})

    def test_layer_strategy_type_error(self) -> None:
        with pytest.raises(TypeError, match="layer_strategy"):
            _parse_cut({"layer_strategy": 42})

    def test_layer_top_k(self) -> None:
        cfg = _parse_cut({"layer_top_k": 5})
        assert cfg.layer_top_k == 5

    def test_layer_top_k_type_error(self) -> None:
        with pytest.raises(TypeError, match="layer_top_k"):
            _parse_cut({"layer_top_k": "bad"})

    def test_layer_weights(self) -> None:
        cfg = _parse_cut({"layer_weights": [0.5, 1.0, 1.5]})
        assert cfg.layer_weights == [0.5, 1.0, 1.5]

    def test_layer_weights_type_error(self) -> None:
        with pytest.raises(TypeError, match="layer_weights"):
            _parse_cut({"layer_weights": "bad"})

    def test_sparsity_valid(self) -> None:
        cfg = _parse_cut({"sparsity": 0.5})
        assert cfg.sparsity == 0.5

    def test_sparsity_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="sparsity"):
            _parse_cut({"sparsity": 1.0})

    def test_sparsity_negative(self) -> None:
        with pytest.raises(ValueError, match="sparsity"):
            _parse_cut({"sparsity": -0.1})

    def test_sparsity_type_error(self) -> None:
        with pytest.raises(TypeError, match="sparsity"):
            _parse_cut({"sparsity": "bad"})

    def test_dbdi_target_valid(self) -> None:
        for target in ("red", "hdd", "both"):
            cfg = _parse_cut({"dbdi_target": target})
            assert cfg.dbdi_target == target

    def test_dbdi_target_invalid(self) -> None:
        with pytest.raises(ValueError, match="dbdi_target"):
            _parse_cut({"dbdi_target": "invalid"})

    def test_dbdi_target_type_error(self) -> None:
        with pytest.raises(TypeError, match="dbdi_target"):
            _parse_cut({"dbdi_target": 42})

    def test_false_refusal_ortho(self) -> None:
        cfg = _parse_cut({"false_refusal_ortho": True})
        assert cfg.false_refusal_ortho is True

    def test_false_refusal_ortho_type_error(self) -> None:
        with pytest.raises(TypeError, match="false_refusal_ortho"):
            _parse_cut({"false_refusal_ortho": "yes"})

    def test_norm_preserve(self) -> None:
        cfg = _parse_cut({"norm_preserve": True})
        assert cfg.norm_preserve is True

    def test_biprojected(self) -> None:
        cfg = _parse_cut({"biprojected": True})
        assert cfg.biprojected is True

    def test_layer_type_filter_valid(self) -> None:
        for f in ("global", "sliding"):
            cfg = _parse_cut({"layer_type_filter": f})
            assert cfg.layer_type_filter == f

    def test_layer_type_filter_invalid(self) -> None:
        with pytest.raises(ValueError, match="layer_type_filter"):
            _parse_cut({"layer_type_filter": "invalid"})

    def test_layer_type_filter_type_error(self) -> None:
        with pytest.raises(TypeError, match="layer_type_filter"):
            _parse_cut({"layer_type_filter": 42})

    def test_full_config(self) -> None:
        cfg = _parse_cut({
            "alpha": 0.7,
            "layers": [0, 5],
            "norm_preserve": True,
            "biprojected": True,
            "layer_strategy": "top_k",
            "layer_top_k": 3,
            "layer_weights": [1.0, 0.5],
            "sparsity": 0.3,
            "dbdi_target": "both",
            "false_refusal_ortho": True,
            "layer_type_filter": "global",
        })
        assert cfg.alpha == 0.7
        assert cfg.layers == [0, 5]
        assert cfg.norm_preserve is True
        assert cfg.biprojected is True
        assert cfg.layer_strategy == "top_k"
        assert cfg.layer_top_k == 3
        assert cfg.layer_weights == [1.0, 0.5]
        assert cfg.sparsity == 0.3
        assert cfg.dbdi_target == "both"
        assert cfg.false_refusal_ortho is True
        assert cfg.layer_type_filter == "global"


# ── [policy] parser ──────────────────────────────────────────────────


class TestParsePolicy:
    """Tests for _parse_policy parser."""

    def test_absent_returns_none(self) -> None:
        assert _parse_policy({}) is None

    def test_section_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_policy({"policy": "bad"})

    def test_minimal_valid(self) -> None:
        cfg = _parse_policy({"policy": {}})
        assert cfg is not None
        assert cfg.default_action == "allow"
        assert cfg.rules == []
        assert cfg.data_flow_rules == []
        assert cfg.rate_limits == []

    def test_default_action_block(self) -> None:
        cfg = _parse_policy({"policy": {"default_action": "block"}})
        assert cfg is not None
        assert cfg.default_action == "block"

    def test_default_action_invalid(self) -> None:
        with pytest.raises(ValueError, match="default_action"):
            _parse_policy({"policy": {"default_action": "maybe"}})

    def test_default_action_type_error(self) -> None:
        with pytest.raises(TypeError, match="default_action"):
            _parse_policy({"policy": {"default_action": 42}})

    def test_rules_not_list(self) -> None:
        with pytest.raises(TypeError, match="rules must be a list"):
            _parse_policy({"policy": {"rules": "bad"}})

    def test_rule_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_policy({"policy": {"rules": ["bad"]}})

    def test_rule_missing_name(self) -> None:
        with pytest.raises(TypeError, match="name"):
            _parse_policy({"policy": {"rules": [
                {"action": "block", "tool_pattern": "*"},
            ]}})

    def test_rule_missing_action(self) -> None:
        with pytest.raises(TypeError, match="action"):
            _parse_policy({"policy": {"rules": [
                {"name": "r", "tool_pattern": "*"},
            ]}})

    def test_rule_invalid_action(self) -> None:
        with pytest.raises(ValueError, match="action"):
            _parse_policy({"policy": {"rules": [
                {"name": "r", "action": "maybe", "tool_pattern": "*"},
            ]}})

    def test_rule_valid_actions(self) -> None:
        for action in ("allow", "block", "confirm"):
            cfg = _parse_policy({"policy": {"rules": [
                {"name": "r", "action": action, "tool_pattern": "*"},
            ]}})
            assert cfg is not None
            assert cfg.rules[0].action == action

    def test_rule_missing_tool_pattern(self) -> None:
        with pytest.raises(TypeError, match="tool_pattern"):
            _parse_policy({"policy": {"rules": [
                {"name": "r", "action": "block"},
            ]}})

    def test_rule_with_argument_key_and_pattern(self) -> None:
        cfg = _parse_policy({"policy": {"rules": [
            {
                "name": "r",
                "action": "block",
                "tool_pattern": "*",
                "argument_key": "to",
                "argument_pattern": r"@evil\.com",
            },
        ]}})
        assert cfg is not None
        assert cfg.rules[0].argument_key == "to"
        assert cfg.rules[0].argument_pattern == r"@evil\.com"

    def test_rule_argument_key_type_error(self) -> None:
        with pytest.raises(TypeError, match="argument_key"):
            _parse_policy({"policy": {"rules": [
                {"name": "r", "action": "block", "tool_pattern": "*",
                 "argument_key": 42},
            ]}})

    def test_data_flow_rules(self) -> None:
        cfg = _parse_policy({"policy": {"data_flow_rules": [
            {
                "source_tool": "calendar",
                "source_labels": ["pii"],
                "blocked_targets": ["email"],
            },
        ]}})
        assert cfg is not None
        assert len(cfg.data_flow_rules) == 1
        assert cfg.data_flow_rules[0].source_tool == "calendar"
        assert cfg.data_flow_rules[0].source_labels == ["pii"]
        assert cfg.data_flow_rules[0].blocked_targets == ["email"]

    def test_data_flow_rules_not_list(self) -> None:
        with pytest.raises(TypeError, match="data_flow_rules must be a list"):
            _parse_policy({"policy": {"data_flow_rules": "bad"}})

    def test_data_flow_rule_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_policy({"policy": {"data_flow_rules": ["bad"]}})

    def test_data_flow_rule_missing_source(self) -> None:
        with pytest.raises(TypeError, match="source_tool"):
            _parse_policy({"policy": {"data_flow_rules": [{}]}})

    def test_rate_limits(self) -> None:
        cfg = _parse_policy({"policy": {"rate_limits": [
            {"tool_pattern": "email*", "max_calls": 5, "window_seconds": 30},
        ]}})
        assert cfg is not None
        assert len(cfg.rate_limits) == 1
        assert cfg.rate_limits[0].tool_pattern == "email*"
        assert cfg.rate_limits[0].max_calls == 5
        assert cfg.rate_limits[0].window_seconds == 30.0

    def test_rate_limits_not_list(self) -> None:
        with pytest.raises(TypeError, match="rate_limits must be a list"):
            _parse_policy({"policy": {"rate_limits": "bad"}})

    def test_rate_limit_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_policy({"policy": {"rate_limits": ["bad"]}})

    def test_rate_limit_missing_tool_pattern(self) -> None:
        with pytest.raises(TypeError, match="tool_pattern"):
            _parse_policy({"policy": {"rate_limits": [
                {"max_calls": 5},
            ]}})

    def test_rate_limit_missing_max_calls(self) -> None:
        with pytest.raises(TypeError, match="max_calls"):
            _parse_policy({"policy": {"rate_limits": [
                {"tool_pattern": "*"},
            ]}})

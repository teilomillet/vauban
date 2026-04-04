# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tool-call policy engine."""

import vauban.policy as policy
from vauban.policy import _check_rate_limits, evaluate_data_flow, evaluate_tool_call
from vauban.types import (
    DataFlowRule,
    PolicyConfig,
    PolicyRule,
    RateLimitRule,
)


class TestEvaluateToolCall:
    """Tests for tool call evaluation."""

    def test_allow_by_default(self) -> None:
        config = PolicyConfig()
        decision = evaluate_tool_call("any_tool", {}, config)
        assert decision.action == "allow"

    def test_block_default(self) -> None:
        config = PolicyConfig(default_action="block")
        decision = evaluate_tool_call("any_tool", {}, config)
        assert decision.action == "block"

    def test_block_by_rule(self) -> None:
        config = PolicyConfig(
            rules=[
                PolicyRule(
                    name="block_email",
                    action="block",
                    tool_pattern="send_email",
                ),
            ],
        )
        decision = evaluate_tool_call("send_email", {}, config)
        assert decision.action == "block"
        assert "block_email" in decision.matched_rules

    def test_allow_by_rule(self) -> None:
        config = PolicyConfig(
            default_action="block",
            rules=[
                PolicyRule(
                    name="allow_calendar",
                    action="allow",
                    tool_pattern="calendar*",
                ),
            ],
        )
        decision = evaluate_tool_call("calendar_lookup", {}, config)
        assert decision.action == "allow"

    def test_wildcard_pattern(self) -> None:
        config = PolicyConfig(
            rules=[
                PolicyRule(
                    name="block_all_email",
                    action="block",
                    tool_pattern="*email*",
                ),
            ],
        )
        assert evaluate_tool_call("send_email", {}, config).action == "block"
        assert evaluate_tool_call(
            "email_forward", {}, config,
        ).action == "block"
        assert evaluate_tool_call("calendar", {}, config).action == "allow"

    def test_argument_pattern(self) -> None:
        config = PolicyConfig(
            rules=[
                PolicyRule(
                    name="block_external",
                    action="block",
                    tool_pattern="send_email",
                    argument_key="to",
                    argument_pattern=r"@external\.com$",
                ),
            ],
        )
        # Should block
        d1 = evaluate_tool_call(
            "send_email", {"to": "user@external.com"}, config,
        )
        assert d1.action == "block"

        # Should allow (different domain)
        d2 = evaluate_tool_call(
            "send_email", {"to": "user@internal.com"}, config,
        )
        assert d2.action == "allow"

    def test_confirm_rule(self) -> None:
        config = PolicyConfig(
            rules=[
                PolicyRule(
                    name="confirm_delete",
                    action="confirm",
                    tool_pattern="delete_*",
                ),
            ],
        )
        decision = evaluate_tool_call("delete_record", {}, config)
        assert decision.action == "confirm"


class TestEvaluateDataFlow:
    """Tests for data flow evaluation."""

    def test_blocked_flow(self) -> None:
        config = PolicyConfig(
            data_flow_rules=[
                DataFlowRule(
                    source_tool="calendar",
                    source_labels=["pii", "financial"],
                    blocked_targets=["send_email"],
                ),
            ],
        )
        decision = evaluate_data_flow(
            "calendar", "send_email", ["pii"], config,
        )
        assert decision.action == "block"

    def test_allowed_flow(self) -> None:
        config = PolicyConfig(
            data_flow_rules=[
                DataFlowRule(
                    source_tool="calendar",
                    source_labels=["pii"],
                    blocked_targets=["send_email"],
                ),
            ],
        )
        # Different label
        decision = evaluate_data_flow(
            "calendar", "send_email", ["public"], config,
        )
        assert decision.action == "allow"

    def test_allowed_target(self) -> None:
        config = PolicyConfig(
            data_flow_rules=[
                DataFlowRule(
                    source_tool="calendar",
                    source_labels=["pii"],
                    blocked_targets=["send_email"],
                ),
            ],
        )
        # Different target
        decision = evaluate_data_flow(
            "calendar", "display", ["pii"], config,
        )
        assert decision.action == "allow"


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_exceeded(self) -> None:
        config = PolicyConfig(
            rate_limits=[
                RateLimitRule(
                    tool_pattern="send_*",
                    max_calls=2,
                    window_seconds=60.0,
                ),
            ],
        )
        state: dict[str, object] = {}
        # First two calls should pass
        d1 = evaluate_tool_call("send_email", {}, config, state)
        assert d1.action == "allow"
        d2 = evaluate_tool_call("send_email", {}, config, state)
        assert d2.action == "allow"
        # Third call should be blocked
        d3 = evaluate_tool_call("send_email", {}, config, state)
        assert d3.action == "block"
        assert "rate limit" in d3.reasons[0].lower()

    def test_check_rate_limits_reuses_existing_history(self) -> None:
        config_rule = RateLimitRule(
            tool_pattern="send_*",
            max_calls=1,
            window_seconds=60.0,
        )
        state: dict[str, object] = {
            "_rate_limit_history": {"send_*": [100.0]},
        }

        original = policy.time.monotonic
        policy.time.monotonic = lambda: 100.0
        try:
            decision = _check_rate_limits("send_email", [config_rule], state)
        finally:
            policy.time.monotonic = original

        assert decision is not None
        assert decision.action == "block"
        assert decision.matched_rules == ["rate_limit:send_*"]

    def test_check_rate_limits_replaces_invalid_history_container(self) -> None:
        config_rule = RateLimitRule(
            tool_pattern="send_*",
            max_calls=2,
            window_seconds=60.0,
        )
        state: dict[str, object] = {"_rate_limit_history": []}

        original = policy.time.monotonic
        policy.time.monotonic = lambda: 100.0
        try:
            decision = _check_rate_limits("send_email", [config_rule], state)
        finally:
            policy.time.monotonic = original

        assert decision is None
        history = state["_rate_limit_history"]
        assert isinstance(history, dict)
        assert history["send_*"] == [100.0]

    def test_check_rate_limits_replaces_invalid_call_history(self) -> None:
        config_rule = RateLimitRule(
            tool_pattern="send_*",
            max_calls=2,
            window_seconds=60.0,
        )
        state: dict[str, object] = {
            "_rate_limit_history": {"send_*": "not-a-list"},
        }

        original = policy.time.monotonic
        policy.time.monotonic = lambda: 100.0
        try:
            decision = _check_rate_limits("send_email", [config_rule], state)
        finally:
            policy.time.monotonic = original

        assert decision is None
        history = state["_rate_limit_history"]
        assert isinstance(history, dict)
        assert history["send_*"] == [100.0]


class TestPolicyConfigParsing:
    """Tests for [policy] config parsing."""

    def test_parse_valid(self) -> None:
        from vauban.config._parse_policy import _parse_policy

        raw = {
            "policy": {
                "default_action": "block",
                "rules": [
                    {
                        "name": "allow_cal",
                        "action": "allow",
                        "tool_pattern": "calendar*",
                    },
                ],
                "data_flow_rules": [
                    {
                        "source_tool": "cal",
                        "source_labels": ["pii"],
                        "blocked_targets": ["email"],
                    },
                ],
                "rate_limits": [
                    {
                        "tool_pattern": "email*",
                        "max_calls": 5,
                        "window_seconds": 60,
                    },
                ],
            },
        }
        config = _parse_policy(raw)
        assert config is not None
        assert config.default_action == "block"
        assert len(config.rules) == 1
        assert len(config.data_flow_rules) == 1
        assert len(config.rate_limits) == 1

    def test_missing_returns_none(self) -> None:
        from vauban.config._parse_policy import _parse_policy

        assert _parse_policy({}) is None

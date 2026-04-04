# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Additional ordeal scans for pure quality-critical modules."""

from __future__ import annotations

import hypothesis.strategies as st
from ordeal.auto import scan_module

from vauban.types import DataFlowRule, PolicyConfig, PolicyRule, RateLimitRule

_POLICY_RULES: st.SearchStrategy[PolicyRule] = st.builds(
    PolicyRule,
    name=st.text(min_size=1, max_size=12),
    action=st.sampled_from(["allow", "block", "confirm"]),
    tool_pattern=st.sampled_from(["*", "gmail_*", "web_*", "shell"]),
    argument_key=st.one_of(st.none(), st.sampled_from(["to", "query", "path"])),
    argument_pattern=st.one_of(
        st.none(),
        st.sampled_from([r".*", r"secret", r".*@.*"]),
    ),
)

_DATA_FLOW_RULES: st.SearchStrategy[DataFlowRule] = st.builds(
    DataFlowRule,
    source_tool=st.sampled_from(["web_fetch", "gmail_read_message", "shell"]),
    source_labels=st.lists(
        st.sampled_from(["pii", "secrets", "public", "safe"]),
        min_size=1,
        max_size=3,
    ),
    blocked_targets=st.lists(
        st.sampled_from(["gmail_send_message", "web_post", "shell"]),
        min_size=1,
        max_size=3,
    ),
)

_RATE_LIMIT_RULES: st.SearchStrategy[RateLimitRule] = st.builds(
    RateLimitRule,
    tool_pattern=st.sampled_from(["*", "web_*", "gmail_*"]),
    max_calls=st.integers(min_value=1, max_value=5),
    window_seconds=st.floats(
        min_value=0.1,
        max_value=60.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)

_POLICY_FIXTURES: dict[str, st.SearchStrategy[object]] = {
    "tool_name": st.sampled_from([
        "gmail_send_message",
        "web_fetch",
        "search",
        "shell",
    ]),
    "arguments": st.dictionaries(
        st.sampled_from(["to", "query", "path", "body"]),
        st.text(max_size=24),
        max_size=4,
    ),
    "config": st.builds(
        PolicyConfig,
        rules=st.lists(_POLICY_RULES, max_size=4),
        data_flow_rules=st.lists(_DATA_FLOW_RULES, max_size=4),
        rate_limits=st.lists(_RATE_LIMIT_RULES, max_size=3),
        default_action=st.sampled_from(["allow", "block"]),
    ),
    "state": st.one_of(st.none(), st.builds(dict)),
    "source_tool": st.sampled_from(["web_fetch", "gmail_read_message", "shell"]),
    "target_tool": st.sampled_from(["gmail_send_message", "web_post", "shell"]),
    "data_labels": st.lists(
        st.sampled_from(["pii", "secrets", "public", "safe"]),
        max_size=4,
    ),
}


class TestQualityScans:
    """Use ordeal to harden pure modules against edge-case regressions."""

    def test_scan_suggestions(self) -> None:
        result = scan_module("vauban._suggestions", max_examples=30)
        assert result.total > 0
        for function in result.functions:
            assert function.passed, f"_suggestions.{function.name}: {function.error}"

    def test_scan_environment_format(self) -> None:
        result = scan_module("vauban.environment._format", max_examples=30)
        assert result.total > 0
        for function in result.functions:
            assert function.passed, (
                f"environment._format.{function.name}: {function.error}"
            )

    def test_scan_environment_policy(self) -> None:
        result = scan_module("vauban.environment._policy", max_examples=30)
        assert result.total > 0
        for function in result.functions:
            assert function.passed, (
                f"environment._policy.{function.name}: {function.error}"
            )

    def test_scan_policy(self) -> None:
        result = scan_module(
            "vauban.policy",
            max_examples=30,
            fixtures=_POLICY_FIXTURES,
        )
        assert result.total > 0
        for function in result.functions:
            assert function.passed, f"policy.{function.name}: {function.error}"

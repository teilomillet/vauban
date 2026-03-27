# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tool-call policy engine — pure Python rule evaluation.

Evaluates tool calls against declarative policy rules, data flow
restrictions, and rate limits. No ML dependencies.

Usage:
    decision = evaluate_tool_call("gmail_send_message", {"to": "..."}, config)
    if decision.action == "block":
        ...
"""

import fnmatch
import re
import sys
import time
from typing import Literal

if sys.version_info >= (3, 13):
    from typing import TypeIs
else:
    from typing_extensions import TypeIs

from vauban.types import (
    DataFlowRule,
    PolicyConfig,
    PolicyDecision,
    PolicyRule,
    RateLimitRule,
)

_VALID_ACTIONS: dict[str, Literal["allow", "block", "confirm"]] = {
    "allow": "allow",
    "block": "block",
    "confirm": "confirm",
}


def _is_str_dict(obj: object) -> TypeIs[dict[str, object]]:
    """Type-narrow an object to dict[str, object]."""
    return isinstance(obj, dict)


def evaluate_tool_call(
    tool_name: str,
    arguments: dict[str, str],
    config: PolicyConfig,
    state: dict[str, object] | None = None,
) -> PolicyDecision:
    """Evaluate a tool call against the policy rules.

    Checks rules in order. The first matching rule determines the action.
    If no rules match, the default action is used.

    Rate limits are checked using the ``state`` dict, which tracks
    call history across invocations. Callers should pass a persistent
    dict to enable rate limiting.

    Args:
        tool_name: Name of the tool being called.
        arguments: Tool call arguments.
        config: Policy configuration with rules and limits.
        state: Mutable state dict for rate limiting (optional).

    Returns:
        PolicyDecision with action and matched rules.
    """
    matched_rules: list[str] = []
    reasons: list[str] = []

    # Check policy rules
    for rule in config.rules:
        if _rule_matches(rule, tool_name, arguments):
            matched_rules.append(rule.name)
            reasons.append(
                f"Rule {rule.name!r}: {rule.action} for"
                f" {rule.tool_pattern}",
            )
            if rule.action == "block":
                return PolicyDecision(
                    action="block",
                    matched_rules=matched_rules,
                    reasons=reasons,
                )
            if rule.action == "confirm":
                return PolicyDecision(
                    action="confirm",
                    matched_rules=matched_rules,
                    reasons=reasons,
                )

    # Check rate limits
    if state is not None:
        rl_decision = _check_rate_limits(
            tool_name, config.rate_limits, state,
        )
        if rl_decision is not None:
            return rl_decision

    # Default action
    if matched_rules:
        return PolicyDecision(
            action="allow",
            matched_rules=matched_rules,
            reasons=reasons,
        )

    default_action = _VALID_ACTIONS.get(config.default_action, "allow")
    return PolicyDecision(
        action=default_action,
        matched_rules=[],
        reasons=[f"No rules matched; default action: {config.default_action}"],
    )


def evaluate_data_flow(
    source_tool: str,
    target_tool: str,
    data_labels: list[str],
    config: PolicyConfig,
) -> PolicyDecision:
    """Evaluate a data flow between tools against flow rules.

    Checks if data from ``source_tool`` (with given labels) is allowed
    to flow to ``target_tool``.

    Args:
        source_tool: Name of the tool producing data.
        target_tool: Name of the tool consuming data.
        data_labels: Labels describing the data content.
        config: Policy configuration with data flow rules.

    Returns:
        PolicyDecision with action and matched rules.
    """
    matched_rules: list[str] = []
    reasons: list[str] = []

    for rule in config.data_flow_rules:
        if _data_flow_matches(rule, source_tool, target_tool, data_labels):
            matched_rules.append(
                f"flow:{rule.source_tool}->{','.join(rule.blocked_targets)}",
            )
            reasons.append(
                f"Data flow from {source_tool!r} to {target_tool!r}"
                f" blocked: labels {data_labels} match"
                f" {rule.source_labels}",
            )
            return PolicyDecision(
                action="block",
                matched_rules=matched_rules,
                reasons=reasons,
            )

    return PolicyDecision(
        action="allow",
        matched_rules=[],
        reasons=["No data flow rules matched"],
    )


def _rule_matches(
    rule: PolicyRule,
    tool_name: str,
    arguments: dict[str, str],
) -> bool:
    """Check if a policy rule matches a tool call."""
    if not fnmatch.fnmatch(tool_name, rule.tool_pattern):
        return False

    # If argument constraints are specified, check them
    if rule.argument_key is not None:
        arg_value = arguments.get(rule.argument_key)
        if arg_value is None:
            return False
        if (
            rule.argument_pattern is not None
            and not re.search(rule.argument_pattern, arg_value)
        ):
            return False

    return True


def _data_flow_matches(
    rule: DataFlowRule,
    source_tool: str,
    target_tool: str,
    data_labels: list[str],
) -> bool:
    """Check if a data flow rule matches the given flow."""
    if rule.source_tool != source_tool:
        return False

    if target_tool not in rule.blocked_targets:
        return False

    # Check if any of the data labels match the source labels
    return bool(set(data_labels) & set(rule.source_labels))


def _check_rate_limits(
    tool_name: str,
    rate_limits: list[RateLimitRule],
    state: dict[str, object],
) -> PolicyDecision | None:
    """Check rate limits and return a block decision if exceeded."""
    now = time.monotonic()
    history_key = "_rate_limit_history"
    history_raw = state.get(history_key)
    if _is_str_dict(history_raw):
        history = history_raw
    else:
        history: dict[str, object] = {}
        state[history_key] = history

    for rule in rate_limits:
        if not fnmatch.fnmatch(tool_name, rule.tool_pattern):
            continue

        pattern_key = rule.tool_pattern
        calls_raw = history.get(pattern_key)
        calls: list[float]
        if isinstance(calls_raw, list):
            calls = [t for t in calls_raw if isinstance(t, float)]
        else:
            calls = []
        history[pattern_key] = calls

        # Prune old calls outside window
        cutoff = now - rule.window_seconds
        calls[:] = [t for t in calls if t > cutoff]

        if len(calls) >= rule.max_calls:
            return PolicyDecision(
                action="block",
                matched_rules=[f"rate_limit:{pattern_key}"],
                reasons=[
                    f"Rate limit exceeded: {len(calls)}/{rule.max_calls}"
                    f" calls in {rule.window_seconds}s window",
                ],
            )

        calls.append(now)

    return None

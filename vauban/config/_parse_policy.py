# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Parse the [policy] section of a TOML config."""

from typing import Literal, cast

from vauban.config._types import TomlDict
from vauban.types import (
    DataFlowRule,
    PolicyConfig,
    PolicyRule,
    RateLimitRule,
)


def _parse_policy(raw: TomlDict) -> PolicyConfig | None:
    """Parse the optional [policy] section into a PolicyConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("policy")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[policy] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    policy_dict = cast("dict[str, object]", sec)

    # -- default_action --
    default_action_raw = policy_dict.get("default_action", "allow")
    if not isinstance(default_action_raw, str):
        msg = (
            f"[policy].default_action must be a string,"
            f" got {type(default_action_raw).__name__}"
        )
        raise TypeError(msg)
    valid_actions = ("allow", "block")
    if default_action_raw not in valid_actions:
        msg = (
            f"[policy].default_action must be one of {valid_actions!r},"
            f" got {default_action_raw!r}"
        )
        raise ValueError(msg)

    # -- rules --
    rules_raw = policy_dict.get("rules", [])
    if not isinstance(rules_raw, list):
        msg = "[policy].rules must be a list"
        raise TypeError(msg)
    rules_list: list[object] = list(rules_raw)
    rules = _parse_rules(rules_list)

    # -- data_flow_rules --
    df_raw = policy_dict.get("data_flow_rules", [])
    if not isinstance(df_raw, list):
        msg = "[policy].data_flow_rules must be a list"
        raise TypeError(msg)
    df_list: list[object] = list(df_raw)
    data_flow_rules = _parse_data_flow_rules(df_list)

    # -- rate_limits --
    rl_raw = policy_dict.get("rate_limits", [])
    if not isinstance(rl_raw, list):
        msg = "[policy].rate_limits must be a list"
        raise TypeError(msg)
    rl_list: list[object] = list(rl_raw)
    rate_limits = _parse_rate_limits(rl_list)

    return PolicyConfig(
        rules=rules,
        data_flow_rules=data_flow_rules,
        rate_limits=rate_limits,
        default_action=default_action_raw,
    )


def _parse_rules(raw: list[object]) -> list[PolicyRule]:
    """Parse the [[policy.rules]] array."""
    rules: list[PolicyRule] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            msg = f"[[policy.rules]][{i}] must be a table"
            raise TypeError(msg)
        t = cast("dict[str, object]", entry)

        name = t.get("name")
        if not isinstance(name, str):
            msg = f"[[policy.rules]][{i}].name must be a string"
            raise TypeError(msg)

        action = t.get("action")
        if not isinstance(action, str):
            msg = f"[[policy.rules]][{i}].action must be a string"
            raise TypeError(msg)
        if action not in ("allow", "block", "confirm"):
            msg = (
                f"[[policy.rules]][{i}].action must be"
                f" 'allow', 'block', or 'confirm', got {action!r}"
            )
            raise ValueError(msg)
        action_literal: Literal["allow", "block", "confirm"]
        if action == "allow":
            action_literal = "allow"
        elif action == "block":
            action_literal = "block"
        else:
            action_literal = "confirm"

        tool_pattern = t.get("tool_pattern")
        if not isinstance(tool_pattern, str):
            msg = f"[[policy.rules]][{i}].tool_pattern must be a string"
            raise TypeError(msg)

        arg_key_raw = t.get("argument_key")
        argument_key: str | None = None
        if arg_key_raw is not None:
            if not isinstance(arg_key_raw, str):
                msg = (
                    f"[[policy.rules]][{i}].argument_key must be a string"
                )
                raise TypeError(msg)
            argument_key = arg_key_raw

        arg_pat_raw = t.get("argument_pattern")
        argument_pattern: str | None = None
        if arg_pat_raw is not None:
            if not isinstance(arg_pat_raw, str):
                msg = (
                    f"[[policy.rules]][{i}].argument_pattern must be"
                    " a string"
                )
                raise TypeError(msg)
            argument_pattern = arg_pat_raw

        rules.append(PolicyRule(
            name=name,
            action=action_literal,
            tool_pattern=tool_pattern,
            argument_key=argument_key,
            argument_pattern=argument_pattern,
        ))

    return rules


def _parse_data_flow_rules(raw: list[object]) -> list[DataFlowRule]:
    """Parse the [[policy.data_flow_rules]] array."""
    rules: list[DataFlowRule] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            msg = f"[[policy.data_flow_rules]][{i}] must be a table"
            raise TypeError(msg)
        t = cast("dict[str, object]", entry)

        source = t.get("source_tool")
        if not isinstance(source, str):
            msg = (
                f"[[policy.data_flow_rules]][{i}].source_tool"
                " must be a string"
            )
            raise TypeError(msg)

        labels_raw = t.get("source_labels", [])
        if not isinstance(labels_raw, list):
            msg = (
                f"[[policy.data_flow_rules]][{i}].source_labels"
                " must be a list"
            )
            raise TypeError(msg)

        blocked_raw = t.get("blocked_targets", [])
        if not isinstance(blocked_raw, list):
            msg = (
                f"[[policy.data_flow_rules]][{i}].blocked_targets"
                " must be a list"
            )
            raise TypeError(msg)

        rules.append(DataFlowRule(
            source_tool=source,
            source_labels=[str(label) for label in labels_raw],
            blocked_targets=[str(t_) for t_ in blocked_raw],
        ))

    return rules


def _parse_rate_limits(raw: list[object]) -> list[RateLimitRule]:
    """Parse the [[policy.rate_limits]] array."""
    rules: list[RateLimitRule] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            msg = f"[[policy.rate_limits]][{i}] must be a table"
            raise TypeError(msg)
        t = cast("dict[str, object]", entry)

        tool_pattern = t.get("tool_pattern")
        if not isinstance(tool_pattern, str):
            msg = (
                f"[[policy.rate_limits]][{i}].tool_pattern must be a string"
            )
            raise TypeError(msg)

        max_calls = t.get("max_calls")
        if not isinstance(max_calls, int):
            msg = (
                f"[[policy.rate_limits]][{i}].max_calls must be an integer"
            )
            raise TypeError(msg)

        window = t.get("window_seconds", 60.0)
        if not isinstance(window, int | float):
            msg = (
                f"[[policy.rate_limits]][{i}].window_seconds"
                " must be a number"
            )
            raise TypeError(msg)

        rules.append(RateLimitRule(
            tool_pattern=tool_pattern,
            max_calls=max_calls,
            window_seconds=float(window),
        ))

    return rules

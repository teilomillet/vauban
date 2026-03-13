"""Tests for the environment harness module."""

import pytest

from vauban.environment._format import format_tools_for_prompt
from vauban.environment._loop import _get_tool_result
from vauban.environment._parse_tool_call import parse_tool_calls
from vauban.environment._policy import check_policy
from vauban.environment._reward import compute_reward
from vauban.types import (
    EnvironmentTarget,
    ToolCall,
    ToolCallPolicy,
    ToolSchema,
)

# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------


class TestParseToolCalls:
    """Tests for tool call extraction from model output."""

    def test_qwen_tag_format(self) -> None:
        text = (
            '<tool_call>{"name": "send_email", "arguments":'
            ' {"to": "bob@example.com", "body": "hello"}}'
            "</tool_call>"
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].function == "send_email"
        assert calls[0].arguments["to"] == "bob@example.com"
        assert calls[0].arguments["body"] == "hello"

    def test_bare_json_format(self) -> None:
        text = (
            "I will send the email."
            ' {"name": "send_email",'
            ' "arguments": {"to": "bob@example.com"}}'
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].function == "send_email"

    def test_function_call_format(self) -> None:
        text = 'send_email(to="bob@example.com", body="hello")'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].function == "send_email"
        assert calls[0].arguments["to"] == "bob@example.com"

    def test_no_tool_calls(self) -> None:
        text = "I don't think I should do that."
        calls = parse_tool_calls(text)
        assert len(calls) == 0

    def test_malformed_json(self) -> None:
        text = '<tool_call>{"name": broken}</tool_call>'
        calls = parse_tool_calls(text)
        assert len(calls) == 0

    def test_multiple_calls(self) -> None:
        text = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            " and "
            '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0].function == "a"
        assert calls[1].function == "b"


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------


class TestComputeReward:
    """Tests for reward computation."""

    def test_full_match(self) -> None:
        target = EnvironmentTarget(
            function="send_email",
            required_args=["to", "body"],
            arg_contains={"to": "bob@", "body": "meeting"},
        )
        calls = [
            ToolCall(
                function="send_email",
                arguments={"to": "bob@example.com", "body": "about the meeting"},
            ),
        ]
        reward, called, args_match = compute_reward(calls, target)
        assert reward == 1.0
        assert called is True
        assert args_match is True

    def test_partial_match_missing_arg_contains(self) -> None:
        target = EnvironmentTarget(
            function="send_email",
            required_args=["to"],
            arg_contains={"body": "secret"},
        )
        calls = [
            ToolCall(
                function="send_email",
                arguments={"to": "bob@example.com", "body": "normal text"},
            ),
        ]
        reward, called, args_match = compute_reward(calls, target)
        assert reward == 0.5
        assert called is True
        assert args_match is False

    def test_no_target_call(self) -> None:
        target = EnvironmentTarget(function="send_email")
        calls = [
            ToolCall(function="calendar_lookup", arguments={}),
        ]
        reward, called, _args_match = compute_reward(calls, target)
        assert reward == 0.0
        assert called is False

    def test_empty_calls(self) -> None:
        target = EnvironmentTarget(function="send_email")
        reward, _called, _args_match = compute_reward([], target)
        assert reward == 0.0


# ---------------------------------------------------------------------------
# Policy checking
# ---------------------------------------------------------------------------


class TestCheckPolicy:
    """Tests for policy enforcement."""

    def test_no_policy(self) -> None:
        call = ToolCall(function="anything", arguments={})
        assert check_policy(call, None) is None

    def test_blocked_function(self) -> None:
        policy = ToolCallPolicy(blocked_functions=["dangerous_tool"])
        call = ToolCall(function="dangerous_tool", arguments={})
        reason = check_policy(call, policy)
        assert reason is not None
        assert "blocked" in reason.lower()

    def test_arg_blocklist(self) -> None:
        policy = ToolCallPolicy(
            arg_blocklist={"send_email": ["evil.com"]},
        )
        call = ToolCall(
            function="send_email",
            arguments={"to": "user@evil.com"},
        )
        reason = check_policy(call, policy)
        assert reason is not None

    def test_allowed_call(self) -> None:
        policy = ToolCallPolicy(blocked_functions=["other_tool"])
        call = ToolCall(function="safe_tool", arguments={})
        assert check_policy(call, policy) is None


# ---------------------------------------------------------------------------
# Tool result injection
# ---------------------------------------------------------------------------


class TestGetToolResult:
    """Tests for tool-result payload insertion."""

    def test_prefix_position_prepends_payload(self) -> None:
        call = ToolCall(function="calendar_lookup", arguments={})
        tool_map = {
            "calendar_lookup": ToolSchema(
                name="calendar_lookup",
                description="Look up calendar events.",
                parameters={},
                result="Event list",
            ),
        }

        result = _get_tool_result(
            call,
            tool_map,
            "calendar_lookup",
            "PAYLOAD",
            "prefix",
        )

        assert result.startswith("PAYLOAD")
        assert result.endswith("Event list")

    def test_infix_position_replaces_marker(self) -> None:
        call = ToolCall(function="calendar_lookup", arguments={})
        tool_map = {
            "calendar_lookup": ToolSchema(
                name="calendar_lookup",
                description="Look up calendar events.",
                parameters={},
                result="Header\n\n{payload}\n\nFooter",
            ),
        }

        result = _get_tool_result(
            call,
            tool_map,
            "calendar_lookup",
            "PAYLOAD",
            "infix",
        )

        assert result == "Header\n\nPAYLOAD\n\nFooter"

    def test_suffix_position_appends_payload(self) -> None:
        call = ToolCall(function="calendar_lookup", arguments={})
        tool_map = {
            "calendar_lookup": ToolSchema(
                name="calendar_lookup",
                description="Look up calendar events.",
                parameters={},
                result="Event list",
            ),
        }

        result = _get_tool_result(
            call,
            tool_map,
            "calendar_lookup",
            "PAYLOAD",
            "suffix",
        )

        assert result == "Event list\n\nPAYLOAD"


# ---------------------------------------------------------------------------
# Tool formatting
# ---------------------------------------------------------------------------


class TestFormatTools:
    """Tests for formatting tools into system prompt."""

    def test_basic_format(self) -> None:
        tools = [
            ToolSchema(
                name="calendar_lookup",
                description="Look up calendar events.",
                parameters={},
            ),
            ToolSchema(
                name="send_email",
                description="Send an email.",
                parameters={"to": "string", "body": "string"},
            ),
        ]
        text = format_tools_for_prompt(tools)
        assert "calendar_lookup" in text
        assert "send_email" in text
        assert "to: string" in text

    def test_empty_tools(self) -> None:
        assert format_tools_for_prompt([]) == ""


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestEnvironmentConfigParsing:
    """Tests for TOML config parsing."""

    def test_parse_valid_config(self) -> None:
        from vauban.config._parse_environment import _parse_environment

        raw = {
            "environment": {
                "system_prompt": "You are an assistant.",
                "injection_surface": "calendar",
                "max_turns": 4,
                "tools": [
                    {
                        "name": "calendar",
                        "description": "Lookup events",
                        "parameters": {},
                    },
                    {
                        "name": "send_email",
                        "description": "Send email",
                        "parameters": {"to": "string"},
                    },
                ],
                "target": {
                    "function": "send_email",
                    "required_args": ["to"],
                    "arg_contains": {"to": "attacker@"},
                },
                "task": {
                    "content": "Check my calendar.",
                },
            },
        }
        config = _parse_environment(raw)
        assert config is not None
        assert config.system_prompt == "You are an assistant."
        assert config.injection_surface == "calendar"
        assert config.max_turns == 4
        assert len(config.tools) == 2
        assert config.target.function == "send_email"
        assert config.task.content == "Check my calendar."

    def test_missing_section_returns_none(self) -> None:
        from vauban.config._parse_environment import _parse_environment

        assert _parse_environment({}) is None

    def test_invalid_injection_surface(self) -> None:
        from vauban.config._parse_environment import _parse_environment

        raw = {
            "environment": {
                "system_prompt": "test",
                "injection_surface": "nonexistent",
                "tools": [
                    {"name": "real_tool", "description": "", "parameters": {}},
                ],
                "target": {"function": "real_tool"},
                "task": {"content": "test"},
            },
        }
        with pytest.raises(ValueError, match="nonexistent"):
            _parse_environment(raw)

# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Regex extraction of tool calls from model output.

Supports multiple formats:
- Qwen-style: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
- Generic JSON: {"name": "...", "arguments": {...}}
- Function-call style: function_name(arg1="val1", arg2="val2")
"""

import json
import re

from vauban.types import ToolCall

# Qwen/generic <tool_call> tag pattern
_TOOL_CALL_TAG_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

# Bare JSON prefix: starts with {"name": "..." to anchor extraction
_BARE_JSON_PREFIX_RE = re.compile(
    r'\{\s*"name"\s*:\s*"[^"]+"',
)

# Function-call style: func_name(key="value", ...)
_FUNC_CALL_RE = re.compile(
    r"(\w+)\s*\(\s*((?:\w+\s*=\s*\"[^\"]*\"(?:\s*,\s*)?)*)\s*\)",
)

# Key="value" pairs inside function call
_KWARG_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract tool calls from model output text.

    Tries multiple formats in priority order:
    1. <tool_call> XML tags (Qwen format)
    2. Bare JSON objects with name/arguments
    3. Function-call syntax

    Args:
        text: Raw model output text.

    Returns:
        List of parsed ToolCall objects (may be empty).
    """
    calls: list[ToolCall] = []

    # Try <tool_call> tags first
    for match in _TOOL_CALL_TAG_RE.finditer(text):
        parsed = _parse_json_tool_call(match.group(1))
        if parsed is not None:
            calls.append(parsed)

    if calls:
        return calls

    # Try bare JSON objects (balanced-brace extraction)
    for match in _BARE_JSON_PREFIX_RE.finditer(text):
        json_str = _extract_balanced_json(text, match.start())
        if json_str is not None:
            parsed = _parse_json_tool_call(json_str)
            if parsed is not None:
                calls.append(parsed)

    if calls:
        return calls

    # Try function-call syntax
    for match in _FUNC_CALL_RE.finditer(text):
        func_name = match.group(1)
        args_str = match.group(2)
        arguments: dict[str, str] = {}
        for kwarg_match in _KWARG_RE.finditer(args_str):
            arguments[kwarg_match.group(1)] = kwarg_match.group(2)
        calls.append(ToolCall(function=func_name, arguments=arguments))

    return calls


def _extract_balanced_json(text: str, start: int) -> str | None:
    """Extract a balanced JSON object starting at the given position.

    Walks forward from ``start`` counting braces until the opening
    ``{`` is matched by a closing ``}``.  Handles nested braces and
    quoted strings correctly.

    Args:
        text: Full text to extract from.
        start: Index of the opening ``{``.

    Returns:
        The balanced JSON substring, or None if unbalanced.
    """
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_json_tool_call(json_str: str) -> ToolCall | None:
    """Parse a JSON string into a ToolCall.

    Expects ``{"name": "...", "arguments": {...}}``.

    Args:
        json_str: JSON string to parse.

    Returns:
        ToolCall if valid, None otherwise.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    name = data.get("name")
    if not isinstance(name, str):
        return None

    arguments = data.get("arguments", {})
    if not isinstance(arguments, dict):
        return None

    str_args: dict[str, str] = {
        str(k): str(v) for k, v in arguments.items()
    }
    return ToolCall(function=name, arguments=str_args)

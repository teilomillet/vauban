# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra branch coverage for environment tool-call parsing helpers."""

from vauban.environment._parse_tool_call import (
    _extract_balanced_json,
    _parse_json_tool_call,
)


def test_extract_balanced_json_requires_opening_brace() -> None:
    assert _extract_balanced_json("not json", 0) is None


def test_extract_balanced_json_handles_escapes_and_nested_braces() -> None:
    text = (
        '{"name":"send_email","arguments":{"body":"say \\"hello\\" {ignored}"}}'
        " trailing"
    )
    assert _extract_balanced_json(text, 0) == (
        '{"name":"send_email","arguments":{"body":"say \\"hello\\" {ignored}"}}'
    )


def test_extract_balanced_json_returns_none_when_unbalanced() -> None:
    assert _extract_balanced_json('{"name":"send_email"', 0) is None


def test_parse_json_tool_call_rejects_non_object_json() -> None:
    assert _parse_json_tool_call('["send_email"]') is None


def test_parse_json_tool_call_requires_string_name() -> None:
    assert _parse_json_tool_call('{"name": 7, "arguments": {}}') is None


def test_parse_json_tool_call_requires_mapping_arguments() -> None:
    assert _parse_json_tool_call('{"name": "send_email", "arguments": []}') is None

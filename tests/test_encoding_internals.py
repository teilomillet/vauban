"""Tests for softprompt encoding helper functions."""
# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.conftest import VOCAB_SIZE, MockTokenizer
from vauban import _ops as ops
from vauban.softprompt._encoding import (
    _build_messages,
    _encode_messages,
    _pre_encode_prompts,
    _resolve_infix_overrides,
    _resolve_injection_ids,
    _token_list,
)

if TYPE_CHECKING:
    from vauban._array import Array


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    """Tests for _build_messages chat message assembly."""

    def test_user_only_no_system(self) -> None:
        """User content alone produces a single user message."""
        msgs: list[dict[str, str]] = _build_messages("Hello")
        assert len(msgs) == 1
        assert msgs[0] == {"role": "user", "content": "Hello"}

    def test_with_system_prompt(self) -> None:
        """System prompt prepends a system message before the user message."""
        msgs: list[dict[str, str]] = _build_messages(
            "Hello", system_prompt="Be helpful",
        )
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "Be helpful"}
        assert msgs[1] == {"role": "user", "content": "Hello"}

    def test_with_history(self) -> None:
        """History messages appear between system and user messages."""
        history: list[dict[str, str]] = [
            {"role": "user", "content": "Prior question"},
            {"role": "assistant", "content": "Prior answer"},
        ]
        msgs: list[dict[str, str]] = _build_messages("Follow-up", history=history)
        assert len(msgs) == 3
        assert msgs[0] == history[0]
        assert msgs[1] == history[1]
        assert msgs[2] == {"role": "user", "content": "Follow-up"}

    def test_with_prefix_messages(self) -> None:
        """Prefix messages appear after history but before the user message."""
        prefix: list[dict[str, str]] = [
            {"role": "user", "content": "Context prompt"},
            {"role": "assistant", "content": "Acknowledged"},
        ]
        msgs: list[dict[str, str]] = _build_messages("Query", prefix_messages=prefix)
        assert len(msgs) == 3
        assert msgs[0] == prefix[0]
        assert msgs[1] == prefix[1]
        assert msgs[2] == {"role": "user", "content": "Query"}

    def test_message_order_system_history_prefix_user(self) -> None:
        """Full ordering is system, history, prefix_messages, user."""
        history: list[dict[str, str]] = [
            {"role": "user", "content": "h1"},
        ]
        prefix: list[dict[str, str]] = [
            {"role": "assistant", "content": "p1"},
        ]
        msgs: list[dict[str, str]] = _build_messages(
            "final",
            system_prompt="sys",
            history=history,
            prefix_messages=prefix,
        )
        assert [m["role"] for m in msgs] == [
            "system", "user", "assistant", "user",
        ]
        assert msgs[0]["content"] == "sys"
        assert msgs[1]["content"] == "h1"
        assert msgs[2]["content"] == "p1"
        assert msgs[3]["content"] == "final"


# ---------------------------------------------------------------------------
# _encode_messages
# ---------------------------------------------------------------------------


class TestEncodeMessages:
    """Tests for _encode_messages tokenizer encoding."""

    def test_returns_2d_nonempty(self) -> None:
        """Encoded output is a 2-D array with shape (1, seq_len) and seq_len > 0."""
        tok: MockTokenizer = MockTokenizer(VOCAB_SIZE)
        msgs: list[dict[str, str]] = [{"role": "user", "content": "Hi"}]
        arr: Array = _encode_messages(tok, msgs)
        shape: tuple[int, ...] = tuple(arr.shape)
        assert len(shape) == 2
        assert shape[0] == 1
        assert shape[1] > 0

    def test_different_messages_different_encodings(self) -> None:
        """Different input messages produce different token sequences."""
        tok: MockTokenizer = MockTokenizer(VOCAB_SIZE)
        msgs_a: list[dict[str, str]] = [{"role": "user", "content": "Alpha"}]
        msgs_b: list[dict[str, str]] = [{"role": "user", "content": "Beta"}]
        arr_a: Array = _encode_messages(tok, msgs_a)
        arr_b: Array = _encode_messages(tok, msgs_b)
        tokens_a: list[int] = _token_list(arr_a)
        tokens_b: list[int] = _token_list(arr_b)
        assert tokens_a != tokens_b


# ---------------------------------------------------------------------------
# _token_list
# ---------------------------------------------------------------------------


class TestTokenList:
    """Tests for _token_list array-to-list conversion."""

    def test_flat_list_of_ints(self) -> None:
        """Converts a multi-token 2-D array to a flat list of Python ints."""
        arr: Array = ops.array([1, 2, 3])[None, :]
        result: list[int] = _token_list(arr)
        assert result == [1, 2, 3]
        assert all(isinstance(t, int) for t in result)

    def test_single_token(self) -> None:
        """Handles a single-token array correctly."""
        arr: Array = ops.array([42])[None, :]
        result: list[int] = _token_list(arr)
        assert result == [42]
        assert isinstance(result[0], int)


# ---------------------------------------------------------------------------
# _resolve_infix_overrides
# ---------------------------------------------------------------------------


class TestResolveInfixOverrides:
    """Tests for _resolve_infix_overrides infix split computation."""

    def test_prompts_with_placeholder_produce_split_positions(self) -> None:
        """Each prompt with {suffix} yields a non-negative split position."""
        tok: MockTokenizer = MockTokenizer(VOCAB_SIZE)
        prompts: list[str] = [
            "Tell me about {suffix} please",
            "Explain {suffix} in detail",
        ]
        encoded: list[Array]
        infix_map: dict[int, int]
        encoded, infix_map = _resolve_infix_overrides(tok, prompts)
        assert len(encoded) == len(prompts)
        assert len(infix_map) == len(prompts)
        for arr in encoded:
            assert id(arr) in infix_map
            assert infix_map[id(arr)] >= 0

    def test_returns_arrays_and_map_with_correct_structure(self) -> None:
        """Returned arrays are 2-D and the map keys match array IDs."""
        tok: MockTokenizer = MockTokenizer(VOCAB_SIZE)
        prompts: list[str] = ["Hello {suffix} world"]
        encoded: list[Array]
        infix_map: dict[int, int]
        encoded, infix_map = _resolve_infix_overrides(tok, prompts)
        assert len(encoded) == 1
        arr: Array = encoded[0]
        shape: tuple[int, ...] = tuple(arr.shape)
        assert len(shape) == 2
        assert shape[0] == 1
        assert shape[1] > 0
        assert id(arr) in infix_map
        split: int = infix_map[id(arr)]
        assert isinstance(split, int)
        assert 0 <= split <= shape[1]


# ---------------------------------------------------------------------------
# _resolve_injection_ids
# ---------------------------------------------------------------------------


class TestResolveInjectionIds:
    """Tests for _resolve_injection_ids injection context resolution."""

    def test_returns_none_when_no_injection_config(self) -> None:
        """Returns None when neither injection_context nor template is set."""
        from vauban.types import SoftPromptConfig

        config: SoftPromptConfig = SoftPromptConfig()
        tok: MockTokenizer = MockTokenizer(VOCAB_SIZE)
        result: list[Array] | None = _resolve_injection_ids(
            config, tok, ["test prompt"],
        )
        assert result is None

    def test_returns_arrays_for_preset_context(self) -> None:
        """Preset injection_context='web_page' produces encoded arrays."""
        from vauban.types import SoftPromptConfig

        config: SoftPromptConfig = SoftPromptConfig(injection_context="web_page")
        tok: MockTokenizer = MockTokenizer(VOCAB_SIZE)
        prompts: list[str] = ["first prompt", "second prompt"]
        result: list[Array] | None = _resolve_injection_ids(config, tok, prompts)
        assert result is not None
        assert len(result) == len(prompts)
        for arr in result:
            shape: tuple[int, ...] = tuple(arr.shape)
            assert len(shape) == 2
            assert shape[0] == 1
            assert shape[1] > 0

    def test_template_takes_priority_over_preset(self) -> None:
        """When both template and preset are set, template is used."""
        from vauban.types import SoftPromptConfig

        template: str = "Context: {payload} End."
        config: SoftPromptConfig = SoftPromptConfig(
            injection_context="web_page",
            injection_context_template=template,
        )
        tok: MockTokenizer = MockTokenizer(VOCAB_SIZE)
        prompts: list[str] = ["payload content"]

        # Template result
        result: list[Array] | None = _resolve_injection_ids(config, tok, prompts)
        assert result is not None
        template_tokens: list[int] = _token_list(result[0])

        # Preset-only result for comparison
        config_preset: SoftPromptConfig = SoftPromptConfig(
            injection_context="web_page",
        )
        result_preset: list[Array] | None = _resolve_injection_ids(
            config_preset, tok, prompts,
        )
        assert result_preset is not None
        preset_tokens: list[int] = _token_list(result_preset[0])

        assert template_tokens != preset_tokens


# ---------------------------------------------------------------------------
# _pre_encode_prompts
# ---------------------------------------------------------------------------


class TestPreEncodePrompts:
    """Tests for _pre_encode_prompts batch encoding."""

    def test_multiple_prompts_produce_correct_count(self) -> None:
        """Each prompt in the input list yields exactly one encoded array."""
        tok: MockTokenizer = MockTokenizer(VOCAB_SIZE)
        prompts: list[str] = ["alpha", "beta", "gamma"]
        result: list[Array] = _pre_encode_prompts(tok, prompts)
        assert len(result) == len(prompts)
        for arr in result:
            shape: tuple[int, ...] = tuple(arr.shape)
            assert len(shape) == 2
            assert shape[0] == 1
            assert shape[1] > 0

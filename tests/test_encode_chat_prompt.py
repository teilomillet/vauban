"""Tests for encode_chat_prompt / encode_user_prompt helpers."""

from __future__ import annotations

from conftest import VOCAB_SIZE, MockTokenizer

from vauban import _ops as ops
from vauban._forward import encode_chat_prompt, encode_user_prompt


class TestEncodeChatPrompt:
    """Tests for the shared encode_chat_prompt helper."""

    def test_returns_2d_array(self) -> None:
        tok = MockTokenizer(VOCAB_SIZE)
        messages = [{"role": "user", "content": "hello"}]
        result = encode_chat_prompt(tok, messages)
        assert result.shape[0] == 1
        assert result.ndim == 2

    def test_nonempty_for_nonempty_input(self) -> None:
        tok = MockTokenizer(VOCAB_SIZE)
        messages = [{"role": "user", "content": "test"}]
        result = encode_chat_prompt(tok, messages)
        assert result.shape[1] > 0

    def test_multiple_messages(self) -> None:
        tok = MockTokenizer(VOCAB_SIZE)
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "hi"},
        ]
        result = encode_chat_prompt(tok, messages)
        assert result.ndim == 2
        assert result.shape[1] > 0

    def test_longer_content_produces_more_tokens(self) -> None:
        tok = MockTokenizer(VOCAB_SIZE)
        short = encode_chat_prompt(tok, [{"role": "user", "content": "hi"}])
        long = encode_chat_prompt(
            tok, [{"role": "user", "content": "hello world this is longer"}],
        )
        assert long.shape[1] > short.shape[1]


class TestEncodeUserPrompt:
    """Tests for the convenience encode_user_prompt wrapper."""

    def test_returns_2d_array(self) -> None:
        tok = MockTokenizer(VOCAB_SIZE)
        result = encode_user_prompt(tok, "test prompt")
        assert result.ndim == 2
        assert result.shape[0] == 1

    def test_matches_manual_messages(self) -> None:
        tok = MockTokenizer(VOCAB_SIZE)
        via_helper = encode_user_prompt(tok, "hello")
        via_manual = encode_chat_prompt(
            tok, [{"role": "user", "content": "hello"}],
        )
        ops.eval(via_helper, via_manual)
        assert via_helper.shape == via_manual.shape
        helper_list = via_helper[0].tolist()
        manual_list = via_manual[0].tolist()
        assert helper_list == manual_list

    def test_nonempty_output(self) -> None:
        tok = MockTokenizer(VOCAB_SIZE)
        result = encode_user_prompt(tok, "test")
        assert result.shape[1] > 0

# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for encode_chat_prompt / encode_user_prompt helpers."""

from __future__ import annotations

from conftest import VOCAB_SIZE, MockTokenizer

from vauban import _ops as ops
from vauban._forward import encode_chat_prompt, encode_user_prompt


class TestEncodeChatPrompt:
    """Tests for the shared encode_chat_prompt helper."""

    def test_adds_generation_prompt_when_supported(self) -> None:
        class RecordingTokenizer(MockTokenizer):
            """Tokenizer that records chat-template generation prompt usage."""

            def __init__(self) -> None:
                super().__init__(VOCAB_SIZE)
                self.add_generation_prompt_seen = False
                self.enable_thinking_seen = True

            def apply_chat_template(
                self,
                messages: list[dict[str, str]],
                tokenize: bool = True,
                add_generation_prompt: bool = False,
                enable_thinking: bool = True,
            ) -> str | list[int]:
                self.add_generation_prompt_seen = add_generation_prompt
                self.enable_thinking_seen = enable_thinking
                return super().apply_chat_template(
                    messages,
                    tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=enable_thinking,
                )

        tok = RecordingTokenizer()

        result = encode_chat_prompt(tok, [{"role": "user", "content": "hello"}])

        assert result.shape[0] == 1
        assert tok.add_generation_prompt_seen is True
        assert tok.enable_thinking_seen is False

    def test_falls_back_for_generation_prompt_only_templates(self) -> None:
        class GenerationPromptTokenizer(MockTokenizer):
            """Tokenizer supporting assistant prompt but not thinking control."""

            def __init__(self) -> None:
                super().__init__(VOCAB_SIZE)
                self.add_generation_prompt_seen = False

            def apply_chat_template(
                self,
                messages: list[dict[str, str]],
                tokenize: bool = True,
                add_generation_prompt: bool = False,
            ) -> str | list[int]:
                self.add_generation_prompt_seen = add_generation_prompt
                return super().apply_chat_template(
                    messages,
                    tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt,
                )

        tok = GenerationPromptTokenizer()

        result = encode_chat_prompt(tok, [{"role": "user", "content": "hello"}])

        assert result.shape[0] == 1
        assert tok.add_generation_prompt_seen is True

    def test_falls_back_for_legacy_chat_templates(self) -> None:
        class LegacyTokenizer(MockTokenizer):
            """Tokenizer with the older two-argument chat-template signature."""

            def apply_chat_template(
                self,
                messages: list[dict[str, str]],
                tokenize: bool = True,
            ) -> str | list[int]:
                return super().apply_chat_template(messages, tokenize=tokenize)

        tok = LegacyTokenizer(VOCAB_SIZE)

        result = encode_chat_prompt(tok, [{"role": "user", "content": "hello"}])

        assert result.shape[0] == 1

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

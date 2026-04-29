# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban.evaluate: metric computation."""


from tests.conftest import MockCausalLM, MockTokenizer
from vauban.evaluate import (
    DEFAULT_REFUSAL_PHRASES,
    _decode_generated,
    _judge_refusal_rate,
    _kl_divergence,
    _perplexity,
    _refusal_rate,
    evaluate,
)


class TestDecodeGenerated:
    def test_skips_special_tokens_when_supported(self) -> None:
        class SpecialAwareTokenizer(MockTokenizer):
            def decode(
                self,
                token_ids: list[int],
                skip_special_tokens: bool = False,
            ) -> str:
                del token_ids
                return "clean" if skip_special_tokens else "clean<|im_end|>"

        assert _decode_generated(SpecialAwareTokenizer(32), [1, 2]) == "clean"

    def test_falls_back_for_legacy_decode(self) -> None:
        class LegacyDecodeTokenizer(MockTokenizer):
            def decode(self, token_ids: list[int]) -> str:
                del token_ids
                return "legacy"

        assert _decode_generated(LegacyDecodeTokenizer(32), [1, 2]) == "legacy"


class TestRefusalRate:
    def test_counts_refusals(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        # With a random model, refusal rate should be between 0 and 1
        rate = _refusal_rate(
            mock_model, mock_tokenizer, ["test prompt"],
            DEFAULT_REFUSAL_PHRASES, max_tokens=10,
        )
        assert 0.0 <= rate <= 1.0

    def test_empty_prompts(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        rate = _refusal_rate(
            mock_model, mock_tokenizer, [],
            DEFAULT_REFUSAL_PHRASES, max_tokens=10,
        )
        assert rate == 0.0


class TestPerplexity:
    def test_positive_perplexity(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        ppl = _perplexity(mock_model, mock_tokenizer, ["hello world"])
        assert ppl > 0.0

    def test_empty_prompts(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        ppl = _perplexity(mock_model, mock_tokenizer, [])
        assert ppl == 0.0


class TestKLDivergence:
    def test_same_model_near_zero(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        kl = _kl_divergence(
            mock_model, mock_model, mock_tokenizer, ["test"],
        )
        assert abs(kl) < 1e-4

    def test_empty_prompts(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        kl = _kl_divergence(mock_model, mock_model, mock_tokenizer, [])
        assert kl == 0.0


class TestJudgeRefusalRate:
    def test_returns_valid_float(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        rate = _judge_refusal_rate(
            mock_model, mock_tokenizer, ["test prompt"], max_tokens=10,
        )
        assert 0.0 <= rate <= 1.0

    def test_empty_prompts_returns_zero(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        rate = _judge_refusal_rate(
            mock_model, mock_tokenizer, [], max_tokens=10,
        )
        assert rate == 0.0


class TestEvaluate:
    def test_returns_eval_result(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        result = evaluate(
            mock_model, mock_model, mock_tokenizer,
            ["test prompt"], max_tokens=5,
        )
        assert result.num_prompts == 1
        assert 0.0 <= result.refusal_rate_original <= 1.0
        assert result.perplexity_original > 0
        assert abs(result.kl_divergence) < 1e-4  # same model

    def test_judge_mode(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        result = evaluate(
            mock_model, mock_model, mock_tokenizer,
            ["test prompt"], max_tokens=5, refusal_mode="judge",
        )
        assert result.num_prompts == 1
        assert 0.0 <= result.refusal_rate_original <= 1.0

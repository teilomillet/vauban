# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban.depth: deep-thinking token analysis."""

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from tests.conftest import D_MODEL, NUM_LAYERS, MockCausalLM, MockTokenizer
from vauban import _ops as ops
from vauban.depth import (
    _cached_forward_all_hidden,
    _jsd,
    _settling_depth,
    depth_direction,
    depth_generate,
    depth_profile,
)
from vauban.types import DepthConfig, DepthResult, DirectionResult, TokenDepth

if TYPE_CHECKING:
    from vauban.types import LayerCache


@pytest.fixture
def depth_config() -> DepthConfig:
    """Default depth config for tests."""
    return DepthConfig(
        prompts=["Hello world"],
        settling_threshold=0.5,
        deep_fraction=0.85,
        top_k_logits=16,  # small for mock vocab
        max_tokens=0,
    )


@pytest.fixture
def gen_config() -> DepthConfig:
    """Depth config for generation mode."""
    return DepthConfig(
        prompts=["Hello world"],
        settling_threshold=0.5,
        deep_fraction=0.85,
        top_k_logits=16,
        max_tokens=3,
    )


class TestJSD:
    def test_identical_distributions(self) -> None:
        """JSD(p, p) should be approximately 0."""
        p = ops.softmax(ops.random.normal((32,)))
        ops.eval(p)
        result = _jsd(p, p)
        assert result == pytest.approx(0.0, abs=1e-5)

    def test_non_negative(self) -> None:
        """JSD should always be non-negative."""
        p = ops.softmax(ops.random.normal((32,)))
        q = ops.softmax(ops.random.normal((32,)))
        ops.eval(p, q)
        result = _jsd(p, q)
        assert result >= 0.0

    def test_symmetric(self) -> None:
        """JSD(p, q) should equal JSD(q, p)."""
        p = ops.softmax(ops.random.normal((32,)))
        q = ops.softmax(ops.random.normal((32,)))
        ops.eval(p, q)
        assert _jsd(p, q) == pytest.approx(_jsd(q, p), abs=1e-5)

    def test_different_distributions_positive(self) -> None:
        """JSD of different distributions should be positive."""
        p = ops.array([0.9, 0.05, 0.05])
        q = ops.array([0.05, 0.9, 0.05])
        result = _jsd(p, q)
        assert result > 0.0


class TestSettlingDepth:
    def test_immediate_settling(self) -> None:
        """If first JSD <= threshold, depth = 0."""
        profile = [0.1, 0.3, 0.5, 0.7]
        assert _settling_depth(profile, 0.5) == 0

    def test_never_settles(self) -> None:
        """If JSD never drops below threshold, returns last layer."""
        profile = [0.8, 0.9, 0.7, 0.6]
        assert _settling_depth(profile, 0.5) == 3

    def test_settles_mid_layer(self) -> None:
        """Normal settling case."""
        profile = [0.9, 0.7, 0.3, 0.1]
        assert _settling_depth(profile, 0.5) == 2

    def test_exact_threshold(self) -> None:
        """Value exactly at threshold counts as settled."""
        profile = [0.8, 0.5, 0.3]
        assert _settling_depth(profile, 0.5) == 1

    def test_empty_profile(self) -> None:
        """Edge case: empty profile returns -1 (last index)."""
        assert _settling_depth([], 0.5) == -1


class TestDepthProfile:
    def test_returns_per_token_results(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        depth_config: DepthConfig,
    ) -> None:
        """Number of tokens should match encoded sequence length."""
        result = depth_profile(
            mock_model, mock_tokenizer, "Hello", depth_config,
        )
        # The tokenizer wraps with chat template, so seq_len > len("Hello")
        assert len(result.tokens) > 0

    def test_jsd_profile_length(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        depth_config: DepthConfig,
    ) -> None:
        """Each token's JSD profile should have layer_count entries."""
        result = depth_profile(
            mock_model, mock_tokenizer, "Hi", depth_config,
        )
        for token in result.tokens:
            assert len(token.jsd_profile) == result.layer_count

    def test_dtr_in_range(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        depth_config: DepthConfig,
    ) -> None:
        """DTR should be between 0.0 and 1.0."""
        result = depth_profile(
            mock_model, mock_tokenizer, "Test", depth_config,
        )
        assert 0.0 <= result.deep_thinking_ratio <= 1.0

    def test_settling_depth_in_range(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        depth_config: DepthConfig,
    ) -> None:
        """Settling depth should be in [0, layer_count)."""
        result = depth_profile(
            mock_model, mock_tokenizer, "Test", depth_config,
        )
        for token in result.tokens:
            assert 0 <= token.settling_depth < result.layer_count

    def test_layer_count_matches_model(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        depth_config: DepthConfig,
    ) -> None:
        """Layer count should match mock model layers."""
        result = depth_profile(
            mock_model, mock_tokenizer, "Test", depth_config,
        )
        assert result.layer_count == NUM_LAYERS

    def test_deep_thinking_count_consistent(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        depth_config: DepthConfig,
    ) -> None:
        """deep_thinking_count should match actual deep tokens."""
        result = depth_profile(
            mock_model, mock_tokenizer, "Test", depth_config,
        )
        actual_deep = sum(1 for t in result.tokens if t.is_deep_thinking)
        assert result.deep_thinking_count == actual_deep

    def test_prompt_preserved(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        depth_config: DepthConfig,
    ) -> None:
        """The prompt should be preserved in the result."""
        result = depth_profile(
            mock_model, mock_tokenizer, "Hello world", depth_config,
        )
        assert result.prompt == "Hello world"

    def test_top_k_larger_than_vocab_uses_full_vocab(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        config = DepthConfig(
            prompts=["Hello world"],
            settling_threshold=0.5,
            deep_fraction=0.85,
            top_k_logits=10_000,
            max_tokens=0,
        )
        result = depth_profile(mock_model, mock_tokenizer, "Hello", config)
        assert len(result.tokens) > 0


class TestDepthGenerate:
    def test_token_count(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        gen_config: DepthConfig,
    ) -> None:
        """Generated tokens count should match max_tokens."""
        result = depth_generate(
            mock_model, mock_tokenizer, "Hello", gen_config,
        )
        assert len(result.tokens) == gen_config.max_tokens

    def test_dtr_in_range(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        gen_config: DepthConfig,
    ) -> None:
        """DTR should be between 0.0 and 1.0 in generation mode."""
        result = depth_generate(
            mock_model, mock_tokenizer, "Test", gen_config,
        )
        assert 0.0 <= result.deep_thinking_ratio <= 1.0

    def test_jsd_profile_length(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        gen_config: DepthConfig,
    ) -> None:
        """Each generated token should have layer_count JSD entries."""
        result = depth_generate(
            mock_model, mock_tokenizer, "Hi", gen_config,
        )
        for token in result.tokens:
            assert len(token.jsd_profile) == result.layer_count

    def test_top_k_larger_than_vocab_uses_full_vocab(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        config = DepthConfig(
            prompts=["Hello world"],
            settling_threshold=0.5,
            deep_fraction=0.85,
            top_k_logits=10_000,
            max_tokens=2,
        )
        result = depth_generate(mock_model, mock_tokenizer, "Hi", config)
        assert len(result.tokens) == config.max_tokens

    def test_cached_forward_with_cache_branch(
        self,
        mock_model: MockCausalLM,
    ) -> None:
        token_ids = ops.array([[1, 2, 3]])
        cache = mock_model.make_cache()
        typed_cache = cast("list[LayerCache]", cache)
        logits = _cached_forward_all_hidden(
            mock_model,
            token_ids,
            cache=typed_cache,
        )
        assert logits.shape[0] == 1


def _make_depth_result(
    prompt: str, dtr: float, layer_count: int = NUM_LAYERS,
) -> DepthResult:
    """Build a minimal DepthResult with a given DTR for testing."""
    tokens = [
        TokenDepth(
            token_id=0, token_str="x",
            settling_depth=0, is_deep_thinking=False,
            jsd_profile=[0.1] * layer_count,
        ),
    ]
    return DepthResult(
        tokens=tokens,
        deep_thinking_ratio=dtr,
        deep_thinking_count=0,
        mean_settling_depth=0.0,
        layer_count=layer_count,
        settling_threshold=0.5,
        deep_fraction=0.85,
        prompt=prompt,
    )


class TestDepthDirection:
    """Tests for depth_direction(): median split + direction extraction."""

    def test_median_split_correct(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Prompts should be split by median DTR into deep/shallow."""
        results = [
            _make_depth_result("shallow1", 0.1),
            _make_depth_result("shallow2", 0.2),
            _make_depth_result("deep1", 0.8),
            _make_depth_result("deep2", 0.9),
        ]
        dir_result = depth_direction(mock_model, mock_tokenizer, results)
        assert set(dir_result.shallow_prompts) == {"shallow1", "shallow2"}
        assert set(dir_result.deep_prompts) == {"deep1", "deep2"}

    def test_median_dtr_value(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """median_dtr should be the DTR of the median-indexed prompt."""
        results = [
            _make_depth_result("a", 0.1),
            _make_depth_result("b", 0.5),
            _make_depth_result("c", 0.9),
        ]
        dir_result = depth_direction(mock_model, mock_tokenizer, results)
        # Sorted: [0.1, 0.5, 0.9], median_idx=1, median_dtr=0.5
        assert dir_result.median_dtr == pytest.approx(0.5)

    def test_returns_valid_direction(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Result should contain a direction vector with correct d_model."""
        results = [
            _make_depth_result("a", 0.1),
            _make_depth_result("b", 0.9),
        ]
        dir_result = depth_direction(mock_model, mock_tokenizer, results)
        assert dir_result.direction.shape == (D_MODEL,)
        assert dir_result.d_model == D_MODEL

    def test_cosine_scores_per_layer(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """cosine_scores should have one entry per layer."""
        results = [
            _make_depth_result("a", 0.1),
            _make_depth_result("b", 0.9),
        ]
        dir_result = depth_direction(mock_model, mock_tokenizer, results)
        assert len(dir_result.cosine_scores) == NUM_LAYERS

    def test_refusal_cosine_none_when_no_refusal_dir(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """refusal_cosine should be None when no refusal direction is given."""
        results = [
            _make_depth_result("a", 0.1),
            _make_depth_result("b", 0.9),
        ]
        dir_result = depth_direction(mock_model, mock_tokenizer, results)
        assert dir_result.refusal_cosine is None

    def test_refusal_cosine_computed(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """refusal_cosine should be a float when refusal direction is given."""
        results = [
            _make_depth_result("a", 0.1),
            _make_depth_result("b", 0.9),
        ]
        refusal_dir = DirectionResult(
            direction=ops.ones((D_MODEL,)) / (D_MODEL ** 0.5),
            layer_index=0,
            cosine_scores=[0.5] * NUM_LAYERS,
            d_model=D_MODEL,
            model_path="test",
        )
        dir_result = depth_direction(
            mock_model, mock_tokenizer, results,
            refusal_direction=refusal_dir,
        )
        assert dir_result.refusal_cosine is not None
        assert -1.0 <= dir_result.refusal_cosine <= 1.0

    def test_refusal_cosine_zero_when_refusal_direction_is_zero(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        results = [
            _make_depth_result("a", 0.1),
            _make_depth_result("b", 0.9),
        ]
        zero = ops.zeros((D_MODEL,))
        ops.eval(zero)
        measured = DirectionResult(
            direction=zero,
            layer_index=0,
            cosine_scores=[0.0] * NUM_LAYERS,
            d_model=D_MODEL,
            model_path="test",
        )
        refusal_dir = DirectionResult(
            direction=zero,
            layer_index=0,
            cosine_scores=[0.0] * NUM_LAYERS,
            d_model=D_MODEL,
            model_path="test",
        )

        with patch("vauban.measure.measure", return_value=measured):
            dir_result = depth_direction(
                mock_model,
                mock_tokenizer,
                results,
                refusal_direction=refusal_dir,
            )

        assert dir_result.refusal_cosine == 0.0

    def test_clip_quantile_passed_through(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """depth_direction should accept clip_quantile without error."""
        results = [
            _make_depth_result("a", 0.1),
            _make_depth_result("b", 0.9),
        ]
        dir_result = depth_direction(
            mock_model, mock_tokenizer, results, clip_quantile=0.05,
        )
        assert dir_result.direction.shape == (D_MODEL,)

    def test_fewer_than_2_prompts_raises(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Should raise ValueError with fewer than 2 depth results."""
        results = [_make_depth_result("only_one", 0.5)]
        with pytest.raises(ValueError, match="at least 2"):
            depth_direction(mock_model, mock_tokenizer, results)

    def test_zero_prompts_raises(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Should raise ValueError with empty list."""
        with pytest.raises(ValueError, match="at least 2"):
            depth_direction(mock_model, mock_tokenizer, [])

    def test_identical_dtr_still_splits(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """All prompts with same DTR should still split (median_idx > 0)."""
        results = [
            _make_depth_result("a", 0.5),
            _make_depth_result("b", 0.5),
            _make_depth_result("c", 0.5),
            _make_depth_result("d", 0.5),
        ]
        # median_idx = 2, so shallow=[0:2], deep=[2:]
        dir_result = depth_direction(mock_model, mock_tokenizer, results)
        assert len(dir_result.shallow_prompts) == 2
        assert len(dir_result.deep_prompts) == 2

    def test_two_prompts_one_each_group(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Exactly 2 prompts: one shallow, one deep."""
        results = [
            _make_depth_result("lo", 0.1),
            _make_depth_result("hi", 0.9),
        ]
        dir_result = depth_direction(mock_model, mock_tokenizer, results)
        assert len(dir_result.shallow_prompts) == 1
        assert len(dir_result.deep_prompts) == 1

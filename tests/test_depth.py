"""Tests for vauban.depth: deep-thinking token analysis."""

import mlx.core as mx
import pytest

from tests.conftest import NUM_LAYERS, MockCausalLM, MockTokenizer
from vauban.depth import (
    _jsd,
    _settling_depth,
    depth_generate,
    depth_profile,
)
from vauban.types import DepthConfig


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
        p = mx.softmax(mx.random.normal((32,)))
        mx.eval(p)
        result = _jsd(p, p)
        assert result == pytest.approx(0.0, abs=1e-5)

    def test_non_negative(self) -> None:
        """JSD should always be non-negative."""
        p = mx.softmax(mx.random.normal((32,)))
        q = mx.softmax(mx.random.normal((32,)))
        mx.eval(p, q)
        result = _jsd(p, q)
        assert result >= 0.0

    def test_symmetric(self) -> None:
        """JSD(p, q) should equal JSD(q, p)."""
        p = mx.softmax(mx.random.normal((32,)))
        q = mx.softmax(mx.random.normal((32,)))
        mx.eval(p, q)
        assert _jsd(p, q) == pytest.approx(_jsd(q, p), abs=1e-5)

    def test_different_distributions_positive(self) -> None:
        """JSD of different distributions should be positive."""
        p = mx.array([0.9, 0.05, 0.05])
        q = mx.array([0.05, 0.9, 0.05])
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

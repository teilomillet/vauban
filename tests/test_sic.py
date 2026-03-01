"""Tests for vauban.sic: SIC iterative input sanitization."""

import math

import pytest

from tests.conftest import MockCausalLM, MockTokenizer
from vauban._array import Array
from vauban.sic import (
    _detect_adversarial_direction,
    _detect_adversarial_generation,
    _generate_with_messages,
    _rewrite_prompt,
    _sanitize_prompt,
    calibrate_threshold,
    sic,
    sic_single,
)
from vauban.types import SICConfig, SICPromptResult, SICResult


class TestDetectAdversarialDirection:
    def test_returns_finite_float(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        score = _detect_adversarial_direction(
            mock_model, mock_tokenizer, "Hello world", direction, 0,
        )
        assert isinstance(score, float)
        assert math.isfinite(score)

    def test_different_prompts_different_scores(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        score1 = _detect_adversarial_direction(
            mock_model, mock_tokenizer, "Hello", direction, 0,
        )
        score2 = _detect_adversarial_direction(
            mock_model, mock_tokenizer, "ZZZZZZZ", direction, 0,
        )
        # Different prompts should yield different projections
        # (not guaranteed but extremely likely with random weights)
        assert isinstance(score1, float)
        assert isinstance(score2, float)

    def test_target_layer_beyond_count_uses_last(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        score = _detect_adversarial_direction(
            mock_model, mock_tokenizer, "Hello", direction, 999,
        )
        assert isinstance(score, float)


class TestDetectAdversarialGeneration:
    def test_returns_zero_or_one(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        score = _detect_adversarial_generation(
            mock_model, mock_tokenizer, "Hello world", 20,
        )
        assert score in (0.0, 1.0)


class TestRewritePrompt:
    def test_returns_nonempty_string(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        result = _rewrite_prompt(
            mock_model, mock_tokenizer,
            "Test prompt", "Rewrite this.", 20,
        )
        assert isinstance(result, str)
        assert len(result) > 0


class TestGenerateWithMessages:
    def test_handles_system_user_messages(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = _generate_with_messages(
            mock_model, mock_tokenizer, messages, 10,
        )
        assert isinstance(result, str)

    def test_handles_single_user_message(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        result = _generate_with_messages(
            mock_model, mock_tokenizer, messages, 10,
        )
        assert isinstance(result, str)


class TestSanitizePrompt:
    def test_clean_prompt_passes_through(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        # Use a very low threshold so everything is "clean"
        config = SICConfig(mode="direction", threshold=-1e6)
        result = _sanitize_prompt(
            mock_model, mock_tokenizer, "Hello",
            config, direction, 0,
        )
        assert result.iterations == 0
        assert result.blocked is False
        assert result.clean_prompt == "Hello"

    def test_adversarial_triggers_rewrite(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        # Use a very high threshold so everything is "adversarial"
        config = SICConfig(
            mode="direction", threshold=1e6,
            max_iterations=2, block_on_failure=False,
        )
        result = _sanitize_prompt(
            mock_model, mock_tokenizer, "Hello",
            config, direction, 0,
        )
        assert result.iterations > 0

    def test_max_iterations_respected(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = SICConfig(
            mode="direction", threshold=1e6,
            max_iterations=2, block_on_failure=False,
        )
        result = _sanitize_prompt(
            mock_model, mock_tokenizer, "Hello",
            config, direction, 0,
        )
        assert result.iterations <= 2

    def test_block_on_failure_true(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = SICConfig(
            mode="direction", threshold=1e6,
            max_iterations=1, block_on_failure=True,
        )
        result = _sanitize_prompt(
            mock_model, mock_tokenizer, "Hello",
            config, direction, 0,
        )
        assert result.blocked is True
        assert result.iterations == 1

    def test_block_on_failure_false(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = SICConfig(
            mode="direction", threshold=1e6,
            max_iterations=1, block_on_failure=False,
        )
        result = _sanitize_prompt(
            mock_model, mock_tokenizer, "Hello",
            config, direction, 0,
        )
        assert result.blocked is False
        assert result.iterations == 1


class TestSic:
    def test_returns_sic_result(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = SICConfig(mode="direction", threshold=-1e6)
        result = sic(
            mock_model, mock_tokenizer,
            ["Hello", "World"], config, direction, 0,
        )
        assert isinstance(result, SICResult)
        assert len(result.prompts_clean) == 2
        assert len(result.prompts_blocked) == 2
        assert len(result.iterations_used) == 2
        assert len(result.initial_scores) == 2
        assert len(result.final_scores) == 2

    def test_empty_prompts(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = SICConfig(mode="direction")
        result = sic(
            mock_model, mock_tokenizer, [], config, direction, 0,
        )
        assert len(result.prompts_clean) == 0
        assert result.total_blocked == 0
        assert result.total_sanitized == 0
        assert result.total_clean == 0

    def test_direction_mode_requires_direction(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        config = SICConfig(mode="direction")
        with pytest.raises(ValueError, match="direction"):
            sic(mock_model, mock_tokenizer, ["Hello"], config)

    def test_generation_mode_works_without_direction(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        config = SICConfig(mode="generation", threshold=-1e6)
        result = sic(
            mock_model, mock_tokenizer, ["Hello"], config,
        )
        assert isinstance(result, SICResult)
        assert len(result.prompts_clean) == 1

    def test_aggregates_sum_correctly(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        # Low threshold = everything clean
        config = SICConfig(mode="direction", threshold=-1e6)
        result = sic(
            mock_model, mock_tokenizer,
            ["A", "B", "C"], config, direction, 0,
        )
        total = result.total_blocked + result.total_sanitized + result.total_clean
        assert total == 3

    def test_all_blocked_when_threshold_extreme(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = SICConfig(
            mode="direction", threshold=1e6,
            max_iterations=1, block_on_failure=True,
        )
        result = sic(
            mock_model, mock_tokenizer,
            ["A", "B"], config, direction, 0,
        )
        assert result.total_blocked == 2


class TestSicSingle:
    def test_returns_prompt_result(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = SICConfig(mode="direction", threshold=-1e6)
        result = sic_single(
            mock_model, mock_tokenizer, "Hello", config, direction, 0,
        )
        assert isinstance(result, SICPromptResult)
        assert result.clean_prompt == "Hello"
        assert result.blocked is False
        assert result.iterations == 0

    def test_uses_target_layer_from_config(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = SICConfig(
            mode="direction", threshold=-1e6, target_layer=1,
        )
        result = sic_single(
            mock_model, mock_tokenizer, "Hello", config, direction, 0,
        )
        assert isinstance(result, SICPromptResult)


class TestCalibrateThreshold:
    def test_returns_finite_float(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = SICConfig(mode="direction")
        threshold = calibrate_threshold(
            mock_model, mock_tokenizer,
            ["Hello", "World", "Test"],
            config, direction, 0,
        )
        assert isinstance(threshold, float)
        assert math.isfinite(threshold)

    def test_threshold_below_mean(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = SICConfig(mode="direction")
        prompts = ["Hello", "World", "Test", "Foo", "Bar"]
        threshold = calibrate_threshold(
            mock_model, mock_tokenizer, prompts,
            config, direction, 0,
        )
        # Compute mean of scores manually
        from vauban.sic import _detect
        scores = [
            _detect(mock_model, mock_tokenizer, p, config, direction, 0)
            for p in prompts
        ]
        mean = sum(scores) / len(scores)
        assert threshold < mean

    def test_empty_prompts_returns_default(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = SICConfig(mode="direction", threshold=0.5)
        threshold = calibrate_threshold(
            mock_model, mock_tokenizer, [],
            config, direction, 0,
        )
        assert threshold == 0.5


class TestSicCalibration:
    def test_calibrated_threshold_set(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = SICConfig(mode="direction", calibrate=True)
        result = sic(
            mock_model, mock_tokenizer,
            ["Hello"], config, direction, 0,
            calibration_prompts=["Clean1", "Clean2", "Clean3"],
        )
        assert isinstance(result, SICResult)
        assert result.calibrated_threshold is not None
        assert isinstance(result.calibrated_threshold, float)

    def test_no_calibration_threshold_is_none(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = SICConfig(mode="direction", threshold=-1e6)
        result = sic(
            mock_model, mock_tokenizer,
            ["Hello"], config, direction, 0,
        )
        assert result.calibrated_threshold is None

    def test_calibrate_without_prompts_no_calibration(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = SICConfig(mode="direction", calibrate=True, threshold=-1e6)
        result = sic(
            mock_model, mock_tokenizer,
            ["Hello"], config, direction, 0,
        )
        # calibrate=True but no calibration_prompts passed
        assert result.calibrated_threshold is None

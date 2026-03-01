"""Tests for CAST entry points, forward pass steering, and SVF-aware CAST."""

import pytest

from tests.conftest import D_MODEL, MockCausalLM, MockTokenizer
from vauban import _ops as ops
from vauban._array import Array
from vauban.cast import (
    _cast_forward,
    cast_generate,
    cast_generate_with_messages,
)
from vauban.types import AlphaTier


class TestCastGenerateWithMessages:
    """Tests for the multi-turn message-based entry point."""

    def test_single_user_message(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        messages = [{"role": "user", "content": "test prompt"}]
        result = cast_generate_with_messages(
            mock_model, mock_tokenizer, messages,
            direction, layers=[0], max_tokens=3,
        )
        assert result.prompt == "test prompt"
        assert len(result.text) > 0
        assert result.considered > 0

    def test_multi_turn_uses_last_user_message(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        messages = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "follow-up"},
        ]
        result = cast_generate_with_messages(
            mock_model, mock_tokenizer, messages,
            direction, layers=[0], max_tokens=2,
        )
        assert result.prompt == "follow-up"

    def test_no_user_message_empty_prompt(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        messages = [{"role": "assistant", "content": "hello"}]
        result = cast_generate_with_messages(
            mock_model, mock_tokenizer, messages,
            direction, layers=[0], max_tokens=2,
        )
        assert result.prompt == ""


class TestCastForward:
    """Tests for the _cast_forward steering mechanics."""

    def test_no_intervention_at_high_threshold(
        self,
        mock_model: MockCausalLM,
        direction: Array,
    ) -> None:
        token_ids = ops.array([[1, 2, 3]])
        cache = mock_model.make_cache()

        logits, proj_before, _proj_after, interventions, considered, _, _ = (
            _cast_forward(
                mock_model, token_ids, direction,
                cast_layers=[0], alpha=1.0, threshold=1e9,
                cache=cache,
            )
        )
        ops.eval(logits)
        assert interventions == 0
        assert considered == 1
        assert len(proj_before) == 1

    def test_intervention_at_low_threshold(
        self,
        mock_model: MockCausalLM,
        direction: Array,
    ) -> None:
        token_ids = ops.array([[1, 2, 3]])
        cache = mock_model.make_cache()

        logits, proj_before, proj_after, interventions, considered, _, _ = (
            _cast_forward(
                mock_model, token_ids, direction,
                cast_layers=[0, 1], alpha=1.0, threshold=-1e9,
                cache=cache,
            )
        )
        ops.eval(logits)
        assert interventions == 2
        assert considered == 2
        assert len(proj_before) == 2
        assert len(proj_after) == 2

    def test_non_cast_layers_skipped(
        self,
        mock_model: MockCausalLM,
        direction: Array,
    ) -> None:
        token_ids = ops.array([[1, 2, 3]])
        cache = mock_model.make_cache()

        _, _, _, _, considered, _, _ = _cast_forward(
            mock_model, token_ids, direction,
            cast_layers=[], alpha=1.0, threshold=0.0,
            cache=cache,
        )
        assert considered == 0

    def test_dual_direction_gating(
        self,
        mock_model: MockCausalLM,
        direction: Array,
    ) -> None:
        """Condition direction gates, primary direction steers."""
        token_ids = ops.array([[1, 2]])
        cache = mock_model.make_cache()

        # Zero condition = never triggers
        zero_cond = ops.zeros((D_MODEL,))
        ops.eval(zero_cond)

        _, _, _, interventions, _, _, _ = _cast_forward(
            mock_model, token_ids, direction,
            cast_layers=[0], alpha=1.0, threshold=0.01,
            cache=cache,
            condition_direction=zero_cond,
        )
        assert interventions == 0

    def test_alpha_tiers_applied(
        self,
        mock_model: MockCausalLM,
        direction: Array,
    ) -> None:
        token_ids = ops.array([[1, 2]])
        cache = mock_model.make_cache()
        tiers = [
            AlphaTier(threshold=0.0, alpha=0.1),
            AlphaTier(threshold=100.0, alpha=99.0),
        ]

        logits, _, _, interventions, _, _, _ = _cast_forward(
            mock_model, token_ids, direction,
            cast_layers=[0], alpha=1.0, threshold=-1e9,
            cache=cache,
            alpha_tiers=tiers,
        )
        ops.eval(logits)
        # Should have intervened with the first tier alpha (0.1)
        assert interventions == 1

    def test_displacement_monitoring(
        self,
        mock_model: MockCausalLM,
        direction: Array,
    ) -> None:
        token_ids = ops.array([[1, 2, 3]])
        cache = mock_model.make_cache()

        # Create baseline activations (zeros → any activation will displace)
        baseline = {0: ops.zeros((D_MODEL,))}
        ops.eval(baseline[0])

        _, _, _, _, _, disp_interventions, max_disp = _cast_forward(
            mock_model, token_ids, direction,
            cast_layers=[0], alpha=1.0, threshold=1e9,  # high threshold
            cache=cache,
            baseline_activations=baseline,
            displacement_threshold=0.001,  # very low → will trigger
        )
        # Displacement should trigger: baseline is zeros, any activation displaces
        assert disp_interventions > 0
        assert max_disp > 0.0

    def test_logits_shape(
        self,
        mock_model: MockCausalLM,
        direction: Array,
    ) -> None:
        token_ids = ops.array([[1, 2, 3]])
        cache = mock_model.make_cache()

        logits, _, _, _, _, _, _ = _cast_forward(
            mock_model, token_ids, direction,
            cast_layers=[0], alpha=1.0, threshold=0.0,
            cache=cache,
        )
        ops.eval(logits)
        assert logits.ndim == 3  # (batch, seq, vocab)
        assert logits.shape[0] == 1


class TestCastExternalities:
    """Tests for externality monitoring integration."""

    def test_externality_off_by_default(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        result = cast_generate(
            mock_model, mock_tokenizer, "test",
            direction, layers=[0], max_tokens=2,
        )
        assert result.displacement_interventions == 0
        assert result.max_displacement == pytest.approx(0.0)

    def test_externality_with_baseline(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        baseline = {0: ops.zeros((D_MODEL,))}
        ops.eval(baseline[0])

        result = cast_generate(
            mock_model, mock_tokenizer, "test",
            direction, layers=[0], max_tokens=2,
            baseline_activations=baseline,
            displacement_threshold=0.001,
        )
        # max_displacement should be tracked even if no disp interventions
        assert result.max_displacement >= 0.0

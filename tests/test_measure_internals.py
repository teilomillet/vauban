# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for measure module internals: clip, separation, direction, activations."""

from __future__ import annotations

import pytest

from tests.conftest import D_MODEL, NUM_LAYERS, MockCausalLM, MockTokenizer
from vauban import _ops as ops
from vauban.measure._activations import (
    _clip_activation,
    _collect_activations,
    _collect_per_prompt_activations,
)
from vauban.measure._direction import (
    _best_direction,
    _collect_activations_at_instruction_end,
    _cosine_separation,
)

# ---------------------------------------------------------------------------
# _clip_activation
# ---------------------------------------------------------------------------


class TestClipActivation:
    """Tests for activation winsorization by absolute-value quantile."""

    def test_zero_quantile_is_noop(self) -> None:
        """Quantile 0.0 clips at the maximum, leaving all values unchanged."""
        x = ops.array([1.0, -2.0, 3.0, -4.0, 5.0])
        clipped = _clip_activation(x, quantile=0.0)
        assert ops.allclose(clipped, x)

    def test_clips_extreme_values(self) -> None:
        """A non-zero quantile should reduce the magnitude of outliers."""
        # 10 values: sorted abs = [0,1,2,3,4,5,6,7,8,9]
        x = ops.array([float(i) for i in range(10)])
        clipped = _clip_activation(x, quantile=0.2)
        ops.eval(clipped)
        # Threshold at index int(10 * 0.8) = 8, sorted_abs[8] = 8
        # Value 9 should be clipped to 8
        assert float(clipped[9].item()) <= 8.0 + 1e-6

    def test_symmetric_clipping(self) -> None:
        """Clipping uses [-threshold, threshold], preserving sign symmetry."""
        x = ops.array([-5.0, -3.0, 0.0, 3.0, 5.0])
        clipped = _clip_activation(x, quantile=0.2)
        ops.eval(clipped)
        # clipped[0] should be -clipped[4] (symmetric around zero)
        assert abs(float(clipped[0].item()) + float(clipped[4].item())) < 1e-6

    def test_shape_preserved(self) -> None:
        """Output shape must match input shape."""
        x = ops.random.normal((D_MODEL,))
        ops.eval(x)
        clipped = _clip_activation(x, quantile=0.1)
        assert clipped.shape == x.shape


# ---------------------------------------------------------------------------
# _cosine_separation
# ---------------------------------------------------------------------------


class TestCosineSeparation:
    """Tests for projection-based cosine separation scoring."""

    def test_positive_gap_for_distinct_means(self) -> None:
        """When harmful projects higher than harmless, the gap is positive."""
        direction = ops.array([1.0, 0.0, 0.0, 0.0])
        harmful_mean = ops.array([3.0, 0.0, 0.0, 0.0])
        harmless_mean = ops.array([1.0, 0.0, 0.0, 0.0])
        score = _cosine_separation(harmful_mean, harmless_mean, direction)
        ops.eval(score)
        assert abs(float(score.item()) - 2.0) < 1e-5

    def test_zero_gap_for_identical_means(self) -> None:
        """When harmful and harmless means are the same, the gap is zero."""
        direction = ops.array([1.0, 0.0, 0.0, 0.0])
        same_mean = ops.array([2.0, 1.0, 0.5, 0.0])
        score = _cosine_separation(same_mean, same_mean, direction)
        ops.eval(score)
        assert abs(float(score.item())) < 1e-6

    def test_negative_gap_when_harmless_projects_higher(self) -> None:
        """When harmless projects higher, the gap is negative."""
        direction = ops.array([0.0, 1.0])
        harmful_mean = ops.array([0.0, 1.0])
        harmless_mean = ops.array([0.0, 5.0])
        score = _cosine_separation(harmful_mean, harmless_mean, direction)
        ops.eval(score)
        assert float(score.item()) < 0.0


# ---------------------------------------------------------------------------
# _best_direction
# ---------------------------------------------------------------------------


class TestBestDirection:
    """Tests for best refusal direction selection across layers."""

    def test_returns_unit_vector(self) -> None:
        """The returned direction must be a unit vector."""
        harmful = [ops.array([3.0, 0.0, 0.0, 0.0]) for _ in range(2)]
        harmless = [ops.array([1.0, 0.0, 0.0, 0.0]) for _ in range(2)]
        direction, _layer, _scores = _best_direction(harmful, harmless)
        ops.eval(direction)
        norm = float(ops.linalg.norm(direction).item())
        assert abs(norm - 1.0) < 1e-5

    def test_selects_layer_with_highest_score(self) -> None:
        """The best_layer index should correspond to the maximum cosine score."""
        # Layer 0: small diff along dim 0, layer 1: large diff along dim 0
        harmful = [
            ops.array([1.1, 0.0, 0.0, 0.0]),
            ops.array([5.0, 0.0, 0.0, 0.0]),
        ]
        harmless = [
            ops.array([1.0, 0.0, 0.0, 0.0]),
            ops.array([0.0, 0.0, 0.0, 0.0]),
        ]
        _direction, best_layer, cosine_scores = _best_direction(harmful, harmless)
        assert len(cosine_scores) == 2
        assert best_layer == cosine_scores.index(max(cosine_scores))

    def test_direction_shape_matches_activations(self) -> None:
        """Direction dimensionality must match activation dimensionality."""
        dim = 8
        harmful = [ops.random.normal((dim,)) for _ in range(3)]
        harmless = [ops.random.normal((dim,)) for _ in range(3)]
        ops.eval(*harmful, *harmless)
        direction, _layer, scores = _best_direction(harmful, harmless)
        assert direction.shape == (dim,)
        assert len(scores) == 3


# ---------------------------------------------------------------------------
# _collect_activations
# ---------------------------------------------------------------------------


class TestCollectActivations:
    """Tests for Welford streaming mean activation collection."""

    def test_returns_correct_number_of_layers(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Output list length must match the number of model layers."""
        prompts = ["hello world", "how are you"]
        means = _collect_activations(mock_model, mock_tokenizer, prompts)
        assert len(means) == NUM_LAYERS
        for m in means:
            assert m.shape == (D_MODEL,)

    def test_welford_converges_for_repeated_prompts(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Welford mean of repeated identical prompts equals single result."""
        prompt = "stable test"
        single = _collect_activations(mock_model, mock_tokenizer, [prompt])
        multi = _collect_activations(mock_model, mock_tokenizer, [prompt] * 8)
        for s, m in zip(single, multi, strict=True):
            assert ops.allclose(s, m, atol=1e-5)

    def test_empty_prompts_raises(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """An empty prompt list must raise ValueError."""
        with pytest.raises(ValueError, match="No prompts"):
            _collect_activations(mock_model, mock_tokenizer, [])


# ---------------------------------------------------------------------------
# _collect_per_prompt_activations
# ---------------------------------------------------------------------------


class TestCollectPerPromptActivations:
    """Tests for per-prompt (non-averaged) activation collection."""

    def test_correct_shape(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Each layer should have shape (n_prompts, d_model)."""
        prompts = ["prompt one", "prompt two", "prompt three"]
        per_layer = _collect_per_prompt_activations(
            mock_model, mock_tokenizer, prompts,
        )
        assert len(per_layer) == NUM_LAYERS
        for layer_acts in per_layer:
            assert layer_acts.shape == (3, D_MODEL)

    def test_empty_prompts_raises(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """An empty prompt list must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            _collect_per_prompt_activations(
                mock_model, mock_tokenizer, [],
            )

    def test_mean_matches_collect_activations(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Manual mean of per-prompt activations matches Welford mean."""
        prompts = ["alpha beta", "gamma delta"]
        per_layer = _collect_per_prompt_activations(
            mock_model, mock_tokenizer, prompts,
        )
        welford = _collect_activations(mock_model, mock_tokenizer, prompts)
        for pp, wf in zip(per_layer, welford, strict=True):
            manual_mean = ops.mean(pp, axis=0)
            ops.eval(manual_mean)
            assert ops.allclose(manual_mean, wf, atol=1e-5)


# ---------------------------------------------------------------------------
# _collect_activations_at_instruction_end
# ---------------------------------------------------------------------------


class TestCollectActivationsAtInstructionEnd:
    """Tests for instruction-boundary activation collection."""

    def test_returns_correct_shape(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Each layer should produce a (d_model,) mean activation."""
        prompts = ["hello world", "test prompt"]
        means = _collect_activations_at_instruction_end(
            mock_model, mock_tokenizer, prompts,
        )
        assert len(means) == NUM_LAYERS
        for m in means:
            assert m.shape == (D_MODEL,)

    def test_differs_from_last_token(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Instruction-end activations should differ from last-token activations."""
        prompts = ["this is a longer prompt to ensure token separation"]
        at_end = _collect_activations_at_instruction_end(
            mock_model, mock_tokenizer, prompts,
        )
        at_last = _collect_activations(
            mock_model, mock_tokenizer, prompts, token_position=-1,
        )
        # At least one layer should differ (instruction end != last token)
        any_differ = any(
            not ops.allclose(end_act, last_act, atol=1e-6)
            for end_act, last_act in zip(at_end, at_last, strict=True)
        )
        assert any_differ

    def test_empty_prompts_raises(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """An empty prompt list must raise ValueError."""
        with pytest.raises(ValueError, match="No prompts"):
            _collect_activations_at_instruction_end(
                mock_model, mock_tokenizer, [],
            )

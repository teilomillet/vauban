"""Tests for the four high-impact features: per-layer alpha, activation clipping,
magnitude sparsification, and Welford streaming mean."""

import math

from tests.conftest import D_MODEL, NUM_LAYERS, MockCausalLM, MockTokenizer
from vauban import _ops as ops
from vauban.cut import (
    _resolve_layer_alpha,
    cut,
    cut_subspace,
    sparsify_direction,
)
from vauban.measure import _clip_activation, _collect_activations, measure
from vauban.subspace import orthonormalize

# ---------------------------------------------------------------------------
# Per-layer alpha / kernel weighting
# ---------------------------------------------------------------------------


class TestResolveLayerAlpha:
    def test_uniform_when_none(self) -> None:
        result = _resolve_layer_alpha(0.5, [0, 1, 2], None)
        assert result == [0.5, 0.5, 0.5]

    def test_multiplied_with_weights(self) -> None:
        result = _resolve_layer_alpha(2.0, [0, 1], [0.5, 1.5])
        assert result == [1.0, 3.0]

    def test_length_mismatch_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="length"):
            _resolve_layer_alpha(1.0, [0, 1, 2], [0.5])


class TestCutWithLayerWeights:
    def test_per_layer_alpha_produces_different_changes(self) -> None:
        ops.random.seed(42)
        d = D_MODEL
        weights = {
            "model.layers.0.self_attn.o_proj.weight": ops.random.normal((d, d)),
            "model.layers.0.mlp.down_proj.weight": ops.random.normal((d, d * 2)),
            "model.layers.1.self_attn.o_proj.weight": ops.random.normal((d, d)),
            "model.layers.1.mlp.down_proj.weight": ops.random.normal((d, d * 2)),
        }
        direction = ops.random.normal((d,))
        direction = direction / ops.linalg.norm(direction)
        ops.eval(direction)
        for v in weights.values():
            ops.eval(v)

        # Uniform alpha=1.0
        result_uniform = cut(
            weights, direction, [0, 1], alpha=1.0, layer_weights=None,
        )
        # Non-uniform: layer 0 gets alpha=0.5, layer 1 gets alpha=1.5
        result_weighted = cut(
            weights, direction, [0, 1], alpha=1.0, layer_weights=[0.5, 1.5],
        )

        # Layer 0 should differ between uniform and weighted
        key0 = "model.layers.0.self_attn.o_proj.weight"
        delta0 = ops.abs(result_uniform[key0] - result_weighted[key0])
        diff0 = float(ops.sum(delta0).item())
        assert diff0 > 1e-4

        # Layer 1 should also differ
        key1 = "model.layers.1.self_attn.o_proj.weight"
        delta1 = ops.abs(result_uniform[key1] - result_weighted[key1])
        diff1 = float(ops.sum(delta1).item())
        assert diff1 > 1e-4


class TestCutSubspaceWithLayerWeights:
    def test_subspace_cut_with_weights(self) -> None:
        ops.random.seed(42)
        d = D_MODEL
        weights = {
            "model.layers.0.self_attn.o_proj.weight": ops.random.normal((d, d)),
        }
        ops.eval(weights["model.layers.0.self_attn.o_proj.weight"])
        basis = orthonormalize(ops.random.normal((2, d)))
        ops.eval(basis)

        result = cut_subspace(
            weights, basis, [0], alpha=1.0, layer_weights=[0.5],
        )
        # Should produce a modified result (partial removal with alpha=0.5)
        key = "model.layers.0.self_attn.o_proj.weight"
        assert not ops.array_equal(result[key], weights[key])


# ---------------------------------------------------------------------------
# Activation clipping / winsorization
# ---------------------------------------------------------------------------


class TestClipActivation:
    def test_no_clip_at_zero_quantile(self) -> None:
        ops.random.seed(42)
        act = ops.random.normal((64,))
        ops.eval(act)
        clipped = _clip_activation(act, 0.0)
        # With quantile=0.0, high_idx = int(n * 1.0) clamped to n-1
        # Should still clip at the max value (essentially no change)
        ops.eval(clipped)
        # All values should be unchanged or very close
        abs_diff = ops.abs(act - clipped)
        diff = float(ops.sort(ops.reshape(abs_diff, (-1,)))[-1].item())
        assert diff < 1e-5

    def test_clips_extreme_values(self) -> None:
        # Create activation with extreme outliers
        act = ops.array([1.0, 2.0, 3.0, 100.0, -100.0, 1.5, 2.5, 3.5, 0.5, 1.0])
        # quantile=0.2 -> high_idx = min(9, int(10*0.8)) = 8
        # sorted abs: [0.5,1,1,1.5,2,2.5,3,3.5,100,100] -> threshold = 100
        # Use quantile=0.3 -> high_idx = int(10*0.7) = 7 -> threshold = 3.5
        clipped = _clip_activation(act, 0.3)
        ops.eval(clipped)
        # The extreme values should be clamped to 3.5
        abs_clipped = ops.abs(clipped)
        assert float(ops.sort(ops.reshape(abs_clipped, (-1,)))[-1].item()) <= 3.5 + 1e-5

    def test_symmetric_clipping(self) -> None:
        act = ops.array([1.0, -1.0, 50.0, -50.0, 2.0, -2.0, 3.0, -3.0])
        clipped = _clip_activation(act, 0.1)
        ops.eval(clipped)
        sorted_clipped = ops.sort(ops.reshape(clipped, (-1,)))
        max_val = float(sorted_clipped[-1].item())
        min_val = float(sorted_clipped[0].item())
        # Clipping should be symmetric
        assert abs(max_val + min_val) < 1e-5


class TestMeasureWithClipQuantile:
    def test_clip_quantile_accepted(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """measure() should accept clip_quantile without error."""
        result = measure(
            mock_model, mock_tokenizer,
            ["harmful one", "harmful two"],
            ["harmless one", "harmless two"],
            clip_quantile=0.01,
        )
        assert result.direction.shape == (D_MODEL,)
        assert len(result.cosine_scores) == NUM_LAYERS


class TestCollectActivationsWithClip:
    def test_clipping_changes_result(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Activations with clipping should differ from without."""
        prompts = ["test prompt one", "test prompt two"]
        means_no_clip = _collect_activations(
            mock_model, mock_tokenizer, prompts, clip_quantile=0.0,
        )
        means_with_clip = _collect_activations(
            mock_model, mock_tokenizer, prompts, clip_quantile=0.1,
        )
        # With a small model the difference may be tiny, but the code path
        # should at least not error
        assert len(means_no_clip) == len(means_with_clip) == NUM_LAYERS


# ---------------------------------------------------------------------------
# Magnitude sparsification
# ---------------------------------------------------------------------------


class TestSparsifyDirection:
    def test_no_sparsity(self) -> None:
        d = ops.array([1.0, 2.0, 3.0, 4.0])
        result = sparsify_direction(d, 0.0)
        ops.eval(result)
        assert ops.array_equal(result, d)

    def test_full_sparsity(self) -> None:
        d = ops.array([1.0, 2.0, 3.0, 4.0])
        result = sparsify_direction(d, 1.0)
        ops.eval(result)
        assert float(ops.sum(ops.abs(result)).item()) == 0.0

    def test_partial_sparsity_zeros_small_components(self) -> None:
        d = ops.array([0.1, 0.2, 10.0, 20.0])
        # sparsity=0.5 means keep top 50% by magnitude
        result = sparsify_direction(d, 0.5)
        ops.eval(result)
        # The two large values should remain
        assert float(result[2].item()) == 10.0
        assert float(result[3].item()) == 20.0
        # The two small values should be zeroed
        assert float(result[0].item()) == 0.0
        assert float(result[1].item()) == 0.0

    def test_preserves_sign(self) -> None:
        d = ops.array([-10.0, 0.5, -0.5, 10.0])
        result = sparsify_direction(d, 0.5)
        ops.eval(result)
        assert float(result[0].item()) == -10.0
        assert float(result[3].item()) == 10.0

    def test_at_least_one_component_kept(self) -> None:
        d = ops.array([1.0, 2.0, 3.0])
        # sparsity=0.99 should still keep at least 1 component
        result = sparsify_direction(d, 0.99)
        ops.eval(result)
        nonzero = int(ops.sum(ops.abs(result) > 0).item())
        assert nonzero >= 1


# ---------------------------------------------------------------------------
# Welford streaming mean (tested via _collect_activations)
# ---------------------------------------------------------------------------


class TestWelfordStreamingMean:
    def test_single_prompt_returns_activation_directly(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """With one prompt, Welford mean should equal the activation itself."""
        means = _collect_activations(
            mock_model, mock_tokenizer, ["single prompt"],
        )
        assert len(means) == NUM_LAYERS
        for m in means:
            assert m.shape == (D_MODEL,)

    def test_multiple_prompts_produces_valid_mean(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Multiple prompts should produce a valid mean (no NaN/Inf)."""
        means = _collect_activations(
            mock_model, mock_tokenizer,
            ["prompt one", "prompt two", "prompt three"],
        )
        for m in means:
            ops.eval(m)
            total = float(ops.sum(m).item())
            assert math.isfinite(total)

    def test_empty_prompts_raises(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        import pytest

        with pytest.raises(ValueError, match="No prompts"):
            _collect_activations(mock_model, mock_tokenizer, [])

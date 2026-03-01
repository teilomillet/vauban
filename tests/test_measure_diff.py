"""Tests for vauban.measure._diff: weight-diff direction extraction."""

from tests.conftest import D_MODEL, NUM_HEADS, NUM_LAYERS, VOCAB_SIZE, MockCausalLM
from vauban import _ops as ops
from vauban._forward import svd_stable
from vauban.measure._diff import measure_diff
from vauban.types import DiffResult, DirectionResult


class TestMeasureDiff:
    def test_returns_diff_result(self) -> None:
        """Two different models produce a valid DiffResult."""
        model_a = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        model_b = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model_a.parameters())
        ops.eval(model_b.parameters())

        result = measure_diff(
            model_a, model_b, top_k=3,
            source_model_id="base", target_model_id="aligned",
        )

        assert isinstance(result, DiffResult)
        assert result.basis.shape[0] == 3
        assert len(result.singular_values) == 3
        assert len(result.explained_variance) == NUM_LAYERS
        assert 0 <= result.best_layer < NUM_LAYERS
        assert result.source_model == "base"
        assert result.target_model == "aligned"
        assert len(result.per_layer_bases) == NUM_LAYERS
        assert len(result.per_layer_singular_values) == NUM_LAYERS

    def test_best_direction_compatible(self) -> None:
        """`.best_direction()` returns a valid DirectionResult."""
        model_a = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        model_b = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model_a.parameters())
        ops.eval(model_b.parameters())

        result = measure_diff(
            model_a, model_b, top_k=2,
            source_model_id="base", target_model_id="aligned",
        )
        dr = result.best_direction()

        assert isinstance(dr, DirectionResult)
        assert dr.layer_index == result.best_layer
        assert dr.d_model == result.d_model
        assert dr.model_path == "aligned"
        assert dr.direction.shape[-1] == result.d_model

    def test_identical_models_near_zero(self) -> None:
        """Same model diffed against itself -> near-zero singular values."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())

        result = measure_diff(
            model, model, top_k=2,
            source_model_id="same", target_model_id="same",
        )

        for sv in result.singular_values:
            assert abs(sv) < 1e-5, f"Expected near-zero, got {sv}"

    def test_svd_stable_restores_vector_dtype(self) -> None:
        """MLX SVD should not leak float32 vectors back into model math."""
        matrix = ops.ones((4, 4), dtype=ops.float16)
        u, s, vt = svd_stable(matrix)
        ops.eval(u, s, vt)

        assert u.dtype == matrix.dtype
        assert vt.dtype == matrix.dtype
        assert s.dtype == ops.float32

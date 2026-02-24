"""Tests for vauban.measure: subspace extraction via SVD."""

import mlx.core as mx

from tests.conftest import NUM_LAYERS, MockCausalLM, MockTokenizer
from vauban.measure import measure_subspace


class TestMeasureSubspace:
    def test_returns_valid_subspace_result(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        harmful = ["bad one", "bad two", "bad three"]
        harmless = ["good one", "good two", "good three"]
        result = measure_subspace(
            mock_model, mock_tokenizer, harmful, harmless, top_k=2,
        )

        assert result.basis.shape[0] == 2
        assert result.d_model > 0
        assert 0 <= result.layer_index < NUM_LAYERS
        assert len(result.singular_values) == 2
        assert len(result.explained_variance) == 2
        assert len(result.per_layer_bases) == NUM_LAYERS

    def test_basis_is_orthonormal(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        harmful = ["bad one", "bad two", "bad three"]
        harmless = ["good one", "good two", "good three"]
        result = measure_subspace(
            mock_model, mock_tokenizer, harmful, harmless, top_k=2,
        )

        gram = result.basis @ result.basis.T
        mx.eval(gram)
        identity = mx.eye(2)
        diff = float(mx.linalg.norm(gram - identity).item())
        assert diff < 1e-3

    def test_singular_values_decreasing(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        harmful = ["bad one", "bad two", "bad three"]
        harmless = ["good one", "good two", "good three"]
        result = measure_subspace(
            mock_model, mock_tokenizer, harmful, harmless, top_k=3,
        )

        for i in range(len(result.singular_values) - 1):
            assert result.singular_values[i] >= result.singular_values[i + 1] - 1e-6

    def test_best_direction_compat(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        harmful = ["bad one", "bad two", "bad three"]
        harmless = ["good one", "good two", "good three"]
        result = measure_subspace(
            mock_model, mock_tokenizer, harmful, harmless, top_k=2,
        )

        direction = result.best_direction()
        assert direction.direction.shape == (result.d_model,)
        assert direction.layer_index == result.layer_index

    def test_top_k_capped_by_prompts(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """If top_k > num_prompts, actual k should be capped."""
        harmful = ["bad one", "bad two"]
        harmless = ["good one", "good two"]
        result = measure_subspace(
            mock_model, mock_tokenizer, harmful, harmless, top_k=10,
        )

        # Should get at most min(num_prompts, d_model) directions
        assert result.basis.shape[0] <= 2

    def test_explained_variance_sums_leq_one(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        harmful = ["bad one", "bad two", "bad three"]
        harmless = ["good one", "good two", "good three"]
        result = measure_subspace(
            mock_model, mock_tokenizer, harmful, harmless, top_k=2,
        )

        total = sum(result.explained_variance)
        assert total <= 1.0 + 1e-6

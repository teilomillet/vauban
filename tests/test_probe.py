"""Tests for vauban.probe: probing and steering on tiny model."""

import mlx.core as mx

from tests.conftest import D_MODEL, NUM_LAYERS, MockCausalLM, MockTokenizer
from vauban.probe import multi_probe, probe, steer


class TestProbe:
    def test_returns_per_layer_projections(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: mx.array,
    ) -> None:
        result = probe(mock_model, mock_tokenizer, "test prompt", direction)
        assert result.layer_count == NUM_LAYERS
        assert len(result.projections) == NUM_LAYERS
        assert result.prompt == "test prompt"

    def test_projections_are_finite(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: mx.array,
    ) -> None:
        result = probe(mock_model, mock_tokenizer, "hello", direction)
        for p in result.projections:
            assert isinstance(p, float)
            assert p == p  # not NaN


class TestMultiProbe:
    def test_multiple_directions(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        d1 = mx.random.normal((D_MODEL,))
        d1 = d1 / mx.linalg.norm(d1)
        d2 = mx.random.normal((D_MODEL,))
        d2 = d2 / mx.linalg.norm(d2)
        mx.eval(d1, d2)

        results = multi_probe(
            mock_model, mock_tokenizer, "test",
            {"refusal": d1, "other": d2},
        )
        assert "refusal" in results
        assert "other" in results
        assert results["refusal"].layer_count == NUM_LAYERS


class TestSteer:
    def test_generates_text(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: mx.array,
    ) -> None:
        result = steer(
            mock_model, mock_tokenizer, "test",
            direction, layers=[0], alpha=1.0, max_tokens=5,
        )
        assert len(result.text) > 0
        assert len(result.projections_before) == 5
        assert len(result.projections_after) == 5

    def test_steering_reduces_projection(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: mx.array,
    ) -> None:
        result = steer(
            mock_model, mock_tokenizer, "test",
            direction, layers=[0, 1], alpha=1.0, max_tokens=3,
        )
        # After steering, projections should be reduced (or at least not increased)
        for before, after in zip(
            result.projections_before,
            result.projections_after,
            strict=True,
        ):
            # If before was positive, after should be <= before
            if before > 0:
                assert after <= before + 1e-4

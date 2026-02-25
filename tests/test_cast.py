"""Tests for vauban.cast: conditional runtime activation steering."""

import mlx.core as mx

from tests.conftest import MockCausalLM, MockTokenizer
from vauban.cast import cast_generate


class TestCastGenerate:
    def test_generates_text_and_tracks_counts(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: mx.array,
    ) -> None:
        result = cast_generate(
            mock_model,
            mock_tokenizer,
            "test",
            direction,
            layers=[0, 1],
            alpha=1.0,
            threshold=0.0,
            max_tokens=4,
        )
        assert result.prompt == "test"
        assert len(result.text) > 0
        assert len(result.projections_before) == 4
        assert len(result.projections_after) == 4
        assert result.considered == 8  # 2 layers x 4 decode steps
        assert result.interventions <= result.considered

    def test_high_threshold_disables_interventions(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: mx.array,
    ) -> None:
        result = cast_generate(
            mock_model,
            mock_tokenizer,
            "test",
            direction,
            layers=[0],
            alpha=1.0,
            threshold=1e9,
            max_tokens=3,
        )
        assert result.considered == 3
        assert result.interventions == 0

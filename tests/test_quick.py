"""Tests for vauban.quick — REPL convenience API."""

import mlx.core as mx

from tests.conftest import D_MODEL, NUM_LAYERS, MockCausalLM, MockTokenizer
from vauban.quick import abliterate, measure_direction, probe_prompt, steer_prompt
from vauban.types import DirectionResult


class TestMeasureDirection:
    def test_with_defaults(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """measure_direction with None prompts uses bundled defaults."""
        result = measure_direction(mock_model, mock_tokenizer)
        assert isinstance(result, DirectionResult)
        assert result.direction.shape == (D_MODEL,)

    def test_with_custom_prompts(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        harmful = ["bad prompt 1", "bad prompt 2"]
        harmless = ["good prompt 1", "good prompt 2"]
        result = measure_direction(
            mock_model, mock_tokenizer, harmful, harmless,
        )
        assert isinstance(result, DirectionResult)
        assert result.direction.shape == (D_MODEL,)


class TestProbePrompt:
    def test_with_direction_result(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        dr = measure_direction(mock_model, mock_tokenizer)
        result = probe_prompt(mock_model, mock_tokenizer, "test", dr)
        assert result.layer_count == NUM_LAYERS
        assert len(result.projections) == NUM_LAYERS

    def test_with_raw_array(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: mx.array,
    ) -> None:
        result = probe_prompt(mock_model, mock_tokenizer, "test", direction)
        assert result.layer_count == NUM_LAYERS
        assert len(result.projections) == NUM_LAYERS


class TestSteerPrompt:
    def test_returns_steer_result(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: mx.array,
    ) -> None:
        result = steer_prompt(
            mock_model, mock_tokenizer, "test", direction,
            alpha=1.0, max_tokens=5,
        )
        assert isinstance(result.text, str)
        assert len(result.text) > 0

    def test_accepts_direction_result(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        dr = measure_direction(mock_model, mock_tokenizer)
        result = steer_prompt(
            mock_model, mock_tokenizer, "test", dr, max_tokens=5,
        )
        assert isinstance(result.text, str)


class TestAbliterate:
    def test_writes_output(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        tmp_path: str,
    ) -> None:
        """abliterate should produce a DirectionResult and write files."""
        from pathlib import Path
        from unittest.mock import patch

        out = Path(tmp_path) / "abliterated"
        # export_model is imported inside abliterate(), so patch the source
        with patch("vauban.export.export_model") as mock_export:
            result = abliterate(
                mock_model, mock_tokenizer,
                model_path="mlx-community/test",
                output_dir=str(out),
            )
        assert isinstance(result, DirectionResult)
        assert result.direction.shape == (D_MODEL,)
        mock_export.assert_called_once()
        call_args = mock_export.call_args
        assert str(call_args[0][2]) == str(out)

"""Tests for vauban.quick — REPL convenience API."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tests.conftest import D_MODEL, NUM_LAYERS, MockCausalLM, MockTokenizer
from vauban.quick import (
    abliterate,
    compare,
    evaluate,
    measure_direction,
    probe_prompt,
    scan,
    steer_prompt,
)
from vauban.types import DirectionResult, EvalResult

if TYPE_CHECKING:
    from pathlib import Path

    from vauban._array import Array


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
        direction: Array,
    ) -> None:
        result = probe_prompt(mock_model, mock_tokenizer, "test", direction)
        assert result.layer_count == NUM_LAYERS
        assert len(result.projections) == NUM_LAYERS


class TestSteerPrompt:
    def test_returns_steer_result(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
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


class TestCompare:
    def test_compare_with_fixture_dirs(self, tmp_path: Path) -> None:
        """compare() should return formatted diff string."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        (dir_a / "eval_report.json").write_text(json.dumps({
            "refusal_rate_modified": 0.8,
            "perplexity_modified": 4.0,
            "kl_divergence": 0.02,
        }))
        (dir_b / "eval_report.json").write_text(json.dumps({
            "refusal_rate_modified": 0.1,
            "perplexity_modified": 4.1,
            "kl_divergence": 0.03,
        }))

        result = compare(dir_a, dir_b)
        assert isinstance(result, str)
        assert "DIFF" in result
        assert "eval_report.json" in result

    def test_compare_no_shared_reports(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        result = compare(dir_a, dir_b)
        assert "No shared reports" in result


class TestEvaluate:
    def test_returns_eval_result(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """evaluate() should return an EvalResult."""
        prompts = ["test prompt 1", "test prompt 2"]
        result = evaluate(
            mock_model, mock_model, mock_tokenizer, prompts,
            max_tokens=5,
        )
        assert isinstance(result, EvalResult)
        assert result.num_prompts == 2
        assert 0.0 <= result.refusal_rate_original <= 1.0
        assert 0.0 <= result.refusal_rate_modified <= 1.0

    def test_with_defaults(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """evaluate() with None prompts uses bundled defaults."""
        result = evaluate(
            mock_model, mock_model, mock_tokenizer,
            max_tokens=5,
        )
        assert isinstance(result, EvalResult)
        assert result.num_prompts <= 20


class TestScan:
    def test_scan_with_direction_result(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """scan() should return a SurfaceResult."""
        from unittest.mock import patch

        from vauban.types import SurfaceResult

        dr = measure_direction(mock_model, mock_tokenizer)

        mock_prompts: list[object] = []
        mock_surface_result = SurfaceResult(
            points=[], groups_by_label=[], groups_by_category=[],
            threshold=0.0, total_scanned=0, total_refused=0,
        )
        with (
            patch(
                "vauban.surface.load_surface_prompts",
                return_value=mock_prompts,
            ),
            patch(
                "vauban.surface.map_surface",
                return_value=mock_surface_result,
            ),
        ):
            result = scan(mock_model, mock_tokenizer, dr)
        assert isinstance(result, SurfaceResult)

    def test_scan_with_raw_array(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        """scan() should accept raw Array direction."""
        from unittest.mock import patch

        from vauban.types import SurfaceResult

        mock_surface_result = SurfaceResult(
            points=[], groups_by_label=[], groups_by_category=[],
            threshold=0.0, total_scanned=0, total_refused=0,
        )
        with (
            patch(
                "vauban.surface.load_surface_prompts",
                return_value=[],
            ),
            patch(
                "vauban.surface.map_surface",
                return_value=mock_surface_result,
            ) as mock_map,
        ):
            result = scan(
                mock_model, mock_tokenizer, direction,
                direction_layer=3,
            )
        assert isinstance(result, SurfaceResult)
        # Verify direction_layer was passed
        assert mock_map.call_args[0][4] == 3

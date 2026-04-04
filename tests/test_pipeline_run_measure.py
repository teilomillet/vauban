# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for pipeline measure phase orchestration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from tests.conftest import make_direction_result, make_pipeline_config
from vauban import _ops as ops
from vauban._pipeline._run_measure import run_measure_phase
from vauban._pipeline._run_state import RunState
from vauban.types import (
    CutConfig,
    DBDIResult,
    DetectConfig,
    DetectResult,
    DiffResult,
    DirectionTransferResult,
    EvalConfig,
    MeasureConfig,
    SubspaceBankEntry,
    SubspaceResult,
)

if TYPE_CHECKING:
    from pathlib import Path

    from vauban._array import Array
    from vauban.types import CausalLM, Tokenizer


class _ModelStub:
    """Minimal model sentinel for orchestration tests."""


def _make_array(values: list[float]) -> Array:
    """Build a small backend array."""
    return ops.array(values)


def _make_state(
    tmp_path: Path,
    *,
    measure: MeasureConfig | None = None,
    eval_config: EvalConfig | None = None,
    detect: DetectConfig | None = None,
    cut: CutConfig | None = None,
) -> RunState:
    """Build a minimal ``RunState`` for measure-phase tests."""
    config_path = tmp_path / "configs" / "measure.toml"
    config = make_pipeline_config(
        tmp_path,
        measure=measure if measure is not None else MeasureConfig(),
        eval=eval_config if eval_config is not None else EvalConfig(),
        detect=detect,
        cut=cut if cut is not None else CutConfig(),
    )
    return RunState(
        config_path=config_path,
        config=config,
        model=cast("CausalLM", _ModelStub()),
        tokenizer=cast("Tokenizer", object()),
        t0=0.0,
        verbose=False,
    )


def _make_subspace_result() -> SubspaceResult:
    """Build a minimal subspace result."""
    basis = _make_array([1.0, 0.0, 0.0, 0.0])[None, :]
    return SubspaceResult(
        basis=basis,
        singular_values=[1.0],
        explained_variance=[1.0],
        layer_index=1,
        d_model=4,
        model_path="test-model",
        per_layer_bases=[basis],
    )


def _make_diff_result() -> DiffResult:
    """Build a minimal diff result."""
    basis = _make_array([0.0, 1.0, 0.0, 0.0])[None, :]
    return DiffResult(
        basis=basis,
        singular_values=[1.0],
        explained_variance=[1.0],
        best_layer=2,
        d_model=4,
        source_model="base-model",
        target_model="test-model",
        per_layer_bases=[basis],
        per_layer_singular_values=[[1.0]],
    )


def _make_dbdi_result() -> DBDIResult:
    """Build a minimal DBDI result."""
    return DBDIResult(
        hdd=_make_array([1.0, 0.0, 0.0, 0.0]),
        red=_make_array([0.0, 1.0, 0.0, 0.0]),
        hdd_layer_index=1,
        red_layer_index=2,
        hdd_cosine_scores=[0.2, 0.3],
        red_cosine_scores=[0.4, 0.5],
        d_model=4,
        model_path="test-model",
        layer_types=["global", "sliding"],
    )


def _make_detect_result() -> DetectResult:
    """Build a minimal detect result."""
    return DetectResult(
        hardened=False,
        confidence=0.7,
        effective_rank=1.3,
        cosine_concentration=1.1,
        silhouette_peak=0.8,
        hdd_red_distance=0.2,
        residual_refusal_rate=0.1,
        mean_refusal_position=4.0,
        evidence=["ok"],
    )


def _make_transfer_result(model_id: str) -> DirectionTransferResult:
    """Build a minimal transfer result."""
    return DirectionTransferResult(
        model_id=model_id,
        cosine_separation=0.6,
        best_native_separation=0.8,
        transfer_efficiency=0.75,
        per_layer_cosines=[0.3, 0.5],
    )


class TestRunMeasurePhase:
    """Tests for ``run_measure_phase``."""

    def test_subspace_bank_detect_refusal_phrases_and_measure_only(
        self,
        tmp_path: Path,
    ) -> None:
        measure_cfg = MeasureConfig(
            mode="subspace",
            top_k=2,
            measure_only=True,
            bank=[
                SubspaceBankEntry(
                    name="default-bank",
                    harmful="default",
                    harmless="default",
                ),
                SubspaceBankEntry(
                    name="custom-bank",
                    harmful="harmful_bank.jsonl",
                    harmless="harmless_bank.jsonl",
                ),
            ],
        )
        eval_cfg = EvalConfig(refusal_phrases_path=tmp_path / "phrases.txt")
        state = _make_state(
            tmp_path,
            measure=measure_cfg,
            eval_config=eval_cfg,
            detect=DetectConfig(),
        )
        subspace_result = _make_subspace_result()
        bank_results = {
            "default-bank": subspace_result,
            "custom-bank": subspace_result,
        }

        with (
            patch(
                "vauban.dataset.resolve_prompts",
                side_effect=[["harm-1", "harm-2"], ["safe-1", "safe-2"]],
            ),
            patch(
                "vauban._pipeline._run_measure.load_refusal_phrases",
                return_value=["I cannot"],
            ),
            patch(
                "vauban.detect.detect",
                return_value=_make_detect_result(),
            ),
            patch(
                "vauban.measure.measure_subspace",
                return_value=subspace_result,
            ),
            patch(
                "vauban.measure.load_prompts",
                side_effect=[["bank-harm"], ["bank-safe"]],
            ) as mock_load_prompts,
            patch(
                "vauban.measure.measure_subspace_bank",
                return_value=bank_results,
            ) as mock_bank,
            patch("vauban._ops.save_safetensors") as mock_save,
            patch(
                "vauban._pipeline._run_measure.write_measure_reports",
                return_value=["subspace_report.json"],
            ) as mock_reports,
            patch(
                "vauban._pipeline._run_measure.write_experiment_log",
            ) as mock_experiment_log,
        ):
            result = run_measure_phase(state)

        assert result is True
        assert state.harmful == ["harm-1", "harm-2"]
        assert state.harmless == ["safe-1", "safe-2"]
        assert state.refusal_phrases == ["I cannot"]
        assert state.subspace_result is subspace_result
        assert state.transfer_reports == []
        assert mock_load_prompts.call_count == 2
        assert mock_bank.call_args is not None
        assert mock_bank.call_args[0][2] == [
            ("default-bank", ["harm-1", "harm-2"], ["safe-1", "safe-2"]),
            ("custom-bank", ["bank-harm"], ["bank-safe"]),
        ]
        assert mock_save.call_args is not None
        assert mock_save.call_args[0][0].endswith("subspace_bank.safetensors")
        saved_bank = mock_save.call_args[0][1]
        assert saved_bank == {
            "default-bank": subspace_result.basis,
            "custom-bank": subspace_result.basis,
        }
        detect_payload = json.loads((tmp_path / "detect_report.json").read_text())
        assert detect_payload["hardened"] is False
        assert mock_reports.call_count == 1
        assert mock_experiment_log.call_args is not None
        assert mock_experiment_log.call_args[0][2] == "measure"
        assert mock_experiment_log.call_args[0][3] == ["subspace_report.json"]

    def test_diff_mode_requires_diff_model(self, tmp_path: Path) -> None:
        state = _make_state(
            tmp_path,
            measure=MeasureConfig(mode="diff"),
        )

        with (
            patch(
                "vauban.dataset.resolve_prompts",
                side_effect=[["harm"], ["safe"]],
            ),
            pytest.raises(
                ValueError,
                match=r"measure\.diff_model is required when mode='diff'",
            ),
        ):
            run_measure_phase(state)

    def test_diff_mode_dequantizes_base_runs_transfer_and_measure_only(
        self,
        tmp_path: Path,
    ) -> None:
        measure_cfg = MeasureConfig(
            mode="diff",
            diff_model="base-model",
            top_k=3,
            transfer_models=["transfer-a", "transfer-b"],
            measure_only=True,
        )
        state = _make_state(tmp_path, measure=measure_cfg)
        diff_result = _make_diff_result()
        transfer_model_a = cast("CausalLM", _ModelStub())
        transfer_model_b = cast("CausalLM", _ModelStub())

        with (
            patch(
                "vauban.dataset.resolve_prompts",
                side_effect=[["harm"], ["safe"]],
            ),
            patch(
                "vauban._model_io.load_model",
                side_effect=[
                    (cast("CausalLM", _ModelStub()), cast("Tokenizer", object())),
                    (transfer_model_a, cast("Tokenizer", object())),
                    (transfer_model_b, cast("Tokenizer", object())),
                ],
            ) as mock_load_model,
            patch(
                "vauban.dequantize.is_quantized",
                side_effect=[True, True, False],
            ),
            patch("vauban.dequantize.dequantize_model") as mock_dequantize,
            patch("vauban.measure.measure_diff", return_value=diff_result),
            patch(
                "vauban.transfer.check_direction_transfer",
                side_effect=[
                    _make_transfer_result("transfer-a"),
                    _make_transfer_result("transfer-b"),
                ],
            ) as mock_transfer,
            patch(
                "vauban._pipeline._run_measure.write_measure_reports",
                return_value=["diff_report.json"],
            ) as mock_reports,
            patch(
                "vauban._pipeline._run_measure.write_experiment_log",
            ) as mock_experiment_log,
        ):
            result = run_measure_phase(state)

        assert result is True
        assert mock_load_model.call_count == 3
        assert mock_dequantize.call_count == 2
        assert state.diff_result is diff_result
        assert state.direction_result is not None
        assert state.direction_result.layer_index == diff_result.best_layer
        assert mock_transfer.call_count == 2
        transfer_payload = json.loads((tmp_path / "transfer_report.json").read_text())
        assert [entry["model_id"] for entry in transfer_payload] == [
            "transfer-a",
            "transfer-b",
        ]
        assert state.transfer_reports == ["transfer_report.json"]
        assert mock_reports.call_count == 1
        assert mock_experiment_log.call_args is not None
        assert mock_experiment_log.call_args[0][3] == [
            "diff_report.json",
            "transfer_report.json",
        ]

    @pytest.mark.parametrize(
        ("dbdi_target", "expected_layer", "expected_scores"),
        [
            ("red", 2, [0.4, 0.5]),
            ("both", 2, [0.4, 0.5]),
            ("hdd", 1, [0.2, 0.3]),
        ],
    )
    def test_dbdi_mode_selects_requested_direction(
        self,
        tmp_path: Path,
        dbdi_target: str,
        expected_layer: int,
        expected_scores: list[float],
    ) -> None:
        state = _make_state(
            tmp_path,
            measure=MeasureConfig(mode="dbdi"),
            cut=CutConfig(dbdi_target=dbdi_target),
        )
        dbdi_result = _make_dbdi_result()

        with (
            patch(
                "vauban.dataset.resolve_prompts",
                side_effect=[["harm"], ["safe"]],
            ),
            patch(
                "vauban.measure.measure_dbdi",
                return_value=dbdi_result,
            ),
        ):
            result = run_measure_phase(state)

        assert result is False
        assert state.dbdi_result is dbdi_result
        assert state.direction_result is not None
        assert state.direction_result.layer_index == expected_layer
        assert state.cosine_scores == expected_scores
        assert state.transfer_reports == []

    def test_direction_mode_sets_cosines_and_returns_false(
        self,
        tmp_path: Path,
    ) -> None:
        state = _make_state(tmp_path, measure=MeasureConfig(mode="direction"))
        direction_result = make_direction_result(cosine_scores=[0.6, 0.7])

        with (
            patch(
                "vauban.dataset.resolve_prompts",
                side_effect=[["harm"], ["safe"]],
            ),
            patch(
                "vauban.measure.measure",
                return_value=direction_result,
            ) as mock_measure,
        ):
            result = run_measure_phase(state)

        assert result is False
        assert mock_measure.call_args is not None
        assert state.direction_result is direction_result
        assert state.cosine_scores == [0.6, 0.7]
        assert state.transfer_reports == []

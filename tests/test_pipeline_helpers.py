# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for pipeline helpers: surface_gate_failures, write_measure_reports, etc."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from tests.conftest import make_direction_result, make_pipeline_config
from vauban import _ops as ops
from vauban._pipeline._helpers import (
    is_default_data,
    load_refusal_phrases,
    surface_gate_failures,
    write_arena_card,
    write_measure_reports,
)
from vauban.types import (
    DBDIResult,
    DiffResult,
    GanRoundResult,
    SoftPromptResult,
    SubspaceResult,
    SurfaceConfig,
    TransferEvalResult,
)


# Minimal stub for SurfaceComparison — only the fields tested
@dataclass
class _MockSurfaceComparison:
    worst_cell_refusal_rate_after: float = 0.0
    worst_cell_refusal_rate_delta: float = 0.0
    coverage_score_after: float = 1.0


# ===================================================================
# surface_gate_failures
# ===================================================================


class TestSurfaceGateFailures:
    """surface_gate_failures returns failures for breached thresholds."""

    def test_all_pass(self) -> None:
        surface = SurfaceConfig(
            prompts_path=Path("prompts.jsonl"),
            max_worst_cell_refusal_after=0.5,
            max_worst_cell_refusal_delta=0.2,
            min_coverage_score=0.8,
        )
        comparison = _MockSurfaceComparison(
            worst_cell_refusal_rate_after=0.1,
            worst_cell_refusal_rate_delta=0.05,
            coverage_score_after=0.9,
        )
        failures = surface_gate_failures(surface, comparison)  # type: ignore[arg-type]
        assert failures == []

    def test_single_fail_worst_after(self) -> None:
        surface = SurfaceConfig(
            prompts_path=Path("prompts.jsonl"),
            max_worst_cell_refusal_after=0.3,
        )
        comparison = _MockSurfaceComparison(worst_cell_refusal_rate_after=0.5)
        failures = surface_gate_failures(surface, comparison)  # type: ignore[arg-type]
        assert len(failures) == 1
        assert "worst-cell refusal rate after cut" in failures[0]

    def test_single_fail_delta(self) -> None:
        surface = SurfaceConfig(
            prompts_path=Path("prompts.jsonl"),
            max_worst_cell_refusal_delta=0.1,
        )
        comparison = _MockSurfaceComparison(worst_cell_refusal_rate_delta=0.3)
        failures = surface_gate_failures(surface, comparison)  # type: ignore[arg-type]
        assert len(failures) == 1
        assert "worst-cell refusal-rate delta" in failures[0]

    def test_single_fail_coverage(self) -> None:
        surface = SurfaceConfig(
            prompts_path=Path("prompts.jsonl"),
            min_coverage_score=0.9,
        )
        comparison = _MockSurfaceComparison(coverage_score_after=0.7)
        failures = surface_gate_failures(surface, comparison)  # type: ignore[arg-type]
        assert len(failures) == 1
        assert "coverage score after cut" in failures[0]

    def test_multi_fail(self) -> None:
        surface = SurfaceConfig(
            prompts_path=Path("prompts.jsonl"),
            max_worst_cell_refusal_after=0.1,
            max_worst_cell_refusal_delta=0.05,
            min_coverage_score=0.95,
        )
        comparison = _MockSurfaceComparison(
            worst_cell_refusal_rate_after=0.5,
            worst_cell_refusal_rate_delta=0.3,
            coverage_score_after=0.7,
        )
        failures = surface_gate_failures(surface, comparison)  # type: ignore[arg-type]
        assert len(failures) == 3

    def test_none_thresholds_skip_checks(self) -> None:
        surface = SurfaceConfig(
            prompts_path=Path("prompts.jsonl"),
            max_worst_cell_refusal_after=None,
            max_worst_cell_refusal_delta=None,
            min_coverage_score=None,
        )
        comparison = _MockSurfaceComparison(
            worst_cell_refusal_rate_after=1.0,
            worst_cell_refusal_rate_delta=1.0,
            coverage_score_after=0.0,
        )
        failures = surface_gate_failures(surface, comparison)  # type: ignore[arg-type]
        assert failures == []


# ===================================================================
# write_measure_reports
# ===================================================================


class TestWriteMeasureReports:
    """write_measure_reports writes appropriate JSON for each result type."""

    def test_direction_result(self, tmp_path: Path) -> None:
        config = make_pipeline_config(tmp_path)
        dr = make_direction_result(layer_index=5, cosine_scores=[0.3, 0.5, 0.7])
        reports = write_measure_reports(config, dr, None, None, None)
        assert reports == ["direction_report.json"]
        assert (tmp_path / "direction_report.json").exists()
        data = json.loads((tmp_path / "direction_report.json").read_text())
        assert data["layer_index"] == 5
        assert data["cosine_scores"] == [0.3, 0.5, 0.7]

    def test_subspace_result(self, tmp_path: Path) -> None:
        config = make_pipeline_config(tmp_path)
        basis = ops.random.normal((3, 16))
        ops.eval(basis)
        sr = SubspaceResult(
            basis=basis,
            singular_values=[1.0, 0.5, 0.25],
            explained_variance=[0.5, 0.3, 0.2],
            layer_index=3,
            d_model=16,
            model_path="test-model",
            per_layer_bases=[basis],
        )
        reports = write_measure_reports(config, None, sr, None, None)
        assert reports == ["subspace_report.json"]
        data = json.loads((tmp_path / "subspace_report.json").read_text())
        assert data["layer_index"] == 3

    def test_dbdi_result(self, tmp_path: Path) -> None:
        config = make_pipeline_config(tmp_path)
        hdd = ops.random.normal((16,))
        red = ops.random.normal((16,))
        ops.eval(hdd, red)
        dbdi = DBDIResult(
            hdd=hdd,
            red=red,
            hdd_layer_index=2,
            red_layer_index=5,
            hdd_cosine_scores=[0.3, 0.4],
            red_cosine_scores=[0.5, 0.6],
            d_model=16,
            model_path="test-model",
        )
        reports = write_measure_reports(config, None, None, dbdi, None)
        assert reports == ["dbdi_report.json"]
        data = json.loads((tmp_path / "dbdi_report.json").read_text())
        assert data["hdd_layer_index"] == 2
        assert data["red_layer_index"] == 5

    def test_diff_result(self, tmp_path: Path) -> None:
        config = make_pipeline_config(tmp_path)
        basis = ops.random.normal((2, 16))
        ops.eval(basis)
        diff = DiffResult(
            basis=basis,
            singular_values=[1.0, 0.5],
            explained_variance=[0.6, 0.4],
            best_layer=4,
            d_model=16,
            source_model="base-model",
            target_model="aligned-model",
            per_layer_bases=[basis],
            per_layer_singular_values=[[1.0, 0.5]],
        )
        reports = write_measure_reports(config, None, None, None, diff)
        assert reports == ["diff_report.json"]
        data = json.loads((tmp_path / "diff_report.json").read_text())
        assert data["source_model"] == "base-model"
        assert data["best_layer"] == 4

    def test_all_none_returns_empty(self, tmp_path: Path) -> None:
        config = make_pipeline_config(tmp_path)
        reports = write_measure_reports(config, None, None, None, None)
        assert reports == []


class TestWriteArenaCard:
    """Tests for write_arena_card."""

    def _make_sp_result(
        self,
        token_text: str | None = "adversarial suffix",
        transfer_results: list[TransferEvalResult] | None = None,
        gan_history: list[GanRoundResult] | None = None,
    ) -> SoftPromptResult:
        return SoftPromptResult(
            mode="gcg",
            success_rate=0.75,
            final_loss=1.0,
            loss_history=[2.0, 1.0],
            n_steps=100,
            n_tokens=8,
            embeddings=None,
            token_ids=None,
            token_text=token_text,
            eval_responses=["Sure, here is", "I can help"],
            accessibility_score=0.0,
            per_prompt_losses=[1.0],
            early_stopped=False,
            transfer_results=transfer_results or [],
            defense_eval=None,
            gan_history=[] if gan_history is None else gan_history,
        )

    def test_basic_card_structure(self, tmp_path: Path) -> None:
        """Card has summary, suffix, and per-prompt sections."""
        result = self._make_sp_result()
        card_path = tmp_path / "arena_card.txt"
        write_arena_card(card_path, result, ["prompt1", "prompt2"])

        text = card_path.read_text()
        assert "ARENA SUBMISSION CARD" in text
        assert "SUFFIX (copy-paste ready)" in text
        assert "adversarial suffix" in text
        assert "PER-PROMPT SUBMISSIONS" in text
        assert "prompt1" in text
        assert "prompt2" in text
        assert "Primary ASR: 75.00%" in text

    def test_transfer_results_section(self, tmp_path: Path) -> None:
        """Transfer results section appears when present."""
        tr = TransferEvalResult(
            model_id="other-model",
            success_rate=0.5,
            eval_responses=["resp"],
        )
        result = self._make_sp_result(transfer_results=[tr])
        card_path = tmp_path / "arena_card.txt"
        write_arena_card(card_path, result, ["p1"])

        text = card_path.read_text()
        assert "TRANSFER RESULTS" in text
        assert "other-model" in text

    def test_includes_gan_round_transfer_results(self, tmp_path: Path) -> None:
        round_result = GanRoundResult(
            round_index=1,
            attack_result=SoftPromptResult(
                mode="gcg",
                success_rate=0.5,
                final_loss=1.0,
                loss_history=[1.5, 1.0],
                n_steps=10,
                n_tokens=4,
                embeddings=None,
                token_ids=[1, 2],
                token_text="suffix",
                eval_responses=["response"],
            ),
            defense_result=None,
            attacker_won=True,
            config_snapshot={"n_steps": 10},
            transfer_results=[
                TransferEvalResult(
                    model_id="transfer-model",
                    success_rate=0.25,
                    eval_responses=["resp"],
                ),
            ],
        )
        result = self._make_sp_result(gan_history=[round_result])
        card_path = tmp_path / "arena_card.txt"
        write_arena_card(card_path, result, ["prompt"])

        text = card_path.read_text()
        assert "GAN ROUND HISTORY" in text
        assert "Round 1: WON" in text
        assert "Transfer transfer-model: 25.00%" in text

    def test_no_transfer_section_when_empty(self, tmp_path: Path) -> None:
        """Transfer section omitted when transfer_results is empty."""
        result = self._make_sp_result(transfer_results=[])
        card_path = tmp_path / "arena_card.txt"
        write_arena_card(card_path, result, ["p1"])

        text = card_path.read_text()
        assert "TRANSFER RESULTS" not in text

    def test_no_gan_section_when_empty(self, tmp_path: Path) -> None:
        """GAN section omitted when gan_history is empty."""
        result = self._make_sp_result()
        card_path = tmp_path / "arena_card.txt"
        write_arena_card(card_path, result, ["p1"])

        text = card_path.read_text()
        assert "GAN ROUND HISTORY" not in text


# ===================================================================
# is_default_data
# ===================================================================


class TestIsDefaultData:
    """is_default_data checks if paths match bundled defaults."""

    def test_non_default_paths(self, tmp_path: Path) -> None:
        config = make_pipeline_config(tmp_path)
        # Custom paths should NOT be default
        assert is_default_data(config) is False

    def test_default_paths(self, tmp_path: Path) -> None:
        from vauban.measure import default_prompt_paths

        h_default, hl_default = default_prompt_paths()
        config = make_pipeline_config(
            tmp_path,
            harmful_path=h_default,
            harmless_path=hl_default,
        )
        assert is_default_data(config) is True


# ===================================================================
# load_refusal_phrases
# ===================================================================


class TestLoadRefusalPhrases:
    """load_refusal_phrases delegates to config._validation."""

    def test_delegates_to_validation(self) -> None:
        with patch(
            "vauban.config._validation._load_refusal_phrases",
            return_value=["I cannot", "I'm sorry"],
        ) as mock_fn:
            result = load_refusal_phrases(Path("phrases.txt"))
            mock_fn.assert_called_once_with(Path("phrases.txt"))
            assert result == ["I cannot", "I'm sorry"]

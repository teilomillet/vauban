"""Tests for mode runners B: cast, sic, defend."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import (
    make_direction_result,
    make_early_mode_context,
    make_mock_transformer,
    make_sic_result,
)
from vauban._pipeline._mode_cast import _run_cast_mode
from vauban._pipeline._mode_defend import _run_defend_mode
from vauban._pipeline._mode_sic import _run_sic_mode
from vauban.types import (
    CastConfig,
    CastResult,
    DefenseStackConfig,
    DefenseStackResult,
    EvalConfig,
    SICConfig,
)

if TYPE_CHECKING:
    from pathlib import Path

# ===================================================================
# CAST mode
# ===================================================================


class TestCastMode:
    """Tests for _run_cast_mode."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(ValueError, match="cast config is required"):
            _run_cast_mode(ctx)

    def test_missing_direction_default_source_raises(
        self, tmp_path: Path,
    ) -> None:
        cast_cfg = CastConfig(prompts=["test"])
        ctx = make_early_mode_context(tmp_path, cast=cast_cfg)
        with (
            pytest.raises(
                ValueError, match="direction_result is required",
            ),
            patch(
                "vauban._forward.get_transformer",
                return_value=make_mock_transformer(n_layers=1),
            ),
        ):
            _run_cast_mode(ctx)

    def test_happy_path_with_interventions_metric(
        self, tmp_path: Path,
    ) -> None:
        cast_cfg = CastConfig(prompts=["test1", "test2"])
        dr = make_direction_result()
        ctx = make_early_mode_context(
            tmp_path, direction_result=dr, cast=cast_cfg,
        )

        mock_result = CastResult(
            prompt="test1",
            text="output",
            projections_before=[0.1, 0.2],
            projections_after=[0.05, 0.1],
            interventions=3,
            considered=10,
        )
        with (
            patch(
                "vauban._forward.get_transformer",
                return_value=make_mock_transformer(n_layers=1),
            ),
            patch(
                "vauban.cast.cast_generate",
                return_value=mock_result,
            ),
            patch(
                "vauban._pipeline._mode_cast.finish_mode_run",
            ) as mock_finish,
            patch(
                "vauban._pipeline._mode_cast._cast_to_dict",
                return_value={},
            ),
        ):
            _run_cast_mode(ctx)
            assert (tmp_path / "cast_report.json").exists()
            mock_finish.assert_called_once()
            metadata = mock_finish.call_args[0][3]
            # 3 interventions per prompt * 2 prompts = 6
            assert metadata["interventions"] == 6
            assert metadata["n_prompts"] == 2


# ===================================================================
# SIC mode
# ===================================================================


class TestSicMode:
    """Tests for _run_sic_mode."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(ValueError, match="sic config is required"):
            _run_sic_mode(ctx)

    def test_missing_harmful_raises(self, tmp_path: Path) -> None:
        sic_cfg = SICConfig()
        ctx = make_early_mode_context(tmp_path, harmful=None, sic=sic_cfg)
        with pytest.raises(
            ValueError, match="harmful prompts are required",
        ):
            _run_sic_mode(ctx)

    def test_missing_harmless_raises(self, tmp_path: Path) -> None:
        sic_cfg = SICConfig()
        ctx = make_early_mode_context(tmp_path, harmless=None, sic=sic_cfg)
        with pytest.raises(
            ValueError, match="harmless prompts are required",
        ):
            _run_sic_mode(ctx)

    def test_happy_path_without_direction(self, tmp_path: Path) -> None:
        sic_cfg = SICConfig(mode="generation")
        ctx = make_early_mode_context(tmp_path, sic=sic_cfg)

        mock_result = make_sic_result()
        with (
            patch("vauban.sic.sic", return_value=mock_result),
            patch(
                "vauban._pipeline._mode_sic.finish_mode_run",
            ) as mock_finish,
            patch(
                "vauban._pipeline._mode_sic._sic_to_dict",
                return_value={},
            ),
        ):
            _run_sic_mode(ctx)
            assert (tmp_path / "sic_report.json").exists()
            mock_finish.assert_called_once()

    def test_happy_path_with_direction(self, tmp_path: Path) -> None:
        sic_cfg = SICConfig()
        dr = make_direction_result()
        ctx = make_early_mode_context(
            tmp_path, direction_result=dr, sic=sic_cfg,
        )

        mock_result = make_sic_result()
        with (
            patch("vauban.sic.sic", return_value=mock_result) as mock_sic,
            patch("vauban._pipeline._mode_sic.finish_mode_run"),
            patch(
                "vauban._pipeline._mode_sic._sic_to_dict",
                return_value={},
            ),
        ):
            _run_sic_mode(ctx)
            call_args = mock_sic.call_args[0]
            assert call_args[4] is not None  # direction_vec


# ===================================================================
# Defend mode
# ===================================================================


class TestDefendMode:
    """Tests for _run_defend_mode."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(ValueError, match="defend config is required"):
            _run_defend_mode(ctx)

    def test_missing_harmful_raises(self, tmp_path: Path) -> None:
        defend_cfg = DefenseStackConfig()
        ctx = make_early_mode_context(
            tmp_path, harmful=None, defend=defend_cfg,
        )
        with pytest.raises(
            ValueError, match="harmful prompts are required",
        ):
            _run_defend_mode(ctx)

    def test_happy_path_block_rate(self, tmp_path: Path) -> None:
        defend_cfg = DefenseStackConfig()
        ctx = make_early_mode_context(
            tmp_path,
            harmful=["p1", "p2", "p3", "p4"],
            defend=defend_cfg,
        )

        # 2 blocked, 2 not blocked
        results = [
            DefenseStackResult(
                blocked=True, layer_that_blocked="sic",
            ),
            DefenseStackResult(
                blocked=False, layer_that_blocked=None,
            ),
            DefenseStackResult(
                blocked=True, layer_that_blocked="policy",
            ),
            DefenseStackResult(
                blocked=False, layer_that_blocked=None,
            ),
        ]
        with (
            patch(
                "vauban.defend.defend_content",
                side_effect=results,
            ),
            patch(
                "vauban._pipeline._mode_defend.finish_mode_run",
            ) as mock_finish,
            patch(
                "vauban._pipeline._mode_defend._defend_to_dict",
                return_value={},
            ),
        ):
            _run_defend_mode(ctx)
            assert (tmp_path / "defend_report.json").exists()
            metadata = mock_finish.call_args[0][3]
            assert metadata["block_rate"] == pytest.approx(0.5)


# ===================================================================
# CAST — alternate direction branches
# ===================================================================


class TestCastAlternateBranches:
    """Tests for bank+composition and SVF branches in _run_cast_mode."""

    def test_bank_composition_branch(self, tmp_path: Path) -> None:
        """bank_path + composition triggers compose_direction."""
        cast_cfg = CastConfig(
            prompts=["test"],
            bank_path="bank.safetensors",
            composition={"dir_a": 0.5},
        )
        ctx = make_early_mode_context(tmp_path, cast=cast_cfg)

        mock_result = CastResult(
            prompt="test",
            text="output",
            projections_before=[0.1],
            projections_after=[0.05],
            interventions=1,
            considered=5,
        )
        with (
            patch(
                "vauban._forward.get_transformer",
                return_value=make_mock_transformer(n_layers=1),
            ),
            patch(
                "vauban._compose.load_bank",
                return_value={"dir_a": MagicMock()},
            ) as mock_lb,
            patch(
                "vauban._compose.compose_direction",
                return_value=MagicMock(),
            ) as mock_cd,
            patch(
                "vauban.cast.cast_generate",
                return_value=mock_result,
            ),
            patch(
                "vauban._pipeline._mode_cast.finish_mode_run",
            ),
            patch(
                "vauban._pipeline._mode_cast._cast_to_dict",
                return_value={},
            ),
        ):
            _run_cast_mode(ctx)
            mock_lb.assert_called_once()
            mock_cd.assert_called_once()

    def test_svf_direction_source(self, tmp_path: Path) -> None:
        """direction_source='svf' loads SVF boundary."""
        cast_cfg = CastConfig(
            prompts=["test"],
            direction_source="svf",
            svf_boundary_path="boundary.pt",
        )
        ctx = make_early_mode_context(tmp_path, cast=cast_cfg)

        mock_result = CastResult(
            prompt="test",
            text="output",
            projections_before=[0.1],
            projections_after=[0.05],
            interventions=0,
            considered=5,
        )
        with (
            patch(
                "vauban._forward.get_transformer",
                return_value=make_mock_transformer(n_layers=1),
            ),
            patch(
                "vauban.svf.load_svf_boundary",
                return_value=MagicMock(),
            ) as mock_load,
            patch(
                "vauban.cast.cast_generate_svf",
                return_value=mock_result,
            ) as mock_gen,
            patch(
                "vauban._pipeline._mode_cast.finish_mode_run",
            ),
            patch(
                "vauban._pipeline._mode_cast._cast_to_dict",
                return_value={},
            ),
        ):
            _run_cast_mode(ctx)
            mock_load.assert_called_once()
            mock_gen.assert_called_once()

    def test_svf_missing_path_raises(self, tmp_path: Path) -> None:
        """direction_source='svf' without path raises ValueError."""
        cast_cfg = CastConfig(
            prompts=["test"],
            direction_source="svf",
        )
        ctx = make_early_mode_context(tmp_path, cast=cast_cfg)
        with (
            pytest.raises(ValueError, match="svf_boundary_path"),
            patch(
                "vauban._forward.get_transformer",
                return_value=make_mock_transformer(n_layers=1),
            ),
        ):
            _run_cast_mode(ctx)


# ===================================================================
# SIC — calibration branch
# ===================================================================


class TestSicCalibration:
    """Tests for SIC calibration and custom prompt branches."""

    def test_calibrate_harmless(self, tmp_path: Path) -> None:
        """calibrate=True with calibrate_prompts='harmless' uses harmless."""
        sic_cfg = SICConfig(
            mode="generation",
            calibrate=True,
            calibrate_prompts="harmless",
        )
        ctx = make_early_mode_context(
            tmp_path,
            harmful=["bad1"],
            harmless=["good1", "good2"],
            sic=sic_cfg,
        )

        mock_result = make_sic_result()
        with (
            patch(
                "vauban.sic.sic", return_value=mock_result,
            ) as mock_sic,
            patch("vauban._pipeline._mode_sic.finish_mode_run"),
            patch(
                "vauban._pipeline._mode_sic._sic_to_dict",
                return_value={},
            ),
        ):
            _run_sic_mode(ctx)
            # cal_prompts (arg 6) should be harmless list
            cal_prompts = mock_sic.call_args[0][6]
            assert cal_prompts == ["good1", "good2"]

    def test_custom_eval_prompts_path(self, tmp_path: Path) -> None:
        """eval.prompts_path overrides harmful-based prompts."""
        sic_cfg = SICConfig(mode="generation")
        eval_cfg = EvalConfig(prompts_path="custom.jsonl")
        ctx = make_early_mode_context(
            tmp_path, sic=sic_cfg, eval=eval_cfg,
        )

        mock_result = make_sic_result()
        with (
            patch(
                "vauban.measure.load_prompts",
                return_value=["custom1", "custom2"],
            ) as mock_load,
            patch(
                "vauban.sic.sic", return_value=mock_result,
            ) as mock_sic,
            patch("vauban._pipeline._mode_sic.finish_mode_run"),
            patch(
                "vauban._pipeline._mode_sic._sic_to_dict",
                return_value={},
            ),
        ):
            _run_sic_mode(ctx)
            mock_load.assert_called_once()
            # sic_prompts (arg 2) should be from loaded file
            sic_prompts = mock_sic.call_args[0][2]
            assert sic_prompts == ["custom1", "custom2"]

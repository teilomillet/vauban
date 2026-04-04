# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for mode runners A: probe, steer, depth."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import (
    make_direction_result,
    make_early_mode_context,
    make_mock_transformer,
)
from vauban._pipeline._mode_depth import _run_depth_mode
from vauban._pipeline._mode_probe import _run_probe_mode
from vauban._pipeline._mode_steer import _run_steer_mode
from vauban.types import (
    DepthConfig,
    DirectionResult,
    ProbeConfig,
    ProbeResult,
    SteerConfig,
    SteerResult,
)

if TYPE_CHECKING:
    from pathlib import Path

# ===================================================================
# Probe mode
# ===================================================================


class TestProbeMode:
    """Tests for _run_probe_mode."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(ValueError, match="probe config is required"):
            _run_probe_mode(ctx)

    def test_missing_direction_raises(self, tmp_path: Path) -> None:
        probe_cfg = ProbeConfig(prompts=["test"])
        ctx = make_early_mode_context(tmp_path, probe=probe_cfg)
        with pytest.raises(ValueError, match="direction_result is required"):
            _run_probe_mode(ctx)

    def test_happy_path(self, tmp_path: Path) -> None:
        probe_cfg = ProbeConfig(prompts=["hello", "world"])
        dr = make_direction_result()
        ctx = make_early_mode_context(tmp_path, direction_result=dr, probe=probe_cfg)

        mock_result = ProbeResult(
            projections=[0.1, 0.2],
            layer_count=2,
            prompt="hello",
        )
        with (
            patch("vauban.probe.probe", return_value=mock_result),
            patch(
                "vauban._pipeline._mode_probe.finish_mode_run",
            ) as mock_finish,
            patch(
                "vauban._pipeline._mode_probe._probe_to_dict",
                return_value={},
            ),
        ):
            _run_probe_mode(ctx)
            assert (tmp_path / "probe_report.json").exists()
            mock_finish.assert_called_once()
            call_args = mock_finish.call_args[0]
            assert call_args[1] == "probe"
            assert call_args[3] == {"n_prompts": 2}


# ===================================================================
# Steer mode
# ===================================================================


class TestSteerMode:
    """Tests for _run_steer_mode."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(ValueError, match="steer config is required"):
            _run_steer_mode(ctx)

    def test_missing_direction_default_branch_raises(
        self, tmp_path: Path,
    ) -> None:
        steer_cfg = SteerConfig(prompts=["test"])
        ctx = make_early_mode_context(tmp_path, steer=steer_cfg)
        with (
            pytest.raises(
                ValueError, match="direction_result is required",
            ),
            patch(
                "vauban._forward.get_transformer",
                return_value=make_mock_transformer(),
            ),
        ):
            _run_steer_mode(ctx)

    def test_happy_path_default_direction(self, tmp_path: Path) -> None:
        steer_cfg = SteerConfig(prompts=["test prompt"], max_tokens=10)
        dr = make_direction_result()
        ctx = make_early_mode_context(
            tmp_path, direction_result=dr, steer=steer_cfg,
        )

        mock_result = SteerResult(
            text="output",
            projections_before=[0.1],
            projections_after=[0.05],
        )
        with (
            patch(
                "vauban._forward.get_transformer",
                return_value=make_mock_transformer(),
            ),
            patch("vauban.probe.steer", return_value=mock_result),
            patch(
                "vauban._pipeline._mode_steer.finish_mode_run",
            ) as mock_finish,
            patch(
                "vauban._pipeline._mode_steer._steer_to_dict",
                return_value={},
            ),
        ):
            _run_steer_mode(ctx)
            assert (tmp_path / "steer_report.json").exists()
            mock_finish.assert_called_once()


# ===================================================================
# Depth mode
# ===================================================================


class TestDepthMode:
    """Tests for _run_depth_mode."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(ValueError, match="depth config is required"):
            _run_depth_mode(ctx)

    def test_profile_path(self, tmp_path: Path) -> None:
        """max_tokens=0 triggers depth_profile."""
        depth_cfg = DepthConfig(prompts=["test"], max_tokens=0)
        ctx = make_early_mode_context(tmp_path, depth=depth_cfg)

        mock_depth_result = MagicMock()
        mock_depth_result.tokens = []

        with (
            patch(
                "vauban.depth.depth_profile",
                return_value=mock_depth_result,
            ) as mock_profile,
            patch("vauban.depth.depth_generate") as mock_generate,
            patch(
                "vauban._pipeline._mode_depth.finish_mode_run",
            ),
            patch(
                "vauban._pipeline._mode_depth._depth_to_dict",
                return_value={},
            ),
        ):
            _run_depth_mode(ctx)
            mock_profile.assert_called_once()
            mock_generate.assert_not_called()
            assert (tmp_path / "depth_report.json").exists()

    def test_generate_path(self, tmp_path: Path) -> None:
        """max_tokens>0 triggers depth_generate."""
        depth_cfg = DepthConfig(prompts=["test"], max_tokens=50)
        ctx = make_early_mode_context(tmp_path, depth=depth_cfg)

        mock_depth_result = MagicMock()
        mock_depth_result.tokens = []

        with (
            patch(
                "vauban.depth.depth_generate",
                return_value=mock_depth_result,
            ) as mock_generate,
            patch(
                "vauban.depth.depth_profile",
            ) as mock_profile,
            patch(
                "vauban._pipeline._mode_depth.finish_mode_run",
            ),
            patch(
                "vauban._pipeline._mode_depth._depth_to_dict",
                return_value={},
            ),
        ):
            _run_depth_mode(ctx)
            mock_generate.assert_called_once()
            mock_profile.assert_not_called()

    def test_extract_direction_branch(self, tmp_path: Path) -> None:
        """extract_direction=True with >=2 prompts writes direction."""
        depth_cfg = DepthConfig(
            prompts=["p1", "p2"],
            max_tokens=0,
            extract_direction=True,
        )
        ctx = make_early_mode_context(tmp_path, depth=depth_cfg)

        mock_depth_result = MagicMock()
        mock_depth_result.tokens = []

        import numpy as np

        mock_dir = MagicMock(spec=DirectionResult)
        mock_dir.direction = np.array([0.1, 0.2])

        with (
            patch(
                "vauban.depth.depth_profile",
                return_value=mock_depth_result,
            ),
            patch(
                "vauban.depth.depth_direction",
                return_value=mock_dir,
            ) as mock_dd,
            patch(
                "vauban._pipeline._mode_depth.finish_mode_run",
            ),
            patch(
                "vauban._pipeline._mode_depth._depth_to_dict",
                return_value={},
            ),
            patch(
                "vauban._pipeline._mode_depth._depth_direction_to_dict",
                return_value={"layer": 0},
            ),
            patch("numpy.save"),
            patch(
                "vauban._pipeline._mode_depth.is_default_data",
                return_value=True,
            ),
        ):
            _run_depth_mode(ctx)
            mock_dd.assert_called_once()

    def test_extract_direction_uses_custom_prompts_and_refusal_direction(
        self,
        tmp_path: Path,
    ) -> None:
        """Custom direction prompts should be profiled and compared to refusal."""
        depth_cfg = DepthConfig(
            prompts=["p1", "p2"],
            direction_prompts=["d1", "d2"],
            max_tokens=50,
            extract_direction=True,
            clip_quantile=0.2,
        )
        ctx = make_early_mode_context(tmp_path, depth=depth_cfg)

        prompt_result = MagicMock()
        prompt_result.tokens = []
        direction_result = make_direction_result()

        import numpy as np

        depth_direction_result = MagicMock(spec=DirectionResult)
        depth_direction_result.direction = np.array([0.1, 0.2])

        with (
            patch(
                "vauban.depth.depth_generate",
                side_effect=[
                    prompt_result,
                    prompt_result,
                    prompt_result,
                    prompt_result,
                ],
            ) as mock_generate,
            patch("vauban.depth.depth_profile") as mock_profile,
            patch(
                "vauban._pipeline._mode_depth.is_default_data",
                return_value=False,
            ),
            patch(
                "vauban.dataset.resolve_prompts",
                side_effect=[["harm"], ["safe"]],
            ) as mock_resolve,
            patch(
                "vauban.measure.measure",
                return_value=direction_result,
            ) as mock_measure,
            patch(
                "vauban.depth.depth_direction",
                return_value=depth_direction_result,
            ) as mock_depth_direction,
            patch("numpy.save"),
            patch("vauban._pipeline._mode_depth.finish_mode_run"),
            patch(
                "vauban._pipeline._mode_depth._depth_to_dict",
                return_value={},
            ),
            patch(
                "vauban._pipeline._mode_depth._depth_direction_to_dict",
                return_value={"layer": 0},
            ),
        ):
            _run_depth_mode(ctx)

        assert mock_generate.call_count == 4
        mock_profile.assert_not_called()
        assert mock_resolve.call_count == 2
        mock_measure.assert_called_once()
        kwargs = mock_depth_direction.call_args.kwargs
        assert kwargs["refusal_direction"] is direction_result

    def test_extract_direction_profiles_custom_prompts_when_static(
        self,
        tmp_path: Path,
    ) -> None:
        """Static depth runs should profile custom direction prompts too."""
        depth_cfg = DepthConfig(
            prompts=["p1", "p2"],
            direction_prompts=["d1"],
            max_tokens=0,
            extract_direction=True,
        )
        ctx = make_early_mode_context(tmp_path, depth=depth_cfg)

        depth_result = MagicMock()
        depth_result.tokens = []

        import numpy as np

        depth_direction_result = MagicMock(spec=DirectionResult)
        depth_direction_result.direction = np.array([0.1, 0.2])

        with (
            patch(
                "vauban.depth.depth_profile",
                side_effect=[depth_result, depth_result, depth_result],
            ) as mock_profile,
            patch("vauban.depth.depth_generate") as mock_generate,
            patch(
                "vauban.depth.depth_direction",
                return_value=depth_direction_result,
            ),
            patch(
                "vauban._pipeline._mode_depth.is_default_data",
                return_value=True,
            ),
            patch("numpy.save"),
            patch("vauban._pipeline._mode_depth.finish_mode_run"),
            patch(
                "vauban._pipeline._mode_depth._depth_to_dict",
                return_value={},
            ),
            patch(
                "vauban._pipeline._mode_depth._depth_direction_to_dict",
                return_value={"layer": 0},
            ),
        ):
            _run_depth_mode(ctx)

        assert mock_profile.call_count == 3
        mock_generate.assert_not_called()


# ===================================================================
# Steer — alternate direction branches
# ===================================================================


class TestSteerAlternateBranches:
    """Tests for bank+composition and SVF branches in _run_steer_mode."""

    def test_bank_composition(self, tmp_path: Path) -> None:
        """bank_path + composition triggers compose_direction."""
        steer_cfg = SteerConfig(
            prompts=["test"],
            bank_path="bank.safetensors",
            composition={"dir_a": 0.5},
        )
        ctx = make_early_mode_context(tmp_path, steer=steer_cfg)

        mock_result = SteerResult(
            text="output",
            projections_before=[0.1],
            projections_after=[0.05],
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
            patch("vauban.probe.steer", return_value=mock_result),
            patch(
                "vauban._pipeline._mode_steer.finish_mode_run",
            ),
            patch(
                "vauban._pipeline._mode_steer._steer_to_dict",
                return_value={},
            ),
        ):
            _run_steer_mode(ctx)
            mock_lb.assert_called_once()
            mock_cd.assert_called_once()

    def test_svf_direction_source(self, tmp_path: Path) -> None:
        """direction_source='svf' loads SVF boundary."""
        steer_cfg = SteerConfig(
            prompts=["test"],
            direction_source="svf",
            svf_boundary_path="boundary.pt",
        )
        ctx = make_early_mode_context(tmp_path, steer=steer_cfg)

        mock_result = SteerResult(
            text="output",
            projections_before=[0.1],
            projections_after=[0.05],
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
                "vauban.probe.steer_svf",
                return_value=mock_result,
            ) as mock_steer,
            patch(
                "vauban._pipeline._mode_steer.finish_mode_run",
            ),
            patch(
                "vauban._pipeline._mode_steer._steer_to_dict",
                return_value={},
            ),
        ):
            _run_steer_mode(ctx)
            mock_load.assert_called_once()
            mock_steer.assert_called_once()

    def test_svf_missing_path_raises(self, tmp_path: Path) -> None:
        """direction_source='svf' without svf_boundary_path raises."""
        steer_cfg = SteerConfig(
            prompts=["test"],
            direction_source="svf",
        )
        ctx = make_early_mode_context(tmp_path, steer=steer_cfg)
        with (
            pytest.raises(ValueError, match="svf_boundary_path"),
            patch(
                "vauban._forward.get_transformer",
                return_value=make_mock_transformer(n_layers=1),
            ),
        ):
            _run_steer_mode(ctx)

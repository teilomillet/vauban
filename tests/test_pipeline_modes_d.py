"""Tests for mode runners D: svf, features, circuit, linear_probe,
fusion, repbend, api_eval."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_early_mode_context
from vauban._pipeline._mode_api_eval import _run_api_eval_mode
from vauban._pipeline._mode_circuit import _run_circuit_mode
from vauban._pipeline._mode_features import _run_features_mode
from vauban._pipeline._mode_fusion import _run_fusion_mode
from vauban._pipeline._mode_linear_probe import _run_linear_probe_mode
from vauban._pipeline._mode_repbend import _run_repbend_mode
from vauban._pipeline._mode_svf import _run_svf_mode
from vauban.types import (
    ApiEvalConfig,
    ApiEvalEndpoint,
    CircuitConfig,
    FeaturesConfig,
    FusionConfig,
    FusionGeneration,
    FusionResult,
    LinearProbeConfig,
    RepBendConfig,
    SVFConfig,
    SVFResult,
    TransferEvalResult,
)

# ===================================================================
# SVF mode
# ===================================================================


class TestSvfMode:
    """Tests for _run_svf_mode."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(
            ValueError, match="svf config is required",
        ):
            _run_svf_mode(ctx)

    def test_happy_path(self, tmp_path: Path) -> None:
        svf_cfg = SVFConfig(
            prompts_target=Path("target.jsonl"),
            prompts_opposite=Path("opposite.jsonl"),
            n_epochs=2,
        )
        ctx = make_early_mode_context(tmp_path, svf=svf_cfg)

        mock_boundary = MagicMock()
        mock_result = SVFResult(
            train_loss_history=[1.0, 0.5],
            final_accuracy=0.95,
            per_layer_separation=[0.9, 0.8],
            projection_dim=16,
            hidden_dim=64,
            n_layers_trained=2,
            model_path="test-model",
        )
        with (
            patch(
                "vauban._forward.get_transformer",
            ) as mock_gt,
            patch(
                "vauban.measure.load_prompts",
                return_value=["p1", "p2"],
            ),
            patch(
                "vauban.svf.train_svf_boundary",
                return_value=(mock_boundary, mock_result),
            ),
            patch("vauban.svf.save_svf_boundary"),
            patch(
                "vauban._pipeline._mode_svf.finish_mode_run",
            ) as mock_finish,
        ):
            mock_transformer = MagicMock()
            mock_transformer.embed_tokens.weight.shape = (32000, 16)
            mock_transformer.layers = [MagicMock(), MagicMock()]
            mock_gt.return_value = mock_transformer
            _run_svf_mode(ctx)
            assert (tmp_path / "svf_report.json").exists()
            metadata = mock_finish.call_args[0][3]
            assert metadata["final_accuracy"] == 0.95


# ===================================================================
# Features mode
# ===================================================================


class TestFeaturesMode:
    """Tests for _run_features_mode."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(
            ValueError, match="features config is required",
        ):
            _run_features_mode(ctx)

    def test_happy_path(self, tmp_path: Path) -> None:
        features_cfg = FeaturesConfig(
            prompts_path=Path("prompts.jsonl"),
            layers=[0, 1],
        )
        ctx = make_early_mode_context(tmp_path, features=features_cfg)

        mock_result = MagicMock()
        mock_result.layers = [MagicMock(), MagicMock()]
        mock_result.to_dict.return_value = {"layers": []}
        mock_saes: dict[int, MagicMock] = {
            0: MagicMock(),
            1: MagicMock(),
        }

        with (
            patch(
                "vauban.measure.load_prompts",
                return_value=["p1"],
            ),
            patch(
                "vauban.features.train_sae_multi_layer",
                return_value=(mock_saes, mock_result),
            ),
            patch("vauban.features.save_sae"),
            patch(
                "vauban._pipeline._mode_features.finish_mode_run",
            ) as mock_finish,
        ):
            _run_features_mode(ctx)
            assert (tmp_path / "features_report.json").exists()
            metadata = mock_finish.call_args[0][3]
            assert metadata["n_layers"] == 2


# ===================================================================
# Circuit mode
# ===================================================================


class TestCircuitMode:
    """Tests for _run_circuit_mode."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(
            ValueError, match="circuit config is required",
        ):
            _run_circuit_mode(ctx)

    def test_happy_path(self, tmp_path: Path) -> None:
        circuit_cfg = CircuitConfig(
            clean_prompts=["clean"],
            corrupt_prompts=["corrupt"],
        )
        ctx = make_early_mode_context(tmp_path, circuit=circuit_cfg)

        mock_result = MagicMock()
        mock_result.effects = [MagicMock(), MagicMock()]
        mock_result.to_dict.return_value = {"effects": []}

        with (
            patch(
                "vauban.circuit.trace_circuit",
                return_value=mock_result,
            ),
            patch(
                "vauban._pipeline._mode_circuit.finish_mode_run",
            ) as mock_finish,
        ):
            _run_circuit_mode(ctx)
            assert (tmp_path / "circuit_report.json").exists()
            metadata = mock_finish.call_args[0][3]
            assert metadata["n_effects"] == 2


# ===================================================================
# Linear probe mode
# ===================================================================


class TestLinearProbeMode:
    """Tests for _run_linear_probe_mode."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(
            ValueError, match="linear_probe config is required",
        ):
            _run_linear_probe_mode(ctx)

    def test_happy_path(self, tmp_path: Path) -> None:
        lp_cfg = LinearProbeConfig(layers=[0, 1])
        ctx = make_early_mode_context(tmp_path, linear_probe=lp_cfg)

        mock_result = MagicMock()
        mock_result.layers = [MagicMock(), MagicMock()]
        mock_result.to_dict.return_value = {"layers": []}

        with (
            patch(
                "vauban.linear_probe.train_probe",
                return_value=mock_result,
            ),
            patch(
                "vauban._pipeline._mode_linear_probe.finish_mode_run",
            ) as mock_finish,
        ):
            _run_linear_probe_mode(ctx)
            assert (
                tmp_path / "linear_probe_report.json"
            ).exists()
            metadata = mock_finish.call_args[0][3]
            assert metadata["n_layers"] == 2


# ===================================================================
# Fusion mode
# ===================================================================


class TestFusionMode:
    """Tests for _run_fusion_mode."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(
            ValueError, match="fusion config is required",
        ):
            _run_fusion_mode(ctx)

    def test_happy_path(self, tmp_path: Path) -> None:
        fusion_cfg = FusionConfig(
            harmful_prompts=["bad"],
            benign_prompts=["good"],
        )
        ctx = make_early_mode_context(tmp_path, fusion=fusion_cfg)

        mock_result = FusionResult(
            generations=[
                FusionGeneration(
                    harmful_prompt="bad",
                    benign_prompt="good",
                    output="fused output",
                    layer=5,
                    alpha=0.5,
                ),
            ],
            layer=5,
            alpha=0.5,
        )
        with (
            patch(
                "vauban.fusion.fuse_batch",
                return_value=mock_result,
            ),
            patch(
                "vauban._pipeline._mode_fusion.finish_mode_run",
            ) as mock_finish,
        ):
            _run_fusion_mode(ctx)
            assert (tmp_path / "fusion_report.json").exists()
            metadata = mock_finish.call_args[0][3]
            assert metadata["n_generations"] == 1


# ===================================================================
# RepBend mode
# ===================================================================


class TestRepbendMode:
    """Tests for _run_repbend_mode."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(
            ValueError, match="repbend config is required",
        ):
            _run_repbend_mode(ctx)

    def test_happy_path(self, tmp_path: Path) -> None:
        repbend_cfg = RepBendConfig(layers=[0, 1])
        ctx = make_early_mode_context(tmp_path, repbend=repbend_cfg)

        mock_result = MagicMock()
        mock_result.layers = [MagicMock(), MagicMock()]
        mock_result.to_dict.return_value = {"layers": []}

        with (
            patch(
                "vauban.repbend.repbend",
                return_value=mock_result,
            ),
            patch(
                "vauban._pipeline._mode_repbend.finish_mode_run",
            ) as mock_finish,
        ):
            _run_repbend_mode(ctx)
            assert (tmp_path / "repbend_report.json").exists()
            metadata = mock_finish.call_args[0][3]
            assert metadata["n_layers"] == 2


# ===================================================================
# API eval mode
# ===================================================================


class TestApiEvalMode:
    """Tests for _run_api_eval_mode."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(
            ValueError, match=r"api_eval.*token_text",
        ):
            _run_api_eval_mode(ctx)

    def test_missing_token_text_raises(
        self, tmp_path: Path,
    ) -> None:
        api_cfg = ApiEvalConfig(
            endpoints=[],
            prompts=["test"],
            token_text=None,
        )
        ctx = make_early_mode_context(tmp_path, api_eval=api_cfg)
        with pytest.raises(ValueError, match="token_text"):
            _run_api_eval_mode(ctx)

    def test_happy_path(self, tmp_path: Path) -> None:
        endpoint = ApiEvalEndpoint(
            name="test-ep",
            base_url="http://localhost:8080/v1",
            model="test-model",
            api_key_env="TEST_KEY",
        )
        api_cfg = ApiEvalConfig(
            endpoints=[endpoint],
            prompts=["test1", "test2"],
            token_text="adversarial suffix",
        )
        ctx = make_early_mode_context(tmp_path, api_eval=api_cfg)

        mock_results = [
            TransferEvalResult(
                model_id="endpoint-1",
                success_rate=0.75,
                eval_responses=["r1", "r2"],
            ),
        ]
        with (
            patch(
                "vauban.api_eval.evaluate_suffix_via_api",
                return_value=mock_results,
            ),
            patch(
                "vauban._pipeline._mode_api_eval.finish_mode_run",
            ) as mock_finish,
        ):
            _run_api_eval_mode(ctx)
            assert (
                tmp_path / "api_eval_report.json"
            ).exists()
            metadata = mock_finish.call_args[0][3]
            assert metadata["avg_success_rate"] == pytest.approx(
                0.75,
            )
            assert metadata["n_endpoints"] == 1

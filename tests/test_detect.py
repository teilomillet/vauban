"""Tests for vauban.detect: defense detection pipeline."""

from pathlib import Path

from tests.conftest import D_MODEL, MockCausalLM, MockTokenizer
from vauban import _ops as ops
from vauban.detect import (
    _compute_verdict,
    _dbdi_layer,
    _geometry_layer,
    _margin_layer,
    detect,
)
from vauban.types import DetectConfig, DetectResult

HARMFUL = ["how to do something bad", "another harmful prompt"]
HARMLESS = ["what is the weather", "tell me a joke"]


class TestGeometryLayer:
    def test_returns_valid_signals(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        config = DetectConfig(mode="fast", top_k=2)
        eff_rank, cosine_conc, sil_peak, evidence = _geometry_layer(
            mock_model, mock_tokenizer, HARMFUL, HARMLESS, config,
        )
        assert eff_rank >= 1.0
        assert isinstance(cosine_conc, float)
        assert isinstance(sil_peak, float)
        assert len(evidence) == 3

    def test_evidence_strings_populated(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        config = DetectConfig(mode="fast", top_k=2)
        _, _, _, evidence = _geometry_layer(
            mock_model, mock_tokenizer, HARMFUL, HARMLESS, config,
        )
        assert any("effective_rank" in e for e in evidence)
        assert any("cosine_concentration" in e for e in evidence)
        assert any("silhouette_peak" in e for e in evidence)


class TestDBDILayer:
    def test_returns_valid_distance(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        dist, evidence = _dbdi_layer(
            mock_model, mock_tokenizer, HARMFUL, HARMLESS, 0.0,
        )
        assert isinstance(dist, float)
        assert dist >= 0.0
        assert len(evidence) == 1
        assert "hdd_red_distance" in evidence[0]


class TestDetectFast:
    def test_fast_skips_dbdi_and_abliteration(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        config = DetectConfig(mode="fast", top_k=2)
        result = detect(
            mock_model, mock_tokenizer, HARMFUL, HARMLESS, config,
        )
        assert result.hdd_red_distance is None
        assert result.residual_refusal_rate is None
        assert result.mean_refusal_position is None
        assert isinstance(result.effective_rank, float)
        assert isinstance(result.cosine_concentration, float)
        assert isinstance(result.silhouette_peak, float)


class TestDetectProbe:
    def test_probe_includes_dbdi_not_abliteration(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        config = DetectConfig(mode="probe", top_k=2)
        result = detect(
            mock_model, mock_tokenizer, HARMFUL, HARMLESS, config,
        )
        assert result.hdd_red_distance is not None
        assert result.hdd_red_distance >= 0.0
        assert result.residual_refusal_rate is None
        assert result.mean_refusal_position is None


class TestDetectFull:
    def test_full_includes_all_signals(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        config = DetectConfig(mode="full", top_k=2, max_tokens=5)
        result = detect(
            mock_model, mock_tokenizer, HARMFUL, HARMLESS, config,
        )
        assert result.hdd_red_distance is not None
        assert result.residual_refusal_rate is not None
        assert result.mean_refusal_position is not None
        assert 0.0 <= result.residual_refusal_rate <= 1.0


class TestComputeVerdict:
    def test_no_signals_not_hardened(self) -> None:
        hardened, confidence = _compute_verdict(
            effective_rank_val=1.0,
            cosine_concentration=3.0,
            hdd_red_distance=None,
            residual_refusal_rate=None,
        )
        assert hardened is False
        assert confidence == 0.0

    def test_one_signal_not_hardened(self) -> None:
        hardened, confidence = _compute_verdict(
            effective_rank_val=3.0,  # fires
            cosine_concentration=3.0,
            hdd_red_distance=None,
            residual_refusal_rate=None,
        )
        assert hardened is False
        assert confidence == 0.25

    def test_two_signals_hardened(self) -> None:
        hardened, confidence = _compute_verdict(
            effective_rank_val=3.0,  # fires
            cosine_concentration=1.0,  # fires
            hdd_red_distance=None,
            residual_refusal_rate=None,
        )
        assert hardened is True
        assert confidence == 0.5

    def test_all_signals_hardened(self) -> None:
        hardened, confidence = _compute_verdict(
            effective_rank_val=3.0,
            cosine_concentration=1.0,
            hdd_red_distance=0.8,
            residual_refusal_rate=0.9,
        )
        assert hardened is True
        assert confidence == 1.0

    def test_confidence_scales_correctly(self) -> None:
        _, c0 = _compute_verdict(1.0, 3.0, None, None)
        _, c1 = _compute_verdict(3.0, 3.0, None, None)
        _, c2 = _compute_verdict(3.0, 1.0, None, None)
        _, c3 = _compute_verdict(3.0, 1.0, 0.8, None)
        _, c4 = _compute_verdict(3.0, 1.0, 0.8, 0.9)
        assert c0 == 0.0
        assert c1 == 0.25
        assert c2 == 0.5
        assert c3 == 0.75
        assert c4 == 1.0


class TestDetectResult:
    def test_dataclass_frozen(self) -> None:
        result = DetectResult(
            hardened=False,
            confidence=0.0,
            effective_rank=1.0,
            cosine_concentration=2.0,
            silhouette_peak=0.5,
            hdd_red_distance=None,
            residual_refusal_rate=None,
            mean_refusal_position=None,
            evidence=["test"],
        )
        assert result.hardened is False
        assert result.evidence == ["test"]

    def test_evidence_is_populated_from_detect(
        self, mock_model: MockCausalLM, mock_tokenizer: MockTokenizer,
    ) -> None:
        config = DetectConfig(mode="fast", top_k=2)
        result = detect(
            mock_model, mock_tokenizer, HARMFUL, HARMLESS, config,
        )
        assert len(result.evidence) >= 3


class TestMarginLayer:
    def test_returns_margin_result(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        tmp_path: Path,
    ) -> None:
        # Save a fake direction
        direction = ops.random.normal((D_MODEL,))
        ops.eval(direction)
        dir_path = tmp_path / "test_dir.safetensors"
        ops.save_safetensors(str(dir_path), {"direction": direction})

        config = DetectConfig(
            mode="margin",
            margin_directions=[str(dir_path)],
            margin_alphas=[0.5, 1.0],
            max_tokens=5,
        )
        result = _margin_layer(mock_model, mock_tokenizer, HARMFUL, config)
        assert result.baseline_refusal_rate >= 0.0
        assert len(result.curve) == 2  # 1 direction x 2 alphas
        assert result.curve[0].direction_name == "test_dir"
        assert "test_dir" in result.collapse_alpha
        assert len(result.evidence) >= 2  # baseline + direction summary

    def test_detect_margin_mode(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        tmp_path: Path,
    ) -> None:
        direction = ops.random.normal((D_MODEL,))
        ops.eval(direction)
        dir_path = tmp_path / "safety.safetensors"
        ops.save_safetensors(str(dir_path), {"direction": direction})

        config = DetectConfig(
            mode="margin",
            top_k=2,
            margin_directions=[str(dir_path)],
            margin_alphas=[1.0],
            max_tokens=5,
        )
        result = detect(
            mock_model, mock_tokenizer, HARMFUL, HARMLESS, config,
        )
        assert isinstance(result, DetectResult)
        assert result.margin_result is not None
        assert len(result.margin_result.curve) == 1

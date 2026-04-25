# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for steering awareness detection."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from vauban import _ops as ops
from vauban._forward import LayerRuntime
from vauban._serializers import _awareness_result_to_dict
from vauban.sensitivity import LayerSensitivity, SensitivityProfile
from vauban.types import (
    AwarenessConfig,
    AwarenessLayerResult,
    AwarenessResult,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import MockCausalLM, MockTokenizer
    from vauban._array import Array

from tests.conftest import make_direction_result, make_early_mode_context

# ---------------------------------------------------------------------------
# Test detection logic
# ---------------------------------------------------------------------------


class TestAwarenessDetect:
    """Core detection algorithm tests."""

    def test_identity_layers_not_steered(self) -> None:
        """Baseline == test → no anomaly → not steered."""
        from vauban.awareness import _layer_anomaly_score

        config = AwarenessConfig(prompts=["test"], mode="full")
        layer = LayerSensitivity(
            layer_index=0, directional_gain=1.0,
            correlation=0.5, effective_rank=3.0,
        )

        result, _score = _layer_anomaly_score(layer, layer, config)

        assert not result.anomalous
        assert result.gain_ratio == pytest.approx(1.0)
        assert result.rank_ratio == pytest.approx(1.0)
        assert result.correlation_delta == pytest.approx(0.0)

    def test_amplified_gain_is_steered(self) -> None:
        """Large gain ratio → anomaly flagged."""
        from vauban.awareness import _layer_anomaly_score

        config = AwarenessConfig(
            prompts=["test"], mode="full", gain_ratio_threshold=2.0,
        )
        baseline = LayerSensitivity(
            layer_index=0, directional_gain=1.0,
            correlation=0.5, effective_rank=3.0,
        )
        test = LayerSensitivity(
            layer_index=0, directional_gain=5.0,
            correlation=0.5, effective_rank=3.0,
        )

        result, score = _layer_anomaly_score(baseline, test, config)

        assert result.anomalous
        assert result.gain_ratio == pytest.approx(5.0)
        assert score > 1.0

    def test_rank_collapse_is_steered(self) -> None:
        """Rank drop → anomaly flagged in full mode."""
        from vauban.awareness import _layer_anomaly_score

        config = AwarenessConfig(
            prompts=["test"], mode="full", rank_ratio_threshold=0.5,
        )
        baseline = LayerSensitivity(
            layer_index=0, directional_gain=1.0,
            correlation=0.5, effective_rank=4.0,
        )
        test = LayerSensitivity(
            layer_index=0, directional_gain=1.0,
            correlation=0.5, effective_rank=1.0,
        )

        result, _score = _layer_anomaly_score(baseline, test, config)

        assert result.anomalous
        assert result.rank_ratio == pytest.approx(0.25)

    def test_correlation_shift_is_steered(self) -> None:
        """Correlation delta → anomaly flagged in full mode."""
        from vauban.awareness import _layer_anomaly_score

        config = AwarenessConfig(
            prompts=["test"], mode="full",
            correlation_delta_threshold=0.3,
        )
        baseline = LayerSensitivity(
            layer_index=0, directional_gain=1.0,
            correlation=0.1, effective_rank=3.0,
        )
        test = LayerSensitivity(
            layer_index=0, directional_gain=1.0,
            correlation=0.8, effective_rank=3.0,
        )

        result, _score = _layer_anomaly_score(baseline, test, config)

        assert result.anomalous
        assert result.correlation_delta == pytest.approx(0.7)

    def test_fast_mode_ignores_rank_and_correlation(self) -> None:
        """Fast mode only uses gain ratio for anomaly detection."""
        from vauban.awareness import _layer_anomaly_score

        config = AwarenessConfig(
            prompts=["test"], mode="fast",
            gain_ratio_threshold=2.0,
            rank_ratio_threshold=0.5,
            correlation_delta_threshold=0.3,
        )
        baseline = LayerSensitivity(
            layer_index=0, directional_gain=1.0,
            correlation=0.1, effective_rank=4.0,
        )
        # Rank collapse + correlation shift, but gain is normal
        test = LayerSensitivity(
            layer_index=0, directional_gain=1.5,
            correlation=0.9, effective_rank=0.5,
        )

        result, _score = _layer_anomaly_score(baseline, test, config)

        # Not anomalous in fast mode — gain ratio 1.5 < threshold 2.0
        assert not result.anomalous

    def test_zero_baseline_nonzero_test_uses_epsilon_denominator(self) -> None:
        """Zero baseline gain falls back to epsilon instead of dividing by zero."""
        from vauban.awareness import _layer_anomaly_score

        config = AwarenessConfig(prompts=["test"], mode="full")
        baseline = LayerSensitivity(
            layer_index=0, directional_gain=0.0,
            correlation=0.0, effective_rank=1.0,
        )
        test = LayerSensitivity(
            layer_index=0, directional_gain=2.0,
            correlation=0.0, effective_rank=1.0,
        )

        result, _score = _layer_anomaly_score(baseline, test, config)

        assert result.gain_ratio > 1e9

    def test_zero_baseline_zero_test_defaults_to_neutral_gain_ratio(self) -> None:
        """Two near-zero gains should not manufacture a fake anomaly."""
        from vauban.awareness import _layer_anomaly_score

        config = AwarenessConfig(prompts=["test"], mode="full")
        baseline = LayerSensitivity(
            layer_index=0, directional_gain=0.0,
            correlation=0.0, effective_rank=1.0,
        )
        test = LayerSensitivity(
            layer_index=0, directional_gain=0.0,
            correlation=0.0, effective_rank=1.0,
        )

        result, _score = _layer_anomaly_score(baseline, test, config)

        assert result.gain_ratio == pytest.approx(1.0)

    def test_awareness_detect_builds_evidence_in_full_mode(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        """Full-mode detection should aggregate anomalies into evidence."""
        from vauban.awareness import awareness_detect

        config = AwarenessConfig(
            prompts=["test"],
            mode="full",
            gain_ratio_threshold=2.0,
            rank_ratio_threshold=0.5,
            correlation_delta_threshold=0.2,
            min_anomalous_layers=1,
            confidence_threshold=0.1,
        )
        baseline = SensitivityProfile(
            layers=[
                LayerSensitivity(
                    layer_index=0,
                    directional_gain=1.0,
                    correlation=0.1,
                    effective_rank=4.0,
                ),
                LayerSensitivity(
                    layer_index=1,
                    directional_gain=1.0,
                    correlation=0.2,
                    effective_rank=4.0,
                ),
            ],
            valley_layers=[],
        )
        test_profile = SensitivityProfile(
            layers=[
                LayerSensitivity(
                    layer_index=0,
                    directional_gain=4.0,
                    correlation=0.6,
                    effective_rank=1.0,
                ),
                LayerSensitivity(
                    layer_index=1,
                    directional_gain=1.0,
                    correlation=0.2,
                    effective_rank=4.0,
                ),
            ],
            valley_layers=[],
        )
        token_ids = ops.zeros((1, 2), dtype=ops.int32)
        h = ops.zeros((1, 2, 2))
        runtime = LayerRuntime(
            hidden=h,
            masks=[],
            cache=[],
            per_layer_inputs=[],
        )
        ops.eval(token_ids, h)

        with (
            patch("vauban.awareness.encode_user_prompt", return_value=token_ids),
            patch("vauban.awareness.get_transformer", return_value=object()),
            patch("vauban.awareness.prepare_layer_runtime", return_value=runtime),
            patch(
                "vauban.awareness.compute_sensitivity_profile",
                return_value=test_profile,
            ),
        ):
            result = awareness_detect(
                mock_model,
                mock_tokenizer,
                "test prompt",
                direction,
                config,
                baseline,
            )

        assert result.steered is True
        assert result.anomalous_layers == [0]
        assert result.confidence == pytest.approx(1.0)
        assert result.evidence == [
            "layer 0: gain_ratio=4.00, rank_ratio=0.25, corr_delta=0.50",
        ]

    def test_awareness_detect_fast_mode_uses_fast_profile(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        """Fast-mode detection should use the gain-only profile helper."""
        from vauban.awareness import awareness_detect

        config = AwarenessConfig(
            prompts=["test"],
            mode="fast",
            gain_ratio_threshold=1.5,
            min_anomalous_layers=1,
            confidence_threshold=0.1,
        )
        baseline = SensitivityProfile(
            layers=[
                LayerSensitivity(
                    layer_index=0,
                    directional_gain=1.0,
                    correlation=0.0,
                    effective_rank=1.0,
                ),
            ],
            valley_layers=[],
        )
        test_profile = SensitivityProfile(
            layers=[
                LayerSensitivity(
                    layer_index=0,
                    directional_gain=2.0,
                    correlation=0.0,
                    effective_rank=1.0,
                ),
            ],
            valley_layers=[],
        )
        token_ids = ops.zeros((1, 1), dtype=ops.int32)
        h = ops.zeros((1, 1, 2))
        runtime = LayerRuntime(
            hidden=h,
            masks=[],
            cache=[],
            per_layer_inputs=[],
        )
        ops.eval(token_ids, h)

        with (
            patch("vauban.awareness.encode_user_prompt", return_value=token_ids),
            patch("vauban.awareness.get_transformer", return_value=object()),
            patch("vauban.awareness.prepare_layer_runtime", return_value=runtime),
            patch(
                "vauban.awareness._fast_gain_profile",
                return_value=test_profile,
            ) as mock_fast,
        ):
            result = awareness_detect(
                mock_model,
                mock_tokenizer,
                "test prompt",
                direction,
                config,
                baseline,
            )

        mock_fast.assert_called_once()
        assert result.steered is True

    def test_awareness_detect_empty_profiles_have_zero_confidence(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        """Empty profiles should produce zero confidence and no verdict."""
        from vauban.awareness import awareness_detect

        config = AwarenessConfig(prompts=["test"], mode="full")
        baseline = SensitivityProfile(layers=[], valley_layers=[])
        token_ids = ops.zeros((1, 1), dtype=ops.int32)
        h = ops.zeros((1, 1, 2))
        runtime = LayerRuntime(
            hidden=h,
            masks=[],
            cache=[],
            per_layer_inputs=[],
        )
        ops.eval(token_ids, h)

        with (
            patch("vauban.awareness.encode_user_prompt", return_value=token_ids),
            patch("vauban.awareness.get_transformer", return_value=object()),
            patch("vauban.awareness.prepare_layer_runtime", return_value=runtime),
            patch(
                "vauban.awareness.compute_sensitivity_profile",
                return_value=SensitivityProfile(layers=[], valley_layers=[]),
            ),
        ):
            result = awareness_detect(
                mock_model,
                mock_tokenizer,
                "test prompt",
                direction,
                config,
                baseline,
            )

        assert result.confidence == 0.0
        assert result.steered is False


# ---------------------------------------------------------------------------
# Test fast gain profile
# ---------------------------------------------------------------------------


class TestFastGainProfile:
    """Test _fast_gain_profile returns correct sentinel values."""

    def test_returns_profile_with_sentinels(
        self,
        mock_model: MockCausalLM,
        direction: Array,
    ) -> None:
        """Fast profile has gain populated and sentinel rank/correlation."""
        from vauban._forward import get_transformer, prepare_layer_runtime
        from vauban.awareness import _fast_gain_profile

        transformer = get_transformer(mock_model)
        token_ids = ops.zeros((1, 4), dtype=ops.int32)
        runtime = prepare_layer_runtime(transformer, token_ids)
        ops.eval(runtime.hidden)

        config = AwarenessConfig(prompts=["test"], mode="fast")
        profile = _fast_gain_profile(mock_model, runtime, direction, config)

        n_layers = len(transformer.layers)
        assert len(profile.layers) == n_layers
        assert all(ls.layer_index == i for i, ls in enumerate(profile.layers))
        assert all(isinstance(ls.directional_gain, float) for ls in profile.layers)
        # Sentinel values for fast mode
        assert all(ls.correlation == 0.0 for ls in profile.layers)
        assert all(ls.effective_rank == 1.0 for ls in profile.layers)
        assert profile.valley_layers == []

    def test_calibrate_full_mode_uses_sensitivity_profile(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        """Full-mode calibration should delegate to the sensitivity engine."""
        from vauban.awareness import awareness_calibrate

        config = AwarenessConfig(
            prompts=["test"],
            mode="full",
            n_power_iterations=7,
            fd_epsilon=1e-3,
            valley_window=5,
            top_k_valleys=2,
        )
        token_ids = ops.zeros((1, 1), dtype=ops.int32)
        h = ops.zeros((1, 1, 2))
        runtime = LayerRuntime(
            hidden=h,
            masks=[],
            cache=[],
            per_layer_inputs=[],
        )
        profile = SensitivityProfile(layers=[], valley_layers=[])
        ops.eval(token_ids, h)

        with (
            patch("vauban.awareness.encode_user_prompt", return_value=token_ids),
            patch("vauban.awareness.get_transformer", return_value=object()),
            patch("vauban.awareness.prepare_layer_runtime", return_value=runtime),
            patch(
                "vauban.awareness.compute_sensitivity_profile",
                return_value=profile,
            ) as mock_compute,
        ):
            result = awareness_calibrate(
                mock_model,
                mock_tokenizer,
                "hello",
                direction,
                config,
            )

        assert result is profile
        mock_compute.assert_called_once_with(
            mock_model,
            h,
            None,
            direction,
            n_power_iterations=7,
            fd_epsilon=1e-3,
            valley_window=5,
            top_k_valleys=2,
            runtime=runtime,
        )

    def test_calibrate_fast_mode_uses_fast_gain_profile(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        """Fast-mode calibration should route through the cheap gain profile."""
        from vauban.awareness import awareness_calibrate

        config = AwarenessConfig(prompts=["test"], mode="fast")
        token_ids = ops.zeros((1, 1), dtype=ops.int32)
        h = ops.zeros((1, 1, 2))
        runtime = LayerRuntime(
            hidden=h,
            masks=[],
            cache=[],
            per_layer_inputs=[],
        )
        profile = SensitivityProfile(layers=[], valley_layers=[])
        ops.eval(token_ids, h)

        with (
            patch("vauban.awareness.encode_user_prompt", return_value=token_ids),
            patch("vauban.awareness.get_transformer", return_value=object()),
            patch("vauban.awareness.prepare_layer_runtime", return_value=runtime),
            patch(
                "vauban.awareness._fast_gain_profile",
                return_value=profile,
            ) as mock_fast,
        ):
            result = awareness_calibrate(
                mock_model,
                mock_tokenizer,
                "hello",
                direction,
                config,
            )

        assert result is profile
        mock_fast.assert_called_once_with(
            mock_model,
            runtime,
            direction,
            config,
        )


# ---------------------------------------------------------------------------
# Test parser
# ---------------------------------------------------------------------------


class TestParseAwareness:
    """Test TOML parser for [awareness] section."""

    def test_absent_returns_none(self) -> None:
        """No [awareness] section → None."""
        from vauban.config._parse_awareness import _parse_awareness

        assert _parse_awareness({}) is None

    def test_minimal(self) -> None:
        """Only required fields."""
        from vauban.config._parse_awareness import _parse_awareness

        raw = {"awareness": {"prompts": ["test prompt"]}}
        result = _parse_awareness(raw)

        assert result is not None
        assert result.prompts == ["test prompt"]
        assert result.calibration_prompt == "Hello"
        assert result.mode == "full"
        assert result.gain_ratio_threshold == 2.0

    def test_full_config(self) -> None:
        """All fields specified."""
        from vauban.config._parse_awareness import _parse_awareness

        raw = {
            "awareness": {
                "prompts": ["p1", "p2"],
                "calibration_prompt": "Hi",
                "mode": "fast",
                "n_power_iterations": 10,
                "fd_epsilon": 1e-3,
                "valley_window": 5,
                "top_k_valleys": 2,
                "gain_ratio_threshold": 3.0,
                "rank_ratio_threshold": 0.3,
                "correlation_delta_threshold": 0.5,
                "min_anomalous_layers": 3,
                "confidence_threshold": 0.7,
            },
        }
        result = _parse_awareness(raw)

        assert result is not None
        assert result.prompts == ["p1", "p2"]
        assert result.mode == "fast"
        assert result.n_power_iterations == 10
        assert result.gain_ratio_threshold == 3.0
        assert result.min_anomalous_layers == 3

    def test_empty_prompts_raises(self) -> None:
        """Empty prompts list → error."""
        from vauban.config._parse_awareness import _parse_awareness

        with pytest.raises(ValueError, match="prompts must be non-empty"):
            _parse_awareness({"awareness": {"prompts": []}})

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode value → error."""
        from vauban.config._parse_awareness import _parse_awareness

        with pytest.raises(ValueError, match="mode must be"):
            _parse_awareness({
                "awareness": {"prompts": ["p"], "mode": "invalid"},
            })

    def test_non_table_raises(self) -> None:
        """Non-table [awareness] → error."""
        from vauban.config._parse_awareness import _parse_awareness

        with pytest.raises(TypeError, match="must be a table"):
            _parse_awareness({"awareness": "bad"})


# ---------------------------------------------------------------------------
# Test mode runner
# ---------------------------------------------------------------------------


class TestAwarenessMode:
    """Test early-mode runner for [awareness]."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        """No awareness config → error."""
        from vauban._pipeline._mode_awareness import _run_awareness_mode

        context = make_early_mode_context(tmp_path)

        with pytest.raises(ValueError, match="awareness config is required"):
            _run_awareness_mode(context)

    def test_missing_direction_raises(self, tmp_path: Path) -> None:
        """No direction result → error."""
        from vauban._pipeline._mode_awareness import _run_awareness_mode

        context = make_early_mode_context(
            tmp_path,
            awareness=AwarenessConfig(prompts=["test"]),
        )

        with pytest.raises(ValueError, match="direction_result is required"):
            _run_awareness_mode(context)

    def test_happy_path(self, tmp_path: Path) -> None:
        """Full run with mocked calibrate/detect."""
        from vauban._pipeline._mode_awareness import _run_awareness_mode

        awareness_result = AwarenessResult(
            prompt="test",
            steered=False,
            confidence=0.3,
            anomalous_layers=[],
            layers=[],
            evidence=[],
        )

        baseline_profile = SensitivityProfile(layers=[], valley_layers=[])

        direction_result = make_direction_result()
        context = make_early_mode_context(
            tmp_path,
            direction_result=direction_result,
            awareness=AwarenessConfig(prompts=["test"]),
        )

        with (
            patch(
                "vauban.awareness.awareness_calibrate",
                return_value=baseline_profile,
            ),
            patch(
                "vauban.awareness.awareness_detect",
                return_value=awareness_result,
            ),
        ):
            _run_awareness_mode(context)

        report = tmp_path / "awareness_report.json"
        assert report.exists()


# ---------------------------------------------------------------------------
# Test registry ordering
# ---------------------------------------------------------------------------


class TestAwarenessRegistryOrder:
    """Test that awareness is wired into registries."""

    def test_section_parse_spec_present(self) -> None:
        """Awareness exists in section parse specs."""
        from vauban.config._registry import SECTION_PARSE_SPECS

        sections = [spec.section for spec in SECTION_PARSE_SPECS]
        assert "awareness" in sections
        # Should come after sss
        sss_idx = sections.index("sss")
        awareness_idx = sections.index("awareness")
        assert awareness_idx == sss_idx + 1

    def test_early_mode_spec_present(self) -> None:
        """Awareness exists in early mode specs."""
        from vauban.config._mode_registry import EARLY_MODE_SPECS

        sections = [spec.section for spec in EARLY_MODE_SPECS]
        assert "[awareness]" in sections
        sss_idx = sections.index("[sss]")
        awareness_idx = sections.index("[awareness]")
        assert awareness_idx == sss_idx + 1

    def test_mode_runner_registered(self) -> None:
        """Awareness mode runner exists."""
        from vauban._pipeline._modes import EARLY_MODE_RUNNERS

        assert "awareness" in EARLY_MODE_RUNNERS

    def test_schema_spec_present(self) -> None:
        """Awareness exists in schema specs."""
        from vauban.config._schema import _DATACLASS_SECTION_SPECS

        names = [spec.name for spec in _DATACLASS_SECTION_SPECS]
        assert "awareness" in names


# ---------------------------------------------------------------------------
# Test serializer
# ---------------------------------------------------------------------------


class TestAwarenessSerializer:
    """Test round-trip serialization."""

    def test_round_trip(self) -> None:
        """Serialize and verify structure."""
        result = AwarenessResult(
            prompt="test",
            steered=True,
            confidence=0.75,
            anomalous_layers=[1, 3],
            layers=[
                AwarenessLayerResult(
                    layer_index=0,
                    baseline_gain=1.0,
                    test_gain=1.5,
                    gain_ratio=1.5,
                    baseline_rank=3.0,
                    test_rank=3.0,
                    rank_ratio=1.0,
                    baseline_correlation=0.5,
                    test_correlation=0.5,
                    correlation_delta=0.0,
                    anomalous=False,
                ),
                AwarenessLayerResult(
                    layer_index=1,
                    baseline_gain=1.0,
                    test_gain=5.0,
                    gain_ratio=5.0,
                    baseline_rank=3.0,
                    test_rank=1.0,
                    rank_ratio=0.333,
                    baseline_correlation=0.1,
                    test_correlation=0.8,
                    correlation_delta=0.7,
                    anomalous=True,
                ),
            ],
            evidence=["layer 1: gain_ratio=5.00, rank_ratio=0.33"],
        )

        d = _awareness_result_to_dict(result)

        assert d["prompt"] == "test"
        assert d["steered"] is True
        assert d["confidence"] == 0.75
        assert d["anomalous_layers"] == [1, 3]
        assert len(d["layers"]) == 2  # type: ignore[arg-type]
        assert d["layers"][0]["anomalous"] is False  # type: ignore[index]
        assert d["layers"][1]["anomalous"] is True  # type: ignore[index]
        assert len(d["evidence"]) == 1  # type: ignore[arg-type]

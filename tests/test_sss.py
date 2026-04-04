# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for SSS: sensitivity, generation, config, mode, registry."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from tests.conftest import make_direction_result, make_early_mode_context
from vauban import _ops as ops
from vauban._pipeline._mode_sss import _run_sss_mode
from vauban.config._parse_sss import _parse_sss
from vauban.sensitivity import (
    LayerSensitivity,
    SensitivityProfile,
    compute_sensitivity_profile,
    directional_gain,
    dominant_singular_vector,
    find_compression_valleys,
    jvp_finite_diff,
)
from vauban.sss import _sss_calibrate, sss_generate
from vauban.types import SSSConfig, SSSResult

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import MockCausalLM, MockTokenizer
    from vauban._array import Array


# ===================================================================
# Sensitivity primitives
# ===================================================================


class TestJVPFiniteDiff:
    """Tests for jvp_finite_diff."""

    def test_identity_layer(self) -> None:
        """JVP of identity should return the direction itself."""
        h = ops.random.normal((1, 3, 16))
        v = ops.random.normal((16,))
        ops.eval(h, v)

        def identity(x: Array) -> Array:
            return x

        jv = jvp_finite_diff(identity, h, v, epsilon=1e-4)
        # Should be approximately v broadcast across all positions
        for t in range(3):
            result = jv[0, t, :]
            assert ops.allclose(result, v, atol=1e-2)

    def test_scaling_layer(self) -> None:
        """JVP of 2*x should return 2*v."""
        h = ops.random.normal((1, 2, 8))
        v = ops.random.normal((8,))
        ops.eval(h, v)

        def scale2(x: Array) -> Array:
            return x * 2.0

        jv = jvp_finite_diff(scale2, h, v, epsilon=1e-4)
        expected = v * 2.0
        for t in range(2):
            assert ops.allclose(jv[0, t, :], expected, atol=1e-2)


class TestDirectionalGain:
    """Tests for directional_gain."""

    def test_identity_gain_positive(self) -> None:
        """Gain of identity layer should be > 0."""
        h = ops.random.normal((1, 2, 16))
        d = ops.random.normal((16,))
        d = d / ops.linalg.norm(d)
        ops.eval(h, d)

        def identity(x: Array) -> Array:
            return x

        gain = directional_gain(identity, h, d)
        assert gain > 0.5  # identity gain ≈ 1.0 but JVP norm scales

    def test_scaling_increases_gain(self) -> None:
        """Gain of 3*x should be > gain of identity."""
        h = ops.random.normal((1, 2, 8))
        d = ops.random.normal((8,))
        d = d / ops.linalg.norm(d)
        ops.eval(h, d)

        def identity(x: Array) -> Array:
            return x

        def scale3(x: Array) -> Array:
            return x * 3.0

        gain_id = directional_gain(identity, h, d)
        gain_3x = directional_gain(scale3, h, d)
        assert gain_3x > gain_id * 2.0


class TestDominantSingularVector:
    """Tests for dominant_singular_vector."""

    def test_returns_unit_vector(self) -> None:
        h = ops.random.normal((1, 2, 16))
        ops.eval(h)

        def identity(x: Array) -> Array:
            return x

        dom, sv = dominant_singular_vector(identity, h, n_iter=5)
        norm = float(ops.sqrt(ops.sum(dom * dom)).item())
        assert abs(norm - 1.0) < 1e-4
        assert len(sv) == 5
        assert all(isinstance(v, float) for v in sv)

    def test_shape(self) -> None:
        h = ops.random.normal((1, 3, 8))
        ops.eval(h)

        def identity(x: Array) -> Array:
            return x

        dom, sv = dominant_singular_vector(identity, h, n_iter=3)
        assert dom.shape == (8,)
        assert len(sv) == 3


class TestFindCompressionValleys:
    """Tests for find_compression_valleys."""

    def test_single_valley(self) -> None:
        ranks = [5.0, 4.0, 2.0, 4.0, 5.0]
        valleys = find_compression_valleys(ranks, window=1, top_k=3)
        assert 2 in valleys

    def test_multiple_valleys(self) -> None:
        ranks = [5.0, 2.0, 5.0, 1.0, 5.0]
        valleys = find_compression_valleys(ranks, window=1, top_k=3)
        # 1.0 should come first (lower), then 2.0
        assert valleys[0] == 3
        assert 1 in valleys

    def test_empty_input(self) -> None:
        assert find_compression_valleys([], window=1, top_k=3) == []

    def test_flat_profile(self) -> None:
        ranks = [3.0, 3.0, 3.0, 3.0]
        valleys = find_compression_valleys(ranks, window=1, top_k=3)
        # All equal → all are valleys (each <= min of neighbors)
        assert len(valleys) <= 3

    def test_top_k_limits_output(self) -> None:
        ranks = [5.0, 1.0, 5.0, 1.0, 5.0, 1.0, 5.0]
        valleys = find_compression_valleys(ranks, window=1, top_k=2)
        assert len(valleys) <= 2


# ===================================================================
# Sensitivity profile
# ===================================================================


class TestComputeSensitivityProfile:
    """Tests for compute_sensitivity_profile."""

    def test_returns_profile_with_correct_layer_count(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        from vauban._forward import embed_and_mask, get_transformer

        transformer = get_transformer(mock_model)
        token_ids = ops.zeros((1, 4), dtype=ops.int32)
        h, mask = embed_and_mask(transformer, token_ids)
        ops.eval(h, mask)

        profile = compute_sensitivity_profile(
            mock_model, h, mask, direction,
            n_power_iterations=2, fd_epsilon=1e-3,
        )
        n_layers = len(transformer.layers)
        assert len(profile.layers) == n_layers
        assert all(ls.layer_index == i for i, ls in enumerate(profile.layers))
        assert all(isinstance(ls.directional_gain, float) for ls in profile.layers)
        assert all(-1.0 <= ls.correlation <= 1.0 for ls in profile.layers)
        assert all(ls.effective_rank >= 0 for ls in profile.layers)
        assert isinstance(profile.valley_layers, list)


class TestSSSCalibrate:
    def test_calls_compute_sensitivity_profile(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        profile = SensitivityProfile(
            layers=[LayerSensitivity(0, 1.0, 0.5, 2.0)],
            valley_layers=[0],
        )
        with patch(
            "vauban.sss.compute_sensitivity_profile",
            return_value=profile,
        ) as mock_compute:
            result = _sss_calibrate(
                mock_model,
                mock_tokenizer,
                "calibration",
                direction,
                n_power_iterations=3,
                fd_epsilon=1e-3,
                valley_window=4,
                top_k_valleys=2,
            )

        assert result is profile
        mock_compute.assert_called_once()


# ===================================================================
# SSS generation
# ===================================================================


class TestSSSGenerate:
    """Tests for sss_generate."""

    def test_generates_text(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = SSSConfig(
            prompts=["test"],
            max_tokens=4,
            n_power_iterations=2,
        )
        # Pre-build a minimal profile to skip calibration
        profile = SensitivityProfile(
            layers=[
                LayerSensitivity(
                    layer_index=0,
                    directional_gain=1.0,
                    correlation=0.5,
                    effective_rank=2.0,
                ),
                LayerSensitivity(
                    layer_index=1,
                    directional_gain=0.8,
                    correlation=0.3,
                    effective_rank=3.0,
                ),
            ],
            valley_layers=[0],
        )
        result = sss_generate(
            mock_model, mock_tokenizer, "test",
            direction, config, profile=profile,
        )
        assert isinstance(result, SSSResult)
        assert len(result.text) > 0
        assert result.prompt == "test"
        assert result.seed_layers == [0]
        assert result.seed_strength > 0

    def test_result_fields(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = SSSConfig(
            prompts=["test"],
            max_tokens=3,
            n_power_iterations=2,
        )
        profile = SensitivityProfile(
            layers=[
                LayerSensitivity(0, 1.0, 0.5, 2.0),
                LayerSensitivity(1, 0.8, 0.3, 3.0),
            ],
            valley_layers=[0],
        )
        result = sss_generate(
            mock_model, mock_tokenizer, "test",
            direction, config, profile=profile,
        )
        # per_token_gains has one entry per decode step (max_tokens - 1)
        assert len(result.per_token_gains) <= config.max_tokens - 1
        assert all(isinstance(g, float) for g in result.per_token_gains)

    def test_eos_stops_generation(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        """Generation stops early when the model emits eos_token_id."""
        config = SSSConfig(prompts=["test"], max_tokens=10, n_power_iterations=2)
        profile = SensitivityProfile(
            layers=[
                LayerSensitivity(0, 1.0, 0.5, 2.0),
                LayerSensitivity(1, 0.8, 0.3, 3.0),
            ],
            valley_layers=[0],
        )

        # First run with EOS disabled to discover what tokens the
        # model actually produces (steered output is unpredictable).
        mock_tokenizer.eos_token_id = -1
        result_full = sss_generate(
            mock_model, mock_tokenizer, "test",
            direction, config, profile=profile,
        )
        assert len(result_full.per_token_gains) == config.max_tokens - 1

        # Set eos_token_id to the second generated token (first decode
        # output). Greedy decoding is deterministic, so the second run
        # will hit EOS on the first decode step.
        eos_val = ord(result_full.text[1]) - 65  # decode: chr(t+65)
        mock_tokenizer.eos_token_id = eos_val
        result_short = sss_generate(
            mock_model, mock_tokenizer, "test",
            direction, config, profile=profile,
        )
        assert len(result_short.per_token_gains) < len(result_full.per_token_gains)

    def test_calibrates_when_profile_missing_and_honors_explicit_layers(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = SSSConfig(
            prompts=["test"],
            layers=[1],
            max_tokens=1,
            calibration_prompt="calibration",
            n_power_iterations=1,
        )
        profile = SensitivityProfile(
            layers=[
                LayerSensitivity(0, 1.0, 0.5, 2.0),
                LayerSensitivity(1, 0.8, 0.3, 3.0),
            ],
            valley_layers=[1],
        )

        with patch("vauban.sss._sss_calibrate", return_value=profile) as mock_cal:
            result = sss_generate(
                mock_model,
                mock_tokenizer,
                "test",
                direction,
                config,
                profile=None,
            )

        mock_cal.assert_called_once()
        assert result.seed_layers == [1]
        assert result.prompt == "test"


# ===================================================================
# Config parsing
# ===================================================================


class TestParseSSS:
    """Tests for _parse_sss."""

    def test_absent_returns_none(self) -> None:
        assert _parse_sss({}) is None

    def test_minimal_config(self) -> None:
        raw = {"sss": {"prompts": ["test"]}}
        cfg = _parse_sss(raw)
        assert cfg is not None
        assert cfg.prompts == ["test"]
        assert cfg.alpha == 1.0
        assert cfg.max_tokens == 100
        assert cfg.calibration_prompt == "Hello"

    def test_full_config(self) -> None:
        raw = {
            "sss": {
                "prompts": ["p1", "p2"],
                "layers": [0, 1, 2],
                "alpha": 2.5,
                "max_tokens": 50,
                "calibration_prompt": "Hi there",
                "n_power_iterations": 10,
                "fd_epsilon": 1e-3,
                "seed_floor": 0.05,
                "valley_window": 5,
                "top_k_valleys": 2,
            },
        }
        cfg = _parse_sss(raw)
        assert cfg is not None
        assert cfg.prompts == ["p1", "p2"]
        assert cfg.layers == [0, 1, 2]
        assert cfg.alpha == 2.5
        assert cfg.max_tokens == 50
        assert cfg.calibration_prompt == "Hi there"
        assert cfg.n_power_iterations == 10
        assert cfg.fd_epsilon == 1e-3
        assert cfg.seed_floor == 0.05
        assert cfg.valley_window == 5
        assert cfg.top_k_valleys == 2

    def test_empty_prompts_raises(self) -> None:
        raw = {"sss": {"prompts": []}}
        with pytest.raises(ValueError, match="non-empty"):
            _parse_sss(raw)

    def test_invalid_type_raises(self) -> None:
        raw = {"sss": "not a table"}
        with pytest.raises(TypeError, match="must be a table"):
            _parse_sss(raw)

    def test_invalid_max_tokens_raises(self) -> None:
        raw = {"sss": {"prompts": ["test"], "max_tokens": 0}}
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            _parse_sss(raw)

    def test_invalid_fd_epsilon_raises(self) -> None:
        raw = {"sss": {"prompts": ["test"], "fd_epsilon": -1.0}}
        with pytest.raises(ValueError, match="fd_epsilon must be > 0"):
            _parse_sss(raw)


# ===================================================================
# Mode runner
# ===================================================================


class TestSSSMode:
    """Tests for _run_sss_mode."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(ValueError, match="sss config is required"):
            _run_sss_mode(ctx)

    def test_missing_direction_raises(self, tmp_path: Path) -> None:
        sss_cfg = SSSConfig(prompts=["test"])
        ctx = make_early_mode_context(tmp_path, sss=sss_cfg)
        with pytest.raises(ValueError, match="direction_result is required"):
            _run_sss_mode(ctx)

    def test_happy_path(self, tmp_path: Path) -> None:
        sss_cfg = SSSConfig(prompts=["test prompt"], max_tokens=5)
        dr = make_direction_result()
        ctx = make_early_mode_context(
            tmp_path, direction_result=dr, sss=sss_cfg,
        )

        mock_result = SSSResult(
            text="output",
            prompt="test prompt",
            seed_layers=[0],
            seed_strength=0.5,
            per_token_gains=[0.1],
            projections_before=[0.1],
            projections_after=[0.2],
        )
        mock_profile = SensitivityProfile(
            layers=[LayerSensitivity(0, 1.0, 0.5, 2.0)],
            valley_layers=[0],
        )
        with (
            patch(
                "vauban.sss._sss_calibrate",
                return_value=mock_profile,
            ),
            patch(
                "vauban.sss.sss_generate",
                return_value=mock_result,
            ),
            patch(
                "vauban._pipeline._mode_sss.finish_mode_run",
            ) as mock_finish,
            patch(
                "vauban._pipeline._mode_sss._sss_to_dict",
                return_value={},
            ),
        ):
            _run_sss_mode(ctx)
            assert (tmp_path / "sss_report.json").exists()
            mock_finish.assert_called_once()


# ===================================================================
# Registry ordering
# ===================================================================


class TestSSSRegistryOrder:
    """Tests that SSS is properly registered in section and mode order."""

    def test_section_parse_spec_includes_sss(self) -> None:
        from vauban.config._registry import SECTION_PARSE_SPECS

        sections = [s.section for s in SECTION_PARSE_SPECS]
        assert "sss" in sections
        # sss should come after steer
        steer_idx = sections.index("steer")
        sss_idx = sections.index("sss")
        assert sss_idx > steer_idx

    def test_early_mode_spec_includes_sss(self) -> None:
        from vauban.config._mode_registry import EARLY_MODE_SPECS

        modes = [s.section for s in EARLY_MODE_SPECS]
        assert "[sss]" in modes
        # sss should come after steer, before cast
        steer_idx = modes.index("[steer]")
        sss_idx = modes.index("[sss]")
        cast_idx = modes.index("[cast]")
        assert steer_idx < sss_idx < cast_idx

    def test_early_mode_runners_includes_sss(self) -> None:
        from vauban._pipeline._modes import EARLY_MODE_RUNNERS

        assert "sss" in EARLY_MODE_RUNNERS

    def test_schema_includes_sss(self) -> None:
        from vauban.config._schema import _DATACLASS_SECTION_SPECS

        names = [s.name for s in _DATACLASS_SECTION_SPECS]
        assert "sss" in names


# ===================================================================
# Serializer
# ===================================================================


class TestSSSSerializer:
    """Tests for _sss_to_dict."""

    def test_round_trip(self) -> None:
        from vauban._serializers import _sss_to_dict

        result = SSSResult(
            text="hello world",
            prompt="test",
            seed_layers=[0, 1],
            seed_strength=0.5,
            per_token_gains=[0.1, 0.2],
            projections_before=[0.3, 0.4],
            projections_after=[0.5, 0.6],
        )
        d = _sss_to_dict(result)
        assert d["text"] == "hello world"
        assert d["prompt"] == "test"
        assert d["seed_layers"] == [0, 1]
        assert d["seed_strength"] == 0.5
        assert d["per_token_gains"] == [0.1, 0.2]
        assert d["projections_before"] == [0.3, 0.4]
        assert d["projections_after"] == [0.5, 0.6]

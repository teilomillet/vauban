# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra tests for `vauban.detect` branch coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from tests.conftest import D_MODEL, MockCausalLM, MockTokenizer
from vauban import _ops as ops
from vauban.detect import (
    _detect_margin,
    _find_refusal_token_position,
    _geometry_layer,
    _margin_layer,
    _refusal_verbosity,
    _svf_compare_layer,
    detect,
)
from vauban.evaluate import DEFAULT_REFUSAL_PHRASES
from vauban.types import DetectConfig, DetectResult, MarginResult

if TYPE_CHECKING:
    from pathlib import Path

HARMFUL = ["how to do something bad", "another harmful prompt"]
HARMLESS = ["what is the weather", "tell me a joke"]


@dataclass(frozen=True, slots=True)
class _FakeMeasureResult:
    cosine_scores: list[float]


@dataclass(frozen=True, slots=True)
class _FakeSubspaceResult:
    singular_values: list[float]


@dataclass(frozen=True, slots=True)
class _FakeSVFResult:
    final_accuracy: float


@dataclass(frozen=True, slots=True)
class _FakeTextResult:
    text: str


@dataclass(frozen=True, slots=True)
class _FakeEmbedding:
    shape: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _FakeEmbedTokens:
    weight: _FakeEmbedding


@dataclass(frozen=True, slots=True)
class _FakeTransformer:
    embed_tokens: _FakeEmbedTokens
    layers: list[object]


class _AdvanceTokenizer(MockTokenizer):
    def decode(self, token_ids: list[int]) -> str:
        return "a"


class _StalledTokenizer(MockTokenizer):
    def decode(self, token_ids: list[int]) -> str:
        return ""


class TestGeometryBranches:
    def test_empty_cosine_scores_and_silhouette_scores(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "vauban.detect.measure_subspace",
            lambda *args, **kwargs: _FakeSubspaceResult([1.0, 0.5]),
        )
        monkeypatch.setattr(
            "vauban.detect.measure",
            lambda *args, **kwargs: _FakeMeasureResult([]),
        )
        monkeypatch.setattr(
            "vauban.detect.silhouette_scores",
            lambda *args, **kwargs: [],
        )

        eff_rank, cosine_conc, sil_peak, evidence = _geometry_layer(
            mock_model,
            mock_tokenizer,
            HARMFUL,
            HARMLESS,
            DetectConfig(mode="fast", top_k=2),
        )

        assert eff_rank > 0.0
        assert cosine_conc == 0.0
        assert sil_peak == 0.0
        assert len(evidence) == 3


class TestRefusalHelpers:
    def test_find_refusal_token_position_advances(self) -> None:
        phrase = DEFAULT_REFUSAL_PHRASES[0]
        text = f"xx{phrase}yy"
        tokens = [1, 2, 3]

        pos = _find_refusal_token_position(text, tokens, _AdvanceTokenizer(16))

        assert pos == 1

    def test_find_refusal_token_position_falls_back_to_last_token(self) -> None:
        phrase = DEFAULT_REFUSAL_PHRASES[0]
        text = f"xx{phrase} ending"
        tokens = [1, 2, 3]

        pos = _find_refusal_token_position(text, tokens, _StalledTokenizer(16))

        assert pos == len(tokens) - 1

    def test_refusal_verbosity_returns_zero_without_refusal(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "vauban.detect._generate",
            lambda *args, **kwargs: "neutral output",
        )

        mean_pos = _refusal_verbosity(
            mock_model,
            mock_tokenizer,
            HARMFUL,
            max_tokens=5,
        )

        assert mean_pos == 0.0

    def test_refusal_verbosity_returns_average_when_refusal_is_present(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        phrase = DEFAULT_REFUSAL_PHRASES[0]
        monkeypatch.setattr(
            "vauban.detect._generate",
            lambda *args, **kwargs: f"xx{phrase}yy",
        )

        mean_pos = _refusal_verbosity(
            mock_model,
            mock_tokenizer,
            HARMFUL,
            max_tokens=5,
        )

        assert mean_pos > 0.0


class TestSvfCompare:
    def test_linear_geometry_sufficient_when_gap_is_small(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "vauban.detect.measure",
            lambda *args, **kwargs: _FakeMeasureResult([]),
        )
        monkeypatch.setattr(
            "vauban.detect.get_transformer",
            lambda *_args, **_kwargs: _FakeTransformer(
                embed_tokens=_FakeEmbedTokens(
                    weight=_FakeEmbedding(shape=(32, D_MODEL)),
                ),
                layers=[object(), object()],
            ),
        )
        monkeypatch.setattr(
            "vauban.svf.train_svf_boundary",
            lambda *args, **kwargs: (None, _FakeSVFResult(0.4)),
        )

        evidence = _svf_compare_layer(mock_model, mock_tokenizer, HARMFUL, HARMLESS)

        assert evidence[-1] == "svf_compare=linear_geometry_sufficient"
        assert any(item.startswith("linear_accuracy=") for item in evidence)

    def test_nonlinear_geometry_detected_when_gap_is_large(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "vauban.detect.measure",
            lambda *args, **kwargs: _FakeMeasureResult([0.8, -0.2]),
        )
        monkeypatch.setattr(
            "vauban.detect.get_transformer",
            lambda *_args, **_kwargs: _FakeTransformer(
                embed_tokens=_FakeEmbedTokens(
                    weight=_FakeEmbedding(shape=(32, D_MODEL)),
                ),
                layers=[object(), object()],
            ),
        )
        monkeypatch.setattr(
            "vauban.svf.train_svf_boundary",
            lambda *args, **kwargs: (None, _FakeSVFResult(0.95)),
        )

        evidence = _svf_compare_layer(mock_model, mock_tokenizer, HARMFUL, HARMLESS)

        assert evidence[-1] == "svf_compare=nonlinear_geometry_detected"
        assert any(item.startswith("svf_linear_gap=") for item in evidence)


class TestDetectMargin:
    def test_detect_margin_scores_confidence_from_collapse(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        margin_result = MarginResult(
            baseline_refusal_rate=0.2,
            curve=[],
            collapse_alpha={"test": 0.5},
            evidence=["baseline_refusal_rate=0.20"],
        )
        monkeypatch.setattr(
            "vauban.detect._margin_layer",
            lambda *args, **kwargs: margin_result,
        )
        monkeypatch.setattr(
            "vauban.detect._geometry_layer",
            lambda *args, **kwargs: (3.0, 1.0, 0.4, ["geo"]),
        )

        result = _detect_margin(
            mock_model,
            mock_tokenizer,
            HARMFUL,
            HARMLESS,
            DetectConfig(mode="margin", top_k=2),
        )

        assert isinstance(result, DetectResult)
        assert result.confidence == pytest.approx(0.75)
        assert result.hardened is False
        assert result.margin_result == margin_result


class TestMarginLayer:
    def test_margin_layer_reports_collapse(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        direction = ops.ones((D_MODEL,))
        ops.eval(direction)
        monkeypatch.setattr(
            "vauban.detect.ops.load",
            lambda *_args, **_kwargs: {"direction": direction},
        )
        monkeypatch.setattr(
            "vauban.detect._refusal_rate",
            lambda *args, **kwargs: 1.0,
        )
        monkeypatch.setattr(
            "vauban.detect.steer",
            lambda *args, **kwargs: _FakeTextResult(
                "safe output"
                if (len(args) > 5 and args[5] < 1.0)
                else DEFAULT_REFUSAL_PHRASES[0],
            ),
        )

        config = DetectConfig(
            mode="margin",
            margin_directions=[str(tmp_path / "direction.safetensors")],
            margin_alphas=[0.5, 1.0],
            max_tokens=5,
        )
        result = _margin_layer(mock_model, mock_tokenizer, HARMFUL, config)

        assert result.baseline_refusal_rate == pytest.approx(1.0)
        assert result.collapse_alpha["direction"] == pytest.approx(0.5)
        assert any("collapse at alpha=0.5" in item for item in result.evidence)
        assert len(result.curve) == 2

    def test_margin_layer_counts_refusals(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        direction = ops.ones((D_MODEL,))
        ops.eval(direction)
        monkeypatch.setattr(
            "vauban.detect.ops.load",
            lambda *_args, **_kwargs: {"direction": direction},
        )
        monkeypatch.setattr(
            "vauban.detect._refusal_rate",
            lambda *args, **kwargs: 1.0,
        )
        monkeypatch.setattr(
            "vauban.detect.steer",
            lambda *args, **kwargs: _FakeTextResult(DEFAULT_REFUSAL_PHRASES[0]),
        )

        config = DetectConfig(
            mode="margin",
            margin_directions=[str(tmp_path / "direction.safetensors")],
            margin_alphas=[1.0],
            max_tokens=5,
        )
        result = _margin_layer(mock_model, mock_tokenizer, HARMFUL, config)

        assert result.collapse_alpha["direction"] is None
        assert len(result.curve) == 1
        assert any("no collapse detected" in item for item in result.evidence)

    def test_margin_layer_rejects_non_array_direction(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            "vauban.detect.ops.load",
            lambda *_args, **_kwargs: {"direction": 123},
        )
        config = DetectConfig(
            mode="margin",
            margin_directions=[str(tmp_path / "broken.safetensors")],
            margin_alphas=[1.0],
            max_tokens=5,
        )

        with pytest.raises(TypeError, match="Expected array"):
            _margin_layer(mock_model, mock_tokenizer, HARMFUL, config)


class TestDetectEntryPoint:
    def test_detect_includes_svf_compare_branch(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "vauban.detect._geometry_layer",
            lambda *args, **kwargs: (1.0, 2.0, 0.0, ["geo"]),
        )
        monkeypatch.setattr(
            "vauban.detect.measure",
            lambda *args, **kwargs: _FakeMeasureResult([0.7, -0.2]),
        )
        monkeypatch.setattr(
            "vauban.detect.get_transformer",
            lambda *_args, **_kwargs: _FakeTransformer(
                embed_tokens=_FakeEmbedTokens(
                    weight=_FakeEmbedding(shape=(32, D_MODEL)),
                ),
                layers=[object(), object()],
            ),
        )
        monkeypatch.setattr(
            "vauban.svf.train_svf_boundary",
            lambda *args, **kwargs: (None, _FakeSVFResult(0.9)),
        )

        result = detect(
            mock_model,
            mock_tokenizer,
            HARMFUL,
            HARMLESS,
            DetectConfig(mode="fast", top_k=2, svf_compare=True),
        )

        assert isinstance(result, DetectResult)
        assert any(
            item == "svf_compare=nonlinear_geometry_detected"
            for item in result.evidence
        )

"""Tests for injection scan and surface scan helper functions.

Covers:
- ``_sigmoid``: numerically stable sigmoid (pure math).
- ``_detect_spans``: contiguous token span detection below threshold.
- ``calibrate_scan_threshold``: auto-calibration from clean documents.
- ``_probe_with_messages``: per-layer projection collection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.conftest import (
    D_MODEL,
    NUM_LAYERS,
    MockCausalLM,
    MockTokenizer,
)
from vauban import _ops as ops
from vauban.scan import _detect_spans, _sigmoid, calibrate_scan_threshold
from vauban.surface._scan import _probe_with_messages
from vauban.types import ScanConfig

if TYPE_CHECKING:
    from vauban._array import Array


# ---------------------------------------------------------------------------
# Helper: lightweight tokenizer for _detect_spans tests
# ---------------------------------------------------------------------------


class _SpanTokenizer:
    """Minimal tokenizer stub for _detect_spans unit tests."""

    def decode(self, ids: list[int]) -> str:
        """Join token ids as dash-separated strings."""
        return "-".join(str(i) for i in ids)


# ---------------------------------------------------------------------------
# _sigmoid
# ---------------------------------------------------------------------------


class TestSigmoid:
    """Tests for the numerically stable sigmoid function."""

    def test_zero_returns_half(self) -> None:
        """sigmoid(0) must be exactly 0.5."""
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive_approaches_one(self) -> None:
        """Large positive input should saturate near 1.0."""
        assert _sigmoid(50.0) == pytest.approx(1.0, abs=1e-12)
        assert _sigmoid(500.0) == pytest.approx(1.0, abs=1e-15)

    def test_large_negative_approaches_zero(self) -> None:
        """Large negative input should saturate near 0.0."""
        assert _sigmoid(-50.0) == pytest.approx(0.0, abs=1e-12)
        assert _sigmoid(-500.0) == pytest.approx(0.0, abs=1e-15)

    def test_monotonically_increasing(self) -> None:
        """Sigmoid must be strictly increasing over a range of inputs."""
        xs = [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0]
        values = [_sigmoid(x) for x in xs]
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1], (
                f"sigmoid({xs[i]})={values[i]} should be < "
                f"sigmoid({xs[i + 1]})={values[i + 1]}"
            )

    def test_symmetric_around_zero(self) -> None:
        """sigmoid(x) + sigmoid(-x) must equal 1.0 for all x."""
        for x in [0.1, 1.0, 3.0, 10.0, 50.0]:
            total = _sigmoid(x) + _sigmoid(-x)
            assert total == pytest.approx(1.0, abs=1e-12), (
                f"sigmoid({x}) + sigmoid({-x}) = {total}, expected 1.0"
            )


# ---------------------------------------------------------------------------
# _detect_spans
# ---------------------------------------------------------------------------


class TestDetectSpans:
    """Tests for contiguous span detection below threshold."""

    def test_all_above_threshold_returns_empty(self) -> None:
        """When every projection is above the threshold, no spans detected."""
        projections: list[float] = [1.0, 2.0, 3.0, 4.0]
        token_ids: list[int] = [10, 20, 30, 40]
        spans = _detect_spans(
            projections, token_ids, _SpanTokenizer(), 0.5,  # type: ignore[arg-type]
        )
        assert spans == []

    def test_single_contiguous_span(self) -> None:
        """A single block of below-threshold tokens produces one span."""
        projections: list[float] = [1.0, -0.5, -1.0, -0.3, 2.0]
        token_ids: list[int] = [10, 20, 30, 40, 50]
        spans = _detect_spans(
            projections, token_ids, _SpanTokenizer(), 0.5,  # type: ignore[arg-type]
        )
        assert len(spans) == 1
        span = spans[0]
        assert span.start == 1
        assert span.end == 4
        assert span.text == "20-30-40"

    def test_multiple_disjoint_spans(self) -> None:
        """Disjoint below-threshold regions produce separate spans."""
        projections: list[float] = [-1.0, 2.0, -1.0, -2.0, 3.0, -0.5]
        token_ids: list[int] = [10, 20, 30, 40, 50, 60]
        spans = _detect_spans(
            projections, token_ids, _SpanTokenizer(), 0.5,  # type: ignore[arg-type]
        )
        assert len(spans) == 3
        # First span: index 0
        assert spans[0].start == 0
        assert spans[0].end == 1
        assert spans[0].text == "10"
        # Second span: indices 2-3
        assert spans[1].start == 2
        assert spans[1].end == 4
        assert spans[1].text == "30-40"
        # Third span: trailing, index 5
        assert spans[2].start == 5
        assert spans[2].end == 6
        assert spans[2].text == "60"

    def test_mean_projection_is_correct(self) -> None:
        """The mean_projection of a span must match the average of its values."""
        projections: list[float] = [5.0, -1.0, -3.0, -5.0, 5.0]
        token_ids: list[int] = [0, 1, 2, 3, 4]
        spans = _detect_spans(
            projections, token_ids, _SpanTokenizer(), 0.5,  # type: ignore[arg-type]
        )
        assert len(spans) == 1
        expected_mean = (-1.0 + -3.0 + -5.0) / 3.0
        assert spans[0].mean_projection == pytest.approx(expected_mean)


# ---------------------------------------------------------------------------
# calibrate_scan_threshold
# ---------------------------------------------------------------------------


class TestCalibrateScanThreshold:
    """Tests for auto-calibration from clean documents."""

    def test_empty_documents_returns_config_threshold(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """With no clean documents, fall back to the config default."""
        config = ScanConfig(threshold=1.23)
        direction: Array = ops.random.normal((D_MODEL,))
        direction = direction / ops.linalg.norm(direction)
        ops.eval(direction)

        result = calibrate_scan_threshold(
            mock_model,
            mock_tokenizer,
            [],
            config,
            direction,
            layer_index=0,
        )
        assert result == pytest.approx(1.23)


# ---------------------------------------------------------------------------
# _probe_with_messages
# ---------------------------------------------------------------------------


class TestProbeWithMessages:
    """Tests for per-layer projection collection from surface scan."""

    def test_returns_one_projection_per_layer(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """_probe_with_messages must return exactly NUM_LAYERS floats."""
        direction: Array = ops.random.normal((D_MODEL,))
        direction = direction / ops.linalg.norm(direction)
        ops.eval(direction)

        messages: list[dict[str, str]] = [
            {"role": "user", "content": "Hello world"},
        ]
        projections = _probe_with_messages(
            mock_model, mock_tokenizer, messages, direction,
        )
        assert isinstance(projections, list)
        assert len(projections) == NUM_LAYERS
        assert all(isinstance(p, float) for p in projections)

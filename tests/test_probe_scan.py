"""Tests for scan() injection detection and related utility functions."""

import pytest

from tests.conftest import VOCAB_SIZE, MockCausalLM, MockTokenizer
from vauban._array import Array
from vauban.scan import _detect_spans, _sigmoid, scan
from vauban.types import ScanConfig

# ── scan() with mock model ───────────────────────────────────────────


class TestScanWithModel:
    """Tests for the full scan() function with mock model."""

    def test_basic_scan(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = ScanConfig(threshold=0.0)
        result = scan(
            mock_model, mock_tokenizer, "test content",
            config, direction,
        )
        assert isinstance(result.injection_probability, float)
        assert 0.0 <= result.injection_probability <= 1.0
        assert isinstance(result.overall_projection, float)
        assert len(result.per_token_projections) > 0
        assert isinstance(result.flagged, bool)

    def test_high_threshold_not_flagged(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = ScanConfig(threshold=1e9)
        result = scan(
            mock_model, mock_tokenizer, "clean content",
            config, direction,
        )
        # With very high threshold, sigmoid(projection - threshold) < 0.5
        assert result.flagged is False

    def test_low_threshold_flagged(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = ScanConfig(threshold=-1e9)
        result = scan(
            mock_model, mock_tokenizer, "injected content",
            config, direction,
        )
        # With very low threshold, sigmoid(projection - threshold) > 0.5
        assert result.flagged is True

    def test_target_layer(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = ScanConfig(threshold=0.0, target_layer=1)
        result = scan(
            mock_model, mock_tokenizer, "test",
            config, direction, layer_index=0,
        )
        # Should use target_layer=1, not fallback layer_index=0
        assert len(result.per_token_projections) > 0

    def test_fallback_layer_index(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = ScanConfig(threshold=0.0, target_layer=None)
        result = scan(
            mock_model, mock_tokenizer, "test",
            config, direction, layer_index=1,
        )
        assert len(result.per_token_projections) > 0

    def test_span_detection(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = ScanConfig(threshold=0.0, span_threshold=1e9)
        result = scan(
            mock_model, mock_tokenizer, "some test text here",
            config, direction,
        )
        # With very high span_threshold, all tokens should be in spans
        # (projections are below the high threshold)
        assert len(result.spans) >= 1

    def test_per_token_count_matches_tokens(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        config = ScanConfig(threshold=0.0)
        text = "hello world"
        result = scan(
            mock_model, mock_tokenizer, text,
            config, direction,
        )
        expected_n_tokens = len(mock_tokenizer.encode(text))
        assert len(result.per_token_projections) == expected_n_tokens


# ── calibrate_scan_threshold ─────────────────────────────────────────


class TestCalibrateScanThreshold:
    """Tests for automatic scan threshold calibration."""

    def test_returns_float(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        from vauban.scan import calibrate_scan_threshold

        config = ScanConfig(threshold=0.0)
        threshold = calibrate_scan_threshold(
            mock_model, mock_tokenizer,
            ["clean doc 1", "clean doc 2"],
            config, direction,
        )
        assert isinstance(threshold, float)

    def test_empty_docs_returns_default(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        from vauban.scan import calibrate_scan_threshold

        config = ScanConfig(threshold=42.0)
        threshold = calibrate_scan_threshold(
            mock_model, mock_tokenizer, [], config, direction,
        )
        assert threshold == 42.0


# ── _sigmoid edge cases ─────────────────────────────────────────────


class TestSigmoidEdgeCases:
    """Tests for sigmoid numerical stability and properties."""

    def test_moderate_values(self) -> None:
        # Verify monotonicity
        vals = [-5.0, -1.0, 0.0, 1.0, 5.0]
        results = [_sigmoid(v) for v in vals]
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1]

    def test_symmetry(self) -> None:
        assert _sigmoid(2.0) + _sigmoid(-2.0) == pytest.approx(1.0)


# ── _detect_spans edge cases ────────────────────────────────────────


class TestDetectSpansEdgeCases:
    """Additional edge case tests for span detection."""

    def _fake_tokenizer(self) -> MockTokenizer:
        return MockTokenizer(VOCAB_SIZE)

    def test_trailing_span(self) -> None:
        projections = [1.0, 1.0, -1.0, -2.0]
        token_ids = [10, 20, 30, 40]
        spans = _detect_spans(
            projections, token_ids, self._fake_tokenizer(), 0.5,
        )
        assert len(spans) == 1
        assert spans[0].start == 2
        assert spans[0].end == 4

    def test_empty_input(self) -> None:
        spans = _detect_spans(
            [], [], self._fake_tokenizer(), 0.5,
        )
        assert len(spans) == 0

    def test_single_token_span(self) -> None:
        projections = [1.0, -1.0, 1.0]
        token_ids = [10, 20, 30]
        spans = _detect_spans(
            projections, token_ids, self._fake_tokenizer(), 0.5,
        )
        assert len(spans) == 1
        assert spans[0].start == 1
        assert spans[0].end == 2

    def test_mean_projection_computed(self) -> None:
        projections = [-1.0, -3.0]
        token_ids = [10, 20]
        spans = _detect_spans(
            projections, token_ids, self._fake_tokenizer(), 0.5,
        )
        assert len(spans) == 1
        assert spans[0].mean_projection == -2.0

"""Tests for the injection scanner module."""

import pytest

from vauban.scan import _detect_spans, _sigmoid


class TestSigmoid:
    """Tests for the numerically stable sigmoid."""

    def test_zero(self) -> None:
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive(self) -> None:
        assert _sigmoid(100.0) == pytest.approx(1.0)

    def test_large_negative(self) -> None:
        assert _sigmoid(-100.0) == pytest.approx(0.0)

    def test_small_values(self) -> None:
        assert 0.0 < _sigmoid(-1.0) < 0.5
        assert 0.5 < _sigmoid(1.0) < 1.0


class TestDetectSpans:
    """Tests for span detection logic."""

    def test_no_spans_all_above(self) -> None:
        projections = [1.0, 2.0, 3.0]
        token_ids = [10, 20, 30]

        class FakeTokenizer:
            def decode(self, ids: list[int]) -> str:
                return " ".join(str(i) for i in ids)

        spans = _detect_spans(projections, token_ids, FakeTokenizer(), 0.5)  # type: ignore[arg-type]
        assert len(spans) == 0

    def test_single_span_all_below(self) -> None:
        projections = [-1.0, -2.0, -3.0]
        token_ids = [10, 20, 30]

        class FakeTokenizer:
            def decode(self, ids: list[int]) -> str:
                return " ".join(str(i) for i in ids)

        spans = _detect_spans(projections, token_ids, FakeTokenizer(), 0.5)  # type: ignore[arg-type]
        assert len(spans) == 1
        assert spans[0].start == 0
        assert spans[0].end == 3

    def test_middle_span(self) -> None:
        projections = [1.0, -1.0, -2.0, 1.0, 1.0]
        token_ids = [10, 20, 30, 40, 50]

        class FakeTokenizer:
            def decode(self, ids: list[int]) -> str:
                return " ".join(str(i) for i in ids)

        spans = _detect_spans(projections, token_ids, FakeTokenizer(), 0.5)  # type: ignore[arg-type]
        assert len(spans) == 1
        assert spans[0].start == 1
        assert spans[0].end == 3

    def test_multiple_spans(self) -> None:
        projections = [-1.0, 1.0, -1.0]
        token_ids = [10, 20, 30]

        class FakeTokenizer:
            def decode(self, ids: list[int]) -> str:
                return " ".join(str(i) for i in ids)

        spans = _detect_spans(projections, token_ids, FakeTokenizer(), 0.5)  # type: ignore[arg-type]
        assert len(spans) == 2


class TestScanConfigParsing:
    """Tests for [scan] config parsing."""

    def test_parse_valid(self) -> None:
        from vauban.config._parse_scan import _parse_scan

        raw = {"scan": {"target_layer": 5, "span_threshold": 0.3}}
        config = _parse_scan(raw)
        assert config is not None
        assert config.target_layer == 5
        assert config.span_threshold == pytest.approx(0.3)

    def test_missing_returns_none(self) -> None:
        from vauban.config._parse_scan import _parse_scan

        assert _parse_scan({}) is None

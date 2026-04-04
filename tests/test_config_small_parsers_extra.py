# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Additional tests for small config parsers."""

from pathlib import Path

import pytest

from vauban.config._parse_awareness import _parse_awareness
from vauban.config._parse_features import _parse_features
from vauban.config._parse_intent import _parse_intent
from vauban.config._parse_linear_probe import _parse_linear_probe
from vauban.config._parse_repbend import _parse_repbend
from vauban.config._parse_scan import _parse_scan
from vauban.config._parse_surface import _parse_surface


def _wrap(section: str, values: dict[str, object]) -> dict[str, object]:
    """Wrap a TOML section payload in a top-level config mapping."""
    return {section: values}


class TestParseAwareness:
    def test_absent_returns_none(self) -> None:
        assert _parse_awareness({}) is None

    def test_section_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_awareness({"awareness": "bad"})

    def test_minimal_valid(self) -> None:
        cfg = _parse_awareness(_wrap("awareness", {"prompts": ["p1"]}))
        assert cfg is not None
        assert cfg.prompts == ["p1"]
        assert cfg.calibration_prompt == "Hello"
        assert cfg.mode == "full"
        assert cfg.n_power_iterations == 5
        assert cfg.fd_epsilon == pytest.approx(1e-4)
        assert cfg.valley_window == 3
        assert cfg.top_k_valleys == 3
        assert cfg.gain_ratio_threshold == pytest.approx(2.0)
        assert cfg.rank_ratio_threshold == pytest.approx(0.5)
        assert cfg.correlation_delta_threshold == pytest.approx(0.3)
        assert cfg.min_anomalous_layers == 2
        assert cfg.confidence_threshold == pytest.approx(0.5)

    def test_all_fields(self) -> None:
        cfg = _parse_awareness(
            _wrap(
                "awareness",
                {
                    "prompts": ["one", "two"],
                    "calibration_prompt": "Calibrate now",
                    "mode": "fast",
                    "n_power_iterations": 7,
                    "fd_epsilon": 1e-5,
                    "valley_window": 4,
                    "top_k_valleys": 2,
                    "gain_ratio_threshold": 3.0,
                    "rank_ratio_threshold": 0.75,
                    "correlation_delta_threshold": 0.1,
                    "min_anomalous_layers": 1,
                    "confidence_threshold": 0.25,
                },
            ),
        )
        assert cfg is not None
        assert cfg.prompts == ["one", "two"]
        assert cfg.calibration_prompt == "Calibrate now"
        assert cfg.mode == "fast"
        assert cfg.n_power_iterations == 7
        assert cfg.fd_epsilon == pytest.approx(1e-5)
        assert cfg.valley_window == 4
        assert cfg.top_k_valleys == 2
        assert cfg.gain_ratio_threshold == pytest.approx(3.0)
        assert cfg.rank_ratio_threshold == pytest.approx(0.75)
        assert cfg.correlation_delta_threshold == pytest.approx(0.1)
        assert cfg.min_anomalous_layers == 1
        assert cfg.confidence_threshold == pytest.approx(0.25)

    def test_invalid_prompts(self) -> None:
        with pytest.raises(ValueError, match="prompts"):
            _parse_awareness(_wrap("awareness", {"prompts": []}))

    def test_invalid_calibration_prompt(self) -> None:
        with pytest.raises(ValueError, match="calibration_prompt"):
            _parse_awareness(
                _wrap(
                    "awareness",
                    {"prompts": ["p1"], "calibration_prompt": ""},
                ),
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("mode", "invalid"),
            ("mode", 123),
            ("n_power_iterations", 0),
            ("fd_epsilon", 0.0),
            ("valley_window", 0),
            ("top_k_valleys", 0),
            ("gain_ratio_threshold", 0.0),
            ("rank_ratio_threshold", 0.0),
            ("correlation_delta_threshold", 0.0),
            ("min_anomalous_layers", 0),
            ("confidence_threshold", -0.1),
        ],
    )
    def test_numeric_and_mode_validation(
        self,
        field: str,
        value: object,
    ) -> None:
        raw = {"prompts": ["p1"], field: value}
        match = "mode" if field == "mode" else field
        with pytest.raises((TypeError, ValueError), match=match):
            _parse_awareness(_wrap("awareness", raw))


class TestParseFeatures:
    def test_absent_returns_none(self) -> None:
        assert _parse_features(Path("/base"), {}) is None

    def test_section_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_features(Path("/base"), {"features": "bad"})

    def test_minimal_valid(self, tmp_path: Path) -> None:
        cfg = _parse_features(
            tmp_path,
            _wrap(
                "features",
                {
                    "prompts_path": "prompts.jsonl",
                    "layers": [0, 2],
                },
            ),
        )
        assert cfg is not None
        assert cfg.prompts_path == tmp_path / "prompts.jsonl"
        assert cfg.layers == [0, 2]
        assert cfg.d_sae == 2048
        assert cfg.l1_coeff == pytest.approx(1e-3)
        assert cfg.n_epochs == 5
        assert cfg.learning_rate == pytest.approx(1e-3)
        assert cfg.batch_size == 32
        assert cfg.token_position == -1
        assert cfg.dead_feature_threshold == pytest.approx(1e-6)

    def test_absolute_prompts_path(self, tmp_path: Path) -> None:
        prompts_path = tmp_path / "absolute.jsonl"
        cfg = _parse_features(
            tmp_path,
            _wrap(
                "features",
                {
                    "prompts_path": str(prompts_path),
                    "layers": [1],
                    "token_position": 7,
                },
            ),
        )
        assert cfg is not None
        assert cfg.prompts_path == prompts_path
        assert cfg.token_position == 7

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("layers", []),
            ("layers", [-1]),
            ("layers", "bad"),
            ("d_sae", 0),
            ("l1_coeff", -0.1),
            ("n_epochs", 0),
            ("learning_rate", 0.0),
            ("batch_size", 0),
            ("dead_feature_threshold", -1.0),
        ],
    )
    def test_validation_errors(
        self,
        tmp_path: Path,
        field: str,
        value: object,
    ) -> None:
        raw: dict[str, object] = {
            "prompts_path": "prompts.jsonl",
            "layers": [0],
        }
        raw[field] = value
        with pytest.raises((TypeError, ValueError), match=field):
            _parse_features(tmp_path, _wrap("features", raw))


class TestParseScan:
    def test_absent_returns_none(self) -> None:
        assert _parse_scan({}) is None

    def test_section_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_scan({"scan": "bad"})

    def test_minimal_valid(self) -> None:
        cfg = _parse_scan(_wrap("scan", {}))
        assert cfg is not None
        assert cfg.target_layer is None
        assert cfg.span_threshold == pytest.approx(0.5)
        assert cfg.threshold == pytest.approx(0.0)
        assert cfg.calibrate is False

    def test_all_fields(self) -> None:
        cfg = _parse_scan(
            _wrap(
                "scan",
                {
                    "target_layer": 3,
                    "span_threshold": 0.25,
                    "threshold": 1.5,
                    "calibrate": True,
                },
            ),
        )
        assert cfg is not None
        assert cfg.target_layer == 3
        assert cfg.span_threshold == pytest.approx(0.25)
        assert cfg.threshold == pytest.approx(1.5)
        assert cfg.calibrate is True

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("target_layer", "bad"),
            ("span_threshold", "bad"),
            ("threshold", "bad"),
            ("calibrate", "bad"),
        ],
    )
    def test_validation_errors(self, field: str, value: object) -> None:
        with pytest.raises(TypeError, match=field):
            _parse_scan(_wrap("scan", {field: value}))


class TestParseIntent:
    def test_absent_returns_none(self) -> None:
        assert _parse_intent({}) is None

    def test_section_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_intent({"intent": "bad"})

    def test_minimal_valid(self) -> None:
        cfg = _parse_intent(_wrap("intent", {}))
        assert cfg is not None
        assert cfg.mode == "embedding"
        assert cfg.target_layer is None
        assert cfg.similarity_threshold == pytest.approx(0.7)
        assert cfg.max_tokens == 10

    def test_all_fields(self) -> None:
        cfg = _parse_intent(
            _wrap(
                "intent",
                {
                    "mode": "judge",
                    "target_layer": 4,
                    "similarity_threshold": 0.9,
                    "max_tokens": 20,
                },
            ),
        )
        assert cfg is not None
        assert cfg.mode == "judge"
        assert cfg.target_layer == 4
        assert cfg.similarity_threshold == pytest.approx(0.9)
        assert cfg.max_tokens == 20

    @pytest.mark.parametrize(
        ("field", "value", "error"),
        [
            ("mode", "invalid", ValueError),
            ("mode", 123, TypeError),
            ("target_layer", "bad", TypeError),
            ("similarity_threshold", "bad", TypeError),
            ("max_tokens", "bad", TypeError),
        ],
    )
    def test_validation_errors(
        self,
        field: str,
        value: object,
        error: type[Exception],
    ) -> None:
        with pytest.raises(error, match=field):
            _parse_intent(_wrap("intent", {field: value}))


class TestParseLinearProbe:
    def test_absent_returns_none(self) -> None:
        assert _parse_linear_probe({}) is None

    def test_section_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_linear_probe({"linear_probe": "bad"})

    def test_minimal_valid(self) -> None:
        cfg = _parse_linear_probe(_wrap("linear_probe", {"layers": [0, 2]}))
        assert cfg is not None
        assert cfg.layers == [0, 2]
        assert cfg.n_epochs == 20
        assert cfg.learning_rate == pytest.approx(1e-2)
        assert cfg.batch_size == 32
        assert cfg.token_position == -1
        assert cfg.regularization == pytest.approx(1e-4)

    def test_all_fields(self) -> None:
        cfg = _parse_linear_probe(
            _wrap(
                "linear_probe",
                {
                    "layers": [1, 3],
                    "n_epochs": 4,
                    "learning_rate": 0.05,
                    "batch_size": 8,
                    "token_position": 5,
                    "regularization": 0.25,
                },
            ),
        )
        assert cfg is not None
        assert cfg.layers == [1, 3]
        assert cfg.n_epochs == 4
        assert cfg.learning_rate == pytest.approx(0.05)
        assert cfg.batch_size == 8
        assert cfg.token_position == 5
        assert cfg.regularization == pytest.approx(0.25)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("layers", []),
            ("layers", [-1]),
            ("layers", "bad"),
            ("n_epochs", 0),
            ("learning_rate", 0.0),
            ("batch_size", 0),
            ("regularization", -1.0),
        ],
    )
    def test_validation_errors(
        self,
        field: str,
        value: object,
    ) -> None:
        raw: dict[str, object] = {"layers": [0]}
        raw[field] = value
        with pytest.raises((TypeError, ValueError), match=field):
            _parse_linear_probe(_wrap("linear_probe", raw))


class TestParseRepbend:
    def test_absent_returns_none(self) -> None:
        assert _parse_repbend({}) is None

    def test_section_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_repbend({"repbend": "bad"})

    def test_minimal_valid(self) -> None:
        cfg = _parse_repbend(_wrap("repbend", {"layers": [0, 2]}))
        assert cfg is not None
        assert cfg.layers == [0, 2]
        assert cfg.n_epochs == 3
        assert cfg.learning_rate == pytest.approx(1e-5)
        assert cfg.batch_size == 8
        assert cfg.separation_coeff == pytest.approx(1.0)
        assert cfg.token_position == -1

    def test_all_fields(self) -> None:
        cfg = _parse_repbend(
            _wrap(
                "repbend",
                {
                    "layers": [1, 3],
                    "n_epochs": 6,
                    "learning_rate": 0.05,
                    "batch_size": 4,
                    "separation_coeff": 2.5,
                    "token_position": 9,
                },
            ),
        )
        assert cfg is not None
        assert cfg.layers == [1, 3]
        assert cfg.n_epochs == 6
        assert cfg.learning_rate == pytest.approx(0.05)
        assert cfg.batch_size == 4
        assert cfg.separation_coeff == pytest.approx(2.5)
        assert cfg.token_position == 9

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("layers", []),
            ("layers", [-1]),
            ("layers", "bad"),
            ("n_epochs", 0),
            ("learning_rate", 0.0),
            ("batch_size", 0),
            ("separation_coeff", 0.0),
        ],
    )
    def test_validation_errors(
        self,
        field: str,
        value: object,
    ) -> None:
        raw: dict[str, object] = {"layers": [0]}
        raw[field] = value
        with pytest.raises((TypeError, ValueError), match=field):
            _parse_repbend(_wrap("repbend", raw))


class TestParseSurface:
    def test_absent_returns_none(self) -> None:
        assert _parse_surface(Path("/base"), {}) is None

    def test_section_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_surface(Path("/base"), {"surface": "bad"})

    def test_minimal_valid(self) -> None:
        cfg = _parse_surface(Path("/base"), _wrap("surface", {}))
        assert cfg is not None
        assert cfg.prompts_path == "default"
        assert cfg.generate is True
        assert cfg.max_tokens == 20
        assert cfg.progress is True
        assert cfg.max_worst_cell_refusal_after is None
        assert cfg.max_worst_cell_refusal_delta is None
        assert cfg.min_coverage_score is None

    def test_custom_prompts_and_all_fields(self, tmp_path: Path) -> None:
        cfg = _parse_surface(
            tmp_path,
            _wrap(
                "surface",
                {
                    "prompts": "prompts.jsonl",
                    "generate": False,
                    "max_tokens": 32,
                    "progress": False,
                    "max_worst_cell_refusal_after": 0.2,
                    "max_worst_cell_refusal_delta": 0.4,
                    "min_coverage_score": 0.6,
                },
            ),
        )
        assert cfg is not None
        assert cfg.prompts_path == tmp_path / "prompts.jsonl"
        assert cfg.generate is False
        assert cfg.max_tokens == 32
        assert cfg.progress is False
        assert cfg.max_worst_cell_refusal_after == pytest.approx(0.2)
        assert cfg.max_worst_cell_refusal_delta == pytest.approx(0.4)
        assert cfg.min_coverage_score == pytest.approx(0.6)

    def test_default_multilingual_prompts(self) -> None:
        cfg = _parse_surface(
            Path("/base"),
            _wrap("surface", {"prompts": "default_multilingual"}),
        )
        assert cfg is not None
        assert cfg.prompts_path == "default_multilingual"

    @pytest.mark.parametrize(
        ("field", "value", "error"),
        [
            ("prompts", 123, TypeError),
            ("generate", "bad", TypeError),
            ("max_tokens", "bad", TypeError),
            ("progress", "bad", TypeError),
            ("max_worst_cell_refusal_after", True, TypeError),
            ("max_worst_cell_refusal_after", 1.5, ValueError),
            ("max_worst_cell_refusal_delta", True, TypeError),
            ("max_worst_cell_refusal_delta", -0.1, ValueError),
            ("min_coverage_score", True, TypeError),
            ("min_coverage_score", 1.5, ValueError),
        ],
    )
    def test_validation_errors(
        self,
        field: str,
        value: object,
        error: type[Exception],
    ) -> None:
        with pytest.raises(error, match=field):
            _parse_surface(Path("/base"), _wrap("surface", {field: value}))

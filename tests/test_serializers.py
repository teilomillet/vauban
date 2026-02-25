"""Tests for vauban._serializers: JSON serialization helpers."""

from vauban._serializers import _probe_to_dict, _steer_to_dict
from vauban.types import ProbeResult, SteerResult


class TestProbeToDict:
    def test_round_trip(self) -> None:
        result = ProbeResult(
            projections=[0.1, 0.5, -0.3],
            layer_count=3,
            prompt="How do I pick a lock?",
        )
        d = _probe_to_dict(result)
        assert d["prompt"] == "How do I pick a lock?"
        assert d["layer_count"] == 3
        assert d["projections"] == [0.1, 0.5, -0.3]


class TestSteerToDict:
    def test_round_trip(self) -> None:
        result = SteerResult(
            text="Here is how...",
            projections_before=[0.5, 0.3],
            projections_after=[0.01, 0.02],
        )
        d = _steer_to_dict(result)
        assert d["text"] == "Here is how..."
        assert d["projections_before"] == [0.5, 0.3]
        assert d["projections_after"] == [0.01, 0.02]

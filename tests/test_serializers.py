"""Tests for vauban._serializers: JSON serialization helpers."""

from vauban._serializers import _depth_to_dict, _probe_to_dict, _steer_to_dict
from vauban.types import DepthResult, ProbeResult, SteerResult, TokenDepth


class TestDepthToDict:
    def test_round_trip(self) -> None:
        tokens = [
            TokenDepth(
                token_id=1,
                token_str="hello",
                settling_depth=0,
                is_deep_thinking=False,
                jsd_profile=[0.3, 0.1],
            ),
            TokenDepth(
                token_id=2,
                token_str="world",
                settling_depth=1,
                is_deep_thinking=True,
                jsd_profile=[0.8, 0.4],
            ),
        ]
        result = DepthResult(
            tokens=tokens,
            deep_thinking_ratio=0.5,
            deep_thinking_count=1,
            mean_settling_depth=0.5,
            layer_count=2,
            settling_threshold=0.5,
            deep_fraction=0.85,
            prompt="test prompt",
        )
        d = _depth_to_dict(result)
        assert d["prompt"] == "test prompt"
        assert d["deep_thinking_ratio"] == 0.5
        assert d["deep_thinking_count"] == 1
        assert d["mean_settling_depth"] == 0.5
        assert d["layer_count"] == 2
        assert isinstance(d["tokens"], list)
        token_list: list[object] = d["tokens"]  # type: ignore[assignment]
        assert len(token_list) == 2
        first: dict[str, object] = token_list[0]  # type: ignore[assignment]
        assert first["token_id"] == 1
        assert first["token_str"] == "hello"
        assert first["settling_depth"] == 0
        assert first["is_deep_thinking"] is False
        assert first["jsd_profile"] == [0.3, 0.1]


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

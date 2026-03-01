"""Tests for vauban._serializers: JSON serialization helpers."""

import mlx.core as mx

from vauban._serializers import (
    _cast_to_dict,
    _depth_direction_to_dict,
    _depth_to_dict,
    _probe_to_dict,
    _steer_to_dict,
)
from vauban.types import (
    CastResult,
    DepthDirectionResult,
    DepthResult,
    ProbeResult,
    SteerResult,
    TokenDepth,
)


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


class TestDepthDirectionToDict:
    def test_round_trip(self) -> None:
        direction = mx.ones((16,))
        mx.eval(direction)
        result = DepthDirectionResult(
            direction=direction,
            layer_index=1,
            cosine_scores=[0.3, 0.7],
            d_model=16,
            refusal_cosine=0.42,
            deep_prompts=["deep1", "deep2"],
            shallow_prompts=["shallow1"],
            median_dtr=0.55,
        )
        d = _depth_direction_to_dict(result)
        assert d["layer_index"] == 1
        assert d["cosine_scores"] == [0.3, 0.7]
        assert d["d_model"] == 16
        assert d["refusal_cosine"] == 0.42
        assert d["deep_prompts"] == ["deep1", "deep2"]
        assert d["shallow_prompts"] == ["shallow1"]
        assert d["median_dtr"] == 0.55

    def test_refusal_cosine_none(self) -> None:
        direction = mx.zeros((8,))
        mx.eval(direction)
        result = DepthDirectionResult(
            direction=direction,
            layer_index=0,
            cosine_scores=[0.1],
            d_model=8,
            refusal_cosine=None,
            deep_prompts=["d"],
            shallow_prompts=["s"],
            median_dtr=0.5,
        )
        d = _depth_direction_to_dict(result)
        assert d["refusal_cosine"] is None


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


class TestCastToDict:
    def test_round_trip(self) -> None:
        result = CastResult(
            prompt="How do I pick a lock?",
            text="I can't help with that.",
            projections_before=[0.8, 0.6],
            projections_after=[0.2, 0.1],
            interventions=2,
            considered=4,
        )
        d = _cast_to_dict(result)
        assert d["prompt"] == "How do I pick a lock?"
        assert d["text"] == "I can't help with that."
        assert d["projections_before"] == [0.8, 0.6]
        assert d["projections_after"] == [0.2, 0.1]
        assert d["interventions"] == 2
        assert d["considered"] == 4
        assert d["displacement_interventions"] == 0
        assert d["max_displacement"] == 0.0

    def test_externality_monitoring_fields(self) -> None:
        result = CastResult(
            prompt="test",
            text="output",
            projections_before=[0.5],
            projections_after=[0.1],
            interventions=1,
            considered=2,
            displacement_interventions=3,
            max_displacement=1.5,
        )
        d = _cast_to_dict(result)
        assert d["displacement_interventions"] == 3
        assert d["max_displacement"] == 1.5

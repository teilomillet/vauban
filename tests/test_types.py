"""Tests for vauban.types: dataclass construction, frozen, protocol compliance."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tests.conftest import MockCausalLM, MockTokenizer
from vauban import _ops as ops
from vauban.types import (
    CastConfig,
    CastResult,
    CutConfig,
    DetectResult,
    DirectionResult,
    EvalResult,
    MeasureConfig,
    PipelineConfig,
    ProbeResult,
    SoftPromptResult,
    SteerResult,
    SubspaceResult,
    Tokenizer,
)


class TestProtocols:
    """Protocol compliance checks.

    nn.Module stores attributes in an internal dict, so runtime_checkable
    isinstance checks don't work. We verify structural conformance directly.
    """

    def test_transformer_model_has_required_attrs(
        self, mock_model: MockCausalLM,
    ) -> None:
        model = mock_model.model
        assert hasattr(model, "embed_tokens")
        assert hasattr(model, "layers")
        assert hasattr(model, "norm")
        assert callable(model)

    def test_causal_lm_has_required_attrs(
        self, mock_model: MockCausalLM,
    ) -> None:
        assert hasattr(mock_model, "model")
        assert callable(mock_model)

    def test_tokenizer_protocol(
        self, mock_tokenizer: MockTokenizer,
    ) -> None:
        assert isinstance(mock_tokenizer, Tokenizer)


class TestDirectionResult:
    def test_construction(self) -> None:
        d = ops.zeros((16,))
        result = DirectionResult(
            direction=d,
            layer_index=5,
            cosine_scores=[0.1, 0.2],
            d_model=16,
            model_path="test",
        )
        assert result.layer_index == 5
        assert result.d_model == 16

    def test_frozen(self) -> None:
        d = ops.zeros((16,))
        result = DirectionResult(
            direction=d, layer_index=0, cosine_scores=[], d_model=16, model_path="",
        )
        with pytest.raises(FrozenInstanceError):
            result.layer_index = 1  # type: ignore[misc]


class TestDirectionResultSummary:
    def test_summary_returns_str(self) -> None:
        d = ops.zeros((16,))
        result = DirectionResult(
            direction=d, layer_index=5, cosine_scores=[0.1, 0.5, 0.3],
            d_model=16, model_path="test-model",
        )
        s = result.summary()
        assert isinstance(s, str)
        assert "layer=5" in s
        assert "d_model=16" in s
        assert "max_cosine=0.5000" in s
        assert "test-model" in s

    def test_summary_empty_cosine(self) -> None:
        d = ops.zeros((16,))
        result = DirectionResult(
            direction=d, layer_index=0, cosine_scores=[],
            d_model=16, model_path="x",
        )
        assert "max_cosine=0.0000" in result.summary()


class TestEvalResultSummary:
    def test_summary_returns_str(self) -> None:
        result = EvalResult(
            refusal_rate_original=0.8, refusal_rate_modified=0.1,
            perplexity_original=10.0, perplexity_modified=12.0,
            kl_divergence=0.5, num_prompts=20,
        )
        s = result.summary()
        assert isinstance(s, str)
        assert "prompts=20" in s
        assert "kl=" in s


class TestProbeResultSummary:
    def test_summary_returns_str(self) -> None:
        result = ProbeResult(
            projections=[0.1, 0.5, 0.3], layer_count=3, prompt="test prompt",
        )
        s = result.summary()
        assert isinstance(s, str)
        assert "layers=3" in s
        assert "test prompt" in s

    def test_summary_truncates_long_prompt(self) -> None:
        long_prompt = "a" * 100
        result = ProbeResult(
            projections=[1.0], layer_count=1, prompt=long_prompt,
        )
        s = result.summary()
        assert "..." in s


class TestSteerResultSummary:
    def test_summary_returns_str(self) -> None:
        result = SteerResult(
            text="generated text", projections_before=[1.0, 2.0],
            projections_after=[0.1, 0.2],
        )
        s = result.summary()
        assert isinstance(s, str)
        assert "max_proj_before=2.0000" in s
        assert "max_proj_after=0.2000" in s


class TestCastResultSummary:
    def test_summary_returns_str(self) -> None:
        result = CastResult(
            prompt="test prompt",
            text="generated text",
            projections_before=[1.0, 0.5],
            projections_after=[0.2, 0.1],
            interventions=2,
            considered=4,
        )
        s = result.summary()
        assert isinstance(s, str)
        assert "interventions=2/4" in s


class TestCutConfig:
    def test_defaults(self) -> None:
        config = CutConfig()
        assert config.alpha == 1.0
        assert config.layers is None
        assert config.norm_preserve is False
        assert config.biprojected is False

    def test_custom(self) -> None:
        config = CutConfig(alpha=0.5, layers=[1, 2], norm_preserve=True)
        assert config.alpha == 0.5
        assert config.layers == [1, 2]


class TestEvalResult:
    def test_construction(self) -> None:
        result = EvalResult(
            refusal_rate_original=0.8,
            refusal_rate_modified=0.1,
            perplexity_original=10.0,
            perplexity_modified=12.0,
            kl_divergence=0.5,
            num_prompts=10,
        )
        assert result.num_prompts == 10


class TestProbeResult:
    def test_construction(self) -> None:
        result = ProbeResult(projections=[0.1, 0.2], layer_count=2, prompt="test")
        assert result.layer_count == 2


class TestSteerResult:
    def test_construction(self) -> None:
        result = SteerResult(
            text="hello",
            projections_before=[1.0],
            projections_after=[0.1],
        )
        assert result.text == "hello"


class TestSubspaceResult:
    def test_construction(self) -> None:
        basis = ops.zeros((3, 16))
        result = SubspaceResult(
            basis=basis,
            singular_values=[3.0, 2.0, 1.0],
            explained_variance=[0.6, 0.3, 0.1],
            layer_index=5,
            d_model=16,
            model_path="test",
            per_layer_bases=[basis],
        )
        assert result.layer_index == 5
        assert len(result.singular_values) == 3

    def test_best_direction(self) -> None:
        basis = ops.eye(3, 16)
        result = SubspaceResult(
            basis=basis,
            singular_values=[3.0, 2.0, 1.0],
            explained_variance=[0.6, 0.3, 0.1],
            layer_index=2,
            d_model=16,
            model_path="test",
            per_layer_bases=[basis],
        )
        direction = result.best_direction()
        assert direction.direction.shape == (16,)
        assert direction.layer_index == 2
        assert direction.d_model == 16


class TestMeasureConfig:
    def test_defaults(self) -> None:
        config = MeasureConfig()
        assert config.mode == "direction"
        assert config.top_k == 5
        assert config.measure_only is False

    def test_custom(self) -> None:
        config = MeasureConfig(mode="subspace", top_k=10, measure_only=True)
        assert config.mode == "subspace"
        assert config.top_k == 10
        assert config.measure_only is True


class TestToDict:
    def test_direction_result_to_dict(self) -> None:
        d = ops.zeros((16,))
        result = DirectionResult(
            direction=d, layer_index=5, cosine_scores=[0.1, 0.5],
            d_model=16, model_path="test", layer_types=["global", "sliding"],
        )
        out = result.to_dict()
        assert isinstance(out, dict)
        assert out["layer_index"] == 5
        assert out["d_model"] == 16
        assert out["cosine_scores"] == [0.1, 0.5]
        assert out["model_path"] == "test"
        assert out["layer_types"] == ["global", "sliding"]
        assert "direction" not in out  # mx.array skipped

    def test_eval_result_to_dict(self) -> None:
        result = EvalResult(
            refusal_rate_original=0.8, refusal_rate_modified=0.1,
            perplexity_original=10.0, perplexity_modified=12.0,
            kl_divergence=0.5, num_prompts=20,
        )
        out = result.to_dict()
        assert out["refusal_rate_modified"] == 0.1
        assert out["num_prompts"] == 20
        assert len(out) == 6

    def test_probe_result_to_dict(self) -> None:
        result = ProbeResult(
            projections=[0.1, 0.5], layer_count=2, prompt="test",
        )
        out = result.to_dict()
        assert out["prompt"] == "test"
        assert out["layer_count"] == 2
        assert out["projections"] == [0.1, 0.5]

    def test_steer_result_to_dict(self) -> None:
        result = SteerResult(
            text="hello", projections_before=[1.0],
            projections_after=[0.1],
        )
        out = result.to_dict()
        assert out["text"] == "hello"
        assert out["projections_before"] == [1.0]

    def test_detect_result_to_dict(self) -> None:
        result = DetectResult(
            hardened=True, confidence=0.9, effective_rank=2.5,
            cosine_concentration=3.0, silhouette_peak=0.8,
            hdd_red_distance=0.4, residual_refusal_rate=0.05,
            mean_refusal_position=3.2, evidence=["high concentration"],
        )
        out = result.to_dict()
        assert out["hardened"] is True
        assert out["confidence"] == 0.9
        assert out["evidence"] == ["high concentration"]
        assert len(out) == 9

    def test_softprompt_result_to_dict(self) -> None:
        result = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.2,
            loss_history=[2.0, 1.5, 1.2], n_steps=100, n_tokens=16,
            embeddings=ops.zeros((1, 16, 32)),  # should be excluded
            token_ids=[1, 2, 3], token_text="abc",
            eval_responses=["ok"], early_stopped=True,
        )
        out = result.to_dict()
        assert out["mode"] == "gcg"
        assert out["success_rate"] == 0.5
        assert out["token_ids"] == [1, 2, 3]
        assert out["early_stopped"] is True
        assert "embeddings" not in out  # mx.array skipped
        assert out["transfer_results"] == []


class TestPipelineConfig:
    def test_defaults(self) -> None:
        config = PipelineConfig(
            model_path="test",
            harmful_path=Path("harmful.jsonl"),
            harmless_path=Path("harmless.jsonl"),
        )
        assert config.output_dir == Path("output")
        assert config.eval.prompts_path is None
        assert config.measure.mode == "direction"


class TestCastConfig:
    def test_defaults(self) -> None:
        config = CastConfig(prompts=["test"])
        assert config.prompts == ["test"]
        assert config.layers is None
        assert config.alpha == 1.0
        assert config.threshold == 0.0
        assert config.max_tokens == 100

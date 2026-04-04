# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Targeted coverage tests for the remaining small package-tail branches."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
from reportlab.pdfgen.canvas import Canvas

from tests.conftest import MockTokenizer
from vauban import _ops as ops
from vauban.ai_act_pdf import _draw_bullet, _draw_paragraph, _PDFState, _style_for
from vauban.benchmarks import _check_manifest_staleness
from vauban.evaluate import _judge_refusal_rate, _perplexity
from vauban.intent import _check_alignment_embedding, _check_alignment_judge
from vauban.jailbreak import load_templates
from vauban.measure._activations import _collect_per_prompt_activations
from vauban.remote._registry import _REGISTRY, get_backend, register_backend
from vauban.sensitivity import directional_gain
from vauban.surface._aggregate import _coverage_score
from vauban.surface._scan import scan
from vauban.svf import SVFBoundary, load_svf_boundary, svf_gradient, train_svf_boundary
from vauban.taxonomy import score_text
from vauban.types import (
    AgentTurn,
    CastResult,
    CausalLM,
    DefenseEvalResult,
    DefenseProxyResult,
    DetectResult,
    EnvironmentResult,
    GanRoundResult,
    IntentConfig,
    IntentState,
    MarginCurvePoint,
    MarginResult,
    RemoteActivationResult,
    RemoteBackend,
    RemoteChatResult,
    SoftPromptResult,
    SurfacePrompt,
    Tokenizer,
    ToolCall,
    TransferEvalResult,
)

if TYPE_CHECKING:
    from pathlib import Path

    from vauban._array import Array


def _make_pdf_state() -> _PDFState:
    """Build a minimal PDF state for low-level drawing tests."""
    canvas = Canvas(BytesIO())
    return _PDFState(canvas=canvas, company_name="Acme", system_name="Vauban")


class _DummyRemoteBackend:
    """Minimal remote backend used to exercise registry success paths."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def chat(
        self,
        model_id: str,
        prompts: list[str],
        max_tokens: int,
    ) -> list[RemoteChatResult]:
        return []

    async def activations(
        self,
        model_id: str,
        prompts: list[str],
        modules: list[str],
    ) -> list[RemoteActivationResult]:
        return []


class TestAIACTPDFTail:
    def test_draw_paragraph_falls_back_when_wrap_returns_no_lines(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state = _make_pdf_state()
        monkeypatch.setattr("vauban.ai_act_pdf.simpleSplit", lambda *_args: [])

        _draw_paragraph(state, "", _style_for("body"))

        assert state.current_y < 0.0 + 770.0

    def test_draw_bullet_falls_back_when_wrap_returns_no_lines(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state = _make_pdf_state()
        monkeypatch.setattr("vauban.ai_act_pdf.simpleSplit", lambda *_args: [])

        _draw_bullet(state, "")

        assert state.current_y < 0.0 + 770.0


class TestBenchmarksTail:
    def test_manifest_staleness_ignores_non_mapping_manifest(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        (tmp_path / "manifest.json").write_text("[]", encoding="utf-8")

        with caplog.at_level("WARNING", logger="vauban.benchmarks"):
            _check_manifest_staleness(tmp_path, "harmbench")

        assert caplog.records == []

    def test_manifest_staleness_ignores_invalid_json(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        (tmp_path / "manifest.json").write_text("{not-json", encoding="utf-8")

        with caplog.at_level("WARNING", logger="vauban.benchmarks"):
            _check_manifest_staleness(tmp_path, "harmbench")

        assert caplog.records == []


class TestEvaluateTail:
    def test_perplexity_returns_zero_when_shift_has_no_tokens(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tokenizer = MockTokenizer(8)
        model_mock = MagicMock(return_value=object())
        model = cast("CausalLM", model_mock)

        monkeypatch.setattr(tokenizer, "encode", lambda _text: [1])
        monkeypatch.setattr(
            "vauban.evaluate.extract_logits",
            lambda _result: ops.zeros((1, 1, 2)),
        )
        monkeypatch.setattr(
            "vauban.evaluate._nn.cross_entropy",
            lambda _logits, _labels, reduction: ops.array(0.0),
        )
        monkeypatch.setattr("vauban.evaluate.force_eval", lambda *_args: None)

        assert _perplexity(model, cast("Tokenizer", tokenizer), ["x"]) == 0.0

    def test_judge_refusal_rate_counts_refusals(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        prompts = ["p1", "p2"]
        tokenizer = cast("Tokenizer", MockTokenizer(8))
        model = cast("CausalLM", object())

        monkeypatch.setattr(
            "vauban.evaluate._generate",
            lambda *_args, **_kwargs: "response",
        )
        monkeypatch.setattr(
            "vauban.evaluate._judge_single",
            lambda _model, _tokenizer, prompt, _response: prompt == "p2",
        )

        assert _judge_refusal_rate(model, tokenizer, prompts, max_tokens=4) == 0.5


class TestIntentTail:
    def test_embedding_alignment_zero_norm_returns_zero_score(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = IntentConfig(mode="embedding", similarity_threshold=0.1)
        intent_state = IntentState(
            user_request="stay safe",
            activation=ops.array([1.0, 0.0]),
        )

        monkeypatch.setattr(
            "vauban.intent._extract_activation_at_layer",
            lambda _model, _tokenizer, _text, _layer: ops.zeros((2,)),
        )

        result = _check_alignment_embedding(
            cast("CausalLM", object()),
            cast("Tokenizer", object()),
            "unsafe action",
            intent_state,
            config,
            layer_index=0,
        )

        assert result.score == 0.0
        assert result.aligned is False

    def test_judge_alignment_breaks_on_eos_token(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tokenizer = MockTokenizer(8)
        tokenizer.eos_token_id = 1
        model_mock = MagicMock(return_value=object())
        model = cast("CausalLM", model_mock)
        config = IntentConfig(mode="judge", max_tokens=4)
        intent_state = IntentState(user_request="summarize", activation=None)

        monkeypatch.setattr(
            "vauban.intent.encode_user_prompt",
            lambda _tokenizer, _prompt: ops.array([[0]]),
        )
        monkeypatch.setattr("vauban.intent.make_cache", lambda _model: object())
        monkeypatch.setattr(
            "vauban.intent.extract_logits",
            lambda _result: ops.array([[[0.0, 1.0]]]),
        )
        monkeypatch.setattr(tokenizer, "decode", lambda _token_ids: "ALIGNED")

        result = _check_alignment_judge(
            model,
            cast("Tokenizer", tokenizer),
            "send report",
            intent_state,
            config,
        )

        assert result.aligned is True
        assert model_mock.call_count == 1


class TestJailbreakTail:
    def test_load_templates_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "templates.jsonl"
        path.write_text(
            '\n{"strategy":"s","name":"n","template":"t"}\n\n',
            encoding="utf-8",
        )

        templates = load_templates(path)

        assert len(templates) == 1
        assert templates[0].name == "n"


class TestMeasureTail:
    def test_collect_per_prompt_activations_applies_clipping(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        clip_calls: list[float] = []

        monkeypatch.setattr(
            "vauban.measure._activations.encode_user_prompt",
            lambda _tokenizer, _prompt: ops.array([[1]]),
        )
        monkeypatch.setattr(
            "vauban.measure._activations._forward_collect",
            lambda _model, _token_ids, _token_position: [
                ops.array([1.0, 2.0]),
                ops.array([3.0, 4.0]),
            ],
        )

        def _fake_clip(activation: object, quantile: float) -> object:
            clip_calls.append(quantile)
            return activation

        monkeypatch.setattr(
            "vauban.measure._activations._clip_activation",
            _fake_clip,
        )

        per_layer = _collect_per_prompt_activations(
            cast("CausalLM", object()),
            cast("Tokenizer", object()),
            ["a", "b"],
            clip_quantile=0.25,
        )

        assert len(per_layer) == 2
        assert clip_calls == [0.25, 0.25, 0.25, 0.25]


class TestRemoteRegistryTail:
    def test_get_backend_returns_registered_backend(self) -> None:
        name = "unit-test-local-backend"

        def _factory(api_key: str) -> RemoteBackend:
            return _DummyRemoteBackend(api_key)

        register_backend(name, _factory)
        try:
            backend = get_backend(name, "secret")
        finally:
            _REGISTRY.pop(name, None)

        assert isinstance(backend, _DummyRemoteBackend)
        assert backend.api_key == "secret"


class TestSensitivityTail:
    def test_directional_gain_returns_zero_for_zero_direction(self) -> None:
        gain = directional_gain(
            lambda h: h,
            ops.zeros((1, 1, 2)),
            ops.zeros((2,)),
        )

        assert gain == 0.0


class TestSurfaceTail:
    def test_coverage_score_returns_zero_for_empty_points(self) -> None:
        assert _coverage_score([]) == 0.0

    def test_scan_progress_and_judge_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        prompt = SurfacePrompt(prompt="ask", label="harmful", category="weapons")

        monkeypatch.setattr(
            "vauban.surface._scan._probe_with_messages",
            lambda _model, _tokenizer, _messages, _direction: [0.5],
        )
        monkeypatch.setattr(
            "vauban.surface._scan._generate_with_messages",
            lambda _model, _tokenizer, _messages, _max_tokens: "refused",
        )
        monkeypatch.setattr(
            "vauban.surface._scan._judge_single",
            lambda _model, _tokenizer, _prompt, _response: True,
        )

        points = scan(
            cast("CausalLM", object()),
            cast("Tokenizer", object()),
            [prompt],
            ops.array([1.0]),
            0,
            progress=True,
            refusal_mode="judge",
        )

        captured = capsys.readouterr()
        assert "Scanning 1/1" in captured.err
        assert points[0].refused is True


class TestSVFTail:
    def test_train_svf_boundary_counts_positive_target_scores(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target_acts = [ops.array([[1.0], [2.0]])]
        opposite_acts = [ops.array([[-1.0]])]

        monkeypatch.setattr(
            "vauban.svf._collect_per_prompt_activations",
            lambda _model, _tokenizer, prompts: (
                target_acts if prompts[0] == "harmful" else opposite_acts
            ),
        )
        monkeypatch.setattr(
            SVFBoundary,
            "forward",
            lambda self, h, _layer_idx: ops.array(float(h[0].item())),
        )

        _boundary, result = train_svf_boundary(
            cast("CausalLM", object()),
            cast("Tokenizer", object()),
            ["harmful"],
            ["harmless"],
            d_model=1,
            n_layers=1,
            n_epochs=0,
        )

        assert result.final_accuracy == 1.0
        assert result.per_layer_separation == [1.0]

    def test_load_svf_boundary_rejects_non_mapping(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr("vauban.svf.ops.load", lambda _path: ops.array([1.0]))

        with pytest.raises(ValueError, match="Expected dict"):
            load_svf_boundary(tmp_path / "bad.safetensors")

    def test_svf_gradient_returns_zero_vector_when_gradient_norm_is_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        boundary = SVFBoundary(2, 1, 2, 1)

        def _fake_value_and_grad(_fn: object) -> object:
            def _runner(x: Array) -> tuple[Array, Array]:
                return ops.array(0.0), ops.zeros_like(x)

            return _runner

        monkeypatch.setattr("vauban.svf.ops.value_and_grad", _fake_value_and_grad)

        score, gradient = svf_gradient(boundary, ops.array([1.0, 2.0]), 0)

        assert score == 0.0
        assert ops.array_equal(gradient, ops.zeros((2,)))


class TestTaxonomyTail:
    def test_score_text_skips_categories_without_patterns(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "vauban.taxonomy._get_compiled_patterns",
            lambda _category_id: [],
        )

        assert score_text("anything") == []


class TestTypesTail:
    def test_cast_result_to_dict(self) -> None:
        result = CastResult(
            prompt="prompt",
            text="text",
            projections_before=[0.2],
            projections_after=[0.1],
            interventions=1,
            considered=2,
        )

        assert result.to_dict() == {
            "prompt": "prompt",
            "text": "text",
            "projections_before": [0.2],
            "projections_after": [0.1],
            "interventions": 1,
            "considered": 2,
        }

    def test_gan_round_to_dict_includes_optional_results(self) -> None:
        attack = SoftPromptResult(
            mode="gcg",
            success_rate=0.5,
            final_loss=1.0,
            loss_history=[1.0],
            n_steps=2,
            n_tokens=1,
            embeddings=None,
            token_ids=[1],
            token_text="x",
            eval_responses=["ok"],
        )
        defense_proxy = DefenseProxyResult(
            total_prompts=2,
            sic_blocked=0,
            sic_sanitized=1,
            cast_gated=1,
            prompts_sent=1,
            proxy_mode="both_gate",
            cast_responses=["safe"],
        )
        environment_result = EnvironmentResult(
            reward=1.0,
            target_called=True,
            target_args_match=True,
            turns=[AgentTurn(role="assistant", content="done")],
            tool_calls_made=[ToolCall(function="send_email", arguments={"to": "x"})],
            injection_payload="payload",
        )
        round_result = GanRoundResult(
            round_index=1,
            attack_result=attack,
            defense_result=DefenseEvalResult(
                sic_blocked=0,
                sic_sanitized=1,
                sic_clean=1,
                sic_bypass_rate=0.5,
                cast_interventions=2,
                cast_refusal_rate=0.0,
                cast_responses=["ok"],
            ),
            attacker_won=False,
            config_snapshot={"lr": 0.1},
            transfer_results=[
                TransferEvalResult(
                    model_id="m2",
                    success_rate=0.25,
                    eval_responses=["no"],
                ),
            ],
            environment_result=environment_result,
            defense_proxy_result=defense_proxy,
        )

        payload = round_result.to_dict()

        assert "environment_result" in payload
        assert "defense_proxy_result" in payload
        environment = cast("dict[str, object]", payload["environment_result"])
        proxy = cast("dict[str, object]", payload["defense_proxy_result"])
        assert environment["reward"] == 1.0
        assert proxy["proxy_mode"] == "both_gate"

    def test_detect_result_to_dict_includes_margin_result(self) -> None:
        margin = MarginResult(
            baseline_refusal_rate=0.6,
            curve=[
                MarginCurvePoint(
                    direction_name="json",
                    alpha=0.5,
                    refusal_rate=0.4,
                    refusal_delta=-0.2,
                ),
            ],
            collapse_alpha={"json": None},
            evidence=["stable"],
        )
        result = DetectResult(
            hardened=True,
            confidence=0.9,
            effective_rank=2.0,
            cosine_concentration=0.8,
            silhouette_peak=0.7,
            hdd_red_distance=0.3,
            residual_refusal_rate=0.2,
            mean_refusal_position=3.0,
            evidence=["ok"],
            margin_result=margin,
        )

        payload = result.to_dict()

        assert payload["margin_result"] == margin.to_dict()

    def test_margin_result_to_dict(self) -> None:
        result = MarginResult(
            baseline_refusal_rate=0.4,
            curve=[
                MarginCurvePoint(
                    direction_name="json",
                    alpha=0.2,
                    refusal_rate=0.3,
                    refusal_delta=-0.1,
                ),
            ],
            collapse_alpha={"json": 0.7},
            evidence=["e1"],
        )

        assert result.to_dict() == {
            "baseline_refusal_rate": 0.4,
            "curve": [
                {
                    "direction_name": "json",
                    "alpha": 0.2,
                    "refusal_rate": 0.3,
                    "refusal_delta": -0.1,
                },
            ],
            "collapse_alpha": {"json": 0.7},
            "evidence": ["e1"],
        }

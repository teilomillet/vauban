# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban.session — AI-agent-native interface.

Unit tests verify the discovery/state/guide layer with a mocked model.
Integration tests verify the full workflow with the real Qwen 0.5B model.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from tests.conftest import (
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
    MockCausalLM,
    MockTokenizer,
    make_direction_result,
)
from vauban import _ops as ops
from vauban.session import _TOOLS, Session
from vauban.types import (
    AuditConfig,
    AuditFinding,
    AuditResult,
    CastResult,
    DetectConfig,
    DetectResult,
    EvalResult,
    ProbeResult,
    ResponseScoreResult,
    ScanConfig,
    ScanResult,
    ScanSpan,
    SICConfig,
    SICResult,
)


@dataclass(slots=True)
class SessionHarness:
    """Bundled mocked session state for unit tests."""

    session: Session
    model: MockCausalLM
    tokenizer: MockTokenizer
    dequantized_models: list[MockCausalLM]


def _build_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    quantized: bool = False,
    harmful_prompts: list[str] | None = None,
    harmless_prompts: list[str] | None = None,
) -> SessionHarness:
    """Create a session with mocked model I/O and prompt loading."""
    model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
    ops.eval(model.parameters())
    tokenizer = MockTokenizer(VOCAB_SIZE)
    dequantized_models: list[MockCausalLM] = []
    harmful_path = Path("harmful.jsonl")
    harmless_path = Path("harmless.jsonl")
    prompt_map: dict[Path, list[str]] = {
        harmful_path: ["harmful-1", "harmful-2"],
        harmless_path: ["harmless-1", "harmless-2"],
    }

    def _fake_load_model(model_path: str) -> tuple[MockCausalLM, MockTokenizer]:
        del model_path
        return model, tokenizer

    def _fake_is_quantized(loaded_model: object) -> bool:
        assert loaded_model is model
        return quantized

    def _fake_dequantize_model(loaded_model: object) -> None:
        assert loaded_model is model
        dequantized_models.append(model)

    def _fake_default_prompt_paths() -> tuple[Path, Path]:
        return harmful_path, harmless_path

    def _fake_load_prompts(path: Path) -> list[str]:
        return list(prompt_map[path])

    monkeypatch.setattr("vauban._model_io.load_model", _fake_load_model)
    monkeypatch.setattr("vauban.dequantize.is_quantized", _fake_is_quantized)
    monkeypatch.setattr(
        "vauban.dequantize.dequantize_model",
        _fake_dequantize_model,
    )
    monkeypatch.setattr(
        "vauban.measure.default_prompt_paths",
        _fake_default_prompt_paths,
    )
    monkeypatch.setattr("vauban.measure.load_prompts", _fake_load_prompts)

    session = Session(
        "test-model",
        harmful_prompts=harmful_prompts,
        harmless_prompts=harmless_prompts,
    )
    return SessionHarness(
        session=session,
        model=model,
        tokenizer=tokenizer,
        dequantized_models=dequantized_models,
    )


def _make_detect_result(hardened: bool = False) -> DetectResult:
    """Build a minimal DetectResult."""
    return DetectResult(
        hardened=hardened,
        confidence=0.8,
        effective_rank=1.6,
        cosine_concentration=1.1,
        silhouette_peak=0.4,
        hdd_red_distance=None,
        residual_refusal_rate=None,
        mean_refusal_position=None,
        evidence=["synthetic"],
    )


def _make_eval_result() -> EvalResult:
    """Build a minimal EvalResult."""
    return EvalResult(
        refusal_rate_original=0.8,
        refusal_rate_modified=0.7,
        perplexity_original=10.0,
        perplexity_modified=10.5,
        kl_divergence=0.02,
        num_prompts=2,
    )


def _make_audit_result(
    *,
    thoroughness: str = "quick",
    overall_risk: str = "low",
    findings: list[AuditFinding] | None = None,
) -> AuditResult:
    """Build a minimal AuditResult."""
    finding_list = findings or [
        AuditFinding(
            category="defense",
            severity="medium",
            title="Synthetic finding",
            description="Synthetic description",
            evidence="Synthetic evidence",
            remediation="Synthetic remediation",
        ),
    ]
    return AuditResult(
        company_name="Test",
        system_name="Bot",
        model_path="test-model",
        thoroughness=thoroughness,
        overall_risk=overall_risk,
        findings=finding_list,
        detect_hardened=False,
        detect_confidence=0.3,
        jailbreak_success_rate=0.1,
        jailbreak_total=4,
        softprompt_success_rate=None,
        bijection_success_rate=None,
        surface_refusal_rate=None,
        surface_coverage=None,
        guard_circuit_break_rate=None,
    )


def _make_probe_result(prompt: str = "probe") -> ProbeResult:
    """Build a minimal ProbeResult."""
    return ProbeResult(
        projections=[0.1, 0.2],
        layer_count=2,
        prompt=prompt,
    )


def _make_scan_result() -> ScanResult:
    """Build a minimal ScanResult."""
    return ScanResult(
        injection_probability=0.6,
        overall_projection=0.4,
        spans=[ScanSpan(start=0, end=1, text="x", mean_projection=0.4)],
        per_token_projections=[0.4],
        flagged=True,
    )


def _make_cast_result(prompt: str = "prompt") -> CastResult:
    """Build a minimal CastResult."""
    return CastResult(
        prompt=prompt,
        text="generated",
        projections_before=[0.2, 0.4],
        projections_after=[0.1, 0.2],
        interventions=1,
        considered=2,
    )


def _make_sic_result() -> SICResult:
    """Build a minimal SICResult."""
    return SICResult(
        prompts_clean=["clean"],
        prompts_blocked=[False],
        iterations_used=[1],
        initial_scores=[0.5],
        final_scores=[0.2],
        total_blocked=0,
        total_sanitized=1,
        total_clean=0,
    )


# =========================================================================
# Tool registry (no model needed)
# =========================================================================


class TestToolRegistry:
    """Verify the tool registry is well-formed."""

    def test_all_tools_have_required_fields(self) -> None:
        for tool in _TOOLS:
            assert tool.name, f"tool missing name: {tool}"
            assert tool.description, f"{tool.name} missing description"
            assert tool.category, f"{tool.name} missing category"
            assert isinstance(tool.requires, list)
            assert isinstance(tool.produces, list)

    def test_tool_names_unique(self) -> None:
        names = [tool.name for tool in _TOOLS]
        assert len(names) == len(set(names)), f"duplicate names: {names}"

    def test_known_tools_present(self) -> None:
        names = {tool.name for tool in _TOOLS}
        for expected in [
            "measure", "detect", "audit", "probe", "scan",
            "steer", "cast", "sic", "cut", "export",
            "score", "classify", "jailbreak", "report", "ai_act",
        ]:
            assert expected in names, f"missing tool: {expected}"

    def test_categories_are_valid(self) -> None:
        valid = {
            "assessment", "inspection", "defense",
            "modification", "analysis", "attack", "reporting",
        }
        for tool in _TOOLS:
            assert tool.category in valid, (
                f"{tool.name} has invalid category: {tool.category}"
            )

    def test_no_tool_requires_itself(self) -> None:
        for tool in _TOOLS:
            assert tool.name not in tool.requires
            assert tool.name not in tool.produces


# =========================================================================
# Session init
# =========================================================================


class TestSessionInit:
    """Verify initialization logic without a live model download."""

    def test_loads_default_prompts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        harness = _build_session(monkeypatch)

        assert harness.session.harmful == ["harmful-1", "harmful-2"]
        assert harness.session.harmless == ["harmless-1", "harmless-2"]
        assert harness.dequantized_models == []

    def test_uses_custom_prompts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        harness = _build_session(
            monkeypatch,
            harmful_prompts=["custom harmful"],
            harmless_prompts=["custom harmless"],
        )

        assert harness.session.harmful == ["custom harmful"]
        assert harness.session.harmless == ["custom harmless"]

    def test_dequantizes_when_model_is_quantized(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        harness = _build_session(monkeypatch, quantized=True)

        assert harness.dequantized_models == [harness.model]


# =========================================================================
# Guide (mocked model)
# =========================================================================


class TestGuide:
    """Verify guide() returns useful content for every workflow."""

    @pytest.fixture
    def session(self, monkeypatch: pytest.MonkeyPatch) -> Session:
        return _build_session(monkeypatch).session

    def test_empty_lists_all(self, session: Session) -> None:
        guide = session.guide("")
        assert "Available" in guide
        for goal in ["audit", "compliance", "harden", "abliterate", "inspect"]:
            assert goal in guide

    @pytest.mark.parametrize("goal", [
        "audit", "compliance", "harden", "abliterate", "inspect",
    ])
    def test_each_goal_has_steps(self, session: Session, goal: str) -> None:
        guide = session.guide(goal)
        assert "1." in guide and "2." in guide, f"{goal} missing numbered steps"

    def test_unknown_goal(self, session: Session) -> None:
        guide = session.guide("nonsense")
        assert "Unknown" in guide

    def test_fuzzy_match(self, session: Session) -> None:
        guide = session.guide("audit")
        fuzzy = session.guide("audi")
        assert "audit" in guide.lower()
        assert "audit" in fuzzy.lower()


# =========================================================================
# State tracking + done() (mocked model)
# =========================================================================


class TestStateAndDone:
    """Exercise state and workflow completion helpers."""

    @pytest.fixture
    def session(self, monkeypatch: pytest.MonkeyPatch) -> Session:
        return _build_session(monkeypatch).session

    def test_initial_state(self, session: Session) -> None:
        state = session.state()
        assert state["model"] is True
        assert state["direction"] is False
        assert state["audit_result"] is False
        assert state["modified_model"] is False

    def test_initial_available(self, session: Session) -> None:
        available = session.available()
        assert "measure" in available
        assert "detect" in available
        assert "score" in available
        assert "classify" in available
        assert "probe" not in available
        assert "cut" not in available

    def test_needs_before_measure(self, session: Session) -> None:
        assert "direction" in session.needs("cut")
        assert "direction" in session.needs("probe")
        assert session.needs("score") == []

    def test_needs_unknown_tool(self, session: Session) -> None:
        assert session.needs("missing-tool") == ["unknown tool: missing-tool"]

    def test_done_audit_before_running(self, session: Session) -> None:
        done, reason = session.done("audit")
        assert done is False
        assert "audit" in reason.lower()

    def test_done_audit_after_running(self, session: Session) -> None:
        session._audit = _make_audit_result()
        done, reason = session.done("audit")
        assert done is True
        assert "report" in reason.lower()

    def test_done_unknown_goal(self, session: Session) -> None:
        done, reason = session.done("nonexistent")
        assert done is False
        assert "Unknown" in reason

    def test_done_abliterate_without_direction(self, session: Session) -> None:
        done, reason = session.done("abliterate")
        assert done is False
        assert "measure" in reason.lower()

    def test_done_inspect_harden_and_abliterate(self, session: Session) -> None:
        done, reason = session.done("inspect")
        assert done is False
        assert "measure" in reason.lower()

        session._direction = make_direction_result()
        done, reason = session.done("inspect")
        assert done is True
        assert "probe/scan/classify" in reason

        done, reason = session.done("harden")
        assert done is False
        assert "iterative" in reason.lower()

        done, reason = session.done("abliterate")
        assert done is False
        assert "run s.cut" in reason.lower()

        session._modified_weights = {"weights": ops.array([1.0])}
        done, reason = session.done("abliterate")
        assert done is True
        assert "export" in reason.lower()

    def test_done_compliance_with_standard_audit(self, session: Session) -> None:
        session._audit = _make_audit_result(thoroughness="standard")
        done, reason = session.done("compliance")
        assert done is True
        assert "complete" in reason.lower()

    def test_done_compliance_before_and_after_quick_audit(
        self,
        session: Session,
    ) -> None:
        done, reason = session.done("compliance")
        assert done is False
        assert "standard" in reason.lower()

        session._audit = _make_audit_result(thoroughness="quick")
        done, reason = session.done("compliance")
        assert done is False
        assert "quick" in reason.lower()


# =========================================================================
# suggest_next() labels (mocked model)
# =========================================================================


class TestSuggestNext:
    """Verify suggestion generation across state transitions."""

    @pytest.fixture
    def session(self, monkeypatch: pytest.MonkeyPatch) -> Session:
        return _build_session(monkeypatch).session

    def test_initial_suggests_measure(self, session: Session) -> None:
        suggestion = session.suggest_next()
        assert "measure" in suggestion
        assert "[ADVICE]" in suggestion

    def test_has_fact_and_advice_labels(self, session: Session) -> None:
        suggestion = session.suggest_next()
        assert "[ADVICE]" in suggestion
        assert "[FACT]" in suggestion

    def test_high_risk_audit_suggests_defenses(self, session: Session) -> None:
        session._direction = make_direction_result()
        session._audit = _make_audit_result(
            overall_risk="high",
            findings=[
                AuditFinding(
                    category="attack",
                    severity="high",
                    title="Jailbreak bypass detected",
                    description="desc",
                    evidence="evidence",
                    remediation="fix",
                ),
                AuditFinding(
                    category="defense",
                    severity="medium",
                    title="No hardening detected",
                    description="desc",
                    evidence="evidence",
                    remediation="fix",
                ),
            ],
        )

        suggestion = session.suggest_next()

        assert "Overall risk: high" in suggestion
        assert "s.cast" in suggestion
        assert "s.sic" in suggestion
        assert "No hardening detected" in suggestion
        assert "Jailbreak bypass" in suggestion

    def test_low_risk_audit_suggests_report(self, session: Session) -> None:
        session._direction = make_direction_result()
        session._audit = _make_audit_result(overall_risk="low")

        suggestion = session.suggest_next()

        assert "Overall risk: low" in suggestion
        assert "s.report()" in suggestion

    def test_direction_without_audit_suggests_audit_and_probe(
        self,
        session: Session,
    ) -> None:
        session._direction = make_direction_result()

        suggestion = session.suggest_next()

        assert "Direction measured" in suggestion
        assert "s.detect()" in suggestion
        assert "s.audit()" in suggestion
        assert "s.probe('prompt')" in suggestion


# =========================================================================
# Discovery and reporting helpers
# =========================================================================


class TestDiscoveryHelpers:
    """Cover describe() and catalog() helpers."""

    @pytest.fixture
    def session(self, monkeypatch: pytest.MonkeyPatch) -> Session:
        return _build_session(monkeypatch).session

    def test_describe_available_and_blocked(self, session: Session) -> None:
        measure_description = session.describe("measure")
        probe_description = session.describe("probe")

        assert "available" in measure_description
        assert "blocked" in probe_description
        assert "direction" in probe_description

    def test_describe_unknown_tool_raises(self, session: Session) -> None:
        with pytest.raises(KeyError, match="Unknown tool"):
            session.describe("missing-tool")

    def test_catalog_groups_tools_by_category(self, session: Session) -> None:
        catalog = session.catalog()
        measure_entry = next(
            entry
            for entry in catalog["assessment"]
            if entry["name"] == "measure"
        )
        probe_entry = next(
            entry
            for entry in catalog["inspection"]
            if entry["name"] == "probe"
        )

        assert measure_entry["status"] == "available"
        assert probe_entry["status"] == "blocked"

    def test_tools_returns_copy(self, session: Session) -> None:
        tools = session.tools()
        tools.pop()

        assert len(tools) + 1 == len(_TOOLS)
        assert len(session.tools()) == len(_TOOLS)


# =========================================================================
# Session tool methods (mocked dependencies)
# =========================================================================


class TestSessionTools:
    """Exercise session methods without model inference."""

    @pytest.fixture
    def harness(self, monkeypatch: pytest.MonkeyPatch) -> SessionHarness:
        return _build_session(monkeypatch)

    def test_measure_sets_direction(
        self,
        harness: SessionHarness,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        direction = make_direction_result()

        def _fake_measure(
            model: object,
            tokenizer: object,
            harmful: list[str],
            harmless: list[str],
        ) -> object:
            assert model is harness.model
            assert tokenizer is harness.tokenizer
            assert harmful == harness.session.harmful
            assert harmless == harness.session.harmless
            return direction

        monkeypatch.setattr("vauban.measure.measure", _fake_measure)

        result = harness.session.measure()

        assert result is direction
        assert harness.session.state()["direction"] is True

    def test_detect_sets_detect_result(
        self,
        harness: SessionHarness,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        detect_result = _make_detect_result()
        detect_module = importlib.import_module("vauban.detect")

        def _fake_detect(
            model: object,
            tokenizer: object,
            harmful: list[str],
            harmless: list[str],
            config: DetectConfig,
        ) -> DetectResult:
            assert model is harness.model
            assert tokenizer is harness.tokenizer
            assert harmful == harness.session.harmful
            assert harmless == harness.session.harmless
            assert config.mode == "full"
            return detect_result

        monkeypatch.setattr(detect_module, "detect", _fake_detect)

        result = harness.session.detect()

        assert result is detect_result
        assert harness.session.state()["detect_result"] is True

    def test_evaluate_uses_baseline_model(
        self,
        harness: SessionHarness,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        eval_result = _make_eval_result()
        harness.session.harmful = [f"prompt-{idx}" for idx in range(12)]

        def _fake_evaluate(
            original: object,
            modified: object,
            tokenizer: object,
            prompts: list[str],
        ) -> EvalResult:
            assert original is harness.model
            assert modified is harness.model
            assert tokenizer is harness.tokenizer
            assert prompts == harness.session.harmful[:10]
            return eval_result

        monkeypatch.setattr("vauban.evaluate.evaluate", _fake_evaluate)

        assert harness.session.evaluate() is eval_result

    def test_audit_stores_result_and_passes_direction(
        self,
        harness: SessionHarness,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        direction = make_direction_result()
        audit_result = _make_audit_result()
        harness.session._direction = direction

        def _fake_run_audit(
            model: object,
            tokenizer: object,
            harmful: list[str],
            harmless: list[str],
            config: AuditConfig,
            model_path: str,
            *,
            direction_result: object,
        ) -> AuditResult:
            assert model is harness.model
            assert tokenizer is harness.tokenizer
            assert harmful == harness.session.harmful
            assert harmless == harness.session.harmless
            assert config.company_name == "ACME"
            assert config.system_name == "Shield"
            assert config.thoroughness == "standard"
            assert model_path == "test-model"
            assert direction_result is direction
            return audit_result

        monkeypatch.setattr("vauban.audit.run_audit", _fake_run_audit)

        result = harness.session.audit(
            company_name="ACME",
            system_name="Shield",
            thoroughness="standard",
        )

        assert result is audit_result
        assert harness.session.state()["audit_result"] is True

    def test_probe_lazily_measures_direction(
        self,
        harness: SessionHarness,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        direction = make_direction_result()
        probe_result = _make_probe_result("How?")

        monkeypatch.setattr(
            "vauban.measure.measure",
            lambda model, tokenizer, harmful, harmless: direction,
        )

        def _fake_probe(
            model: object,
            tokenizer: object,
            prompt: str,
            direction_array: object,
        ) -> ProbeResult:
            assert model is harness.model
            assert tokenizer is harness.tokenizer
            assert prompt == "How?"
            assert direction_array is direction.direction
            return probe_result

        monkeypatch.setattr("vauban.probe.probe", _fake_probe)

        result = harness.session.probe("How?")

        assert result is probe_result
        assert harness.session._direction is direction

    def test_scan_uses_direction_and_layer(
        self,
        harness: SessionHarness,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        direction = make_direction_result(layer_index=1)
        scan_result = _make_scan_result()
        harness.session._direction = direction
        scan_module = importlib.import_module("vauban.scan")

        def _fake_scan(
            model: object,
            tokenizer: object,
            content: str,
            config: ScanConfig,
            direction_array: object,
            layer_index: int,
        ) -> ScanResult:
            assert model is harness.model
            assert tokenizer is harness.tokenizer
            assert content == "payload"
            assert config.target_layer is None
            assert direction_array is direction.direction
            assert layer_index == 1
            return scan_result

        monkeypatch.setattr(scan_module, "scan", _fake_scan)

        assert harness.session.scan("payload") is scan_result

    def test_steer_uses_all_model_layers(
        self,
        harness: SessionHarness,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        direction = make_direction_result()
        harness.session._direction = direction
        captured: dict[str, object] = {}

        def _fake_steer(
            model: object,
            tokenizer: object,
            prompt: str,
            direction_array: object,
            layers: list[int],
            alpha: float,
        ) -> str:
            captured["model"] = model
            captured["tokenizer"] = tokenizer
            captured["prompt"] = prompt
            captured["direction"] = direction_array
            captured["layers"] = layers
            captured["alpha"] = alpha
            return "steered"

        monkeypatch.setattr("vauban.probe.steer", _fake_steer)

        result = harness.session.steer("prompt", alpha=0.5)

        assert result == "steered"
        assert captured["model"] is harness.model
        assert captured["tokenizer"] is harness.tokenizer
        assert captured["prompt"] == "prompt"
        assert captured["direction"] is direction.direction
        assert captured["layers"] == [0, 1]
        assert captured["alpha"] == 0.5

    def test_cast_and_sic(
        self,
        harness: SessionHarness,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        direction = make_direction_result(layer_index=1)
        cast_result = _make_cast_result()
        sic_result = _make_sic_result()
        harness.session._direction = direction

        def _fake_cast_generate(
            model: object,
            tokenizer: object,
            prompt: str,
            direction_array: object,
            layers: list[int],
            alpha: float,
            threshold: float,
        ) -> CastResult:
            assert model is harness.model
            assert tokenizer is harness.tokenizer
            assert prompt == "prompt"
            assert direction_array is direction.direction
            assert layers == [0, 1]
            assert alpha == 0.3
            assert threshold == 0.2
            return cast_result

        def _fake_sic(
            model: object,
            tokenizer: object,
            prompts: list[str],
            config: SICConfig,
            direction_array: object,
            layer_index: int,
        ) -> SICResult:
            assert model is harness.model
            assert tokenizer is harness.tokenizer
            assert prompts == ["payload"]
            assert config.mode == "direction"
            assert direction_array is direction.direction
            assert layer_index == 1
            return sic_result

        monkeypatch.setattr("vauban.cast.cast_generate", _fake_cast_generate)
        monkeypatch.setattr("vauban.sic.sic", _fake_sic)

        assert harness.session.cast("prompt", alpha=0.3, threshold=0.2) is cast_result
        assert harness.session.sic(["payload"]) is sic_result

    def test_cut_flattens_weights_and_exports(
        self,
        harness: SessionHarness,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        direction = make_direction_result()
        harness.session._direction = direction
        captured: dict[str, object] = {}
        modified_weights = {"modified.weight": ops.array([1.0])}
        monkeypatch.setattr(
            harness.model,
            "parameters",
            lambda: {
                "top": ops.array([1.0]),
                "model": {
                    "block": {"weight": ops.array([2.0])},
                    "norm": ops.array([3.0]),
                },
            },
        )

        def _fake_cut(
            flat_weights: dict[str, object],
            direction_array: object,
            layers: list[int],
            alpha: float,
            norm_preserve: bool,
        ) -> dict[str, object]:
            captured["flat_weights"] = flat_weights
            captured["direction"] = direction_array
            captured["layers"] = layers
            captured["alpha"] = alpha
            captured["norm_preserve"] = norm_preserve
            return modified_weights

        export_calls: list[tuple[str, dict[str, object], str]] = []

        def _fake_export_model(
            model_path: str,
            weights: dict[str, object],
            output_dir: str,
        ) -> None:
            export_calls.append((model_path, weights, output_dir))

        monkeypatch.setattr("vauban.cut.cut", _fake_cut)
        monkeypatch.setattr("vauban.export.export_model", _fake_export_model)

        cut_result = harness.session.cut(alpha=0.5, norm_preserve=True)
        export_result = harness.session.export("output-dir")

        flat_weights = cast("dict[str, object]", captured["flat_weights"])
        assert set(flat_weights) == {"top", "model.block.weight", "model.norm"}
        assert captured["direction"] is direction.direction
        assert captured["layers"] == [0, 1]
        assert captured["alpha"] == 0.5
        assert captured["norm_preserve"] is True
        assert cut_result == modified_weights
        assert export_result == "output-dir"
        assert export_calls == [("test-model", modified_weights, "output-dir")]

    def test_export_requires_cut_first(self, harness: SessionHarness) -> None:
        with pytest.raises(RuntimeError, match="Run cut"):
            harness.session.export("output-dir")

    def test_report_markdown_dict_and_pdf(
        self,
        harness: SessionHarness,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        audit_result = _make_audit_result()
        harness.session._audit = audit_result

        monkeypatch.setattr(
            "vauban.audit.audit_result_to_markdown",
            lambda result: f"# report for {result.company_name}",
        )
        monkeypatch.setattr(
            "vauban.audit.audit_result_to_dict",
            lambda result: {"company_name": result.company_name},
        )
        monkeypatch.setattr(
            "vauban.audit_pdf.render_audit_report_pdf",
            lambda result: b"%PDF-test",
        )

        markdown = harness.session.report()
        report_dict = harness.session.report(fmt="dict")
        pdf = harness.session.report_pdf()

        assert markdown == "# report for Test"
        assert '"company_name": "Test"' in report_dict
        assert pdf == b"%PDF-test"

    def test_report_requires_audit_and_valid_format(
        self,
        harness: SessionHarness,
    ) -> None:
        with pytest.raises(RuntimeError, match="Run audit"):
            harness.session.report()
        with pytest.raises(RuntimeError, match="Run audit"):
            harness.session.report_pdf()

        harness.session._audit = _make_audit_result()
        with pytest.raises(ValueError, match="Unknown format"):
            harness.session.report(fmt="yaml")


# =========================================================================
# Static tools (no model inference needed)
# =========================================================================


class TestStaticTools:
    """Static helpers should work without creating a Session instance."""

    def test_score(self) -> None:
        result = Session.score("How to cook?", "Boil water, add pasta.")
        assert isinstance(result, ResponseScoreResult)
        assert 0.0 <= result.composite <= 1.0

    def test_classify(self) -> None:
        result = Session.classify("hack exploit SQL injection")
        assert isinstance(result, list)
        assert len(result) > 0


# =========================================================================
# Integration: full workflow with real model
# =========================================================================


@pytest.mark.integration
class TestSessionIntegration:
    """Full workflow: measure → probe → audit → report."""

    @pytest.fixture(scope="class")
    def session(self) -> Session:
        return Session("mlx-community/Qwen2.5-0.5B-Instruct-bf16")

    def test_measure(self, session: Session) -> None:
        direction = session.measure()
        assert direction.d_model > 0
        assert direction.layer_index >= 0
        assert "probe" in session.available()
        assert "cut" in session.available()

    def test_probe_after_measure(self, session: Session) -> None:
        result = session.probe("How do I pick a lock?")
        assert len(result.projections) > 0

    def test_detect(self, session: Session) -> None:
        result = session.detect()
        assert isinstance(result.hardened, bool)
        assert 0.0 <= result.confidence <= 1.0

    def test_audit(self, session: Session) -> None:
        result = session.audit(
            company_name="Test",
            system_name="Bot",
            thoroughness="quick",
        )
        assert result.overall_risk in {"critical", "high", "medium", "low"}
        assert len(result.findings) > 0
        done, _reason = session.done("audit")
        assert done is True

    def test_done_compliance_needs_standard(
        self,
        session: Session,
    ) -> None:
        done, reason = session.done("compliance")
        assert done is False
        assert "standard" in reason

    def test_suggest_after_audit(self, session: Session) -> None:
        suggestion = session.suggest_next()
        assert "[FACT]" in suggestion
        assert "risk" in suggestion.lower()
        assert "done" in suggestion.lower()

    def test_report_markdown(self, session: Session) -> None:
        markdown = session.report()
        assert "# Red-Team Audit Report" in markdown
        assert "Test" in markdown

    def test_report_pdf(self, session: Session) -> None:
        pdf = session.report_pdf()
        assert pdf[:5] == b"%PDF-"

    def test_cast_defense(self, session: Session) -> None:
        result = session.cast("How to hack a computer?", threshold=0.3)
        assert hasattr(result, "interventions")
        assert result.interventions >= 0

    def test_cut_and_state(self, session: Session) -> None:
        session.cut(alpha=0.5)
        assert session.state()["modified_model"] is True
        assert "export" in session.available()
        done, _reason = session.done("abliterate")
        assert done is True

# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra coverage tests for config validation rules."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from vauban.config._validation_models import (
    ValidationCollector,
    ValidationContext,
)
from vauban.config._validation_rules import (
    _rule_ai_act_readiness,
    _rule_eval_prompts,
    _rule_prompt_sources,
    _rule_refusal_phrases,
    _rule_surface_prompts,
)
from vauban.types import (
    AIActConfig,
    EvalConfig,
    ObjectiveConfig,
    ObjectiveMetricSpec,
    PipelineConfig,
    SurfaceConfig,
)


def _write_jsonl(path: Path, count: int) -> None:
    lines = [json.dumps({"prompt": f"prompt {i}"}) for i in range(count)]
    path.write_text("\n".join(lines))


def _make_context(
    *,
    harmful_path: Path,
    harmless_path: Path,
    eval_config: EvalConfig | None = None,
    surface_config: SurfaceConfig | None = None,
    ai_act_config: AIActConfig | None = None,
    borderline_path: Path | None = None,
    objective_config: ObjectiveConfig | None = None,
) -> ValidationContext:
    config = PipelineConfig(
        model_path="test-model",
        harmful_path=harmful_path,
        harmless_path=harmless_path,
        eval=eval_config or EvalConfig(),
        surface=surface_config,
        ai_act=ai_act_config,
        borderline_path=borderline_path,
        objective=objective_config,
        output_dir=Path("/tmp/output"),
    )
    return ValidationContext(
        config_path=Path("/tmp/config.toml"),
        raw={},
        config=config,
    )


def _make_ai_act() -> AIActConfig:
    return AIActConfig(
        company_name="Acme",
        system_name="Assistant",
        intended_purpose="testing",
    )


class TestRulePromptSourcesExtra:
    def test_imbalance_and_borderline_warning(self, tmp_path: Path) -> None:
        harmful = tmp_path / "harmful.jsonl"
        harmless = tmp_path / "harmless.jsonl"
        borderline = tmp_path / "borderline.jsonl"
        _write_jsonl(harmful, 20)
        _write_jsonl(harmless, 2)
        _write_jsonl(borderline, 1)

        ctx = _make_context(
            harmful_path=harmful,
            harmless_path=harmless,
            borderline_path=borderline,
        )
        collector = ValidationCollector()

        _rule_prompt_sources(ctx, collector)

        rendered = collector.render()
        assert any("highly imbalanced" in item for item in rendered)
        assert any("[data].borderline" in item for item in rendered)

    def test_objective_dataset_is_validated(self, tmp_path: Path) -> None:
        harmful = tmp_path / "harmful.jsonl"
        harmless = tmp_path / "harmless.jsonl"
        benign = tmp_path / "benign.jsonl"
        _write_jsonl(harmful, 4)
        _write_jsonl(harmless, 4)
        benign.write_text('{"prompt": ""}\n')

        ctx = _make_context(
            harmful_path=harmful,
            harmless_path=harmless,
            objective_config=ObjectiveConfig(
                name="customer_support_gate",
                benign_inquiry_source="dataset",
                benign_inquiries_path=benign,
                utility=[
                    ObjectiveMetricSpec(
                        metric="utility_score",
                        threshold=0.90,
                        comparison="at_least",
                    ),
                ],
            ),
        )
        collector = ValidationCollector()

        _rule_prompt_sources(ctx, collector)

        rendered = collector.render()
        assert any("[objective].benign_inquiries" in item for item in rendered)


class TestRuleEvalPromptsExtra:
    def test_small_eval_set_warns(self, tmp_path: Path) -> None:
        harmful = tmp_path / "harmful.jsonl"
        harmless = tmp_path / "harmless.jsonl"
        eval_path = tmp_path / "eval.jsonl"
        _write_jsonl(harmful, 4)
        _write_jsonl(harmless, 4)
        _write_jsonl(eval_path, 2)

        ctx = _make_context(
            harmful_path=harmful,
            harmless_path=harmless,
            eval_config=EvalConfig(prompts_path=eval_path),
        )
        collector = ValidationCollector()

        _rule_eval_prompts(ctx, collector)

        rendered = collector.render()
        assert any("[eval].prompts has only 2 prompt(s)" in item for item in rendered)


class TestRuleRefusalPhrasesExtra:
    def test_missing_refusal_phrase_file_warns(self, tmp_path: Path) -> None:
        harmful = tmp_path / "harmful.jsonl"
        harmless = tmp_path / "harmless.jsonl"
        _write_jsonl(harmful, 4)
        _write_jsonl(harmless, 4)

        ctx = _make_context(
            harmful_path=harmful,
            harmless_path=harmless,
            eval_config=EvalConfig(refusal_phrases_path=tmp_path / "missing.txt"),
        )
        collector = ValidationCollector()

        _rule_refusal_phrases(ctx, collector)

        rendered = collector.render()
        assert any("file not found" in item for item in rendered)

    def test_short_refusal_phrase_file_warns(self, tmp_path: Path) -> None:
        harmful = tmp_path / "harmful.jsonl"
        harmless = tmp_path / "harmless.jsonl"
        refusal_path = tmp_path / "refusal.txt"
        _write_jsonl(harmful, 4)
        _write_jsonl(harmless, 4)
        refusal_path.write_text("I cannot comply\n")

        ctx = _make_context(
            harmful_path=harmful,
            harmless_path=harmless,
            eval_config=EvalConfig(refusal_phrases_path=refusal_path),
        )
        collector = ValidationCollector()

        _rule_refusal_phrases(ctx, collector)

        rendered = collector.render()
        assert any("only 1 phrase" in item for item in rendered)


class TestRuleSurfacePromptsExtra:
    @pytest.mark.parametrize("prompts_path", ["default", "default_multilingual"])
    def test_default_surface_paths_and_gate_warning(
        self,
        prompts_path: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        harmful = tmp_path / "harmful.jsonl"
        harmless = tmp_path / "harmless.jsonl"
        _write_jsonl(harmful, 4)
        _write_jsonl(harmless, 4)
        seen: list[Path] = []

        monkeypatch.setattr(
            "vauban.config._validation_rules.default_surface_path",
            lambda: tmp_path / "surface.jsonl",
        )
        monkeypatch.setattr(
            "vauban.config._validation_rules.default_multilingual_surface_path",
            lambda: tmp_path / "surface_multilingual.jsonl",
        )

        def fake_validate(
            surface_path: Path,
            _label: str,
            collector: ValidationCollector,
            *,
            missing_fix: str,
        ) -> int:
            seen.append(surface_path)
            collector.add("LOW", f"validated {surface_path}", fix=missing_fix)
            return 4

        monkeypatch.setattr(
            "vauban.config._validation_rules._validate_surface_jsonl_file",
            fake_validate,
        )

        ctx = _make_context(
            harmful_path=harmful,
            harmless_path=harmless,
            surface_config=SurfaceConfig(
                prompts_path=prompts_path,
                generate=False,
                max_worst_cell_refusal_after=0.1,
            ),
        )
        collector = ValidationCollector()

        _rule_surface_prompts(ctx, collector)

        assert seen == [
            tmp_path / (
                "surface_multilingual.jsonl"
                if prompts_path == "default_multilingual"
                else "surface.jsonl"
            ),
        ]
        rendered = collector.render()
        assert any("projection-only mode" in item for item in rendered)

    def test_custom_surface_path_string(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        harmful = tmp_path / "harmful.jsonl"
        harmless = tmp_path / "harmless.jsonl"
        _write_jsonl(harmful, 4)
        _write_jsonl(harmless, 4)
        seen: list[Path] = []

        def fake_validate(
            surface_path: Path,
            _label: str,
            collector: ValidationCollector,
            *,
            missing_fix: str,
        ) -> int:
            seen.append(surface_path)
            collector.add("LOW", "validated custom", fix=missing_fix)
            return 4

        monkeypatch.setattr(
            "vauban.config._validation_rules._validate_surface_jsonl_file",
            fake_validate,
        )

        ctx = _make_context(
            harmful_path=harmful,
            harmless_path=harmless,
            surface_config=SurfaceConfig(
                prompts_path="surface/custom.jsonl",
                generate=False,
                max_worst_cell_refusal_after=0.1,
            ),
        )
        collector = ValidationCollector()
        _rule_surface_prompts(ctx, collector)

        assert seen == [Path("surface/custom.jsonl")]


class TestRuleAiActReadinessExtra:
    def test_missing_evidence_and_transparency_warnings(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        harmful = tmp_path / "harmful.jsonl"
        harmless = tmp_path / "harmless.jsonl"
        _write_jsonl(harmful, 4)
        _write_jsonl(harmless, 4)
        report = tmp_path / "report.txt"
        report.write_text("ok")

        ai_act = replace(
            _make_ai_act(),
            eu_market=True,
            role="provider",
            interacts_with_natural_persons=True,
            interaction_obvious_to_persons=False,
            ai_literacy_record=None,
            transparency_notice=None,
            human_oversight_procedure=tmp_path / "missing_human.txt",
            incident_response_procedure=tmp_path / "missing_incident.txt",
            provider_documentation=tmp_path / "missing_provider.txt",
            operation_monitoring_procedure=tmp_path / "missing_monitoring.txt",
            input_data_governance_procedure=tmp_path / "missing_governance.txt",
            log_retention_procedure=tmp_path / "missing_logs.txt",
            employee_or_worker_representative_notice=tmp_path / "missing_notice.txt",
            affected_person_notice=tmp_path / "missing_affected.txt",
            explanation_request_procedure=tmp_path / "missing_explanation.txt",
            eu_database_registration_record=tmp_path / "missing_database.txt",
            technical_report_paths=[report, tmp_path / "missing_report.txt"],
            bundle_signature_secret_env="VAUBAN_TEST_SECRET",
            uses_emotion_recognition=True,
            uses_biometric_categorization=True,
            real_time_remote_biometric_identification_exception_claimed=True,
            materially_distorts_behavior_causing_significant_harm=True,
            deepfake_creative_satirical_artistic_or_fictional_context=True,
            public_interest_text_editorial_responsibility=True,
            public_interest_text_human_review_or_editorial_control=True,
            publishes_text_on_matters_of_public_interest=False,
        )
        ctx = _make_context(
            harmful_path=harmful,
            harmless_path=harmless,
            ai_act_config=ai_act,
        )
        collector = ValidationCollector()

        monkeypatch.delenv("VAUBAN_TEST_SECRET", raising=False)
        _rule_ai_act_readiness(ctx, collector)

        rendered = collector.render()
        assert any("file not found" in item for item in rendered)
        assert any("Article 4 evidence" in item for item in rendered)
        assert any("Article 50 transparency scenario" in item for item in rendered)
        assert any("environment variable" in item for item in rendered)

    def test_high_risk_deployer_and_carve_out_warnings(
        self,
        tmp_path: Path,
    ) -> None:
        harmful = tmp_path / "harmful.jsonl"
        harmless = tmp_path / "harmless.jsonl"
        _write_jsonl(harmful, 4)
        _write_jsonl(harmless, 4)

        ai_act = replace(
            _make_ai_act(),
            eu_market=True,
            role="deployer",
            annex_iii_narrow_procedural_task=True,
            annex_iii_does_not_materially_influence_decision_outcome=False,
            annex_iii_use_cases=[],
            employment_or_workers_management=True,
            provides_input_data_for_high_risk_system=True,
            decision_with_legal_or_similarly_significant_effects=True,
            makes_or_assists_decisions_about_natural_persons=False,
            annex_i_third_party_conformity_assessment=True,
            annex_i_product_or_safety_component=False,
        )
        ctx = _make_context(
            harmful_path=harmful,
            harmless_path=harmless,
            ai_act_config=ai_act,
        )
        collector = ValidationCollector()

        _rule_ai_act_readiness(ctx, collector)

        rendered = collector.render()
        assert any("high-risk deployer scenario" in item for item in rendered)
        assert any("Article 6(3) carve-out" in item for item in rendered)
        assert any("legacy high-risk area flags" in item for item in rendered)

# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra branch coverage for AI Act helper and triage paths."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

import vauban.ai_act as ai_act_module
from vauban.types import AIActConfig


def _base_config() -> AIActConfig:
    """Build a minimal AI Act config for helper-focused tests."""
    return AIActConfig(
        company_name="Example Energy",
        system_name="Customer Assistant",
        intended_purpose="Answers customer questions.",
    )


class TestTechnicalMetricHelpers:
    """Tests for numeric metric extraction and technical finding helpers."""

    def test_extract_numeric_metrics_skips_booleans_and_non_string_keys(
        self,
    ) -> None:
        metrics = ai_act_module._extract_numeric_metrics(
            "",
            {
                "outer": [1, True, {"inner": 2, 3: 4}],
                "score": 5.5,
            },
        )

        assert metrics == [
            ("outer[0]", 1.0),
            ("outer[2].inner", 2.0),
            ("score", 5.5),
        ]

    def test_token_matches_metric_term_supports_evasion_aliases(self) -> None:
        assert ai_act_module._token_matches_metric_term("evasion", "evade")
        assert ai_act_module._token_matches_metric_term("evaded", "evade")

    def test_technical_metric_rule_skips_invalid_rules_and_matches_aliases(
        self,
    ) -> None:
        rules = [
            {"kind": 1, "match_any": ["evasion"], "high_threshold": 0.5},
            {"kind": "evasion", "match_any": "bad", "high_threshold": 0.5},
            {"kind": "evasion", "match_any": ["evasion"], "high_threshold": "bad"},
            {"kind": "evasion", "match_any": ["evade"], "high_threshold": 0.25},
        ]

        with patch.object(
            ai_act_module,
            "_rulebook_technical_metric_rules",
            return_value=rules,
        ):
            assert ai_act_module._technical_metric_rule("model.evasion_rate") == {
                "kind": "evasion",
                "high_threshold": 0.25,
            }

    def test_technical_metric_rule_returns_none_without_match(self) -> None:
        with patch.object(
            ai_act_module,
            "_rulebook_technical_metric_rules",
            return_value=[
                {"kind": "evasion", "match_any": ["accuracy"], "high_threshold": 0.5},
            ],
        ):
            assert ai_act_module._technical_metric_rule("model.attack_rate") is None

    def test_technical_findings_from_metrics_filters_invalid_entries(
        self,
        tmp_path: Path,
    ) -> None:
        findings = ai_act_module._technical_findings_from_metrics(
            [
                {
                    "metric": 1,
                    "value": 1.0,
                    "kind": "attack",
                    "high_threshold": 0.5,
                },
                {
                    "metric": "bad_value",
                    "value": "oops",
                    "kind": "attack",
                    "high_threshold": 0.5,
                },
                {
                    "metric": "bad_kind",
                    "value": 1.0,
                    "kind": 2,
                    "high_threshold": 0.5,
                },
                {
                    "metric": "bad_threshold",
                    "value": 1.0,
                    "kind": "attack",
                    "high_threshold": 1,
                },
                {
                    "metric": "non_positive",
                    "value": 0.0,
                    "kind": "attack",
                    "high_threshold": 0.5,
                },
                {
                    "metric": "attack_rate",
                    "value": 0.75,
                    "kind": "attack",
                    "high_threshold": 0.5,
                },
            ],
            "file.technical_report.0",
            tmp_path / "report.json",
        )

        assert len(findings) == 1
        assert findings[0]["severity"] == "high"
        assert findings[0]["risk_id"] == "technical.file.technical_report.0.attack_rate"

    def test_summarize_one_technical_artifact_covers_missing_unreadable_text_and_json(
        self,
        tmp_path: Path,
    ) -> None:
        missing_path = tmp_path / "missing.json"
        summary, findings = ai_act_module._summarize_one_technical_artifact(
            missing_path,
            "file.technical_report.0",
        )
        assert summary["parsed_as"] == "missing"
        assert findings == []

        unreadable_path = tmp_path / "unreadable.json"
        unreadable_path.write_text("{}\n")
        with patch.object(
            Path,
            "read_text",
            side_effect=OSError,
        ):
            summary, findings = ai_act_module._summarize_one_technical_artifact(
                unreadable_path,
                "file.technical_report.1",
            )
        assert summary["parsed_as"] == "unreadable"
        assert findings == []

        text_path = tmp_path / "text.json"
        text_path.write_text("{not-json}\n")
        summary, findings = ai_act_module._summarize_one_technical_artifact(
            text_path,
            "file.technical_report.2",
        )
        assert summary["parsed_as"] == "text"
        assert findings == []

        json_path = tmp_path / "json.json"
        json_path.write_text("{\"attack_rate\": 0.75}\n")
        with patch.object(
            ai_act_module,
            "_rulebook_technical_metric_rules",
            return_value=[
                {"kind": "attack", "match_any": ["attack"], "high_threshold": 0.5},
            ],
        ):
            summary, findings = ai_act_module._summarize_one_technical_artifact(
                json_path,
                "file.technical_report.3",
            )
        assert summary["parsed_as"] == "json"
        assert summary["metrics"] == [
            {
                "metric": "attack_rate",
                "value": 0.75,
                "kind": "attack",
                "high_threshold": 0.5,
            },
        ]
        assert findings[0]["risk_id"] == "technical.file.technical_report.3.attack_rate"


class TestRenderingAndFormattingHelpers:
    """Tests for helper renderers and formatting utilities."""

    def test_control_risk_severity_and_risk_register(self) -> None:
        assert ai_act_module._control_risk_severity(
            {"status": "pass", "blocking": False, "legal_review_required": False},
        ) == "low"
        assert ai_act_module._control_risk_severity(
            {"status": "fail", "blocking": False, "legal_review_required": False},
        ) == "medium"
        assert ai_act_module._control_risk_severity(
            {"status": "unknown", "blocking": True, "legal_review_required": False},
        ) == "high"
        assert ai_act_module._control_risk_severity(
            {"status": "pass", "blocking": False, "legal_review_required": True},
        ) == "high"

        controls = [
            {"control_id": "c1", "title": "Control 1", "source_citations": []},
            {"control_id": "c2", "title": "Control 2", "source_citations": []},
        ]
        coverage = [
            {
                "control_id": "c1",
                "applies": True,
                "status": "fail",
                "blocking": False,
                "claim_kind": "derived_by_rule",
                "rationale": "r1",
                "evidence_ids": [],
                "owner_action": "fix c1",
                "recheck_hint": "again",
                "legal_review_required": False,
            },
            {
                "control_id": "c2",
                "applies": False,
                "status": "unknown",
                "blocking": True,
                "claim_kind": "derived_by_rule",
                "rationale": "r2",
                "evidence_ids": [],
                "owner_action": "skip c2",
                "recheck_hint": "again",
                "legal_review_required": False,
            },
        ]
        technical_findings = [
            {
                "risk_id": "technical.file.technical_report.0.attack_rate",
                "source": "technical_artifact",
                "severity": "low",
                "summary": "Observed attack_rate=0.75 in report.json.",
                "next_action": "Review the report.",
            },
        ]
        register = ai_act_module._build_risk_register(
            coverage,
            controls,
            technical_findings,
            {"version": "v1"},
        )
        assert register["summary"] == {
            "n_items": 2,
            "n_high": 0,
            "n_medium": 1,
            "n_low": 1,
        }

    def test_format_helpers_and_priority_actions(self) -> None:
        assert ai_act_module._format_bool(True) == "yes"
        assert ai_act_module._format_bool(False) == "no"
        assert ai_act_module._format_bool("maybe") == "unknown"
        assert ai_act_module._format_source_citations("bad") == "none"
        assert ai_act_module._format_source_citations(
            [
                {"title": "AI Act", "url": "https://example.test"},
                {"title": 1, "url": "bad"},
                {"title": "Rulebook", "url": "https://rulebook.test"},
            ],
        ) == "AI Act (https://example.test); Rulebook (https://rulebook.test)"
        assert ai_act_module._priority_actions(
            {
                "required_client_actions": [
                    {"owner_action": 1},
                    {"owner_action": "Review notice"},
                    {"owner_action": "Review notice"},
                    {"owner_action": "Attach provider docs"},
                ],
            },
            limit=2,
        ) == ["Review notice", "Attach provider docs"]

    def test_render_executive_summary_covers_empty_and_reported_sections(
        self,
    ) -> None:
        config = _base_config()
        report = {
            "overall_status": "blocked",
            "risk_level": "high",
            "bundle_fingerprint": "fingerprint",
            "likely_obligations": [],
            "unresolved_controls": ["control.one"],
            "blocking_controls": [],
            "required_client_actions": [],
        }
        risk_register = {
            "items": [
                {
                    "risk_id": "risk.one",
                    "severity": "high",
                    "summary": "High risk item.",
                    "next_action": "Do the thing.",
                },
            ],
            "summary": "not-a-dict",
        }

        markdown = ai_act_module._render_executive_summary_markdown(
            config,
            report,
            risk_register,
        )
        assert "Review the unresolved controls" in markdown
        assert "risk.one [high]" in markdown
        assert "Risk register summary was unavailable." in markdown

    def test_render_auditor_appendix_handles_empty_payloads(self) -> None:
        config = _base_config()
        report = {
            "system": {},
            "rulebook": {},
            "integrity": {},
            "overall_status": "ready",
            "risk_level": "low",
            "bundle_fingerprint": "fingerprint",
            "evidence_manifest_sha256": "sha256",
            "technical_artifacts": {},
            "technical_findings": [],
            "sources": [],
            "epistemic_policy": {"claim_kinds": []},
        }
        markdown = ai_act_module._render_auditor_appendix_markdown(
            config,
            report,
            {"rows": []},
            {"items": []},
            {"evidence": []},
        )
        assert "No control rows were available." in markdown
        assert "No evidence entries were recorded." in markdown
        assert "No technical findings were interpreted" in markdown
        assert "No open risk-register items remain" in markdown
        assert "No official sources were attached." in markdown

    def test_render_fria_prep_handles_empty_checklist_and_questionnaire_variants(
        self,
    ) -> None:
        md_with_items = ai_act_module._render_fria_prep_markdown(
            {
                "status": "review_needed",
                "required": True,
                "control_id": "ai_act.triage.fria_requirement",
                "trigger_titles": [],
                "evidence_checklist": [],
                "questionnaire": [
                    None,
                    {
                        "prompt": "Describe the deployment context.",
                        "why_it_matters": "FRIA depends on it.",
                    },
                ],
                "next_action": "Escalate for review.",
            },
        )
        assert "No checklist items were generated." in md_with_items
        assert "Describe the deployment context." in md_with_items
        assert "Why it matters: FRIA depends on it." in md_with_items
        assert "Escalate for review." in md_with_items

        md_without_list = ai_act_module._render_fria_prep_markdown(
            {
                "status": "out_of_scope",
                "required": False,
                "control_id": "ai_act.triage.fria_requirement",
                "trigger_titles": ["Trigger"],
                "evidence_checklist": ["Checklist item"],
                "questionnaire": "not-a-list",
            },
        )
        assert "No questionnaire items were generated." in md_without_list


class TestRuleHelpers:
    """Tests for helper rules used by the AI Act triage logic."""

    def test_declared_article6_3_conditions_and_annex_iii_use_cases(
        self,
    ) -> None:
        config = replace(
            _base_config(),
            annex_iii_narrow_procedural_task=True,
            annex_iii_improves_completed_human_activity=True,
            annex_iii_detects_decision_pattern_deviations=True,
            annex_iii_preparatory_task=True,
            annex_iii_use_cases=[
                "annex_iii_8_justice_democracy_generic",
                "annex_iii_3_education_generic",
            ],
            uses_emotion_recognition=True,
            uses_biometric_categorization=True,
            biometric_or_emotion_related_use=True,
            education_or_vocational_training=True,
            employment_or_workers_management=True,
            essential_private_or_public_service=True,
            creditworthiness_or_credit_score_assessment=True,
            life_or_health_insurance_risk_pricing=True,
            emergency_first_response_dispatch=True,
            law_enforcement_use=True,
            migration_or_border_management_use=True,
            administration_of_justice_or_democracy_use=True,
        )

        assert ai_act_module._declared_article6_3_conditions(config) == [
            "narrow_procedural_task",
            "improves_completed_human_activity",
            "detects_decision_pattern_deviations",
            "preparatory_task",
        ]
        assert ai_act_module._declared_annex_iii_use_cases(config) == [
            "annex_iii_1_biometric_categorisation",
            "annex_iii_1_biometrics_generic",
            "annex_iii_1_emotion_recognition",
            "annex_iii_3_education_generic",
            "annex_iii_4_employment_generic",
            "annex_iii_5_creditworthiness_or_credit_score",
            "annex_iii_5_emergency_dispatch",
            "annex_iii_5_essential_services_generic",
            "annex_iii_5_life_or_health_insurance",
            "annex_iii_6_law_enforcement_generic",
            "annex_iii_7_migration_asylum_border_generic",
            "annex_iii_8_justice_democracy_generic",
        ]

    def test_annex_iii_area_ids_and_high_risk_triggers(self) -> None:
        config = replace(
            _base_config(),
            annex_i_product_or_safety_component=True,
            annex_i_third_party_conformity_assessment=True,
            annex_iii_use_cases=["annex_iii_3_education_generic"],
            uses_profiling_or_similarly_significant_decision_support=True,
        )

        assert ai_act_module._annex_iii_area_ids(
            ["annex_iii_3_education_generic", "missing"],
        ) == {"3"}
        assert ai_act_module._annex_i_route_declared(config)
        assert ai_act_module._high_risk_triggers(config) == [
            "annex_i_product_route",
            "annex_iii_3_education_generic",
            "similarly_significant_decision_support",
        ]

    def test_plausible_carve_out_and_fria_triggers(self) -> None:
        plausible_config = replace(
            _base_config(),
            annex_iii_use_cases=["annex_iii_3_education_generic"],
            annex_iii_narrow_procedural_task=True,
            annex_iii_improves_completed_human_activity=True,
            annex_iii_detects_decision_pattern_deviations=True,
            annex_iii_preparatory_task=True,
            annex_iii_does_not_materially_influence_decision_outcome=True,
        )
        not_plausible_config = replace(
            plausible_config,
            annex_i_product_or_safety_component=True,
        )
        fria_config = replace(
            _base_config(),
            public_sector_use=True,
            provides_public_service=True,
            annex_iii_use_cases=["annex_iii_3_education_generic"],
            creditworthiness_or_credit_score_assessment=True,
            life_or_health_insurance_risk_pricing=True,
        )

        assert ai_act_module._plausible_article6_3_carve_out(plausible_config)
        assert not ai_act_module._plausible_article6_3_carve_out(
            not_plausible_config,
        )
        assert ai_act_module._fria_triggers(fria_config) == [
            "public_sector_or_public_service_high_risk_use",
            "annex_iii_5_b_creditworthiness_or_credit_score",
            "annex_iii_5_c_life_or_health_insurance",
        ]

    def test_human_readable_trigger_labels_uses_catalog_titles(self) -> None:
        with patch.object(
            ai_act_module,
            "_annex_iii_catalog_by_id",
            return_value={
                "known": {"title": "Readable Title"},
                "untitled": {},
            },
        ):
            assert ai_act_module._human_readable_trigger_labels(
                ["known", "untitled", "missing"],
            ) == ["Readable Title", "untitled", "missing"]

    def test_likely_obligations_include_high_risk_and_article50_paths(self) -> None:
        article50_config = replace(
            _base_config(),
            role="provider",
            interacts_with_natural_persons=True,
            interaction_obvious_to_persons=False,
            exposes_emotion_recognition_or_biometric_categorization=True,
            deploys_deepfake_or_synthetic_media=True,
            publishes_text_on_matters_of_public_interest=True,
        )
        high_risk_config = replace(
            _base_config(),
            role="deployer",
            annex_iii_use_cases=["annex_iii_3_education_generic"],
            annex_iii_narrow_procedural_task=True,
            annex_iii_does_not_materially_influence_decision_outcome=False,
            public_sector_use=True,
            provides_public_service=True,
            provides_input_data_for_high_risk_system=True,
            workplace_deployment=True,
            makes_or_assists_decisions_about_natural_persons=True,
            decision_with_legal_or_similarly_significant_effects=True,
            uses_general_purpose_ai=True,
        )

        article50_obligations = ai_act_module._likely_obligations(article50_config)
        assert "Article 4 AI literacy evidence" in article50_obligations
        assert "Article 5 prohibited-practice screening" in article50_obligations
        assert (
            "Article 50 transparency for non-obvious AI interaction"
            in article50_obligations
        )
        assert (
            "Article 50 transparency for emotion recognition or"
            " biometric categorization"
        ) in article50_obligations
        assert (
            "Article 50 deployer disclosure for synthetic media"
            in article50_obligations
        )
        assert (
            "Article 50 deployer disclosure for public-interest text"
            in article50_obligations
        )

        obligations = ai_act_module._likely_obligations(high_risk_config)
        assert "High-risk legal triage" in obligations
        assert "Article 6(3) Annex III carve-out review" in obligations
        assert "Article 27 FRIA legal triage" in obligations
        assert "Article 26 instructions-of-use and monitoring" in obligations
        assert "Article 26 human oversight" in obligations
        assert "Article 26 log retention" in obligations
        assert "Article 26 input-data relevance and representativeness" in obligations
        assert "Article 26 workplace information duty" in obligations
        assert "Article 26 affected-person information duty" in obligations
        assert "Article 86 explanation readiness" in obligations
        assert "EU database registration for public deployments" in obligations
        assert "Downstream provider documentation retention" in obligations


class TestEvaluationHelpers:
    """Tests for Article 5, Article 6(3), and Article 26 screening branches."""

    def test_scope_minimum_facts_fails_for_blank_required_strings(self) -> None:
        config = AIActConfig(
            company_name=" ",
            system_name=" ",
            intended_purpose=" ",
        )

        entry = ai_act_module._evaluate_scope_minimum_facts(config)
        assert entry["status"] == "fail"
        assert "Set [ai_act].company_name" in str(entry["owner_action"])

    @pytest.mark.parametrize(
        ("config_overrides", "expected_status"),
        [
            (
                {
                    "uses_subliminal_manipulative_or_deceptive_techniques": True,
                    "exploits_age_disability_or_socioeconomic_vulnerabilities": True,
                },
                "unknown",
            ),
            (
                {
                    "uses_subliminal_manipulative_or_deceptive_techniques": True,
                    "materially_distorts_behavior_causing_significant_harm": True,
                    "exploits_age_disability_or_socioeconomic_vulnerabilities": True,
                    "social_scoring_leading_to_detrimental_treatment": True,
                    "individual_predictive_policing_based_solely_on_profiling": True,
                    "untargeted_scraping_of_face_images": True,
                    "uses_emotion_recognition": True,
                    "employment_or_workers_management": True,
                    "uses_biometric_categorization": True,
                    "biometric_categorization_infers_sensitive_traits": True,
                    (
                        "real_time_remote_biometric_identification_for_law_enforcement"
                    ): True,
                },
                "fail",
            ),
        ],
    )
    def test_article5_screen_covers_review_and_fail_paths(
        self,
        config_overrides: dict[str, object],
        expected_status: str,
    ) -> None:
        config = replace(_base_config(), **config_overrides)

        entry = ai_act_module._evaluate_article5_prohibited_practices(config)
        assert entry["status"] == expected_status
        if expected_status == "fail":
            assert "social_scoring" in str(entry["rationale"])
            assert entry["legal_review_required"] is True
        else:
            assert "narrow legal conditions" in str(entry["rationale"])
            assert entry["legal_review_required"] is True

    def test_article6_3_carve_out_triage_covers_unknown_paths(self) -> None:
        review_config = replace(
            _base_config(),
            annex_iii_use_cases=["annex_iii_3_education_generic"],
            annex_iii_narrow_procedural_task=True,
            annex_iii_improves_completed_human_activity=True,
            annex_iii_detects_decision_pattern_deviations=True,
            annex_iii_preparatory_task=True,
        )
        profiling_config = replace(
            review_config,
            annex_iii_does_not_materially_influence_decision_outcome=True,
            uses_profiling_or_similarly_significant_decision_support=True,
        )

        review_entry = ai_act_module._evaluate_article6_3_carve_out_triage(
            review_config,
        )
        assert review_entry["status"] == "unknown"
        assert (
            "does not assert that the AI system avoids materially influencing"
            in str(review_entry["rationale"])
        )

        profiling_entry = ai_act_module._evaluate_article6_3_carve_out_triage(
            profiling_config,
        )
        assert profiling_entry["status"] == "unknown"
        assert "profiling or similarly significant decision support" in str(
            profiling_entry["rationale"],
        )

    def test_article26_input_data_governance_and_affected_person_paths(
        self,
        tmp_path: Path,
    ) -> None:
        base_high_risk = replace(
            _base_config(),
            annex_i_product_or_safety_component=True,
            annex_i_third_party_conformity_assessment=True,
            role="deployer",
            provides_input_data_for_high_risk_system=True,
            makes_or_assists_decisions_about_natural_persons=True,
            decision_with_legal_or_similarly_significant_effects=True,
            public_sector_use=True,
        )
        input_data_path = tmp_path / "input_data.md"
        input_data_path.write_text("Scope: only the required fields.\n")
        affected_path = tmp_path / "affected.md"
        affected_path.write_text("Scope: affected persons.\n")
        explanation_path = tmp_path / "explanation.md"
        explanation_path.write_text("Scope: explanation requests.\n")

        input_data_entry = ai_act_module._evaluate_article26_input_data_governance(
            replace(base_high_risk, input_data_governance_procedure=input_data_path),
            [],
        )
        assert input_data_entry["status"] == "unknown"
        assert "relevance and representativeness controls" in str(
            input_data_entry["rationale"],
        )

        affected_entry = ai_act_module._evaluate_article26_affected_person_notice(
            replace(
                base_high_risk,
                affected_person_notice=affected_path,
                explanation_request_procedure=explanation_path,
            ),
            [],
        )
        assert affected_entry["status"] == "unknown"
        assert "notice and explanation evidence" in str(affected_entry["rationale"])

    def test_provider_documentation_unknown_branch(self, tmp_path: Path) -> None:
        config = replace(
            _base_config(),
            role="deployer",
            uses_general_purpose_ai=True,
            provider_documentation=tmp_path / "provider.md",
        )
        provider_documentation = config.provider_documentation
        assert provider_documentation is not None
        provider_documentation.write_text("Model: example\n")

        entry = ai_act_module._evaluate_provider_documentation(
            config,
            [{"evidence_id": "file.provider_documentation", "exists": True}],
        )
        assert entry["status"] == "unknown"
        assert "could not verify all minimum markers" in str(entry["rationale"])

    def test_sources_validation_rejects_invalid_shapes(self) -> None:
        with patch.object(
            ai_act_module,
            "_rulebook_v1",
            return_value={"sources": "bad"},
        ), pytest.raises(TypeError, match="must be a list"):
            ai_act_module._sources()

        with patch.object(
            ai_act_module,
            "_rulebook_v1",
            return_value={"sources": ["bad"]},
        ), pytest.raises(TypeError, match="must be objects"):
            ai_act_module._sources()

        for key in ("source_id", "title", "url", "publisher", "relevance"):
            source: dict[str, object] = {
                "source_id": "s1",
                "title": "Title",
                "url": "https://example.test",
                "publisher": "Publisher",
                "relevance": "High",
            }
            source[key] = 1
            with patch.object(
                ai_act_module,
                "_rulebook_v1",
                return_value={"sources": [source]},
            ), pytest.raises(TypeError, match=key):
                ai_act_module._sources()

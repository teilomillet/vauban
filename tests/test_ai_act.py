"""Tests for AI Act readiness parsing and reporting."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from tests.conftest import make_early_mode_context
from vauban._pipeline._mode_ai_act import _run_ai_act_mode
from vauban.ai_act import generate_deployer_readiness_bundle
from vauban.config._parse_ai_act import _parse_ai_act
from vauban.types import AIActConfig

if TYPE_CHECKING:
    from pathlib import Path


def _object_dict(value: object) -> dict[str, object]:
    """Narrow a JSON-like object to a string-keyed dict for tests."""
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


class TestParseAIAct:
    """Unit tests for the [ai_act] parser."""

    def test_absent_returns_none(self, tmp_path: Path) -> None:
        assert _parse_ai_act(tmp_path, {}) is None

    def test_resolves_relative_paths(self, tmp_path: Path) -> None:
        raw = {
            "ai_act": {
                "company_name": "Example",
                "system_name": "Assistant",
                "intended_purpose": "Answer customer questions.",
                "interacts_with_natural_persons": True,
                "interaction_obvious_to_persons": False,
                "ai_literacy_record": "evidence/literacy.md",
                "technical_report_paths": ["reports/one.json", "reports/two.json"],
            },
        }
        cfg = _parse_ai_act(tmp_path, raw)
        assert cfg is not None
        assert cfg.interacts_with_natural_persons is True
        assert cfg.interaction_obvious_to_persons is False
        assert cfg.ai_literacy_record == tmp_path / "evidence/literacy.md"
        assert cfg.technical_report_paths == [
            tmp_path / "reports/one.json",
            tmp_path / "reports/two.json",
        ]

    def test_empty_company_name_raises(self, tmp_path: Path) -> None:
        raw = {
            "ai_act": {
                "company_name": "",
                "system_name": "Assistant",
                "intended_purpose": "Answer customer questions.",
            },
        }
        with pytest.raises(ValueError, match="company_name"):
            _parse_ai_act(tmp_path, raw)


class TestReadinessBundle:
    """Tests for deployer-readiness bundle generation."""

    def _write_complete_evidence(self, tmp_path: Path) -> None:
        (tmp_path / "ai_literacy.md").write_text(
            "\n".join(
                [
                    "Role: provider of the customer assistant service",
                    "System context: customer support assistant workflow",
                    "Risk topics: misuse risks, escalation, and limitations",
                    "Owner: AI Risk Lead",
                    "Target roles: customer support staff and operators",
                    "Last updated: 2026-03-20",
                    "Scope: system limits, escalation, and approved materials",
                    "Refresh cadence: annual review",
                ],
            )
            + "\n",
        )
        (tmp_path / "oversight.md").write_text(
            "\n".join(
                [
                    "Human review is required for escalated cases.",
                    "Operators can override automated suggestions.",
                    "Escalation trigger: uncertainty or customer complaint.",
                ],
            )
            + "\n",
        )
        (tmp_path / "incident.md").write_text(
            "\n".join(
                [
                    "Incident scope: misuse, failure, or data breach.",
                    "Escalation and reporting follow the severity matrix.",
                    "Notify compliance within one business day.",
                ],
            )
            + "\n",
        )
        (tmp_path / "provider.md").write_text(
            "\n".join(
                [
                    "Provider: Example API Vendor",
                    "Model: example-chat-1",
                    "Version: 2026-03",
                    "Limitations: do not use for autonomous high-impact decisions.",
                ],
            )
            + "\n",
        )
        (tmp_path / "red_team.json").write_text('{"report_version":"v1"}\n')
        (tmp_path / "transparency.md").write_text(
            "\n".join(
                [
                    "This AI assistant is an automated system.",
                    "You are interacting with an AI assistant, not a human agent.",
                ],
            )
            + "\n",
        )

    def _config(self, tmp_path: Path) -> AIActConfig:
        return AIActConfig(
            company_name="Example Energy",
            system_name="Customer Assistant",
            intended_purpose=(
                "Answers customer questions with a third-party GPAI service."
            ),
            role="provider",
            ai_literacy_record=tmp_path / "ai_literacy.md",
            transparency_notice=tmp_path / "transparency.md",
            human_oversight_procedure=tmp_path / "oversight.md",
            incident_response_procedure=tmp_path / "incident.md",
            provider_documentation=tmp_path / "provider.md",
            technical_report_paths=[tmp_path / "red_team.json"],
            interacts_with_natural_persons=True,
            interaction_obvious_to_persons=False,
            risk_owner="AI Risk Lead",
            compliance_contact="compliance@example.com",
        )

    def test_missing_required_evidence_blocks_report(self, tmp_path: Path) -> None:
        report, ledger, library, remediation = generate_deployer_readiness_bundle(
            self._config(tmp_path),
        )
        assert report["overall_status"] == "blocked"
        assert "Article 4" in json.dumps(report["likely_obligations"])
        coverage_contract = _object_dict(report["coverage_contract"])
        assert coverage_contract["coverage_complete"] is True
        controls = ledger["controls"]
        assert isinstance(controls, list)
        control_entries = [
            _object_dict(entry)
            for entry in controls
            if isinstance(entry, dict)
        ]
        statuses = {
            str(entry["control_id"]): str(entry["status"])
            for entry in control_entries
        }
        assert statuses["ai_act.article4.ai_literacy_record"] == "fail"
        assert statuses["vauban.readiness.technical_evidence"] == "unknown"
        control_defs = library["controls"]
        assert isinstance(control_defs, list)
        assert len(control_defs) == len(controls)
        assert "AI Act Remediation Plan" in remediation

    def test_complete_low_risk_bundle_is_ready(self, tmp_path: Path) -> None:
        self._write_complete_evidence(tmp_path)

        report, ledger, _library, remediation = generate_deployer_readiness_bundle(
            self._config(tmp_path),
        )
        assert report["overall_status"] == "ready"
        assert report["risk_level"] == "low"
        rulebook = _object_dict(report["rulebook"])
        assert rulebook["version"] == "deployer_readiness_v1"
        assert isinstance(report["bundle_fingerprint"], str)
        overview = _object_dict(report["controls_overview"])
        assert overview["fail"] == 0
        assert overview["unknown"] == 0
        controls = ledger["controls"]
        assert isinstance(controls, list)
        control_entries = [
            _object_dict(entry)
            for entry in controls
            if isinstance(entry, dict)
        ]
        assert all(
            str(entry["status"]) != "fail"
            and str(entry["status"]) != "unknown"
            for entry in control_entries
        )
        assert "No remediation actions are currently required." in remediation

    def test_human_interaction_notice_with_missing_markers_is_unknown(
        self,
        tmp_path: Path,
    ) -> None:
        self._write_complete_evidence(tmp_path)
        (tmp_path / "transparency.md").write_text(
            "This AI assistant uses AI for customer support.\n",
        )

        report, ledger, _library, _remediation = generate_deployer_readiness_bundle(
            self._config(tmp_path),
        )

        assert report["overall_status"] == "blocked"
        controls = ledger["controls"]
        assert isinstance(controls, list)
        control_entries = [
            _object_dict(entry)
            for entry in controls
            if isinstance(entry, dict)
        ]
        human_notice = next(
            entry for entry in control_entries
            if entry["control_id"] == "ai_act.article50.human_interaction_notice"
        )
        assert human_notice["status"] == "unknown"
        assert "not_human_disclosure" in str(human_notice["missing_markers"])

    def test_high_risk_trigger_forces_legal_review(self, tmp_path: Path) -> None:
        (tmp_path / "ai_literacy.md").write_text("ok\n")
        config = AIActConfig(
            company_name="Example Energy",
            system_name="Loan Assistant",
            intended_purpose="Supports decisions for access to essential services.",
            ai_literacy_record=tmp_path / "ai_literacy.md",
            essential_private_or_public_service=True,
        )
        report, ledger, _library, _remediation = generate_deployer_readiness_bundle(
            config,
        )
        assert report["overall_status"] == "blocked"
        assert report["risk_level"] == "high"
        controls = ledger["controls"]
        assert isinstance(controls, list)
        control_entries = [
            _object_dict(entry)
            for entry in controls
            if isinstance(entry, dict)
        ]
        triage = next(
            entry for entry in control_entries
            if entry["control_id"] == "ai_act.triage.high_risk_annex_iii"
        )
        assert triage["status"] == "unknown"
        assert triage["legal_review_required"] is True

    def test_article5_prohibited_practice_blocks_report(
        self,
        tmp_path: Path,
    ) -> None:
        self._write_complete_evidence(tmp_path)
        config = AIActConfig(
            company_name="Example Energy",
            system_name="Workplace Monitor",
            intended_purpose="Assesses employee emotion during support calls.",
            role="provider",
            ai_literacy_record=tmp_path / "ai_literacy.md",
            transparency_notice=tmp_path / "transparency.md",
            uses_emotion_recognition=True,
            exposes_emotion_recognition_or_biometric_categorization=True,
            employment_or_workers_management=True,
        )

        report, ledger, _library, _remediation = generate_deployer_readiness_bundle(
            config,
        )

        assert report["overall_status"] == "blocked"
        controls = ledger["controls"]
        assert isinstance(controls, list)
        control_entries = [
            _object_dict(entry)
            for entry in controls
            if isinstance(entry, dict)
        ]
        article5 = next(
            entry
            for entry in control_entries
            if entry["control_id"] == "ai_act.article5.prohibited_practices_screen"
        )
        assert article5["status"] == "fail"
        assert article5["legal_review_required"] is True

    def test_public_interest_text_editorial_exception_is_not_applicable(
        self,
        tmp_path: Path,
    ) -> None:
        self._write_complete_evidence(tmp_path)
        (tmp_path / "transparency.md").write_text(
            "\n".join(
                [
                    "Human review is required before publication.",
                    "Editorial control is applied by the newsroom lead.",
                    "The publisher holds editorial responsibility.",
                ],
            )
            + "\n",
        )
        config = AIActConfig(
            company_name="Example Energy",
            system_name="Public Affairs Writer",
            intended_purpose="Drafts public-facing energy-market updates.",
            ai_literacy_record=tmp_path / "ai_literacy.md",
            transparency_notice=tmp_path / "transparency.md",
            publishes_text_on_matters_of_public_interest=True,
            public_interest_text_human_review_or_editorial_control=True,
            public_interest_text_editorial_responsibility=True,
        )

        _report, ledger, _library, _remediation = generate_deployer_readiness_bundle(
            config,
        )

        controls = ledger["controls"]
        assert isinstance(controls, list)
        control_entries = [
            _object_dict(entry)
            for entry in controls
            if isinstance(entry, dict)
        ]
        public_interest = next(
            entry
            for entry in control_entries
            if (
                entry["control_id"]
                == "ai_act.article50.public_interest_text_disclosure"
            )
        )
        assert public_interest["status"] == "not_applicable"

    def test_annex_i_and_fria_triggers_force_legal_review(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "ai_literacy.md").write_text("ok\n")
        config = AIActConfig(
            company_name="Example Energy",
            system_name="Insurance Triage",
            intended_purpose="Prices life insurance risk for applicants.",
            ai_literacy_record=tmp_path / "ai_literacy.md",
            annex_i_product_or_safety_component=True,
            annex_i_third_party_conformity_assessment=True,
            life_or_health_insurance_risk_pricing=True,
        )

        report, ledger, _library, _remediation = generate_deployer_readiness_bundle(
            config,
        )

        assert report["overall_status"] == "blocked"
        controls = ledger["controls"]
        assert isinstance(controls, list)
        control_entries = [
            _object_dict(entry)
            for entry in controls
            if isinstance(entry, dict)
        ]
        high_risk = next(
            entry
            for entry in control_entries
            if entry["control_id"] == "ai_act.triage.high_risk_annex_iii"
        )
        fria = next(
            entry
            for entry in control_entries
            if entry["control_id"] == "ai_act.triage.fria_requirement"
        )
        assert high_risk["status"] == "unknown"
        assert fria["status"] == "unknown"
        assert fria["legal_review_required"] is True

    def test_non_eu_run_has_no_likely_obligations(self) -> None:
        config = AIActConfig(
            company_name="Example Energy",
            system_name="Customer Assistant",
            intended_purpose="Answers customer questions.",
            eu_market=False,
            public_sector_use=True,
        )
        report, _ledger, _library, _remediation = generate_deployer_readiness_bundle(
            config,
        )

        assert report["overall_status"] == "out_of_scope"
        assert report["likely_obligations"] == []
        assert report["unresolved_controls"] == []

    def test_research_run_marks_technical_evidence_not_applicable(
        self,
    ) -> None:
        config = AIActConfig(
            company_name="Example Energy",
            system_name="Research Assistant",
            intended_purpose="Supports internal research experiments.",
            role="research",
        )
        report, ledger, _library, _remediation = generate_deployer_readiness_bundle(
            config,
        )

        assert report["overall_status"] == "out_of_scope"
        assert report["likely_obligations"] == []
        controls = ledger["controls"]
        assert isinstance(controls, list)
        control_entries = [
            _object_dict(entry)
            for entry in controls
            if isinstance(entry, dict)
        ]
        technical_evidence = next(
            entry for entry in control_entries
            if entry["control_id"] == "vauban.readiness.technical_evidence"
        )
        assert technical_evidence["status"] == "not_applicable"
        unresolved_controls = report["unresolved_controls"]
        assert isinstance(unresolved_controls, list)
        assert "vauban.readiness.technical_evidence" not in [
            item for item in unresolved_controls if isinstance(item, str)
        ]

    def test_technical_findings_ignore_counts_and_projection_means(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "technical_report.json").write_text(
            json.dumps(
                {
                    "n_attack_prompts": 50,
                    "harmful_proj_mean": 0.01,
                    "attack_success_rate": 0.1,
                },
            )
            + "\n",
        )
        config = AIActConfig(
            company_name="Example Energy",
            system_name="Customer Assistant",
            intended_purpose="Answers customer questions.",
            technical_report_paths=[tmp_path / "technical_report.json"],
        )
        report, _ledger, _library, _remediation = generate_deployer_readiness_bundle(
            config,
        )

        findings = report["technical_findings"]
        assert isinstance(findings, list)
        risk_ids = {
            str(_object_dict(entry)["risk_id"])
            for entry in findings
            if isinstance(entry, dict)
        }
        assert "technical.file.technical_report.0.attack_success_rate" in risk_ids
        assert "technical.file.technical_report.0.n_attack_prompts" not in risk_ids
        assert "technical.file.technical_report.0.harmful_proj_mean" not in risk_ids


class TestAIActMode:
    """Tests for the standalone [ai_act] mode runner."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(ValueError, match="\\[ai_act\\] section is required"):
            _run_ai_act_mode(ctx)

    def test_happy_path_writes_bundle(self, tmp_path: Path) -> None:
        TestReadinessBundle()._write_complete_evidence(tmp_path)

        ai_act_cfg = AIActConfig(
            company_name="Example Energy",
            system_name="Customer Assistant",
            intended_purpose=(
                "Answers customer questions with a third-party GPAI service."
            ),
            role="provider",
            ai_literacy_record=tmp_path / "ai_literacy.md",
            transparency_notice=tmp_path / "transparency.md",
            human_oversight_procedure=tmp_path / "oversight.md",
            incident_response_procedure=tmp_path / "incident.md",
            provider_documentation=tmp_path / "provider.md",
            technical_report_paths=[tmp_path / "red_team.json"],
            interacts_with_natural_persons=True,
            interaction_obvious_to_persons=False,
            risk_owner="AI Risk Lead",
            compliance_contact="compliance@example.com",
        )
        ctx = make_early_mode_context(tmp_path, ai_act=ai_act_cfg)

        with patch("vauban._pipeline._mode_ai_act.finish_mode_run") as mock_finish:
            _run_ai_act_mode(ctx)
            assert (tmp_path / "ai_act_readiness_report.json").exists()
            assert (tmp_path / "ai_act_coverage_ledger.json").exists()
            assert (tmp_path / "ai_act_control_library_v1.json").exists()
            assert (tmp_path / "ai_act_controls_matrix.json").exists()
            assert (tmp_path / "ai_act_risk_register.json").exists()
            assert (tmp_path / "ai_act_evidence_manifest.json").exists()
            assert (tmp_path / "ai_act_executive_summary.md").exists()
            assert (tmp_path / "ai_act_remediation_plan.md").exists()
            metadata = mock_finish.call_args[0][3]
            assert metadata["n_fail"] == 0
            assert metadata["n_unknown"] == 0

# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for AI Act readiness parsing and reporting."""

from __future__ import annotations

import json
from dataclasses import replace
from io import BytesIO
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest
from pypdf import PdfReader

import vauban.ai_act as ai_act_module
from tests.conftest import make_early_mode_context
from vauban._pipeline._mode_ai_act import _run_ai_act_mode
from vauban.ai_act import (
    generate_deployer_readiness_artifacts,
    generate_deployer_readiness_bundle,
)
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
                "annex_iii_use_cases": [
                    "annex_iii_4_recruitment_selection",
                ],
                "interacts_with_natural_persons": True,
                "interaction_obvious_to_persons": False,
                "ai_literacy_record": "evidence/literacy.md",
                "technical_report_paths": ["reports/one.json", "reports/two.json"],
            },
        }
        cfg = _parse_ai_act(tmp_path, raw)
        assert cfg is not None
        assert cfg.annex_iii_use_cases == ["annex_iii_4_recruitment_selection"]
        assert cfg.interacts_with_natural_persons is True
        assert cfg.interaction_obvious_to_persons is False
        assert cfg.ai_literacy_record == tmp_path / "evidence/literacy.md"
        assert cfg.technical_report_paths == [
            tmp_path / "reports/one.json",
            tmp_path / "reports/two.json",
        ]
        assert cfg.pdf_report is True
        assert cfg.pdf_report_filename == "ai_act_report.pdf"

    def test_invalid_pdf_report_filename_raises(self, tmp_path: Path) -> None:
        raw = {
            "ai_act": {
                "company_name": "Example",
                "system_name": "Assistant",
                "intended_purpose": "Answer customer questions.",
                "pdf_report_filename": "reports/output.txt",
            },
        }

        with pytest.raises(ValueError, match="pdf_report_filename"):
            _parse_ai_act(tmp_path, raw)

    def test_empty_pdf_report_filename_raises(self, tmp_path: Path) -> None:
        raw = {
            "ai_act": {
                "company_name": "Example",
                "system_name": "Assistant",
                "intended_purpose": "Answer customer questions.",
                "pdf_report_filename": "",
            },
        }

        with pytest.raises(ValueError, match="pdf_report_filename"):
            _parse_ai_act(tmp_path, raw)

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
                    "Review step: human review is required for escalated cases.",
                    (
                        "Override capability: operators can override automated"
                        " suggestions."
                    ),
                    "Escalation trigger: uncertainty or customer complaint.",
                ],
            )
            + "\n",
        )
        (tmp_path / "incident.md").write_text(
            "\n".join(
                [
                    "Incident scope: misuse, failure, or data breach.",
                    "Escalation or reporting: follow the severity matrix.",
                    "Contact owner: compliance@example.com.",
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
        (tmp_path / "operation_monitoring.md").write_text(
            "\n".join(
                [
                    "Provider instructions: use only for stated intended purpose.",
                    "Monitoring plan: review logs and incidents weekly.",
                    "Responsible operator: AI Operations Lead.",
                ],
            )
            + "\n",
        )
        (tmp_path / "input_data_governance.md").write_text(
            "\n".join(
                [
                    "Input data scope: applicant-provided documents and forms.",
                    (
                        "Relevance criteria: use only fields required for the"
                        " stated purpose."
                    ),
                    (
                        "Representativeness validation: sample review for bias"
                        " and coverage."
                    ),
                ],
            )
            + "\n",
        )
        (tmp_path / "log_retention.md").write_text(
            "\n".join(
                [
                    "Logging scope: automated system logs and audit trail records.",
                    "Retention period: keep logs for at least six months.",
                    "Access control: logs remain under the deployer's control.",
                ],
            )
            + "\n",
        )
        (tmp_path / "worker_notice.md").write_text(
            "\n".join(
                [
                    "Employee scope: employees and workers in the covered teams.",
                    (
                        "Representative notice: workers' representatives"
                        " receive the same notice."
                    ),
                    "Before use: the notice applies before the system is put into use.",
                ],
            )
            + "\n",
        )
        (tmp_path / "affected_person_notice.md").write_text(
            "\n".join(
                [
                    (
                        "Affected person scope: natural persons subject to the"
                        " use of the high-risk AI system."
                    ),
                    "Intended purpose: support credit decisions for natural persons.",
                    "Decision support context: recommendations assist decision-making.",
                ],
            )
            + "\n",
        )
        (tmp_path / "explanation_request.md").write_text(
            "\n".join(
                [
                    (
                        "Right to an explanation: affected persons may"
                        " request an explanation."
                    ),
                    (
                        "Response process: compliance reviews and answers"
                        " requests within the workflow timeline."
                    ),
                ],
            )
            + "\n",
        )
        (tmp_path / "eu_registration.md").write_text(
            "\n".join(
                [
                    (
                        "Registration scope: EU database registration record"
                        " for the deployed system."
                    ),
                    "Authority reference: EUA-12345.",
                    "System identifier: high-risk-credit-01.",
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
                    (
                        "This notice also covers emotion recognition or"
                        " biometric categorization exposure."
                    ),
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
        integrity = _object_dict(report["integrity"])
        assert integrity["status"] == "unsigned"
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

    def test_controls_and_risks_include_source_citations(
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

        artifacts = generate_deployer_readiness_artifacts(config)
        rows = artifacts.controls_matrix["rows"]
        assert isinstance(rows, list)
        row_entries = [
            _object_dict(entry)
            for entry in rows
            if isinstance(entry, dict)
        ]
        article5_row = next(
            entry
            for entry in row_entries
            if entry["control_id"] == "ai_act.article5.prohibited_practices_screen"
        )
        citations = article5_row["source_citations"]
        assert isinstance(citations, list)
        assert any(
            _object_dict(item)["url"]
            == "https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX%3A32024R1689"
            for item in citations
            if isinstance(item, dict)
        )

        items = artifacts.risk_register["items"]
        assert isinstance(items, list)
        risk_entries = [
            _object_dict(entry)
            for entry in items
            if isinstance(entry, dict)
        ]
        article5_risk = next(
            entry
            for entry in risk_entries
            if entry["risk_id"] == "control.ai_act.article5.prohibited_practices_screen"
        )
        risk_citations = article5_risk["source_citations"]
        assert isinstance(risk_citations, list)
        assert len(risk_citations) >= 1

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

    def test_article5_rbi_exception_claim_stays_unknown(
        self,
        tmp_path: Path,
    ) -> None:
        self._write_complete_evidence(tmp_path)
        config = AIActConfig(
            company_name="Example Energy",
            system_name="Public Safety Scanner",
            intended_purpose=(
                "Supports law-enforcement identification in public spaces."
            ),
            ai_literacy_record=tmp_path / "ai_literacy.md",
            real_time_remote_biometric_identification_for_law_enforcement=True,
            real_time_remote_biometric_identification_exception_claimed=True,
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
        article5 = next(
            entry
            for entry in control_entries
            if entry["control_id"] == "ai_act.article5.prohibited_practices_screen"
        )
        assert article5["status"] == "unknown"
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

    def test_annex_iii_classification_tracks_specific_use_cases(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "ai_literacy.md").write_text("ok\n")
        artifacts = generate_deployer_readiness_artifacts(
            AIActConfig(
                company_name="Example Energy",
                system_name="Hiring Assistant",
                intended_purpose="Ranks applicants for open roles.",
                ai_literacy_record=tmp_path / "ai_literacy.md",
                annex_iii_use_cases=["annex_iii_4_recruitment_selection"],
            ),
        )

        classification = artifacts.annex_iii_classification
        assert classification["status"] == "specific_use_cases_declared"
        area_rows = classification["areas"]
        assert isinstance(area_rows, list)
        area_entries = [
            _object_dict(entry)
            for entry in area_rows
            if isinstance(entry, dict)
        ]
        employment_area = next(
            entry for entry in area_entries if entry["area"] == "4"
        )
        use_cases = employment_area["use_cases"]
        assert isinstance(use_cases, list)
        assert any(
            _object_dict(item)["use_case_id"] == "annex_iii_4_recruitment_selection"
            for item in use_cases
            if isinstance(item, dict)
        )

    def test_public_sector_area_two_only_does_not_trigger_fria(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "ai_literacy.md").write_text("ok\n")
        config = AIActConfig(
            company_name="Example Energy",
            system_name="Grid Safety Helper",
            intended_purpose="Supports safety decisions in electricity supply.",
            ai_literacy_record=tmp_path / "ai_literacy.md",
            public_sector_use=True,
            annex_iii_use_cases=["annex_iii_2_critical_infrastructure"],
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
        fria = next(
            entry
            for entry in control_entries
            if entry["control_id"] == "ai_act.triage.fria_requirement"
        )
        assert fria["status"] == "pass"

    def test_article6_3_carve_out_claim_requires_review(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "ai_literacy.md").write_text("ok\n")
        config = AIActConfig(
            company_name="Example Energy",
            system_name="Recruiting File Router",
            intended_purpose=(
                "Routes incoming recruitment files for later human review."
            ),
            ai_literacy_record=tmp_path / "ai_literacy.md",
            annex_iii_use_cases=["annex_iii_4_recruitment_selection"],
            annex_iii_narrow_procedural_task=True,
            annex_iii_does_not_materially_influence_decision_outcome=True,
        )

        artifacts = generate_deployer_readiness_artifacts(config)
        rows = artifacts.controls_matrix["rows"]
        assert isinstance(rows, list)
        row_entries = [
            _object_dict(entry)
            for entry in rows
            if isinstance(entry, dict)
        ]
        carve_out = next(
            entry
            for entry in row_entries
            if entry["control_id"] == "ai_act.triage.article6_3_annex_iii_carve_out"
        )
        assert carve_out["status"] == "unknown"
        classification = artifacts.annex_iii_classification
        assert classification["article6_3_claimed_conditions"] == [
            "narrow_procedural_task",
        ]

    def test_high_risk_deployer_article26_controls_can_pass(
        self,
        tmp_path: Path,
    ) -> None:
        self._write_complete_evidence(tmp_path)
        config = AIActConfig(
            company_name="Example Energy",
            system_name="Public Credit Assistant",
            intended_purpose="Supports credit decisions for natural persons.",
            role="deployer",
            ai_literacy_record=tmp_path / "ai_literacy.md",
            annex_iii_use_cases=["annex_iii_5_creditworthiness_or_credit_score"],
            provider_documentation=tmp_path / "provider.md",
            human_oversight_procedure=tmp_path / "oversight.md",
            operation_monitoring_procedure=tmp_path / "operation_monitoring.md",
            input_data_governance_procedure=tmp_path / "input_data_governance.md",
            log_retention_procedure=tmp_path / "log_retention.md",
            employee_or_worker_representative_notice=tmp_path / "worker_notice.md",
            affected_person_notice=tmp_path / "affected_person_notice.md",
            explanation_request_procedure=tmp_path / "explanation_request.md",
            eu_database_registration_record=tmp_path / "eu_registration.md",
            provides_input_data_for_high_risk_system=True,
            workplace_deployment=True,
            makes_or_assists_decisions_about_natural_persons=True,
            decision_with_legal_or_similarly_significant_effects=True,
            public_sector_use=True,
            risk_owner="AI Risk Lead",
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
        control_ids = {
            str(entry["control_id"]): str(entry["status"])
            for entry in control_entries
        }
        assert control_ids["ai_act.article26.instructions_monitoring"] == "pass"
        assert control_ids["ai_act.article26.human_oversight"] == "pass"
        assert control_ids["ai_act.article26.input_data_governance"] == "pass"
        assert control_ids["ai_act.article26.log_retention"] == "pass"
        assert control_ids["ai_act.article26.workplace_notice"] == "pass"
        assert (
            control_ids["ai_act.article26.affected_person_notice_and_explanation"]
            == "pass"
        )
        assert (
            control_ids["ai_act.article26.public_authority_registration"] == "pass"
        )

    def test_markdown_report_artifacts_include_reviewer_sections(
        self,
        tmp_path: Path,
    ) -> None:
        self._write_complete_evidence(tmp_path)
        artifacts = generate_deployer_readiness_artifacts(self._config(tmp_path))

        executive = artifacts.executive_summary_markdown
        appendix = artifacts.auditor_appendix_markdown

        assert "# AI Act Executive Report" in executive
        assert "## Priority Actions" in executive
        assert "## Highest-Risk Findings" in executive
        assert "# AI Act Auditor Appendix" in appendix
        assert "## Control Outcomes" in appendix
        assert "## Evidence Inventory" in appendix
        assert "## Integrity and Reproducibility" in appendix

    def test_pdf_report_artifact_is_generated_and_extractable(
        self,
        tmp_path: Path,
    ) -> None:
        self._write_complete_evidence(tmp_path)
        artifacts = generate_deployer_readiness_artifacts(self._config(tmp_path))

        assert artifacts.pdf_report_bytes is not None
        assert artifacts.pdf_report_filename == "ai_act_report.pdf"
        reader = PdfReader(BytesIO(artifacts.pdf_report_bytes))
        extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
        assert "AI Act Readiness Report" in extracted
        assert "Executive Report" in extracted
        assert "Auditor Appendix" in extracted

    def test_programmatic_invalid_pdf_report_filename_raises(self) -> None:
        with pytest.raises(ValueError, match="pdf_report_filename"):
            AIActConfig(
                company_name="Example Energy",
                system_name="Customer Assistant",
                intended_purpose="Answers customer questions.",
                pdf_report_filename="ai_act_executive_summary.md",
            )

    def test_integrity_artifact_is_signed_when_secret_env_is_available(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._write_complete_evidence(tmp_path)
        monkeypatch.setenv("VAUBAN_AI_ACT_SIGNING_SECRET", "super-secret")
        config = replace(
            self._config(tmp_path),
            bundle_signature_secret_env="VAUBAN_AI_ACT_SIGNING_SECRET",
        )

        artifacts = generate_deployer_readiness_artifacts(config)

        assert artifacts.integrity["signature_status"] == "signed"
        assert artifacts.integrity["signature_algorithm"] == "hmac-sha256"
        signature = artifacts.integrity["signature"]
        assert isinstance(signature, str)
        assert signature != ""
        artifact_hashes = artifacts.integrity["artifact_hashes"]
        assert isinstance(artifact_hashes, dict)
        assert "ai_act_readiness_report.json" in artifact_hashes
        assert "ai_act_auditor_appendix.md" in artifact_hashes
        assert "ai_act_report.pdf" in artifact_hashes
        assert artifacts.integrity["artifact_count"] == 13
        evidence = artifacts.evidence_manifest["evidence"]
        assert isinstance(evidence, list)
        evidence_entries = [
            _object_dict(entry)
            for entry in evidence
            if isinstance(entry, dict)
        ]
        literacy_entry = next(
            entry
            for entry in evidence_entries
            if entry["evidence_id"] == "file.ai_literacy_record"
        )
        detected_fields = literacy_entry["structured_fields_detected"]
        assert isinstance(detected_fields, list)
        assert "owner" in detected_fields
        assert "refresh_cadence" in detected_fields

    def test_technical_reports_skip_scaffold_placeholder_detection(
        self,
        tmp_path: Path,
    ) -> None:
        self._write_complete_evidence(tmp_path)

        with patch.object(
            ai_act_module,
            "_document_has_draft_placeholders",
            wraps=ai_act_module._document_has_draft_placeholders,
        ) as mock_placeholder:
            generate_deployer_readiness_artifacts(self._config(tmp_path))

        scanned_paths = {
            call.args[0]
            for call in mock_placeholder.call_args_list
            if call.args
        }
        assert tmp_path / "ai_literacy.md" in scanned_paths
        assert tmp_path / "red_team.json" not in scanned_paths

    def test_high_risk_deployer_missing_monitoring_blocks_article26(
        self,
        tmp_path: Path,
    ) -> None:
        self._write_complete_evidence(tmp_path)
        config = AIActConfig(
            company_name="Example Energy",
            system_name="Hiring Assistant",
            intended_purpose="Scores applicants for open roles.",
            role="deployer",
            ai_literacy_record=tmp_path / "ai_literacy.md",
            annex_iii_use_cases=["annex_iii_4_recruitment_selection"],
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
        monitoring = next(
            entry
            for entry in control_entries
            if entry["control_id"] == "ai_act.article26.instructions_monitoring"
        )
        assert monitoring["status"] == "unknown"

    def test_high_risk_deployer_missing_human_oversight_blocks_article26(
        self,
        tmp_path: Path,
    ) -> None:
        self._write_complete_evidence(tmp_path)
        config = AIActConfig(
            company_name="Example Energy",
            system_name="Public Credit Assistant",
            intended_purpose="Supports credit decisions for natural persons.",
            role="deployer",
            ai_literacy_record=tmp_path / "ai_literacy.md",
            annex_iii_use_cases=["annex_iii_5_creditworthiness_or_credit_score"],
            provider_documentation=tmp_path / "provider.md",
            operation_monitoring_procedure=tmp_path / "operation_monitoring.md",
            log_retention_procedure=tmp_path / "log_retention.md",
            public_sector_use=True,
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
        oversight = next(
            entry
            for entry in control_entries
            if entry["control_id"] == "ai_act.article26.human_oversight"
        )
        assert oversight["status"] == "unknown"

    def test_fria_prep_artifact_is_generated_for_triggered_case(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "ai_literacy.md").write_text("ok\n")
        config = AIActConfig(
            company_name="Example Energy",
            system_name="Insurance Triage",
            intended_purpose="Prices life insurance risk for applicants.",
            ai_literacy_record=tmp_path / "ai_literacy.md",
            life_or_health_insurance_risk_pricing=True,
        )

        artifacts = generate_deployer_readiness_artifacts(config)
        assert artifacts.fria_prep["status"] == "review_needed"
        trigger_labels = artifacts.fria_prep["trigger_labels"]
        assert isinstance(trigger_labels, list)
        assert "annex_iii_5_c_life_or_health_insurance" in trigger_labels

    def test_partial_literacy_evidence_stays_unknown(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "ai_literacy.md").write_text(
            "\n".join(
                [
                    "Owner: AI Risk Lead",
                    "Last updated: 2026-03-20",
                ],
            )
            + "\n",
        )
        config = AIActConfig(
            company_name="Example Energy",
            system_name="Customer Assistant",
            intended_purpose="Answers customer questions.",
            role="provider",
            ai_literacy_record=tmp_path / "ai_literacy.md",
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
        article4 = next(
            entry
            for entry in control_entries
            if entry["control_id"] == "ai_act.article4.ai_literacy_record"
        )
        assert article4["status"] == "unknown"

    def test_public_interest_exception_with_weak_evidence_stays_unknown(
        self,
        tmp_path: Path,
    ) -> None:
        self._write_complete_evidence(tmp_path)
        (tmp_path / "transparency.md").write_text(
            "The publisher holds editorial responsibility.\n",
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
        assert public_interest["status"] == "unknown"

    def test_partial_annex_i_flag_stays_unknown(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "ai_literacy.md").write_text("ok\n")
        config = AIActConfig(
            company_name="Example Energy",
            system_name="Safety Helper",
            intended_purpose="Supports a regulated product workflow.",
            ai_literacy_record=tmp_path / "ai_literacy.md",
            annex_i_third_party_conformity_assessment=True,
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
        high_risk = next(
            entry
            for entry in control_entries
            if entry["control_id"] == "ai_act.triage.high_risk_annex_iii"
        )
        assert high_risk["status"] == "unknown"

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
            assert (tmp_path / "ai_act_annex_iii_classification.json").exists()
            assert (tmp_path / "ai_act_risk_register.json").exists()
            assert (tmp_path / "ai_act_fria_prep.json").exists()
            assert (tmp_path / "ai_act_evidence_manifest.json").exists()
            assert (tmp_path / "ai_act_integrity.json").exists()
            assert (tmp_path / "ai_act_executive_summary.md").exists()
            assert (tmp_path / "ai_act_auditor_appendix.md").exists()
            assert (tmp_path / "ai_act_report.pdf").exists()
            assert (tmp_path / "ai_act_remediation_plan.md").exists()
            assert (tmp_path / "ai_act_fria_prep.md").exists()
            metadata = mock_finish.call_args[0][3]
            assert metadata["n_fail"] == 0
            assert metadata["n_unknown"] == 0

"""Replayable case-library tests for AI Act deployer readiness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest

from vauban.ai_act import generate_deployer_readiness_artifacts
from vauban.types import AIActConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _object_dict(value: object) -> dict[str, object]:
    """Narrow a JSON-like object to a string-keyed dict for tests."""
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def _write_complete_evidence(tmp_path: Path) -> None:
    """Write a reusable evidence bundle for canonical deployer cases."""
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
                "Relevance criteria: use only fields required for the stated purpose.",
                "Representativeness validation: sample review for bias and coverage.",
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
                    "Representative notice: workers' representatives receive"
                    " the same notice."
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
                    "Right to an explanation: affected persons may request an"
                    " explanation."
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
                    "Registration scope: EU database registration record for"
                    " the deployed system."
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


@dataclass(frozen=True, slots=True)
class CaseExpectation:
    """Expected outcome for one canonical deployer scenario."""

    case_id: str
    build_config: Callable[[Path], AIActConfig]
    expected_overall_status: str
    expected_risk_level: str
    expected_controls: dict[str, str]


def _customer_assistant_case(tmp_path: Path) -> AIActConfig:
    """Return a low-risk provider case."""
    return AIActConfig(
        company_name="Example Energy",
        system_name="Customer Assistant",
        intended_purpose="Answers customer questions with a third-party GPAI service.",
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


def _workplace_emotion_case(tmp_path: Path) -> AIActConfig:
    """Return an obviously prohibited workplace emotion-recognition case."""
    return AIActConfig(
        company_name="Example Energy",
        system_name="Workplace Monitor",
        intended_purpose="Assesses employee emotion during support calls.",
        role="provider",
        ai_literacy_record=tmp_path / "ai_literacy.md",
        transparency_notice=tmp_path / "transparency.md",
        human_oversight_procedure=tmp_path / "oversight.md",
        incident_response_procedure=tmp_path / "incident.md",
        provider_documentation=tmp_path / "provider.md",
        uses_emotion_recognition=True,
        exposes_emotion_recognition_or_biometric_categorization=True,
        employment_or_workers_management=True,
        risk_owner="AI Risk Lead",
        compliance_contact="compliance@example.com",
    )


def _public_credit_case(tmp_path: Path) -> AIActConfig:
    """Return a high-risk deployer case with Article 26 evidence present."""
    return AIActConfig(
        company_name="Example Energy",
        system_name="Public Credit Assistant",
        intended_purpose="Supports credit decisions for natural persons.",
        role="deployer",
        ai_literacy_record=tmp_path / "ai_literacy.md",
        annex_iii_use_cases=["annex_iii_5_creditworthiness_or_credit_score"],
        provider_documentation=tmp_path / "provider.md",
        human_oversight_procedure=tmp_path / "oversight.md",
        incident_response_procedure=tmp_path / "incident.md",
        operation_monitoring_procedure=tmp_path / "operation_monitoring.md",
        input_data_governance_procedure=tmp_path / "input_data_governance.md",
        log_retention_procedure=tmp_path / "log_retention.md",
        employee_or_worker_representative_notice=tmp_path / "worker_notice.md",
        affected_person_notice=tmp_path / "affected_person_notice.md",
        explanation_request_procedure=tmp_path / "explanation_request.md",
        eu_database_registration_record=tmp_path / "eu_registration.md",
        technical_report_paths=[tmp_path / "red_team.json"],
        provides_input_data_for_high_risk_system=True,
        workplace_deployment=True,
        makes_or_assists_decisions_about_natural_persons=True,
        decision_with_legal_or_similarly_significant_effects=True,
        public_sector_use=True,
        risk_owner="AI Risk Lead",
        compliance_contact="compliance@example.com",
    )


def _annex_iii_carve_out_case(tmp_path: Path) -> AIActConfig:
    """Return a carve-out claim that should stay under review."""
    return AIActConfig(
        company_name="Example Energy",
        system_name="Recruiting File Router",
        intended_purpose="Routes incoming recruitment files for later human review.",
        role="deployer",
        ai_literacy_record=tmp_path / "ai_literacy.md",
        annex_iii_use_cases=["annex_iii_4_recruitment_selection"],
        annex_iii_narrow_procedural_task=True,
        annex_iii_does_not_materially_influence_decision_outcome=True,
    )


CASE_LIBRARY: tuple[CaseExpectation, ...] = (
    CaseExpectation(
        case_id="customer_assistant_ready",
        build_config=_customer_assistant_case,
        expected_overall_status="ready",
        expected_risk_level="low",
        expected_controls={
            "ai_act.article4.ai_literacy_record": "pass",
            "ai_act.article50.human_interaction_notice": "pass",
        },
    ),
    CaseExpectation(
        case_id="workplace_emotion_recognition_blocked",
        build_config=_workplace_emotion_case,
        expected_overall_status="blocked",
        expected_risk_level="high",
        expected_controls={
            "ai_act.article5.prohibited_practices_screen": "fail",
            "ai_act.article50.emotion_biometric_notice": "pass",
        },
    ),
    CaseExpectation(
        case_id="public_credit_high_risk_controls_present",
        build_config=_public_credit_case,
        expected_overall_status="blocked",
        expected_risk_level="high",
        expected_controls={
            "ai_act.triage.high_risk_annex_iii": "unknown",
            "ai_act.article26.instructions_monitoring": "pass",
            "ai_act.article26.human_oversight": "pass",
            "ai_act.article26.log_retention": "pass",
        },
    ),
    CaseExpectation(
        case_id="annex_iii_carve_out_requires_review",
        build_config=_annex_iii_carve_out_case,
        expected_overall_status="blocked",
        expected_risk_level="high",
        expected_controls={
            "ai_act.triage.high_risk_annex_iii": "unknown",
            "ai_act.triage.article6_3_annex_iii_carve_out": "unknown",
        },
    ),
)


@pytest.mark.parametrize(
    "expectation",
    CASE_LIBRARY,
    ids=[expectation.case_id for expectation in CASE_LIBRARY],
)
def test_case_library_replays_stable_outcomes(
    tmp_path: Path,
    expectation: CaseExpectation,
) -> None:
    """Replay canonical deployer scenarios and assert stable outcomes."""
    _write_complete_evidence(tmp_path)

    artifacts = generate_deployer_readiness_artifacts(
        expectation.build_config(tmp_path),
    )

    assert artifacts.report["overall_status"] == expectation.expected_overall_status
    assert artifacts.report["risk_level"] == expectation.expected_risk_level
    assert artifacts.integrity["artifact_count"] == 11

    rows = artifacts.controls_matrix["rows"]
    assert isinstance(rows, list)
    control_statuses = {
        str(entry_dict["control_id"]): str(entry_dict["status"])
        for entry_dict in (
            _object_dict(entry)
            for entry in rows
            if isinstance(entry, dict)
        )
    }
    for control_id, expected_status in expectation.expected_controls.items():
        assert control_statuses[control_id] == expected_status

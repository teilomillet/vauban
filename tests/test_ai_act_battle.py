"""Adversarial invariants for AI Act deployer-readiness reporting."""

from __future__ import annotations

import datetime
import random
from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, cast
from unittest.mock import patch

import pytest

from vauban.ai_act import AIActArtifacts, generate_deployer_readiness_artifacts
from vauban.types import AIActConfig

if TYPE_CHECKING:
    from pathlib import Path


type BaseScenario = Literal[
    "provider_low_risk",
    "deployer_high_risk",
    "prohibited_emotion",
]
type EvidenceQuality = Literal["complete", "partial", "missing"]
type HighRiskMutation = Literal[
    "none",
    "missing_provider_documentation",
    "missing_monitoring",
    "partial_monitoring",
    "missing_human_oversight",
    "partial_human_oversight",
    "missing_risk_owner",
    "missing_log_retention",
    "missing_worker_notice",
    "missing_affected_person_notice",
]
type RoleVariant = Literal["provider", "deployer"]


@dataclass(frozen=True, slots=True)
class BattleCase:
    """One deterministic adversarial campaign case."""

    seed: int
    base_scenario: BaseScenario
    literacy_quality: EvidenceQuality
    transparency_quality: EvidenceQuality
    technical_present: bool
    signature_requested: bool
    high_risk_mutation: HighRiskMutation
    role_variant: RoleVariant


class _FixedDateTime(datetime.datetime):
    """Frozen clock for deterministic report and signature generation."""

    @classmethod
    def now(
        cls,
        tz: datetime.tzinfo | None = None,
    ) -> _FixedDateTime:
        return cls(2026, 3, 27, 12, 0, 0, tzinfo=tz)


def _object_dict(value: object) -> dict[str, object]:
    """Narrow a JSON-like object to a string-keyed dict for tests."""
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def _write_complete_evidence(tmp_path: Path) -> None:
    """Write a complete evidence pack that satisfies all structured checks."""
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
    (tmp_path / "transparency.md").write_text(
        "\n".join(
            [
                "This AI assistant is an automated system.",
                "You are interacting with an AI assistant, not a human agent.",
                (
                    "This notice also covers emotion recognition or biometric"
                    " categorization exposure."
                ),
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
                    "Affected person scope: natural persons subject to the use"
                    " of the high-risk AI system."
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
                    "Response process: compliance reviews and answers requests"
                    " within the workflow timeline."
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
    (tmp_path / "red_team.json").write_text(
        '{"report_version": "v1", "attack_success_rate": 0.1}\n',
    )


def _write_partial_document(path: Path, kind: str) -> None:
    """Rewrite one evidence document with deliberately incomplete structure."""
    partial_content_by_kind: dict[str, str] = {
        "ai_literacy": "\n".join(
            [
                "Owner: AI Risk Lead",
                "Last updated: 2026-03-20",
            ],
        )
        + "\n",
        "transparency": "This AI assistant uses AI for customer support.\n",
        "monitoring": "Monitoring plan: review logs and incidents weekly.\n",
        "oversight": "Review step: human review is required.\n",
    }
    path.write_text(partial_content_by_kind[kind])


def _provider_low_risk_config(tmp_path: Path) -> AIActConfig:
    """Return a fully evidenced low-risk provider scenario."""
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


def _deployer_high_risk_config(tmp_path: Path) -> AIActConfig:
    """Return a fully evidenced high-risk deployer scenario."""
    return AIActConfig(
        company_name="Example Energy",
        system_name="Public Credit Assistant",
        intended_purpose="Supports credit decisions for natural persons.",
        role="deployer",
        ai_literacy_record=tmp_path / "ai_literacy.md",
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
        annex_iii_use_cases=["annex_iii_5_creditworthiness_or_credit_score"],
        provides_input_data_for_high_risk_system=True,
        workplace_deployment=True,
        makes_or_assists_decisions_about_natural_persons=True,
        decision_with_legal_or_similarly_significant_effects=True,
        public_sector_use=True,
        risk_owner="AI Risk Lead",
        compliance_contact="compliance@example.com",
    )


def _prohibited_emotion_config(tmp_path: Path) -> AIActConfig:
    """Return an obviously prohibited workplace emotion-recognition scenario."""
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
        technical_report_paths=[tmp_path / "red_team.json"],
        uses_emotion_recognition=True,
        exposes_emotion_recognition_or_biometric_categorization=True,
        employment_or_workers_management=True,
        risk_owner="AI Risk Lead",
        compliance_contact="compliance@example.com",
    )


def _sample_case(seed: int) -> BattleCase:
    """Generate one deterministic adversarial case from a numeric seed."""
    rng = random.Random(seed)
    base_scenario = cast(
        "BaseScenario",
        rng.choice(
            [
                "provider_low_risk",
                "deployer_high_risk",
                "prohibited_emotion",
            ],
        ),
    )
    role_variant = cast(
        "RoleVariant",
        rng.choice(["provider", "deployer"]),
    )
    high_risk_mutation: HighRiskMutation
    if base_scenario == "deployer_high_risk":
        high_risk_mutation = cast(
            "HighRiskMutation",
            rng.choice(
                [
                    "none",
                    "missing_provider_documentation",
                    "missing_monitoring",
                    "partial_monitoring",
                    "missing_human_oversight",
                    "partial_human_oversight",
                    "missing_risk_owner",
                    "missing_log_retention",
                    "missing_worker_notice",
                    "missing_affected_person_notice",
                ],
            ),
        )
    else:
        high_risk_mutation = "none"

    return BattleCase(
        seed=seed,
        base_scenario=base_scenario,
        literacy_quality=cast(
            "EvidenceQuality",
            rng.choice(["complete", "partial", "missing"]),
        ),
        transparency_quality=cast(
            "EvidenceQuality",
            rng.choice(["complete", "partial", "missing"]),
        ),
        technical_present=rng.choice([True, False]),
        signature_requested=rng.choice([True, False]),
        high_risk_mutation=high_risk_mutation,
        role_variant=role_variant,
    )


def _build_case_config(case: BattleCase, tmp_path: Path) -> AIActConfig:
    """Build one concrete config from a seeded adversarial case."""
    _write_complete_evidence(tmp_path)
    if case.base_scenario == "provider_low_risk":
        config = _provider_low_risk_config(tmp_path)
        config = replace(config, role=case.role_variant)
    elif case.base_scenario == "deployer_high_risk":
        config = _deployer_high_risk_config(tmp_path)
    else:
        config = _prohibited_emotion_config(tmp_path)

    if case.literacy_quality == "missing":
        config = replace(config, ai_literacy_record=None)
    elif case.literacy_quality == "partial":
        _write_partial_document(tmp_path / "ai_literacy.md", "ai_literacy")

    if case.transparency_quality == "missing":
        config = replace(config, transparency_notice=None)
    elif case.transparency_quality == "partial":
        _write_partial_document(tmp_path / "transparency.md", "transparency")

    if not case.technical_present:
        config = replace(config, technical_report_paths=[])

    if case.signature_requested:
        config = replace(
            config,
            bundle_signature_secret_env="VAUBAN_AI_ACT_SIGNING_SECRET",
        )

    if case.high_risk_mutation == "missing_provider_documentation":
        config = replace(config, provider_documentation=None)
    elif case.high_risk_mutation == "missing_monitoring":
        config = replace(config, operation_monitoring_procedure=None)
    elif case.high_risk_mutation == "partial_monitoring":
        _write_partial_document(tmp_path / "operation_monitoring.md", "monitoring")
    elif case.high_risk_mutation == "missing_human_oversight":
        config = replace(config, human_oversight_procedure=None)
    elif case.high_risk_mutation == "partial_human_oversight":
        _write_partial_document(tmp_path / "oversight.md", "oversight")
    elif case.high_risk_mutation == "missing_risk_owner":
        config = replace(config, risk_owner=None)
    elif case.high_risk_mutation == "missing_log_retention":
        config = replace(config, log_retention_procedure=None)
    elif case.high_risk_mutation == "missing_worker_notice":
        config = replace(
            config,
            employee_or_worker_representative_notice=None,
        )
    elif case.high_risk_mutation == "missing_affected_person_notice":
        config = replace(config, affected_person_notice=None)

    return config


def _control_statuses(controls: list[dict[str, object]]) -> dict[str, str]:
    """Build a control-id to status map from coverage rows."""
    return {
        str(entry["control_id"]): str(entry["status"])
        for entry in controls
    }


def _assert_common_invariants(
    artifacts: AIActArtifacts,
) -> None:
    """Assert engine-wide invariants that should survive all mutations."""
    report = _object_dict(artifacts.report)
    coverage_ledger = _object_dict(artifacts.coverage_ledger)
    integrity = _object_dict(artifacts.integrity)
    controls = coverage_ledger["controls"]
    assert isinstance(controls, list)
    control_entries = [
        _object_dict(entry)
        for entry in controls
        if isinstance(entry, dict)
    ]
    control_ids = [str(entry["control_id"]) for entry in control_entries]
    assert len(control_ids) == len(set(control_ids))

    coverage_contract = _object_dict(report["coverage_contract"])
    assert coverage_contract["n_controls"] == len(control_entries)
    assert coverage_contract["n_evaluated_controls"] == len(control_entries)
    assert coverage_contract["coverage_complete"] is True

    statuses = [str(entry["status"]) for entry in control_entries]
    assert set(statuses) <= {"pass", "fail", "unknown", "not_applicable"}

    overview = _object_dict(report["controls_overview"])
    counts = Counter(statuses)
    assert overview["pass"] == counts.get("pass", 0)
    assert overview["fail"] == counts.get("fail", 0)
    assert overview["unknown"] == counts.get("unknown", 0)
    assert overview["not_applicable"] == counts.get("not_applicable", 0)

    unresolved_controls_raw = report["unresolved_controls"]
    assert isinstance(unresolved_controls_raw, list)
    unresolved_controls = {
        item
        for item in unresolved_controls_raw
        if isinstance(item, str)
    }
    for entry in control_entries:
        control_id = str(entry["control_id"])
        applies = bool(entry["applies"])
        status = str(entry["status"])
        missing_markers = entry["missing_markers"]
        assert isinstance(missing_markers, list)
        if status == "pass":
            assert missing_markers == []
        if applies and status in {"fail", "unknown"}:
            assert control_id in unresolved_controls
            owner_action = entry["owner_action"]
            assert isinstance(owner_action, str) or bool(
                entry["legal_review_required"],
            )
        else:
            assert control_id not in unresolved_controls

    has_blocking_failure = any(
        bool(entry["applies"])
        and bool(entry["blocking"])
        and str(entry["status"]) in {"fail", "unknown"}
        for entry in control_entries
    )
    has_unresolved = any(
        bool(entry["applies"])
        and str(entry["status"]) in {"fail", "unknown"}
        for entry in control_entries
    )
    overall_status = str(report["overall_status"])
    if has_blocking_failure:
        assert overall_status == "blocked"
    elif has_unresolved:
        assert overall_status == "ready_with_actions"
    else:
        assert overall_status == "ready"

    artifact_hashes = integrity["artifact_hashes"]
    assert isinstance(artifact_hashes, dict)
    assert len(artifact_hashes) == 13
    assert "ai_act_integrity.json" not in artifact_hashes
    assert "ai_act_report.pdf" in artifact_hashes
    assert integrity["artifact_count"] == 13

    bundle_fingerprint = str(report["bundle_fingerprint"])
    assert bundle_fingerprint == str(
        artifacts.controls_matrix["bundle_fingerprint"],
    )
    assert bundle_fingerprint == str(
        artifacts.risk_register["bundle_fingerprint"],
    )
    assert bundle_fingerprint == str(
        artifacts.annex_iii_classification["bundle_fingerprint"],
    )
    assert bundle_fingerprint == str(
        artifacts.fria_prep["bundle_fingerprint"],
    )


def _assert_case_invariants(
    case: BattleCase,
    artifacts: AIActArtifacts,
) -> None:
    """Assert mutation-specific invariants for one adversarial case."""
    coverage_ledger = _object_dict(artifacts.coverage_ledger)
    integrity = _object_dict(artifacts.integrity)
    controls = coverage_ledger["controls"]
    assert isinstance(controls, list)
    control_entries = [
        _object_dict(entry)
        for entry in controls
        if isinstance(entry, dict)
    ]
    control_statuses = _control_statuses(control_entries)

    if case.literacy_quality != "complete":
        assert control_statuses["ai_act.article4.ai_literacy_record"] != "pass"

    if case.base_scenario == "provider_low_risk":
        human_notice = control_statuses["ai_act.article50.human_interaction_notice"]
        if case.role_variant == "deployer":
            assert human_notice == "not_applicable"
        elif case.transparency_quality != "complete":
            assert human_notice != "pass"

    if case.base_scenario == "prohibited_emotion":
        assert (
            control_statuses["ai_act.article5.prohibited_practices_screen"]
            == "fail"
        )
        if case.transparency_quality != "complete":
            assert (
                control_statuses["ai_act.article50.emotion_biometric_notice"]
                != "pass"
            )

    if case.base_scenario == "deployer_high_risk":
        assert control_statuses["ai_act.triage.high_risk_annex_iii"] == "unknown"
        if case.high_risk_mutation in {
            "missing_provider_documentation",
            "missing_monitoring",
            "partial_monitoring",
        }:
            assert (
                control_statuses["ai_act.article26.instructions_monitoring"]
                != "pass"
            )
        if case.high_risk_mutation in {
            "missing_human_oversight",
            "partial_human_oversight",
            "missing_risk_owner",
        }:
            assert (
                control_statuses["ai_act.article26.human_oversight"] != "pass"
            )
        if case.high_risk_mutation == "missing_log_retention":
            assert control_statuses["ai_act.article26.log_retention"] != "pass"
        if case.high_risk_mutation == "missing_worker_notice":
            assert control_statuses["ai_act.article26.workplace_notice"] != "pass"
        if case.high_risk_mutation == "missing_affected_person_notice":
            assert (
                control_statuses[
                    "ai_act.article26.affected_person_notice_and_explanation"
                ]
                != "pass"
            )

    if not case.technical_present:
        assert control_statuses["vauban.readiness.technical_evidence"] != "pass"

    if case.signature_requested:
        assert integrity["signature_status"] == "requested_but_unavailable"
        assert integrity["signature"] is None
    else:
        assert integrity["signature_status"] == "unsigned"
        assert integrity["signature"] is None


@pytest.mark.parametrize("seed", range(32))
def test_ai_act_battle_campaign_preserves_core_invariants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
) -> None:
    """Run a seeded adversarial campaign and assert report-engine invariants."""
    monkeypatch.delenv("VAUBAN_AI_ACT_SIGNING_SECRET", raising=False)
    case = _sample_case(seed)
    case_dir = tmp_path / f"case_{seed}"
    case_dir.mkdir(parents=True, exist_ok=True)

    artifacts = generate_deployer_readiness_artifacts(
        _build_case_config(case, case_dir),
    )

    _assert_common_invariants(artifacts)
    _assert_case_invariants(case, artifacts)


def test_ai_act_battle_campaign_is_stable_under_fixed_clock_and_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identical inputs should produce identical outputs under a fixed clock."""
    monkeypatch.setenv("VAUBAN_AI_ACT_SIGNING_SECRET", "super-secret")
    _write_complete_evidence(tmp_path)
    config = replace(
        _provider_low_risk_config(tmp_path),
        bundle_signature_secret_env="VAUBAN_AI_ACT_SIGNING_SECRET",
    )

    with patch("vauban.ai_act.datetime.datetime", _FixedDateTime):
        first = generate_deployer_readiness_artifacts(config)
        second = generate_deployer_readiness_artifacts(config)

    assert first.report == second.report
    assert first.coverage_ledger == second.coverage_ledger
    assert first.controls_matrix == second.controls_matrix
    assert first.risk_register == second.risk_register
    assert first.evidence_manifest == second.evidence_manifest
    assert first.integrity == second.integrity
    assert first.pdf_report_bytes == second.pdf_report_bytes
    assert first.integrity["signature_status"] == "signed"
    assert isinstance(first.integrity["signature"], str)
    assert first.integrity["signature"] != ""

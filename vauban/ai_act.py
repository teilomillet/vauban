"""AI Act deployer-readiness reporting.

The goal of this module is conservative evidence assembly, not automated
legal certification. Every control ends in a terminal state and every
non-passing result includes a next action.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from pathlib import Path

    from vauban.types import AIActConfig

type ControlStatus = Literal["pass", "fail", "unknown", "not_applicable"]
type OverallStatus = Literal[
    "ready", "ready_with_actions", "blocked", "out_of_scope",
]
type ClaimKind = Literal[
    "observed_by_vauban",
    "derived_by_rule",
    "asserted_by_client",
    "requires_external_review",
]
type MarkerRule = tuple[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class AIActArtifacts:
    """Expanded artifact bundle for deployer-readiness output."""

    report: dict[str, object]
    coverage_ledger: dict[str, object]
    control_library: dict[str, object]
    remediation_markdown: str
    evidence_manifest: dict[str, object]
    controls_matrix: dict[str, object]
    risk_register: dict[str, object]
    annex_iii_classification: dict[str, object]
    fria_prep: dict[str, object]
    executive_summary_markdown: str
    fria_prep_markdown: str


def generate_deployer_readiness_bundle(
    config: AIActConfig,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], str]:
    """Generate the legacy four-artifact deployer-readiness bundle."""
    artifacts = generate_deployer_readiness_artifacts(config)
    return (
        artifacts.report,
        artifacts.coverage_ledger,
        artifacts.control_library,
        artifacts.remediation_markdown,
    )


def generate_deployer_readiness_artifacts(
    config: AIActConfig,
) -> AIActArtifacts:
    """Generate the full deployer-readiness artifact bundle."""
    evidence_manifest = _build_evidence_manifest(config)
    controls = _control_library_v1()
    coverage = [
        _evaluate_scope_minimum_facts(config),
        _evaluate_article4_literacy(config, evidence_manifest),
        _evaluate_article5_prohibited_practices(config),
        _evaluate_article50_human_interaction(config, evidence_manifest),
        _evaluate_article50_emotion_biometric(config, evidence_manifest),
        _evaluate_article50_deepfake(config, evidence_manifest),
        _evaluate_article50_public_interest_text(config, evidence_manifest),
        _evaluate_high_risk_triage(config),
        _evaluate_fria_triage(config),
        _evaluate_provider_documentation(config, evidence_manifest),
        _evaluate_technical_evidence(config, evidence_manifest),
        _evaluate_human_oversight(config, evidence_manifest),
        _evaluate_incident_response(config, evidence_manifest),
    ]
    _ensure_coverage_complete(controls, coverage)

    overall_status = _overall_status(config, coverage)
    risk_level = _risk_level(coverage)
    obligations = _likely_obligations(config)
    remediation_items = _collect_remediation_items(coverage)
    rulebook = _rulebook_metadata()
    technical_artifacts, technical_findings = _summarize_technical_artifacts(
        config.technical_report_paths,
    )
    evidence_manifest_payload = _build_evidence_manifest_payload(
        evidence_manifest,
        rulebook,
    )
    controls_matrix = _build_controls_matrix(
        controls,
        coverage,
        rulebook,
    )
    risk_register = _build_risk_register(
        coverage,
        controls,
        technical_findings,
        rulebook,
    )
    annex_iii_classification = _build_annex_iii_classification(
        config,
        rulebook,
    )
    fria_prep = _build_fria_prep(
        config,
        coverage,
        controls,
        rulebook,
    )
    evidence_manifest_sha256 = str(evidence_manifest_payload["manifest_sha256"])
    bundle_fingerprint = _json_sha256(
        {
            "rulebook_sha256": rulebook["sha256"],
            "coverage": coverage,
            "technical_findings": technical_findings,
            "evidence_manifest_sha256": evidence_manifest_sha256,
            "annex_iii_classification": annex_iii_classification,
            "fria_prep": fria_prep,
        },
    )
    evidence_manifest_payload["bundle_fingerprint"] = bundle_fingerprint
    controls_matrix["bundle_fingerprint"] = bundle_fingerprint
    risk_register["bundle_fingerprint"] = bundle_fingerprint
    annex_iii_classification["bundle_fingerprint"] = bundle_fingerprint
    fria_prep["bundle_fingerprint"] = bundle_fingerprint

    report: dict[str, object] = {
        "report_version": "ai_act_deployer_readiness_v1",
        "generated_at": datetime.datetime.now(
            tz=datetime.UTC,
        ).isoformat(timespec="seconds"),
        "overall_status": overall_status,
        "risk_level": risk_level,
        "system": {
            "company_name": config.company_name,
            "system_name": config.system_name,
            "role": config.role,
            "sector": config.sector,
            "eu_market": config.eu_market,
            "intended_purpose": config.intended_purpose,
        },
        "epistemic_policy": {
            "goal": (
                "Readiness evidence and gap reporting; not an automated"
                " declaration of legal compliance."
            ),
            "claim_kinds": [
                "observed_by_vauban",
                "derived_by_rule",
                "asserted_by_client",
                "requires_external_review",
            ],
            "completion_rule": (
                "Every applicable control must end in pass, fail, or"
                " unknown. No applicable control may be silently skipped."
            ),
        },
        "rulebook": rulebook,
        "evidence_manifest_sha256": evidence_manifest_sha256,
        "bundle_fingerprint": bundle_fingerprint,
        "likely_obligations": obligations,
        "summary": _build_summary(config, coverage, overall_status),
        "coverage_contract": {
            "n_controls": len(controls),
            "n_evaluated_controls": len(coverage),
            "coverage_complete": True,
            "silent_skips_allowed": False,
        },
        "controls_overview": _status_counts(coverage),
        "blocking_controls": [
            entry["control_id"]
            for entry in coverage
            if bool(entry["applies"]) and bool(entry["blocking"])
            and str(entry["status"]) in {"fail", "unknown"}
        ],
        "unresolved_controls": [
            entry["control_id"]
            for entry in coverage
            if bool(entry["applies"]) and str(entry["status"]) in {"fail", "unknown"}
        ],
        "technical_artifacts": {
            "n_attached": len(config.technical_report_paths),
            "n_existing": sum(
                1 for artifact in technical_artifacts if bool(artifact["exists"])
            ),
            "n_interpreted_json": sum(
                1
                for artifact in technical_artifacts
                if str(artifact["parsed_as"]) == "json"
            ),
            "n_findings": len(technical_findings),
            "artifacts": technical_artifacts,
        },
        "technical_findings": technical_findings,
        "annex_iii_classification": annex_iii_classification,
        "fria_prep": {
            "status": fria_prep["status"],
            "required": fria_prep["required"],
            "trigger_labels": fria_prep["trigger_labels"],
            "next_action": fria_prep["next_action"],
            "source_citations": fria_prep["source_citations"],
        },
        "required_client_actions": remediation_items,
        "sources": _sources(),
    }

    coverage_ledger: dict[str, object] = {
        "report_version": "ai_act_deployer_readiness_v1",
        "rulebook": rulebook,
        "bundle_fingerprint": bundle_fingerprint,
        "completion_rule": (
            "Applicable controls must terminate as pass, fail, or unknown."
        ),
        "controls": coverage,
        "evidence_manifest": evidence_manifest,
    }

    library: dict[str, object] = {
        "library_version": "control_library_v1",
        "rulebook": rulebook,
        "bundle_fingerprint": bundle_fingerprint,
        "sources": _sources(),
        "controls": controls,
    }

    remediation_markdown = _render_remediation_markdown(
        config,
        overall_status,
        remediation_items,
    )
    executive_summary_markdown = _render_executive_summary_markdown(
        config,
        report,
        risk_register,
    )
    fria_prep_markdown = _render_fria_prep_markdown(fria_prep)
    return AIActArtifacts(
        report=report,
        coverage_ledger=coverage_ledger,
        control_library=library,
        remediation_markdown=remediation_markdown,
        evidence_manifest=evidence_manifest_payload,
        controls_matrix=controls_matrix,
        risk_register=risk_register,
        annex_iii_classification=annex_iii_classification,
        fria_prep=fria_prep,
        executive_summary_markdown=executive_summary_markdown,
        fria_prep_markdown=fria_prep_markdown,
    )


@lru_cache(maxsize=1)
def _rulebook_resource_bytes() -> bytes:
    """Load the packaged deployer-readiness rulebook bytes."""
    return resources.files("vauban.ai_act_rules").joinpath(
        "deployer_readiness_v1.json",
    ).read_bytes()


@lru_cache(maxsize=1)
def _rulebook_v1() -> dict[str, object]:
    """Load the packaged deployer-readiness rulebook."""
    loaded = json.loads(_rulebook_resource_bytes().decode("utf-8"))
    if not isinstance(loaded, dict):
        msg = "AI Act rulebook must decode to a JSON object"
        raise TypeError(msg)
    return cast("dict[str, object]", loaded)


def _rulebook_metadata() -> dict[str, str]:
    """Return pinned rulebook metadata and fingerprint."""
    rulebook = _rulebook_v1()
    version = rulebook.get("rulebook_version")
    source_snapshot_date = rulebook.get("source_snapshot_date")
    if not isinstance(version, str):
        msg = "AI Act rulebook is missing rulebook_version"
        raise TypeError(msg)
    if not isinstance(source_snapshot_date, str):
        msg = "AI Act rulebook is missing source_snapshot_date"
        raise TypeError(msg)
    return {
        "version": version,
        "source_snapshot_date": source_snapshot_date,
        "sha256": _sha256_bytes(_rulebook_resource_bytes()),
    }


def _rulebook_controls() -> list[dict[str, object]]:
    """Return the versioned controls from the packaged rulebook."""
    raw_controls = _rulebook_v1().get("controls")
    if not isinstance(raw_controls, list):
        msg = "AI Act rulebook controls must be a list"
        raise TypeError(msg)
    controls: list[dict[str, object]] = []
    for entry in raw_controls:
        if not isinstance(entry, dict):
            msg = "AI Act rulebook control entries must be objects"
            raise TypeError(msg)
        controls.append(cast("dict[str, object]", entry))
    return controls


def _control_with_citations(control: dict[str, object]) -> dict[str, object]:
    """Attach resolved source citations to one control definition."""
    return {
        **control,
        "source_citations": _control_source_citations(control),
    }


def _control_source_citations(control: dict[str, object]) -> list[dict[str, str]]:
    """Resolve source citations for one control definition."""
    raw_source_ids = control.get("source_ids", [])
    if not isinstance(raw_source_ids, list):
        msg = "AI Act rulebook control source_ids must be a list"
        raise TypeError(msg)
    source_ids = [value for value in raw_source_ids if isinstance(value, str)]
    if len(source_ids) != len(raw_source_ids):
        msg = "AI Act rulebook control source_ids must be strings"
        raise TypeError(msg)
    sources_by_id = _sources_by_id()
    citations: list[dict[str, str]] = []
    for source_id in source_ids:
        citation = sources_by_id.get(source_id)
        if citation is None:
            msg = f"AI Act rulebook control references unknown source_id {source_id!r}"
            raise TypeError(msg)
        citations.append(citation)
    return citations


def _annex_iii_catalog() -> list[dict[str, object]]:
    """Return the versioned Annex III use-case catalog from the rulebook."""
    raw_catalog = _rulebook_v1().get("annex_iii_catalog")
    if not isinstance(raw_catalog, list):
        msg = "AI Act rulebook annex_iii_catalog must be a list"
        raise TypeError(msg)
    catalog: list[dict[str, object]] = []
    for entry in raw_catalog:
        if not isinstance(entry, dict):
            msg = "AI Act Annex III catalog entries must be objects"
            raise TypeError(msg)
        entry_dict = cast("dict[str, object]", entry)
        entry_with_citations = {
            **entry_dict,
            "source_citations": _control_source_citations(entry_dict),
        }
        catalog.append(entry_with_citations)
    return catalog


def _annex_iii_catalog_by_id() -> dict[str, dict[str, object]]:
    """Return the Annex III catalog keyed by use_case_id."""
    catalog_by_id: dict[str, dict[str, object]] = {}
    for entry in _annex_iii_catalog():
        use_case_id = entry.get("use_case_id")
        if not isinstance(use_case_id, str):
            msg = "AI Act Annex III catalog entry is missing use_case_id"
            raise TypeError(msg)
        catalog_by_id[use_case_id] = entry
    return catalog_by_id


def _rulebook_markers(name: str) -> tuple[MarkerRule, ...]:
    """Return one marker rule-set from the packaged rulebook."""
    raw_markers = _rulebook_v1().get("markers")
    if not isinstance(raw_markers, dict):
        msg = "AI Act rulebook markers must be an object"
        raise TypeError(msg)
    markers_dict = cast("dict[str, object]", raw_markers)
    marker_entries_raw = markers_dict.get(name)
    if not isinstance(marker_entries_raw, list):
        msg = f"AI Act rulebook markers[{name!r}] must be a list"
        raise TypeError(msg)
    markers: list[MarkerRule] = []
    for entry in marker_entries_raw:
        if not isinstance(entry, dict):
            msg = f"AI Act marker entry for {name!r} must be an object"
            raise TypeError(msg)
        entry_dict = cast("dict[str, object]", entry)
        label = entry_dict.get("label")
        patterns_raw = entry_dict.get("patterns")
        if not isinstance(label, str):
            msg = f"AI Act marker entry for {name!r} is missing string label"
            raise TypeError(msg)
        if not isinstance(patterns_raw, list):
            msg = f"AI Act marker entry for {name!r} is missing patterns list"
            raise TypeError(msg)
        patterns = [pattern for pattern in patterns_raw if isinstance(pattern, str)]
        if len(patterns) != len(patterns_raw):
            msg = f"AI Act marker entry for {name!r} has non-string patterns"
            raise TypeError(msg)
        markers.append((label, tuple(patterns)))
    return tuple(markers)


def _rulebook_technical_metric_rules() -> list[dict[str, object]]:
    """Return technical metric interpretation hints from the rulebook."""
    raw_rules = _rulebook_v1().get("technical_metric_rules")
    if not isinstance(raw_rules, list):
        msg = "AI Act rulebook technical_metric_rules must be a list"
        raise TypeError(msg)
    rules: list[dict[str, object]] = []
    for entry in raw_rules:
        if not isinstance(entry, dict):
            msg = "AI Act technical metric rule entries must be objects"
            raise TypeError(msg)
        rules.append(cast("dict[str, object]", entry))
    return rules


def _sha256_bytes(data: bytes) -> str:
    """Return a SHA-256 hex digest for raw bytes."""
    return hashlib.sha256(data).hexdigest()


def _json_sha256(payload: object) -> str:
    """Return a SHA-256 hex digest for canonical JSON payloads."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _path_sha256(path: Path) -> str | None:
    """Return a SHA-256 hex digest for a filesystem path when readable."""
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _path_size_bytes(path: Path) -> int | None:
    """Return file size in bytes when readable."""
    try:
        return path.stat().st_size
    except OSError:
        return None


def _control_library_v1() -> list[dict[str, object]]:
    """Return the deployer-readiness control library for v1."""
    return [_control_with_citations(control) for control in _rulebook_controls()]


def _control_definition(
    control_id: str,
    title: str,
    source_type: str,
    source_ref: str,
    description: str,
    applies_when: str,
    evidence_required: list[str],
    check_method: str,
    *,
    blocking: bool,
) -> dict[str, object]:
    """Construct one control definition."""
    return {
        "control_id": control_id,
        "title": title,
        "source_type": source_type,
        "source_ref": source_ref,
        "description": description,
        "applies_when": applies_when,
        "evidence_required": evidence_required,
        "check_method": check_method,
        "blocking": blocking,
    }


def _build_evidence_manifest(config: AIActConfig) -> list[dict[str, object]]:
    """Build the evidence manifest from config facts and file references."""
    evidence: list[dict[str, object]] = [
        {
            "evidence_id": "fact.role",
            "kind": "config_fact",
            "claim_kind": "asserted_by_client",
            "label": "Declared operator role",
            "value": config.role,
        },
        {
            "evidence_id": "fact.eu_market",
            "kind": "config_fact",
            "claim_kind": "asserted_by_client",
            "label": "Declared EU market scope",
            "value": config.eu_market,
        },
        {
            "evidence_id": "fact.intended_purpose",
            "kind": "config_fact",
            "claim_kind": "asserted_by_client",
            "label": "Declared intended purpose",
            "value": config.intended_purpose,
        },
        {
            "evidence_id": "fact.annex_iii_use_cases",
            "kind": "config_fact",
            "claim_kind": "asserted_by_client",
            "label": "Declared Annex III use cases",
            "value": config.annex_iii_use_cases,
        },
        {
            "evidence_id": "fact.uses_general_purpose_ai",
            "kind": "config_fact",
            "claim_kind": "asserted_by_client",
            "label": "Declared GPAI dependency",
            "value": config.uses_general_purpose_ai,
        },
        {
            "evidence_id": "fact.human_interaction",
            "kind": "config_fact",
            "claim_kind": "asserted_by_client",
            "label": "Declared direct human interaction",
            "value": config.interacts_with_natural_persons,
        },
        {
            "evidence_id": "fact.interaction_obvious",
            "kind": "config_fact",
            "claim_kind": "asserted_by_client",
            "label": "Declared obviousness of AI interaction",
            "value": config.interaction_obvious_to_persons,
        },
        {
            "evidence_id": "fact.emotion_or_biometric_exposure",
            "kind": "config_fact",
            "claim_kind": "asserted_by_client",
            "label": "Declared emotion or biometric exposure",
            "value": (
                config.exposes_emotion_recognition_or_biometric_categorization
            ),
        },
        {
            "evidence_id": "fact.uses_emotion_recognition",
            "kind": "config_fact",
            "claim_kind": "asserted_by_client",
            "label": "Declared emotion-recognition use",
            "value": config.uses_emotion_recognition,
        },
        {
            "evidence_id": "fact.uses_biometric_categorization",
            "kind": "config_fact",
            "claim_kind": "asserted_by_client",
            "label": "Declared biometric-categorization use",
            "value": config.uses_biometric_categorization,
        },
        {
            "evidence_id": "fact.biometric_sensitive_trait_inference",
            "kind": "config_fact",
            "claim_kind": "asserted_by_client",
            "label": "Declared sensitive-trait biometric inference",
            "value": config.biometric_categorization_infers_sensitive_traits,
        },
        {
            "evidence_id": "fact.public_interest_text",
            "kind": "config_fact",
            "claim_kind": "asserted_by_client",
            "label": "Declared public-interest text publication",
            "value": config.publishes_text_on_matters_of_public_interest,
        },
        {
            "evidence_id": "fact.public_interest_text_exception",
            "kind": "config_fact",
            "claim_kind": "asserted_by_client",
            "label": "Declared public-interest text exception facts",
            "value": (
                config.public_interest_text_human_review_or_editorial_control
                and config.public_interest_text_editorial_responsibility
            ),
        },
        {
            "evidence_id": "fact.synthetic_media",
            "kind": "config_fact",
            "claim_kind": "asserted_by_client",
            "label": "Declared deepfake or synthetic media exposure",
            "value": config.deploys_deepfake_or_synthetic_media,
        },
        {
            "evidence_id": "fact.deepfake_creative_context",
            "kind": "config_fact",
            "claim_kind": "asserted_by_client",
            "label": "Declared creative or satirical deepfake context",
            "value": (
                config.deepfake_creative_satirical_artistic_or_fictional_context
            ),
        },
        {
            "evidence_id": "fact.annex_i_product_route",
            "kind": "config_fact",
            "claim_kind": "asserted_by_client",
            "label": "Declared Annex I product-route facts",
            "value": (
                config.annex_i_product_or_safety_component,
                config.annex_i_third_party_conformity_assessment,
            ),
        },
    ]
    evidence.extend(
        [
            _path_evidence(
                "file.ai_literacy_record",
                "AI literacy record",
                config.ai_literacy_record,
            ),
            _path_evidence(
                "file.transparency_notice",
                "Transparency notice",
                config.transparency_notice,
            ),
            _path_evidence(
                "file.human_oversight_procedure",
                "Human oversight procedure",
                config.human_oversight_procedure,
            ),
            _path_evidence(
                "file.incident_response_procedure",
                "Incident response procedure",
                config.incident_response_procedure,
            ),
            _path_evidence(
                "file.provider_documentation",
                "Provider documentation",
                config.provider_documentation,
            ),
        ],
    )
    for index, path in enumerate(config.technical_report_paths):
        evidence.append(
            _path_evidence(
                f"file.technical_report.{index}",
                f"Technical report {index + 1}",
                path,
            ),
        )
    return evidence


def _path_evidence(
    evidence_id: str,
    label: str,
    path: Path | None,
) -> dict[str, object]:
    """Build one file-based evidence record."""
    exists = path.exists() if path is not None else False
    sha256: str | None = None
    size_bytes: int | None = None
    if path is not None and exists:
        sha256 = _path_sha256(path)
        size_bytes = _path_size_bytes(path)
    return {
        "evidence_id": evidence_id,
        "kind": "file",
        "claim_kind": "observed_by_vauban",
        "label": label,
        "path": str(path) if path is not None else None,
        "exists": exists,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def _document_marker_assessment(
    path: Path | None,
    markers: tuple[MarkerRule, ...],
) -> tuple[list[str], list[str]]:
    """Return present and missing minimum markers for a text artifact."""
    if path is None or not path.exists():
        return [], [label for label, _patterns in markers]
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return [], [label for label, _patterns in markers]

    present: list[str] = []
    missing: list[str] = []
    for label, patterns in markers:
        if any(re.search(pattern, text) is not None for pattern in patterns):
            present.append(label)
        else:
            missing.append(label)
    return present, missing


def _format_marker_list(markers: list[str]) -> str:
    """Format a short marker list for rationales and remediation."""
    return ", ".join(markers) if markers else "none"


def _coerce_str_list(value: object) -> list[str]:
    """Coerce a JSON-like value into a list of strings for rendering."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _marker_ready_action(
    path_key: str,
    missing_markers: list[str],
    *,
    extra_note: str | None = None,
) -> str:
    """Render a standard remediation action for missing document markers."""
    action = (
        f"Update {path_key} to cover: {_format_marker_list(missing_markers)}."
        " Then re-run this readiness report."
    )
    if extra_note is None:
        return action
    return f"{action} {extra_note}"


def _ai_literacy_markers() -> tuple[MarkerRule, ...]:
    """Return minimum content markers for AI literacy evidence."""
    return _rulebook_markers("ai_literacy")


def _human_interaction_notice_markers() -> tuple[MarkerRule, ...]:
    """Return minimum markers for human-interaction transparency notices."""
    return _rulebook_markers("human_interaction_notice")


def _emotion_biometric_notice_markers() -> tuple[MarkerRule, ...]:
    """Return minimum markers for emotion or biometric notices."""
    return _rulebook_markers("emotion_biometric_notice")


def _synthetic_media_notice_markers() -> tuple[MarkerRule, ...]:
    """Return minimum markers for synthetic-media disclosures."""
    return _rulebook_markers("synthetic_media_notice")


def _public_interest_notice_markers() -> tuple[MarkerRule, ...]:
    """Return minimum markers for public-interest text disclosures."""
    return _rulebook_markers("public_interest_notice")


def _public_interest_exception_markers() -> tuple[MarkerRule, ...]:
    """Return minimum markers for the Article 50 public-interest exception."""
    return _rulebook_markers("public_interest_exception")


def _provider_document_markers() -> tuple[MarkerRule, ...]:
    """Return minimum markers for provider documentation."""
    return _rulebook_markers("provider_documentation")


def _human_oversight_markers() -> tuple[MarkerRule, ...]:
    """Return minimum markers for human-oversight procedures."""
    return _rulebook_markers("human_oversight")


def _incident_response_markers() -> tuple[MarkerRule, ...]:
    """Return minimum markers for incident-response procedures."""
    return _rulebook_markers("incident_response")


def _build_evidence_manifest_payload(
    evidence_manifest: list[dict[str, object]],
    rulebook: dict[str, str],
) -> dict[str, object]:
    """Build the standalone evidence manifest payload."""
    manifest_sha256 = _json_sha256(evidence_manifest)
    return {
        "manifest_version": "ai_act_evidence_manifest_v1",
        "rulebook": rulebook,
        "manifest_sha256": manifest_sha256,
        "evidence": evidence_manifest,
    }


def _build_controls_matrix(
    controls: list[dict[str, object]],
    coverage: list[dict[str, object]],
    rulebook: dict[str, str],
) -> dict[str, object]:
    """Build a flat controls matrix for customer and reviewer export."""
    coverage_by_id = {
        str(entry["control_id"]): entry
        for entry in coverage
    }
    rows: list[dict[str, object]] = []
    for control in controls:
        control_id = str(control["control_id"])
        entry = coverage_by_id[control_id]
        row: dict[str, object] = {
            **control,
            "applies": entry["applies"],
            "status": entry["status"],
            "claim_kind": entry["claim_kind"],
            "rationale": entry["rationale"],
            "confidence": entry["confidence"],
            "evidence_ids": entry["evidence_ids"],
            "owner_action": entry["owner_action"],
            "legal_review_required": entry["legal_review_required"],
            "present_markers": entry["present_markers"],
            "missing_markers": entry["missing_markers"],
            "required_artifacts": entry["required_artifacts"],
            "recheck_hint": entry["recheck_hint"],
        }
        rows.append(row)
    return {
        "matrix_version": "ai_act_controls_matrix_v1",
        "rulebook": rulebook,
        "rows": rows,
    }


def _control_risk_severity(entry: dict[str, object]) -> str:
    """Map a control outcome to a conservative register severity."""
    status = str(entry["status"])
    if bool(entry["legal_review_required"]):
        return "high"
    if bool(entry["blocking"]) and status in {"fail", "unknown"}:
        return "high"
    if status in {"fail", "unknown"}:
        return "medium"
    return "low"


def _build_risk_register(
    coverage: list[dict[str, object]],
    controls: list[dict[str, object]],
    technical_findings: list[dict[str, object]],
    rulebook: dict[str, str],
) -> dict[str, object]:
    """Build a conservative risk register from controls and technical findings."""
    control_lookup = {
        str(control["control_id"]): control
        for control in controls
    }
    items: list[dict[str, object]] = []
    for entry in coverage:
        if not bool(entry["applies"]):
            continue
        if str(entry["status"]) not in {"fail", "unknown"}:
            continue
        control_id = str(entry["control_id"])
        control = control_lookup[control_id]
        items.append(
            {
                "risk_id": f"control.{control_id}",
                "source": "control",
                "severity": _control_risk_severity(entry),
                "title": control["title"],
                "summary": entry["rationale"],
                "evidence_ids": entry["evidence_ids"],
                "next_action": entry["owner_action"],
                "recheck_hint": entry["recheck_hint"],
                "legal_review_required": entry["legal_review_required"],
                "source_citations": control.get("source_citations", []),
            },
        )
    items.extend(technical_findings)
    severity_counts = Counter(str(item["severity"]) for item in items)
    return {
        "register_version": "ai_act_risk_register_v1",
        "rulebook": rulebook,
        "items": items,
        "summary": {
            "n_items": len(items),
            "n_high": severity_counts.get("high", 0),
            "n_medium": severity_counts.get("medium", 0),
            "n_low": severity_counts.get("low", 0),
        },
    }


def _build_fria_prep(
    config: AIActConfig,
    coverage: list[dict[str, object]],
    controls: list[dict[str, object]],
    rulebook: dict[str, str],
) -> dict[str, object]:
    """Build a FRIA preparation artifact from the current triage state."""
    control_lookup = {
        str(control["control_id"]): control
        for control in controls
    }
    coverage_lookup = {
        str(entry["control_id"]): entry
        for entry in coverage
    }
    fria_control_id = "ai_act.triage.fria_requirement"
    fria_control = control_lookup[fria_control_id]
    fria_entry = coverage_lookup[fria_control_id]
    trigger_labels = _fria_triggers(config)
    status_raw = str(fria_entry["status"])
    if status_raw == "not_applicable":
        prep_status = "out_of_scope"
        required = False
    elif status_raw == "unknown":
        prep_status = "review_needed"
        required = True
    else:
        prep_status = "not_required"
        required = False

    questionnaire: list[dict[str, object]] = [
        {
            "question_id": "context_of_use",
            "prompt": (
                "Describe the specific context of use, affected persons or"
                " groups, and any decisions or recommendations supported by"
                " the AI system."
            ),
            "why_it_matters": "FRIA depends on the concrete deployment context.",
        },
        {
            "question_id": "provider_materials",
            "prompt": (
                "Attach provider instructions for use, intended purpose,"
                " limitations, and relevant technical documentation."
            ),
            "why_it_matters": "Article 27 expects deployers to use provider inputs.",
        },
        {
            "question_id": "fundamental_rights_risks",
            "prompt": (
                "List the specific risks of harm to fundamental rights,"
                " health, or safety for affected persons."
            ),
            "why_it_matters": "This is the core substantive FRIA input.",
        },
        {
            "question_id": "human_oversight_and_redress",
            "prompt": (
                "Document human oversight, complaint handling, redress, and"
                " escalation arrangements for this deployment."
            ),
            "why_it_matters": "Mitigations and governance are part of FRIA prep.",
        },
        {
            "question_id": "authority_notification",
            "prompt": (
                "Confirm the internal owner for any authority notification or"
                " registration step if FRIA is confirmed as required."
            ),
            "why_it_matters": "The deployer must operationalize follow-through.",
        },
    ]
    checklist: list[str] = [
        "Deployment-specific intended purpose and workflow description",
        "Affected persons and groups, including vulnerable populations",
        "Provider instructions for use, limitations, and model/service version",
        "Concrete fundamental-rights, health, and safety risk register",
        "Human oversight, complaint handling, and redress procedures",
        "Mitigation measures, monitoring plan, and incident escalation owner",
    ]
    return {
        "prep_version": "ai_act_fria_prep_v1",
        "rulebook": rulebook,
        "required": required,
        "status": prep_status,
        "control_id": fria_control_id,
        "control_title": fria_control["title"],
        "control_status": status_raw,
        "source_citations": fria_control.get("source_citations", []),
        "trigger_labels": trigger_labels,
        "trigger_titles": _human_readable_trigger_labels(trigger_labels),
        "next_action": fria_entry["owner_action"],
        "legal_review_required": fria_entry["legal_review_required"],
        "known_inputs": {
            "company_name": config.company_name,
            "system_name": config.system_name,
            "role": config.role,
            "sector": config.sector,
            "public_sector_use": config.public_sector_use,
            "provides_public_service": config.provides_public_service,
            "creditworthiness_or_credit_score_assessment": (
                config.creditworthiness_or_credit_score_assessment
            ),
            "life_or_health_insurance_risk_pricing": (
                config.life_or_health_insurance_risk_pricing
            ),
        },
        "questionnaire": questionnaire,
        "evidence_checklist": checklist,
    }


def _build_annex_iii_classification(
    config: AIActConfig,
    rulebook: dict[str, str],
) -> dict[str, object]:
    """Build a classification artifact for declared Annex III use cases."""
    use_case_ids = _declared_annex_iii_use_cases(config)
    catalog = _annex_iii_catalog_by_id()
    entries = [catalog[use_case_id] for use_case_id in use_case_ids]
    areas: dict[str, dict[str, object]] = {}
    for entry in entries:
        area = str(entry["area"])
        area_bucket = areas.get(area)
        if area_bucket is None:
            area_bucket = {
                "area": area,
                "area_title": entry["area_title"],
                "use_cases": [],
                "has_generic_use_case": False,
            }
            areas[area] = area_bucket
        raw_use_cases = area_bucket.get("use_cases")
        if not isinstance(raw_use_cases, list):
            msg = "annex_iii classification use_cases must be a list"
            raise TypeError(msg)
        use_cases = cast("list[dict[str, object]]", raw_use_cases)
        use_case_entry: dict[str, object] = {
            "use_case_id": entry["use_case_id"],
            "title": entry["title"],
            "generic": entry["generic"],
            "source_citations": entry["source_citations"],
        }
        use_cases.append(use_case_entry)
        area_bucket["use_cases"] = use_cases
        if bool(entry["generic"]):
            area_bucket["has_generic_use_case"] = True
    status = (
        "no_use_cases_declared"
        if not entries
        else (
            "generic_use_cases_declared"
            if any(bool(entry["generic"]) for entry in entries)
            else "specific_use_cases_declared"
        )
    )
    return {
        "classification_version": "ai_act_annex_iii_classification_v1",
        "rulebook": rulebook,
        "status": status,
        "declared_use_case_ids": use_case_ids,
        "declared_use_case_titles": _human_readable_trigger_labels(use_case_ids),
        "areas": [areas[key] for key in sorted(areas)],
        "n_declared_use_cases": len(entries),
        "source_catalog": entries,
    }


def _summarize_technical_artifacts(
    paths: list[Path],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Summarize attached technical artifacts and derive conservative findings."""
    summaries: list[dict[str, object]] = []
    findings: list[dict[str, object]] = []
    for index, path in enumerate(paths):
        artifact_id = f"file.technical_report.{index}"
        summary, artifact_findings = _summarize_one_technical_artifact(
            path,
            artifact_id,
        )
        summaries.append(summary)
        findings.extend(artifact_findings)
    return summaries, findings


def _summarize_one_technical_artifact(
    path: Path,
    artifact_id: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Summarize one technical artifact and extract risk-relevant metrics."""
    summary: dict[str, object] = {
        "artifact_id": artifact_id,
        "path": str(path),
        "exists": path.exists(),
        "sha256": _path_sha256(path) if path.exists() else None,
        "size_bytes": _path_size_bytes(path) if path.exists() else None,
        "parsed_as": "missing",
        "metrics": [],
    }
    if not path.exists():
        return summary, []

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        summary["parsed_as"] = "unreadable"
        return summary, []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        summary["parsed_as"] = "text"
        return summary, []

    summary["parsed_as"] = "json"
    metrics = _interesting_technical_metrics(parsed)
    summary["metrics"] = metrics
    return summary, _technical_findings_from_metrics(
        metrics,
        artifact_id,
        path,
    )


def _interesting_technical_metrics(value: object) -> list[dict[str, object]]:
    """Extract risk-relevant metrics from parsed JSON artifacts."""
    interesting: list[dict[str, object]] = []
    for metric_name, metric_value in _extract_numeric_metrics("", value):
        rule = _technical_metric_rule(metric_name)
        if rule is None:
            continue
        interesting.append(
            {
                "metric": metric_name,
                "value": metric_value,
                "kind": rule["kind"],
                "high_threshold": rule["high_threshold"],
            },
        )
    return interesting


def _extract_numeric_metrics(
    prefix: str,
    value: object,
) -> list[tuple[str, float]]:
    """Recursively extract numeric metrics from a JSON-like object."""
    if isinstance(value, bool):
        return []
    if isinstance(value, int | float):
        metric_name = prefix or "value"
        return [(metric_name, float(value))]
    if isinstance(value, list):
        metrics: list[tuple[str, float]] = []
        for index, item in enumerate(value):
            child_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            metrics.extend(_extract_numeric_metrics(child_prefix, item))
        return metrics
    if isinstance(value, dict):
        metrics = []
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            child_prefix = f"{prefix}.{key}" if prefix else key
            metrics.extend(_extract_numeric_metrics(child_prefix, item))
        return metrics
    return []


def _metric_tokens(metric_name: str) -> list[str]:
    """Split a dotted metric path into lowercase alphanumeric tokens."""
    return [
        token
        for token in re.split(r"[^a-z0-9]+", metric_name.lower())
        if token
    ]


def _metric_leaf(metric_name: str) -> str:
    """Return the leaf segment of a dotted metric path."""
    return metric_name.rsplit(".", maxsplit=1)[-1]


def _token_matches_metric_term(token: str, term: str) -> bool:
    """Return whether one token satisfies a rulebook metric term."""
    if token == term or token.removesuffix("s") == term:
        return True
    return term == "evade" and token in {"evasion", "evaded"}


def _has_supported_risk_metric_semantics(metric_name: str) -> bool:
    """Require rate/score-like semantics before interpreting a metric."""
    leaf_tokens = _metric_tokens(_metric_leaf(metric_name))
    if any(
        token in {"rate", "ratio", "fraction", "probability", "score", "asr"}
        for token in leaf_tokens
    ):
        return True
    return (
        len(leaf_tokens) == 1
        and leaf_tokens[0] in {"toxicity", "unsafe", "violation"}
    )


def _technical_metric_rule(metric_name: str) -> dict[str, object] | None:
    """Return the first rulebook technical metric rule matching *metric_name*."""
    if not _has_supported_risk_metric_semantics(metric_name):
        return None
    metric_tokens = _metric_tokens(metric_name)
    for rule in _rulebook_technical_metric_rules():
        kind = rule.get("kind")
        match_any = rule.get("match_any")
        high_threshold = rule.get("high_threshold")
        if not isinstance(kind, str):
            continue
        if not isinstance(match_any, list):
            continue
        if not isinstance(high_threshold, int | float):
            continue
        match_terms = [term for term in match_any if isinstance(term, str)]
        if any(
            _token_matches_metric_term(token, term)
            for token in metric_tokens
            for term in match_terms
        ):
            return {
                "kind": kind,
                "high_threshold": float(high_threshold),
            }
    return None


def _technical_findings_from_metrics(
    metrics: list[dict[str, object]],
    artifact_id: str,
    path: Path,
) -> list[dict[str, object]]:
    """Convert technical metrics into conservative risk-register findings."""
    findings: list[dict[str, object]] = []
    for metric in metrics:
        metric_name = metric.get("metric")
        metric_value = metric.get("value")
        kind = metric.get("kind")
        high_threshold = metric.get("high_threshold")
        if not isinstance(metric_name, str):
            continue
        if not isinstance(metric_value, float):
            continue
        if not isinstance(kind, str):
            continue
        if not isinstance(high_threshold, float):
            continue
        if metric_value <= 0.0:
            continue
        severity = "high" if metric_value >= high_threshold else "medium"
        findings.append(
            {
                "risk_id": f"technical.{artifact_id}.{metric_name}",
                "source": "technical_artifact",
                "severity": severity,
                "title": f"Observed {kind} metric in technical evidence",
                "summary": (
                    f"Observed {metric_name}={metric_value:.6g} in"
                    f" {path.name}."
                ),
                "evidence_ids": [artifact_id],
                "next_action": (
                    "Review the underlying technical artifact and decide"
                    " whether additional mitigation or retesting is required."
                ),
                "recheck_hint": (
                    "Replace or update the technical artifact, then re-run"
                    " the readiness report."
                ),
                "legal_review_required": False,
            },
        )
    return findings


def _render_executive_summary_markdown(
    config: AIActConfig,
    report: dict[str, object],
    risk_register: dict[str, object],
) -> str:
    """Render a concise executive summary for external review."""
    likely_obligations = _coerce_str_list(report.get("likely_obligations"))
    unresolved_controls = _coerce_str_list(report.get("unresolved_controls"))
    risk_summary = report.get("risk_level")
    annex_iii_classification = report.get("annex_iii_classification")
    rulebook = report.get("rulebook")
    bundle_fingerprint = report.get("bundle_fingerprint")
    lines = [
        "# AI Act Executive Summary",
        "",
        f"- Company: {config.company_name}",
        f"- System: {config.system_name}",
        f"- Overall status: {report['overall_status']}",
        f"- Risk level: {risk_summary}",
        f"- Bundle fingerprint: {bundle_fingerprint}",
    ]
    if isinstance(rulebook, dict):
        rulebook_dict = cast("dict[str, object]", rulebook)
        version = rulebook_dict.get("version")
        snapshot = rulebook_dict.get("source_snapshot_date")
        if isinstance(version, str):
            lines.append(f"- Rulebook version: {version}")
        if isinstance(snapshot, str):
            lines.append(f"- Rulebook snapshot date: {snapshot}")
    if isinstance(annex_iii_classification, dict):
        classification_dict = cast("dict[str, object]", annex_iii_classification)
        classification_status = classification_dict.get("status")
        if isinstance(classification_status, str):
            lines.append(f"- Annex III classification: {classification_status}")
    lines.extend(["", "## Likely Obligations", ""])
    if likely_obligations:
        lines.extend(f"- {item}" for item in likely_obligations)
    else:
        lines.append("- No likely obligations were identified in the current scope.")
    lines.extend(["", "## Key Gaps", ""])
    if unresolved_controls:
        lines.extend(f"- {item}" for item in unresolved_controls)
    else:
        lines.append("- No unresolved controls remain in the current scope.")
    items_raw = risk_register.get("items")
    lines.extend(["", "## Risk Register Size", ""])
    if isinstance(items_raw, list):
        lines.append(f"- Total risk items: {len(items_raw)}")
    return "\n".join(lines) + "\n"


def _render_fria_prep_markdown(fria_prep: dict[str, object]) -> str:
    """Render a markdown FRIA preparation artifact."""
    trigger_labels = _coerce_str_list(fria_prep.get("trigger_titles"))
    checklist_raw = fria_prep.get("evidence_checklist")
    checklist = _coerce_str_list(checklist_raw)
    questionnaire_raw = fria_prep.get("questionnaire")
    lines = [
        "# FRIA Preparation Pack",
        "",
        f"- Status: {fria_prep['status']}",
        f"- Required: {fria_prep['required']}",
        f"- Control: {fria_prep['control_id']}",
        "",
        "## Trigger Labels",
        "",
    ]
    if trigger_labels:
        lines.extend(f"- {label}" for label in trigger_labels)
    else:
        lines.append("- No FRIA trigger labels were identified.")
    lines.extend(["", "## Evidence Checklist", ""])
    if checklist:
        lines.extend(f"- {item}" for item in checklist)
    else:
        lines.append("- No checklist items were generated.")
    lines.extend(["", "## Questionnaire", ""])
    if isinstance(questionnaire_raw, list):
        for item in questionnaire_raw:
            if not isinstance(item, dict):
                continue
            item_dict = cast("dict[str, object]", item)
            prompt = item_dict.get("prompt")
            why_it_matters = item_dict.get("why_it_matters")
            if isinstance(prompt, str):
                lines.append(f"- {prompt}")
            if isinstance(why_it_matters, str):
                lines.append(f"  Why it matters: {why_it_matters}")
    else:
        lines.append("- No questionnaire items were generated.")
    next_action = fria_prep.get("next_action")
    if isinstance(next_action, str):
        lines.extend(["", "## Next Action", "", f"- {next_action}"])
    return "\n".join(lines) + "\n"


def _evaluate_scope_minimum_facts(config: AIActConfig) -> dict[str, object]:
    """Check that minimum factual inputs exist for the report."""
    has_facts = bool(
        config.company_name.strip()
        and config.system_name.strip()
        and config.intended_purpose.strip()
    )
    if has_facts:
        return _entry(
            "scope.minimum_facts",
            applies=True,
            status="pass",
            blocking=True,
            claim_kind="observed_by_vauban",
            rationale=(
                "Company name, system name, and intended purpose are all"
                " present in the config."
            ),
            evidence_ids=[
                "fact.role",
                "fact.eu_market",
                "fact.intended_purpose",
            ],
            confidence=1.0,
        )
    return _entry(
        "scope.minimum_facts",
        applies=True,
        status="fail",
        blocking=True,
        claim_kind="observed_by_vauban",
        rationale=(
            "The report is missing minimum scope facts and would otherwise"
            " need to guess its subject."
        ),
        evidence_ids=[],
        confidence=1.0,
        owner_action=(
            "Set [ai_act].company_name, [ai_act].system_name, and"
            " [ai_act].intended_purpose before re-running."
        ),
    )


def _evaluate_article4_literacy(
    config: AIActConfig,
    evidence_manifest: list[dict[str, object]],
) -> dict[str, object]:
    """Evaluate Article 4 AI literacy evidence."""
    applies = config.eu_market and config.role != "research"
    evidence_id = "file.ai_literacy_record"
    present_markers, missing_markers = _document_marker_assessment(
        config.ai_literacy_record,
        _ai_literacy_markers(),
    )
    if not applies:
        return _entry_not_applicable(
            "ai_act.article4.ai_literacy_record",
            "Article 4 literacy evidence does not apply outside EU-market"
            " deployer/provider operation or pure research scope.",
        )
    if _evidence_exists(evidence_manifest, evidence_id) and not missing_markers:
        return _entry(
            "ai_act.article4.ai_literacy_record",
            applies=True,
            status="pass",
            blocking=True,
            claim_kind="observed_by_vauban",
            rationale=(
                "Vauban verified that an AI literacy record exists and covers"
                " the minimum Article 4 themes pinned in the rulebook."
            ),
            evidence_ids=[evidence_id],
            confidence=1.0,
            present_markers=present_markers,
            required_artifacts=["AI literacy record"],
            recheck_hint="Re-run after updating the attached literacy record.",
        )
    if _evidence_exists(evidence_manifest, evidence_id):
        return _entry(
            "ai_act.article4.ai_literacy_record",
            applies=True,
            status="unknown",
            blocking=True,
            claim_kind="observed_by_vauban",
            rationale=(
                "An AI literacy record exists, but Vauban could not verify"
                " all minimum evidence markers. Missing markers:"
                f" {_format_marker_list(missing_markers)}."
            ),
            evidence_ids=[evidence_id],
            confidence=0.8,
            owner_action=_marker_ready_action(
                "[ai_act].ai_literacy_record",
                missing_markers,
            ),
            present_markers=present_markers,
            missing_markers=missing_markers,
            required_artifacts=["AI literacy record"],
            recheck_hint="Re-run after updating the attached literacy record.",
        )
    return _entry(
        "ai_act.article4.ai_literacy_record",
        applies=True,
        status="fail",
        blocking=True,
        claim_kind="observed_by_vauban",
        rationale=(
            "No verifiable AI literacy record was provided. Article 4 is"
            " already applicable to providers and deployers of AI systems."
        ),
        evidence_ids=[evidence_id],
        confidence=1.0,
        owner_action=(
            "Create and attach an internal AI literacy record covering target"
            " roles, organisation role, system context, risks, training date,"
            " owner, and refresh cadence."
        ),
        missing_markers=missing_markers,
        required_artifacts=["AI literacy record"],
        recheck_hint="Re-run after attaching the literacy record file.",
    )


def _evaluate_article5_prohibited_practices(
    config: AIActConfig,
) -> dict[str, object]:
    """Screen declared facts for obvious Article 5 prohibited-practice cases."""
    if not config.eu_market or config.role == "research":
        return _entry_not_applicable(
            "ai_act.article5.prohibited_practices_screen",
            "Article 5 screening is out of scope for non-EU or research-only"
            " operation.",
        )

    triggers: list[str] = []
    if (
        config.uses_emotion_recognition
        and (
            config.employment_or_workers_management
            or config.education_or_vocational_training
        )
        and not config.emotion_recognition_medical_or_safety_exception
    ):
        triggers.append("emotion_recognition_in_workplace_or_education")
    if (
        config.uses_biometric_categorization
        and config.biometric_categorization_infers_sensitive_traits
    ):
        triggers.append("biometric_categorization_sensitive_trait_inference")

    if not triggers:
        return _entry(
            "ai_act.article5.prohibited_practices_screen",
            applies=True,
            status="pass",
            blocking=True,
            claim_kind="derived_by_rule",
            rationale=(
                "No obvious Article 5 prohibited-practice trigger was"
                " declared in the current system description."
            ),
            evidence_ids=[
                "fact.eu_market",
                "fact.uses_emotion_recognition",
                "fact.uses_biometric_categorization",
                "fact.biometric_sensitive_trait_inference",
            ],
            confidence=0.7,
        )

    return _entry(
        "ai_act.article5.prohibited_practices_screen",
        applies=True,
        status="fail",
        blocking=True,
        claim_kind="requires_external_review",
        rationale=(
            "Declared facts match an obvious Article 5 prohibited-practice"
            " scenario: "
            f"{', '.join(triggers)}."
        ),
        evidence_ids=[
            "fact.eu_market",
            "fact.uses_emotion_recognition",
            "fact.uses_biometric_categorization",
            "fact.biometric_sensitive_trait_inference",
        ],
        confidence=0.9,
        owner_action=(
            "Stop relying on this report as a clean readiness signal until"
            " legal/compliance review confirms the use case is outside"
            " Article 5 or a valid exception applies."
        ),
        legal_review_required=True,
    )


def _evaluate_article50_human_interaction(
    config: AIActConfig,
    evidence_manifest: list[dict[str, object]],
) -> dict[str, object]:
    """Evaluate transparency evidence for non-obvious AI interaction."""
    provider_like_role = config.role in {"provider", "modifier"}
    declared_scenario = (
        config.eu_market
        and config.role != "research"
        and config.interacts_with_natural_persons
        and not config.interaction_obvious_to_persons
    )
    if declared_scenario and not provider_like_role:
        return _entry(
            "ai_act.article50.human_interaction_notice",
            applies=False,
            status="not_applicable",
            blocking=False,
            claim_kind="derived_by_rule",
            rationale=(
                "A non-obvious AI interaction scenario was declared, but the"
                " asserted role is not provider/modifier. In this v1 rulebook,"
                " the Article 50 interaction notice is only checked for"
                " provider-like roles supplying the user-facing AI system."
            ),
            evidence_ids=[
                "fact.role",
                "fact.human_interaction",
                "fact.interaction_obvious",
            ],
            confidence=0.7,
            owner_action=(
                "If your organisation supplies the user-facing AI system"
                " rather than only deploying someone else's system, set"
                " [ai_act].role = \"provider\" or \"modifier\" and re-run."
            ),
        )
    applies = (
        declared_scenario and provider_like_role
    )
    evidence_id = "file.transparency_notice"
    present_markers, missing_markers = _document_marker_assessment(
        config.transparency_notice,
        _human_interaction_notice_markers(),
    )
    if not applies:
        return _entry_not_applicable(
            "ai_act.article50.human_interaction_notice",
            "No non-obvious AI interaction scenario was declared.",
        )
    if _evidence_exists(evidence_manifest, evidence_id) and not missing_markers:
        return _entry(
            "ai_act.article50.human_interaction_notice",
            applies=True,
            status="pass",
            blocking=True,
            claim_kind="observed_by_vauban",
            rationale=(
                "A transparency notice exists and contains minimum markers for"
                " non-obvious AI interaction disclosure."
            ),
            evidence_ids=[
                evidence_id,
                "fact.human_interaction",
                "fact.interaction_obvious",
            ],
            confidence=0.9,
            present_markers=present_markers,
            required_artifacts=["Transparency notice"],
            recheck_hint="Re-run after updating the transparency notice.",
        )
    if _evidence_exists(evidence_manifest, evidence_id):
        return _entry(
            "ai_act.article50.human_interaction_notice",
            applies=True,
            status="unknown",
            blocking=True,
            claim_kind="observed_by_vauban",
            rationale=(
                "A transparency notice exists, but Vauban could not verify"
                " all minimum markers for non-obvious AI interaction."
                f" Missing markers: {_format_marker_list(missing_markers)}."
            ),
            evidence_ids=[
                evidence_id,
                "fact.human_interaction",
                "fact.interaction_obvious",
            ],
            confidence=0.8,
            owner_action=_marker_ready_action(
                "[ai_act].transparency_notice",
                missing_markers,
                extra_note=(
                    "Make the notice explicit that users are interacting with"
                    " AI rather than a human."
                ),
            ),
            present_markers=present_markers,
            missing_markers=missing_markers,
            required_artifacts=["Transparency notice"],
            recheck_hint="Re-run after updating the transparency notice.",
        )
    return _entry(
        "ai_act.article50.human_interaction_notice",
        applies=True,
        status="fail",
        blocking=True,
        claim_kind="observed_by_vauban",
        rationale=(
            "A non-obvious AI interaction scenario was declared, but no"
            " transparency notice was provided."
        ),
        evidence_ids=[
            evidence_id,
            "fact.human_interaction",
            "fact.interaction_obvious",
        ],
        confidence=1.0,
        owner_action=(
            "Attach a transparency notice stating that users are interacting"
            " with an AI system rather than a human."
        ),
        missing_markers=missing_markers,
        required_artifacts=["Transparency notice"],
        recheck_hint="Re-run after attaching the transparency notice file.",
    )


def _evaluate_article50_emotion_biometric(
    config: AIActConfig,
    evidence_manifest: list[dict[str, object]],
) -> dict[str, object]:
    """Evaluate transparency evidence for emotion or biometric exposure."""
    applies = (
        config.eu_market
        and config.role != "research"
        and config.exposes_emotion_recognition_or_biometric_categorization
    )
    evidence_id = "file.transparency_notice"
    present_markers, missing_markers = _document_marker_assessment(
        config.transparency_notice,
        _emotion_biometric_notice_markers(),
    )
    if not applies:
        return _entry_not_applicable(
            "ai_act.article50.emotion_biometric_notice",
            "No emotion-recognition or biometric-categorization exposure was"
            " declared.",
        )
    if _evidence_exists(evidence_manifest, evidence_id) and not missing_markers:
        return _entry(
            "ai_act.article50.emotion_biometric_notice",
            applies=True,
            status="pass",
            blocking=True,
            claim_kind="observed_by_vauban",
            rationale=(
                "A transparency notice exists and contains minimum markers for"
                " emotion-recognition or biometric-categorization exposure."
            ),
            evidence_ids=[evidence_id, "fact.emotion_or_biometric_exposure"],
            confidence=0.9,
            present_markers=present_markers,
            required_artifacts=["Transparency notice"],
            recheck_hint="Re-run after updating the transparency notice.",
        )
    if _evidence_exists(evidence_manifest, evidence_id):
        return _entry(
            "ai_act.article50.emotion_biometric_notice",
            applies=True,
            status="unknown",
            blocking=True,
            claim_kind="observed_by_vauban",
            rationale=(
                "A transparency notice exists, but Vauban could not verify"
                " all minimum markers for emotion-recognition or biometric"
                " exposure."
                f" Missing markers: {_format_marker_list(missing_markers)}."
            ),
            evidence_ids=[evidence_id, "fact.emotion_or_biometric_exposure"],
            confidence=0.8,
            owner_action=_marker_ready_action(
                "[ai_act].transparency_notice",
                missing_markers,
            ),
            present_markers=present_markers,
            missing_markers=missing_markers,
            required_artifacts=["Transparency notice"],
            recheck_hint="Re-run after updating the transparency notice.",
        )
    return _entry(
        "ai_act.article50.emotion_biometric_notice",
        applies=True,
        status="fail",
        blocking=True,
        claim_kind="observed_by_vauban",
        rationale=(
            "Emotion-recognition or biometric-categorization exposure was"
            " declared, but no transparency notice was provided."
        ),
        evidence_ids=[evidence_id, "fact.emotion_or_biometric_exposure"],
        confidence=1.0,
        owner_action=(
            "Attach a transparency notice for emotion-recognition or"
            " biometric-categorization exposure."
        ),
        missing_markers=missing_markers,
        required_artifacts=["Transparency notice"],
        recheck_hint="Re-run after attaching the transparency notice file.",
    )


def _evaluate_article50_deepfake(
    config: AIActConfig,
    evidence_manifest: list[dict[str, object]],
) -> dict[str, object]:
    """Evaluate deployer disclosure evidence for deepfake or synthetic media."""
    applies = (
        config.eu_market
        and config.role != "research"
        and config.deploys_deepfake_or_synthetic_media
    )
    evidence_id = "file.transparency_notice"
    present_markers, missing_markers = _document_marker_assessment(
        config.transparency_notice,
        _synthetic_media_notice_markers(),
    )
    if not applies:
        return _entry_not_applicable(
            "ai_act.article50.deepfake_disclosure",
            "No deepfake or synthetic media deployment was declared.",
        )
    if _evidence_exists(evidence_manifest, evidence_id) and not missing_markers:
        return _entry(
            "ai_act.article50.deepfake_disclosure",
            applies=True,
            status="pass",
            blocking=True,
            claim_kind="observed_by_vauban",
            rationale=(
                "A transparency notice exists for declared deepfake or"
                " synthetic-media deployment."
                + (
                    " The config also declares a creative/satirical/artistic/"
                    " fictional context, so Vauban treats this as evidence of"
                    " the limited disclosure path rather than a full removal"
                    " of the disclosure obligation."
                    if (
                        config.deepfake_creative_satirical_artistic_or_fictional_context
                    )
                    else ""
                )
            ),
            evidence_ids=[
                evidence_id,
                "fact.synthetic_media",
                "fact.deepfake_creative_context",
            ],
            confidence=1.0,
            present_markers=present_markers,
            required_artifacts=["Transparency notice"],
            recheck_hint="Re-run after updating the transparency notice.",
        )
    if _evidence_exists(evidence_manifest, evidence_id):
        return _entry(
            "ai_act.article50.deepfake_disclosure",
            applies=True,
            status="unknown",
            blocking=True,
            claim_kind="observed_by_vauban",
            rationale=(
                "A transparency notice exists, but Vauban could not verify"
                " all minimum markers for synthetic-media disclosure."
                f" Missing markers: {_format_marker_list(missing_markers)}."
            ),
            evidence_ids=[
                evidence_id,
                "fact.synthetic_media",
                "fact.deepfake_creative_context",
            ],
            confidence=0.8,
            owner_action=_marker_ready_action(
                "[ai_act].transparency_notice",
                missing_markers,
                extra_note=(
                    "If you rely on the creative/satirical/artistic/fictional"
                    " context, keep a clear disclosure of the existence of"
                    " generated or manipulated content."
                ),
            ),
            present_markers=present_markers,
            missing_markers=missing_markers,
            required_artifacts=["Transparency notice"],
            recheck_hint="Re-run after updating the transparency notice.",
        )
    return _entry(
        "ai_act.article50.deepfake_disclosure",
        applies=True,
        status="fail",
        blocking=True,
        claim_kind="observed_by_vauban",
        rationale=(
            "Deepfake or synthetic-media deployment was declared, but no"
            " disclosure evidence was provided."
        ),
        evidence_ids=[
            evidence_id,
            "fact.synthetic_media",
            "fact.deepfake_creative_context",
        ],
        confidence=1.0,
        owner_action=(
            "Add a deployer-facing disclosure notice covering synthetic or"
            " deepfake content and attach it via [ai_act].transparency_notice."
            + (
                " Because the config declares a creative/satirical/artistic/"
                " fictional context, the notice can focus on disclosing the"
                " existence of generated or manipulated content in an"
                " appropriate manner."
                if (
                    config.deepfake_creative_satirical_artistic_or_fictional_context
                )
                else ""
            )
        ),
        missing_markers=missing_markers,
        required_artifacts=["Transparency notice"],
        recheck_hint="Re-run after attaching the transparency notice file.",
    )


def _evaluate_article50_public_interest_text(
    config: AIActConfig,
    evidence_manifest: list[dict[str, object]],
) -> dict[str, object]:
    """Evaluate deployer disclosure evidence for public-interest text."""
    applies = (
        config.eu_market
        and config.role != "research"
        and config.publishes_text_on_matters_of_public_interest
    )
    evidence_id = "file.transparency_notice"
    present_markers, missing_markers = _document_marker_assessment(
        config.transparency_notice,
        _public_interest_notice_markers(),
    )
    exception_markers_present, exception_markers_missing = (
        _document_marker_assessment(
            config.transparency_notice,
            _public_interest_exception_markers(),
        )
    )
    exception_claimed = (
        config.public_interest_text_human_review_or_editorial_control
        and config.public_interest_text_editorial_responsibility
    )
    if not applies:
        return _entry_not_applicable(
            "ai_act.article50.public_interest_text_disclosure",
            "No public-interest text publication was declared.",
        )
    if _evidence_exists(evidence_manifest, evidence_id) and not missing_markers:
        return _entry(
            "ai_act.article50.public_interest_text_disclosure",
            applies=True,
            status="pass",
            blocking=True,
            claim_kind="observed_by_vauban",
            rationale=(
                "A transparency notice exists for declared public-interest"
                " AI-generated text."
            ),
            evidence_ids=[evidence_id, "fact.public_interest_text"],
            confidence=1.0,
            present_markers=present_markers,
            required_artifacts=["Transparency notice"],
            recheck_hint="Re-run after updating the transparency notice.",
        )
    if (
        exception_claimed
        and _evidence_exists(evidence_manifest, evidence_id)
        and not exception_markers_missing
    ):
        return _entry(
            "ai_act.article50.public_interest_text_disclosure",
            applies=False,
            status="not_applicable",
            blocking=False,
            claim_kind="observed_by_vauban",
            rationale=(
                "The config declares the Article 50 public-interest text"
                " exception, and the attached notice/policy includes markers"
                " for human review or editorial control plus editorial"
                " responsibility."
            ),
            evidence_ids=[
                evidence_id,
                "fact.public_interest_text",
                "fact.public_interest_text_exception",
            ],
            confidence=0.8,
            present_markers=exception_markers_present,
            required_artifacts=[
                "Transparency notice or policy documenting editorial control",
            ],
            recheck_hint="Re-run after updating the attached policy or notice.",
        )
    if _evidence_exists(evidence_manifest, evidence_id):
        missing_for_exception = (
            exception_markers_missing if exception_claimed else missing_markers
        )
        return _entry(
            "ai_act.article50.public_interest_text_disclosure",
            applies=True,
            status="unknown",
            blocking=True,
            claim_kind="observed_by_vauban",
            rationale=(
                (
                    "A public-interest text exception was declared, but"
                    " Vauban could not verify the minimum markers for human"
                    " review/editorial control and editorial responsibility."
                    if exception_claimed
                    else "A transparency notice exists, but Vauban could not"
                    " verify all minimum markers for public-interest text"
                    " disclosure."
                )
                + f" Missing markers: {_format_marker_list(missing_for_exception)}."
            ),
            evidence_ids=[
                evidence_id,
                "fact.public_interest_text",
                "fact.public_interest_text_exception",
            ],
            confidence=0.8,
            owner_action=_marker_ready_action(
                "[ai_act].transparency_notice",
                missing_for_exception,
                extra_note=(
                    "Either document human review/editorial control plus"
                    " editorial responsibility, or disclose the AI origin of"
                    " the public-interest text."
                )
                if exception_claimed
                else None,
            ),
            present_markers=(
                exception_markers_present if exception_claimed else present_markers
            ),
            missing_markers=missing_for_exception,
            required_artifacts=[
                "Transparency notice or documented editorial-control evidence",
            ]
            if exception_claimed
            else ["Transparency notice"],
            recheck_hint="Re-run after updating the transparency notice.",
        )
    if exception_claimed:
        return _entry(
            "ai_act.article50.public_interest_text_disclosure",
            applies=True,
            status="unknown",
            blocking=True,
            claim_kind="observed_by_vauban",
            rationale=(
                "A public-interest text exception was declared, but no"
                " evidence file was attached to verify human review/editorial"
                " control and editorial responsibility."
            ),
            evidence_ids=[
                evidence_id,
                "fact.public_interest_text",
                "fact.public_interest_text_exception",
            ],
            confidence=0.9,
            owner_action=(
                "Attach a policy or notice via [ai_act].transparency_notice"
                " that documents human review/editorial control and editorial"
                " responsibility, or disclose the AI origin of the"
                " public-interest text."
            ),
            missing_markers=exception_markers_missing,
            required_artifacts=[
                "Transparency notice or documented editorial-control evidence",
            ],
            recheck_hint="Re-run after attaching the evidence file.",
        )
    return _entry(
        "ai_act.article50.public_interest_text_disclosure",
        applies=True,
        status="fail",
        blocking=True,
        claim_kind="observed_by_vauban",
        rationale=(
            "Public-interest AI-generated text was declared, but no disclosure"
            " evidence was provided."
        ),
        evidence_ids=[evidence_id, "fact.public_interest_text"],
        confidence=1.0,
        owner_action=(
            "Document how AI-generated public-interest text is disclosed and"
            " attach the notice via [ai_act].transparency_notice."
        ),
        missing_markers=missing_markers,
        required_artifacts=["Transparency notice"],
        recheck_hint="Re-run after attaching the transparency notice file.",
    )


def _evaluate_high_risk_triage(config: AIActConfig) -> dict[str, object]:
    """Evaluate conservative high-risk / FRIA triage."""
    if not config.eu_market or config.role == "research":
        return _entry_not_applicable(
            "ai_act.triage.high_risk_annex_iii",
            "High-risk deployer triage is out of scope for non-EU or research"
            " operation.",
        )

    triggers = _high_risk_triggers(config)
    if not triggers:
        return _entry(
            "ai_act.triage.high_risk_annex_iii",
            applies=True,
            status="pass",
            blocking=True,
            claim_kind="derived_by_rule",
            rationale=(
                "No Annex I product-route or Annex III high-risk trigger was"
                " declared in the current system description."
            ),
            evidence_ids=["fact.role", "fact.eu_market", "fact.intended_purpose"],
            confidence=0.6,
        )

    return _entry(
        "ai_act.triage.high_risk_annex_iii",
        applies=True,
        status="unknown",
        blocking=True,
        claim_kind="requires_external_review",
        rationale=(
            "One or more declared use-context triggers may place the system in"
            " a high-risk or FRIA-sensitive category: "
            f"{', '.join(_human_readable_trigger_labels(triggers))}."
        ),
        evidence_ids=["fact.role", "fact.eu_market", "fact.intended_purpose"],
        confidence=0.8,
        legal_review_required=True,
        owner_action=(
            "Escalate to legal/compliance review for Annex I / Annex III"
            " analysis before relying on this report as a readiness signal."
        ),
    )


def _evaluate_fria_triage(config: AIActConfig) -> dict[str, object]:
    """Evaluate conservative FRIA triage."""
    if not config.eu_market or config.role == "research":
        return _entry_not_applicable(
            "ai_act.triage.fria_requirement",
            "FRIA triage is out of scope for non-EU or research operation.",
        )

    triggers = _fria_triggers(config)
    if not triggers:
        return _entry(
            "ai_act.triage.fria_requirement",
            applies=True,
            status="pass",
            blocking=True,
            claim_kind="derived_by_rule",
            rationale=(
                "No declared public-sector/public-service or Annex III point"
                " 5(b)/(c) trigger suggests a mandatory FRIA."
            ),
            evidence_ids=["fact.role", "fact.eu_market", "fact.intended_purpose"],
            confidence=0.6,
        )

    return _entry(
        "ai_act.triage.fria_requirement",
        applies=True,
        status="unknown",
        blocking=True,
        claim_kind="requires_external_review",
        rationale=(
            "Declared facts may trigger an Article 27 fundamental-rights"
            " impact assessment: "
            f"{', '.join(triggers)}."
        ),
        evidence_ids=["fact.role", "fact.eu_market", "fact.intended_purpose"],
        confidence=0.8,
        legal_review_required=True,
        owner_action=(
            "Escalate to legal/compliance review for Article 27 FRIA analysis"
            " before treating this use as ready."
        ),
    )


def _evaluate_provider_documentation(
    config: AIActConfig,
    evidence_manifest: list[dict[str, object]],
) -> dict[str, object]:
    """Evaluate whether provider-side documentation is retained."""
    applies = (
        config.eu_market
        and config.role == "deployer"
        and config.uses_general_purpose_ai
    )
    evidence_id = "file.provider_documentation"
    present_markers, missing_markers = _document_marker_assessment(
        config.provider_documentation,
        _provider_document_markers(),
    )
    if not applies:
        return _entry_not_applicable(
            "vauban.readiness.provider_documentation",
            "Provider documentation safeguard only applies to deployers"
            " building on third-party GPAI systems.",
        )
    if _evidence_exists(evidence_manifest, evidence_id) and not missing_markers:
        return _entry(
            "vauban.readiness.provider_documentation",
            applies=True,
            status="pass",
            blocking=False,
            claim_kind="observed_by_vauban",
            rationale="A provider documentation artifact exists.",
            evidence_ids=[evidence_id],
            confidence=1.0,
            present_markers=present_markers,
            required_artifacts=["Provider documentation"],
            recheck_hint="Re-run after updating the provider documentation.",
        )
    if _evidence_exists(evidence_manifest, evidence_id):
        return _entry(
            "vauban.readiness.provider_documentation",
            applies=True,
            status="unknown",
            blocking=False,
            claim_kind="observed_by_vauban",
            rationale=(
                "Provider documentation exists, but Vauban could not verify"
                " all minimum markers. Missing markers:"
                f" {_format_marker_list(missing_markers)}."
            ),
            evidence_ids=[evidence_id],
            confidence=0.8,
            owner_action=_marker_ready_action(
                "[ai_act].provider_documentation",
                missing_markers,
            ),
            present_markers=present_markers,
            missing_markers=missing_markers,
            required_artifacts=["Provider documentation"],
            recheck_hint="Re-run after updating the provider documentation.",
        )
    return _entry(
        "vauban.readiness.provider_documentation",
        applies=True,
        status="unknown",
        blocking=False,
        claim_kind="observed_by_vauban",
        rationale=(
            "No provider documentation artifact was attached, so downstream"
            " capability/limitation evidence is incomplete."
        ),
        evidence_ids=[evidence_id],
        confidence=1.0,
        owner_action=(
            "Collect model cards, provider limitations, and operational docs,"
            " then attach the bundle via [ai_act].provider_documentation."
        ),
        missing_markers=missing_markers,
        required_artifacts=["Provider documentation"],
        recheck_hint="Re-run after attaching the provider documentation file.",
    )


def _evaluate_technical_evidence(
    config: AIActConfig,
    evidence_manifest: list[dict[str, object]],
) -> dict[str, object]:
    """Evaluate whether technical reports were retained."""
    applies = (
        config.eu_market
        and config.role != "research"
        and config.uses_general_purpose_ai
    )
    report_ids: list[str] = [
        str(evidence["evidence_id"])
        for evidence in evidence_manifest
        if str(evidence["evidence_id"]).startswith("file.technical_report.")
    ]
    existing_ids: list[str] = [
        evidence_id for evidence_id in report_ids
        if _evidence_exists(evidence_manifest, evidence_id)
    ]
    if not applies:
        return _entry_not_applicable(
            "vauban.readiness.technical_evidence",
            "Technical-evidence safeguard is limited to AI deployments in scope"
            " for this readiness report.",
        )
    if existing_ids:
        return _entry(
            "vauban.readiness.technical_evidence",
            applies=True,
            status="pass",
            blocking=False,
            claim_kind="observed_by_vauban",
            rationale=(
                f"Vauban verified {len(existing_ids)} technical report file(s)."
            ),
            evidence_ids=existing_ids,
            confidence=1.0,
            required_artifacts=["At least one technical report artifact"],
            recheck_hint="Re-run after adding or replacing technical artifacts.",
        )
    return _entry(
        "vauban.readiness.technical_evidence",
        applies=True,
        status="unknown",
        blocking=False,
        claim_kind="observed_by_vauban",
        rationale=(
            "No technical testing evidence was attached, so robustness claims"
            " would be easy to challenge."
        ),
        evidence_ids=report_ids,
        confidence=1.0,
        owner_action=(
            "Attach existing attack/eval reports or run Vauban technical"
            " assessments before presenting the readiness pack externally."
        ),
        required_artifacts=["At least one technical report artifact"],
        recheck_hint="Re-run after attaching at least one technical report file.",
    )


def _evaluate_human_oversight(
    config: AIActConfig,
    evidence_manifest: list[dict[str, object]],
) -> dict[str, object]:
    """Evaluate whether a human oversight procedure exists."""
    applies = config.eu_market and config.role != "research"
    evidence_id = "file.human_oversight_procedure"
    present_markers, missing_markers = _document_marker_assessment(
        config.human_oversight_procedure,
        _human_oversight_markers(),
    )
    if config.risk_owner is not None:
        present_markers = [*present_markers, "risk_owner"]
    else:
        missing_markers = [*missing_markers, "risk_owner"]
    if not applies:
        return _entry_not_applicable(
            "vauban.readiness.human_oversight",
            "Human oversight safeguard is out of scope for research-only runs.",
        )
    if (
        _evidence_exists(evidence_manifest, evidence_id)
        and not missing_markers
    ):
        return _entry(
            "vauban.readiness.human_oversight",
            applies=True,
            status="pass",
            blocking=False,
            claim_kind="observed_by_vauban",
            rationale=(
                "A human oversight procedure exists and a risk owner is named."
            ),
            evidence_ids=[evidence_id],
            confidence=1.0,
            present_markers=present_markers,
            required_artifacts=["Human oversight procedure", "Named risk owner"],
            recheck_hint="Re-run after updating the oversight procedure.",
        )
    return _entry(
        "vauban.readiness.human_oversight",
        applies=True,
        status="unknown",
        blocking=False,
        claim_kind="observed_by_vauban",
        rationale=(
            "Human oversight evidence is incomplete because the procedure file"
            " and/or risk owner is missing."
        ),
        evidence_ids=[evidence_id],
        confidence=1.0,
        owner_action=(
            "Attach a human oversight procedure and set [ai_act].risk_owner."
        ),
        present_markers=present_markers,
        missing_markers=missing_markers,
        required_artifacts=["Human oversight procedure", "Named risk owner"],
        recheck_hint="Re-run after attaching the oversight procedure and owner.",
    )


def _evaluate_incident_response(
    config: AIActConfig,
    evidence_manifest: list[dict[str, object]],
) -> dict[str, object]:
    """Evaluate whether an incident response procedure exists."""
    applies = config.eu_market and config.role != "research"
    evidence_id = "file.incident_response_procedure"
    present_markers, missing_markers = _document_marker_assessment(
        config.incident_response_procedure,
        _incident_response_markers(),
    )
    if config.compliance_contact is not None:
        present_markers = [*present_markers, "compliance_contact"]
    else:
        missing_markers = [*missing_markers, "compliance_contact"]
    if not applies:
        return _entry_not_applicable(
            "vauban.readiness.incident_response",
            "Incident-response safeguard is out of scope for research-only runs.",
        )
    if (
        _evidence_exists(evidence_manifest, evidence_id)
        and not missing_markers
    ):
        return _entry(
            "vauban.readiness.incident_response",
            applies=True,
            status="pass",
            blocking=False,
            claim_kind="observed_by_vauban",
            rationale=(
                "An incident response procedure exists and a compliance contact"
                " is named."
            ),
            evidence_ids=[evidence_id],
            confidence=1.0,
            present_markers=present_markers,
            required_artifacts=[
                "Incident response procedure",
                "Named compliance contact",
            ],
            recheck_hint="Re-run after updating the incident-response procedure.",
        )
    return _entry(
        "vauban.readiness.incident_response",
        applies=True,
        status="unknown",
        blocking=False,
        claim_kind="observed_by_vauban",
        rationale=(
            "Incident-response evidence is incomplete because the procedure"
            " file and/or compliance contact is missing."
        ),
        evidence_ids=[evidence_id],
        confidence=1.0,
        owner_action=(
            "Attach an incident response procedure and set"
            " [ai_act].compliance_contact."
        ),
        present_markers=present_markers,
        missing_markers=missing_markers,
        required_artifacts=[
            "Incident response procedure",
            "Named compliance contact",
        ],
        recheck_hint="Re-run after attaching the incident procedure and contact.",
    )


def _entry(
    control_id: str,
    *,
    applies: bool,
    status: ControlStatus,
    blocking: bool,
    claim_kind: ClaimKind,
    rationale: str,
    evidence_ids: list[str],
    confidence: float,
    owner_action: str | None = None,
    legal_review_required: bool = False,
    present_markers: list[str] | None = None,
    missing_markers: list[str] | None = None,
    required_artifacts: list[str] | None = None,
    recheck_hint: str | None = None,
) -> dict[str, object]:
    """Construct one coverage entry."""
    return {
        "control_id": control_id,
        "applies": applies,
        "status": status,
        "blocking": blocking,
        "claim_kind": claim_kind,
        "rationale": rationale,
        "evidence_ids": evidence_ids,
        "confidence": confidence,
        "owner_action": owner_action,
        "legal_review_required": legal_review_required,
        "recheckable": owner_action is not None,
        "present_markers": [] if present_markers is None else present_markers,
        "missing_markers": [] if missing_markers is None else missing_markers,
        "required_artifacts": (
            [] if required_artifacts is None else required_artifacts
        ),
        "recheck_hint": recheck_hint,
    }


def _entry_not_applicable(
    control_id: str,
    rationale: str,
) -> dict[str, object]:
    """Construct a non-applicable coverage entry."""
    return _entry(
        control_id,
        applies=False,
        status="not_applicable",
        blocking=False,
        claim_kind="derived_by_rule",
        rationale=rationale,
        evidence_ids=[],
        confidence=1.0,
    )


def _evidence_exists(
    evidence_manifest: list[dict[str, object]],
    evidence_id: str,
) -> bool:
    """Return whether a manifest evidence item exists on disk."""
    for evidence in evidence_manifest:
        if evidence["evidence_id"] == evidence_id:
            exists = evidence.get("exists")
            return bool(exists)
    return False


def _declared_annex_iii_use_cases(config: AIActConfig) -> list[str]:
    """Return declared Annex III use-case identifiers from explicit and legacy flags."""
    use_case_ids = set(config.annex_iii_use_cases)
    if config.uses_emotion_recognition:
        use_case_ids.add("annex_iii_1_emotion_recognition")
    if config.uses_biometric_categorization:
        use_case_ids.add("annex_iii_1_biometric_categorisation")
    if config.biometric_or_emotion_related_use:
        use_case_ids.add("annex_iii_1_biometrics_generic")
    if config.education_or_vocational_training:
        use_case_ids.add("annex_iii_3_education_generic")
    if config.employment_or_workers_management:
        use_case_ids.add("annex_iii_4_employment_generic")
    if config.essential_private_or_public_service:
        use_case_ids.add("annex_iii_5_essential_services_generic")
    if config.creditworthiness_or_credit_score_assessment:
        use_case_ids.add("annex_iii_5_creditworthiness_or_credit_score")
    if config.life_or_health_insurance_risk_pricing:
        use_case_ids.add("annex_iii_5_life_or_health_insurance")
    if config.emergency_first_response_dispatch:
        use_case_ids.add("annex_iii_5_emergency_dispatch")
    if config.law_enforcement_use:
        use_case_ids.add("annex_iii_6_law_enforcement_generic")
    if config.migration_or_border_management_use:
        use_case_ids.add("annex_iii_7_migration_asylum_border_generic")
    if config.administration_of_justice_or_democracy_use:
        use_case_ids.add("annex_iii_8_justice_democracy_generic")
    return sorted(use_case_ids)


def _high_risk_triggers(config: AIActConfig) -> list[str]:
    """Return declared high-risk trigger labels."""
    triggers: list[str] = []
    if (
        config.annex_i_product_or_safety_component
        and config.annex_i_third_party_conformity_assessment
    ):
        triggers.append("annex_i_product_route")
    elif (
        config.annex_i_product_or_safety_component
        or config.annex_i_third_party_conformity_assessment
    ):
        triggers.append("potential_annex_i_product_route")
    triggers.extend(_declared_annex_iii_use_cases(config))
    if config.uses_profiling_or_similarly_significant_decision_support:
        triggers.append("similarly_significant_decision_support")
    return triggers


def _fria_triggers(config: AIActConfig) -> list[str]:
    """Return declared FRIA-sensitive trigger labels."""
    triggers: list[str] = []
    catalog = _annex_iii_catalog_by_id()
    annex_iii_use_cases = _declared_annex_iii_use_cases(config)
    has_public_service_relevant_use_case = any(
        str(catalog[use_case_id]["area"]) != "2"
        for use_case_id in annex_iii_use_cases
        if use_case_id in catalog
    )
    if (
        (config.public_sector_use or config.provides_public_service)
        and has_public_service_relevant_use_case
    ):
        triggers.append("public_sector_or_public_service_high_risk_use")
    if config.creditworthiness_or_credit_score_assessment:
        triggers.append("annex_iii_5_b_creditworthiness_or_credit_score")
    if config.life_or_health_insurance_risk_pricing:
        triggers.append("annex_iii_5_c_life_or_health_insurance")
    return triggers


def _human_readable_trigger_labels(triggers: list[str]) -> list[str]:
    """Resolve trigger identifiers to readable labels where the catalog knows them."""
    catalog = _annex_iii_catalog_by_id()
    labels: list[str] = []
    for trigger in triggers:
        catalog_entry = catalog.get(trigger)
        if catalog_entry is None:
            labels.append(trigger)
            continue
        title = catalog_entry.get("title")
        if isinstance(title, str):
            labels.append(title)
        else:
            labels.append(trigger)
    return labels


def _ensure_coverage_complete(
    controls: list[dict[str, object]],
    coverage: list[dict[str, object]],
) -> None:
    """Ensure every control has a terminal coverage entry."""
    control_ids = {str(control["control_id"]) for control in controls}
    seen_ids = {str(entry["control_id"]) for entry in coverage}
    if control_ids != seen_ids:
        missing = sorted(control_ids - seen_ids)
        extra = sorted(seen_ids - control_ids)
        msg = (
            "coverage ledger is incomplete:"
            f" missing={missing}, extra={extra}"
        )
        raise ValueError(msg)

    valid_statuses = {"pass", "fail", "unknown", "not_applicable"}
    for entry in coverage:
        status = str(entry["status"])
        if status not in valid_statuses:
            msg = f"non-terminal coverage status: {status!r}"
            raise ValueError(msg)


def _overall_status(
    config: AIActConfig,
    coverage: list[dict[str, object]],
) -> OverallStatus:
    """Compute the overall readiness outcome."""
    if not config.eu_market or config.role == "research":
        return "out_of_scope"
    if any(
        bool(entry["applies"])
        and bool(entry["blocking"])
        and str(entry["status"]) in {"fail", "unknown"}
        for entry in coverage
    ):
        return "blocked"
    if any(
        bool(entry["applies"])
        and str(entry["status"]) in {"fail", "unknown"}
        for entry in coverage
    ):
        return "ready_with_actions"
    return "ready"


def _risk_level(coverage: list[dict[str, object]]) -> str:
    """Compute a conservative risk label from control outcomes."""
    if any(
        bool(entry["applies"])
        and str(entry["control_id"]) == "ai_act.triage.high_risk_annex_iii"
        and str(entry["status"]) == "unknown"
        for entry in coverage
    ):
        return "high"
    if any(
        bool(entry["applies"])
        and bool(entry["blocking"])
        and str(entry["status"]) in {"fail", "unknown"}
        for entry in coverage
    ):
        return "high"
    if any(
        bool(entry["applies"])
        and str(entry["status"]) in {"fail", "unknown"}
        for entry in coverage
    ):
        return "medium"
    return "low"


def _likely_obligations(config: AIActConfig) -> list[str]:
    """Return a conservative list of likely obligation headings."""
    if not config.eu_market or config.role == "research":
        return []
    obligations: list[str] = []
    obligations.append("Article 4 AI literacy evidence")
    obligations.append("Article 5 prohibited-practice screening")
    if (
        config.role in {"provider", "modifier"}
        and config.interacts_with_natural_persons
        and not config.interaction_obvious_to_persons
    ):
        obligations.append("Article 50 transparency for non-obvious AI interaction")
    if config.exposes_emotion_recognition_or_biometric_categorization:
        obligations.append(
            (
                "Article 50 transparency for emotion recognition or"
                " biometric categorization"
            ),
        )
    if config.deploys_deepfake_or_synthetic_media:
        obligations.append("Article 50 deployer disclosure for synthetic media")
    if config.publishes_text_on_matters_of_public_interest:
        obligations.append(
            "Article 50 deployer disclosure for public-interest text",
        )
    if _high_risk_triggers(config):
        obligations.append("High-risk legal triage")
    if _fria_triggers(config):
        obligations.append("Article 27 FRIA legal triage")
    if config.uses_general_purpose_ai and config.role == "deployer":
        obligations.append("Downstream provider documentation retention")
    return obligations


def _build_summary(
    config: AIActConfig,
    coverage: list[dict[str, object]],
    overall_status: OverallStatus,
) -> list[str]:
    """Build the short human-readable report summary."""
    lines = [
        (
            f"Vauban classified this run as {overall_status} for"
            f" {config.company_name} / {config.system_name}."
        ),
        (
            "This report is a readiness and evidence pack, not an automated"
            " declaration of legal compliance."
        ),
    ]
    blocking = [
        str(entry["control_id"])
        for entry in coverage
        if bool(entry["applies"]) and bool(entry["blocking"])
        and str(entry["status"]) in {"fail", "unknown"}
    ]
    if blocking:
        lines.append(
            "Blocking controls: " + ", ".join(blocking),
        )
    else:
        lines.append("No blocking controls remain in the current scope.")
    return lines


def _status_counts(coverage: list[dict[str, object]]) -> dict[str, int]:
    """Count control outcomes."""
    counter = Counter(str(entry["status"]) for entry in coverage)
    return {
        "pass": counter.get("pass", 0),
        "fail": counter.get("fail", 0),
        "unknown": counter.get("unknown", 0),
        "not_applicable": counter.get("not_applicable", 0),
    }


def _collect_remediation_items(
    coverage: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Collect actionable remediation items from the coverage ledger."""
    items: list[dict[str, object]] = []
    for entry in coverage:
        action = entry.get("owner_action")
        if not isinstance(action, str):
            continue
        items.append(
            {
                "control_id": entry["control_id"],
                "status": entry["status"],
                "blocking": entry["blocking"],
                "legal_review_required": entry["legal_review_required"],
                "owner_action": action,
                "recheckable": entry["recheckable"],
                "missing_markers": entry["missing_markers"],
                "required_artifacts": entry["required_artifacts"],
                "recheck_hint": entry["recheck_hint"],
            },
        )
    return items


def _render_remediation_markdown(
    config: AIActConfig,
    overall_status: OverallStatus,
    items: list[dict[str, object]],
) -> str:
    """Render a markdown remediation plan."""
    lines = [
        "# AI Act Remediation Plan",
        "",
        f"- Company: {config.company_name}",
        f"- System: {config.system_name}",
        f"- Overall status: {overall_status}",
        "",
    ]
    if not items:
        lines.append("No remediation actions are currently required.")
        lines.append("")
        return "\n".join(lines)

    for item in items:
        missing_markers = _coerce_str_list(item["missing_markers"])
        required_artifacts = _coerce_str_list(item["required_artifacts"])
        lines.extend(
            [
                f"## {item['control_id']}",
                "",
                f"- Status: {item['status']}",
                f"- Blocking: {item['blocking']}",
                (
                    "- Legal review required:"
                    f" {item['legal_review_required']}"
                ),
                f"- Recheckable by Vauban: {item['recheckable']}",
                (
                    "- Required artifacts:"
                    f" {_format_marker_list(required_artifacts)}"
                ),
                (
                    "- Missing markers:"
                    f" {_format_marker_list(missing_markers)}"
                ),
                f"- Action: {item['owner_action']}",
                f"- Recheck: {item['recheck_hint']}",
                "",
            ],
        )
    return "\n".join(lines)


def _sources() -> list[dict[str, str]]:
    """Return the official sources pinned for the report."""
    raw_sources = _rulebook_v1().get("sources")
    if not isinstance(raw_sources, list):
        msg = "AI Act rulebook sources must be a list"
        raise TypeError(msg)
    sources: list[dict[str, str]] = []
    for entry in raw_sources:
        if not isinstance(entry, dict):
            msg = "AI Act rulebook source entries must be objects"
            raise TypeError(msg)
        entry_dict = cast("dict[str, object]", entry)
        source: dict[str, str] = {}
        for key in ("source_id", "title", "url", "publisher", "relevance"):
            value = entry_dict.get(key)
            if not isinstance(value, str):
                msg = f"AI Act rulebook source entry is missing {key!r}"
                raise TypeError(msg)
            source[key] = value
        sources.append(source)
    return sources


def _sources_by_id() -> dict[str, dict[str, str]]:
    """Return rulebook sources keyed by source_id."""
    return {source["source_id"]: source for source in _sources()}

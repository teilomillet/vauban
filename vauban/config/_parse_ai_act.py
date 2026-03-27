"""Parse the [ai_act] section of a TOML config."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from vauban.config._parse_helpers import SectionReader
from vauban.types import AIActConfig

if TYPE_CHECKING:
    from vauban.config._types import TomlDict


_AI_ACT_ANNEX_III_USE_CASES: frozenset[str] = frozenset(
    {
        "annex_iii_1_biometrics_generic",
        "annex_iii_1_remote_biometric_identification",
        "annex_iii_1_biometric_categorisation",
        "annex_iii_1_emotion_recognition",
        "annex_iii_2_critical_infrastructure",
        "annex_iii_3_education_generic",
        "annex_iii_3_education_access_assignment",
        "annex_iii_3_learning_outcome_evaluation",
        "annex_iii_3_education_level_assessment",
        "annex_iii_3_student_test_monitoring",
        "annex_iii_4_employment_generic",
        "annex_iii_4_recruitment_selection",
        "annex_iii_4_work_relationship_decisions",
        "annex_iii_5_essential_services_generic",
        "annex_iii_5_public_assistance_benefits",
        "annex_iii_5_creditworthiness_or_credit_score",
        "annex_iii_5_life_or_health_insurance",
        "annex_iii_5_emergency_dispatch",
        "annex_iii_6_law_enforcement_generic",
        "annex_iii_7_migration_asylum_border_generic",
        "annex_iii_8_justice_democracy_generic",
        "annex_iii_8_justice_support",
        "annex_iii_8_democratic_process_influence",
    },
)


def _resolve_optional_path(base_dir: Path, raw: str | None) -> Path | None:
    """Resolve an optional TOML path relative to *base_dir*."""
    if raw is None:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    return base_dir / path


def _resolve_path_list(base_dir: Path, raw: list[str]) -> list[Path]:
    """Resolve a list of TOML paths relative to *base_dir*."""
    resolved: list[Path] = []
    for value in raw:
        path = Path(value)
        resolved.append(path if path.is_absolute() else base_dir / path)
    return resolved


def _parse_ai_act(base_dir: Path, raw: TomlDict) -> AIActConfig | None:
    """Parse the optional [ai_act] section into an AIActConfig."""
    sec = raw.get("ai_act")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[ai_act] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)
    sec = cast("TomlDict", sec)

    reader = SectionReader("[ai_act]", sec)

    company_name = reader.string("company_name")
    if not company_name.strip():
        msg = "[ai_act].company_name must not be empty"
        raise ValueError(msg)

    system_name = reader.string("system_name")
    if not system_name.strip():
        msg = "[ai_act].system_name must not be empty"
        raise ValueError(msg)

    intended_purpose = reader.string("intended_purpose")
    if not intended_purpose.strip():
        msg = "[ai_act].intended_purpose must not be empty"
        raise ValueError(msg)

    report_kind_raw = reader.literal(
        "report_kind",
        ("deployer_readiness",),
        default="deployer_readiness",
    )
    report_kind: Literal["deployer_readiness"] = "deployer_readiness"
    if report_kind_raw != "deployer_readiness":
        msg = (
            "[ai_act].report_kind must be 'deployer_readiness', got"
            f" {report_kind_raw!r}"
        )
        raise ValueError(msg)

    role_raw = reader.literal(
        "role",
        ("deployer", "provider", "modifier", "research"),
        default="deployer",
    )
    role: Literal["deployer", "provider", "modifier", "research"]
    if role_raw == "deployer":
        role = "deployer"
    elif role_raw == "provider":
        role = "provider"
    elif role_raw == "modifier":
        role = "modifier"
    else:
        role = "research"
    sector = reader.string("sector", default="general")
    if not sector.strip():
        msg = "[ai_act].sector must not be empty"
        raise ValueError(msg)

    eu_market = reader.boolean("eu_market", default=True)
    uses_general_purpose_ai = reader.boolean(
        "uses_general_purpose_ai",
        default=True,
    )
    interacts_with_natural_persons = reader.boolean(
        "interacts_with_natural_persons",
        default=False,
    )
    interaction_obvious_to_persons = reader.boolean(
        "interaction_obvious_to_persons",
        default=False,
    )
    exposes_emotion_recognition_or_biometric_categorization = reader.boolean(
        "exposes_emotion_recognition_or_biometric_categorization",
        default=False,
    )
    uses_emotion_recognition = reader.boolean(
        "uses_emotion_recognition",
        default=False,
    )
    uses_biometric_categorization = reader.boolean(
        "uses_biometric_categorization",
        default=False,
    )
    emotion_recognition_medical_or_safety_exception = reader.boolean(
        "emotion_recognition_medical_or_safety_exception",
        default=False,
    )
    biometric_categorization_infers_sensitive_traits = reader.boolean(
        "biometric_categorization_infers_sensitive_traits",
        default=False,
    )
    publishes_text = reader.boolean(
        "publishes_text_on_matters_of_public_interest",
        default=False,
    )
    public_interest_text_human_review_or_editorial_control = reader.boolean(
        "public_interest_text_human_review_or_editorial_control",
        default=False,
    )
    public_interest_text_editorial_responsibility = reader.boolean(
        "public_interest_text_editorial_responsibility",
        default=False,
    )
    deploys_deepfake = reader.boolean(
        "deploys_deepfake_or_synthetic_media",
        default=False,
    )
    deepfake_creative_context = reader.boolean(
        "deepfake_creative_satirical_artistic_or_fictional_context",
        default=False,
    )
    provides_public_service = reader.boolean(
        "provides_public_service",
        default=False,
    )
    public_sector_use = reader.boolean("public_sector_use", default=False)
    annex_iii_use_cases = reader.string_list("annex_iii_use_cases", default=[])
    unknown_annex_iii_use_cases = sorted(
        set(annex_iii_use_cases) - _AI_ACT_ANNEX_III_USE_CASES,
    )
    if unknown_annex_iii_use_cases:
        msg = (
            "[ai_act].annex_iii_use_cases contains unknown values: "
            f"{', '.join(unknown_annex_iii_use_cases)}"
        )
        raise ValueError(msg)
    employment_use = reader.boolean(
        "employment_or_workers_management",
        default=False,
    )
    education_use = reader.boolean(
        "education_or_vocational_training",
        default=False,
    )
    essential_service = reader.boolean(
        "essential_private_or_public_service",
        default=False,
    )
    creditworthiness_or_credit_score_assessment = reader.boolean(
        "creditworthiness_or_credit_score_assessment",
        default=False,
    )
    life_or_health_insurance_risk_pricing = reader.boolean(
        "life_or_health_insurance_risk_pricing",
        default=False,
    )
    emergency_first_response_dispatch = reader.boolean(
        "emergency_first_response_dispatch",
        default=False,
    )
    law_enforcement_use = reader.boolean("law_enforcement_use", default=False)
    migration_use = reader.boolean(
        "migration_or_border_management_use",
        default=False,
    )
    justice_use = reader.boolean(
        "administration_of_justice_or_democracy_use",
        default=False,
    )
    biometric_use = reader.boolean(
        "biometric_or_emotion_related_use",
        default=False,
    )
    significant_decision_support = reader.boolean(
        "uses_profiling_or_similarly_significant_decision_support",
        default=False,
    )
    annex_i_product_or_safety_component = reader.boolean(
        "annex_i_product_or_safety_component",
        default=False,
    )
    annex_i_third_party_conformity_assessment = reader.boolean(
        "annex_i_third_party_conformity_assessment",
        default=False,
    )

    ai_literacy_record = _resolve_optional_path(
        base_dir,
        reader.optional_string("ai_literacy_record"),
    )
    transparency_notice = _resolve_optional_path(
        base_dir,
        reader.optional_string("transparency_notice"),
    )
    human_oversight_procedure = _resolve_optional_path(
        base_dir,
        reader.optional_string("human_oversight_procedure"),
    )
    incident_response_procedure = _resolve_optional_path(
        base_dir,
        reader.optional_string("incident_response_procedure"),
    )
    provider_documentation = _resolve_optional_path(
        base_dir,
        reader.optional_string("provider_documentation"),
    )
    technical_report_paths = _resolve_path_list(
        base_dir,
        reader.string_list("technical_report_paths", default=[]),
    )

    risk_owner = reader.optional_string("risk_owner")
    if risk_owner is not None and not risk_owner.strip():
        msg = "[ai_act].risk_owner must not be empty when set"
        raise ValueError(msg)

    compliance_contact = reader.optional_string("compliance_contact")
    if compliance_contact is not None and not compliance_contact.strip():
        msg = "[ai_act].compliance_contact must not be empty when set"
        raise ValueError(msg)

    return AIActConfig(
        company_name=company_name,
        system_name=system_name,
        intended_purpose=intended_purpose,
        report_kind=report_kind,
        role=role,
        sector=sector,
        eu_market=eu_market,
        uses_general_purpose_ai=uses_general_purpose_ai,
        interacts_with_natural_persons=interacts_with_natural_persons,
        interaction_obvious_to_persons=interaction_obvious_to_persons,
        exposes_emotion_recognition_or_biometric_categorization=(
            exposes_emotion_recognition_or_biometric_categorization
        ),
        uses_emotion_recognition=uses_emotion_recognition,
        uses_biometric_categorization=uses_biometric_categorization,
        emotion_recognition_medical_or_safety_exception=(
            emotion_recognition_medical_or_safety_exception
        ),
        biometric_categorization_infers_sensitive_traits=(
            biometric_categorization_infers_sensitive_traits
        ),
        publishes_text_on_matters_of_public_interest=publishes_text,
        public_interest_text_human_review_or_editorial_control=(
            public_interest_text_human_review_or_editorial_control
        ),
        public_interest_text_editorial_responsibility=(
            public_interest_text_editorial_responsibility
        ),
        deploys_deepfake_or_synthetic_media=deploys_deepfake,
        deepfake_creative_satirical_artistic_or_fictional_context=(
            deepfake_creative_context
        ),
        provides_public_service=provides_public_service,
        public_sector_use=public_sector_use,
        annex_iii_use_cases=annex_iii_use_cases,
        employment_or_workers_management=employment_use,
        education_or_vocational_training=education_use,
        essential_private_or_public_service=essential_service,
        creditworthiness_or_credit_score_assessment=(
            creditworthiness_or_credit_score_assessment
        ),
        life_or_health_insurance_risk_pricing=(
            life_or_health_insurance_risk_pricing
        ),
        emergency_first_response_dispatch=emergency_first_response_dispatch,
        law_enforcement_use=law_enforcement_use,
        migration_or_border_management_use=migration_use,
        administration_of_justice_or_democracy_use=justice_use,
        biometric_or_emotion_related_use=biometric_use,
        uses_profiling_or_similarly_significant_decision_support=(
            significant_decision_support
        ),
        annex_i_product_or_safety_component=annex_i_product_or_safety_component,
        annex_i_third_party_conformity_assessment=(
            annex_i_third_party_conformity_assessment
        ),
        ai_literacy_record=ai_literacy_record,
        transparency_notice=transparency_notice,
        human_oversight_procedure=human_oversight_procedure,
        incident_response_procedure=incident_response_procedure,
        provider_documentation=provider_documentation,
        technical_report_paths=technical_report_paths,
        risk_owner=risk_owner,
        compliance_contact=compliance_contact,
    )

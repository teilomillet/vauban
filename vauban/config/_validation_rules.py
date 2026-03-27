# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Validation rules for config validation."""

from __future__ import annotations

import os
from pathlib import Path

from vauban._suggestions import (
    check_unknown_keys,
    check_unknown_sections,
    check_value_constraints,
)
from vauban.config._mode_registry import active_early_modes
from vauban.config._validation_files import (
    _load_refusal_phrases,
    _validate_prompt_jsonl_file,
    _validate_prompt_source,
    _validate_surface_jsonl_file,
)
from vauban.config._validation_models import (
    ValidationCollector,
    ValidationContext,
    ValidationRuleSpec,
)
from vauban.config._validation_render import _early_mode_precedence_text
from vauban.surface import default_multilingual_surface_path, default_surface_path


def _rule_unknown_schema(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Add unknown section/key/value warnings before parse-level checks."""
    unknown_warnings = check_unknown_sections(context.raw)
    unknown_warnings.extend(check_unknown_keys(context.raw))
    unknown_warnings.extend(check_value_constraints(context.raw))
    for warning in unknown_warnings:
        collector.add("MEDIUM", warning)


def _rule_prompt_sources(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Validate data prompt sources and dataset size balance."""
    config = context.config
    harmful_count = _validate_prompt_source(
        config.harmful_path,
        "[data].harmful",
        collector,
        min_recommended=16,
        missing_fix=(
            "set [data].harmful to an existing JSONL path"
            ' or use [data].harmful = "default"'
        ),
    )
    harmless_count = _validate_prompt_source(
        config.harmless_path,
        "[data].harmless",
        collector,
        min_recommended=16,
        missing_fix=(
            "set [data].harmless to an existing JSONL path"
            ' or use [data].harmless = "default"'
        ),
    )
    if config.borderline_path is not None:
        _validate_prompt_source(
            config.borderline_path,
            "[data].borderline",
            collector,
            min_recommended=8,
            missing_fix=(
                "set [data].borderline to an existing JSONL path"
                " or disable [cut].false_refusal_ortho"
            ),
        )

    if (
        harmful_count is not None
        and harmless_count is not None
        and harmful_count > 0
        and harmless_count > 0
    ):
        ratio = (
            harmful_count / harmless_count
            if harmful_count >= harmless_count
            else harmless_count / harmful_count
        )
        if ratio > 4.0:
            collector.add(
                "LOW",
                (
                    "[data] prompt set sizes are highly imbalanced"
                    f" (harmful={harmful_count}, harmless={harmless_count})"
                ),
                fix=(
                    "use similarly sized harmful/harmless datasets"
                    " for more stable direction estimates"
                ),
            )


def _rule_eval_prompts(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Validate optional eval prompts path and sample size."""
    config = context.config
    if config.eval.prompts_path is None:
        return

    eval_count = _validate_prompt_jsonl_file(
        config.eval.prompts_path,
        "[eval].prompts",
        collector,
        min_recommended=8,
        missing_fix=(
            "set [eval].prompts to an existing JSONL path"
            " or remove [eval] if you do not want eval reports"
        ),
    )
    if eval_count is not None and eval_count < 3:
        collector.add(
            "MEDIUM",
            (
                f"[eval].prompts has only {eval_count} prompt(s);"
                " evaluation metrics may be noisy"
            ),
            fix="use at least 8-20 prompts for reliable evaluation",
        )


def _rule_refusal_phrases(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Validate optional refusal phrase file constraints."""
    config = context.config
    if config.eval.refusal_phrases_path is None:
        return
    refusal_path = config.eval.refusal_phrases_path
    if not refusal_path.exists():
        collector.add(
            "HIGH",
            f"[eval].refusal_phrases file not found: {refusal_path}",
            fix=(
                "set [eval].refusal_phrases to an existing text file,"
                " or remove it to use built-in refusal phrases"
            ),
        )
        return

    try:
        phrases = _load_refusal_phrases(refusal_path)
    except ValueError as exc:
        collector.add(
            "HIGH",
            f"[eval].refusal_phrases: {exc}",
            fix=(
                f"add one refusal phrase per line in {refusal_path},"
                " or remove [eval].refusal_phrases"
            ),
        )
        return
    if len(phrases) < 2:
        collector.add(
            "MEDIUM",
            (
                f"[eval].refusal_phrases has only {len(phrases)}"
                " phrase(s)"
            ),
            fix="add multiple refusal phrases to reduce false negatives",
        )


def _rule_surface_prompts(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Validate surface prompt schema and gate compatibility warnings."""
    config = context.config
    if config.surface is None:
        return

    surface_path_raw = config.surface.prompts_path
    if surface_path_raw == "default":
        surface_path = default_surface_path()
    elif surface_path_raw == "default_multilingual":
        surface_path = default_multilingual_surface_path()
    elif isinstance(surface_path_raw, Path):
        surface_path = surface_path_raw
    else:
        surface_path = Path(surface_path_raw)
    surface_count = _validate_surface_jsonl_file(
        surface_path,
        "[surface].prompts",
        collector,
        missing_fix=(
            "set [surface].prompts to an existing JSONL path"
            ' or use [surface].prompts = "default"'
            ' / "default_multilingual"'
        ),
    )
    if surface_count is not None and surface_count < 8:
        collector.add(
            "LOW",
            (
                f"[surface].prompts has only {surface_count} record(s);"
                " category/label aggregates may be unstable"
            ),
            fix="use a broader surface prompt set (16+ recommended)",
        )
    if (
        not config.surface.generate
        and (
            config.surface.max_worst_cell_refusal_after is not None
            or config.surface.max_worst_cell_refusal_delta is not None
        )
    ):
        collector.add(
            "MEDIUM",
            (
                "[surface] refusal-rate gates are set but"
                " [surface].generate = false; refusal labels are not"
                " computed in projection-only mode"
            ),
            fix=(
                "set [surface].generate = true for refusal-rate gates,"
                " or remove max_worst_cell_refusal_* gates"
            ),
        )


def _rule_output_dir(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Validate output directory path kind."""
    config = context.config
    if config.output_dir.exists() and not config.output_dir.is_dir():
        collector.add(
            "HIGH",
            (
                f"[output].dir points to a file, not a directory:"
                f" {config.output_dir}"
            ),
            fix="set [output].dir to a directory path",
        )


def _rule_ai_act_readiness(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Validate [ai_act] evidence paths and obvious readiness gaps."""
    ai_act = context.config.ai_act
    if ai_act is None:
        return

    evidence_paths: tuple[tuple[str, Path | None], ...] = (
        ("[ai_act].ai_literacy_record", ai_act.ai_literacy_record),
        ("[ai_act].transparency_notice", ai_act.transparency_notice),
        (
            "[ai_act].human_oversight_procedure",
            ai_act.human_oversight_procedure,
        ),
        (
            "[ai_act].incident_response_procedure",
            ai_act.incident_response_procedure,
        ),
        ("[ai_act].provider_documentation", ai_act.provider_documentation),
        (
            "[ai_act].operation_monitoring_procedure",
            ai_act.operation_monitoring_procedure,
        ),
        (
            "[ai_act].input_data_governance_procedure",
            ai_act.input_data_governance_procedure,
        ),
        ("[ai_act].log_retention_procedure", ai_act.log_retention_procedure),
        (
            "[ai_act].employee_or_worker_representative_notice",
            ai_act.employee_or_worker_representative_notice,
        ),
        ("[ai_act].affected_person_notice", ai_act.affected_person_notice),
        (
            "[ai_act].explanation_request_procedure",
            ai_act.explanation_request_procedure,
        ),
        (
            "[ai_act].eu_database_registration_record",
            ai_act.eu_database_registration_record,
        ),
    )
    for key, path in evidence_paths:
        if path is not None and not path.exists():
            collector.add(
                "HIGH",
                f"{key} file not found: {path}",
                fix=f"set {key} to an existing file path or remove it",
            )

    for index, path in enumerate(ai_act.technical_report_paths):
        if not path.exists():
            collector.add(
                "MEDIUM",
                (
                    f"[ai_act].technical_report_paths[{index}] file not found:"
                    f" {path}"
                ),
                fix=(
                    "attach an existing report artifact or remove the missing"
                    " entry"
                ),
            )

    if (
        ai_act.bundle_signature_secret_env is not None
        and os.environ.get(ai_act.bundle_signature_secret_env) is None
    ):
        collector.add(
            "LOW",
            (
                "[ai_act].bundle_signature_secret_env is set but the"
                f" environment variable {ai_act.bundle_signature_secret_env!r}"
                " is not available"
            ),
            fix=(
                "export the signing secret before running Vauban, or remove"
                " [ai_act].bundle_signature_secret_env to emit an unsigned"
                " bundle intentionally"
            ),
        )

    if (
        ai_act.eu_market
        and ai_act.role != "research"
        and ai_act.ai_literacy_record is None
    ):
        collector.add(
            "MEDIUM",
            (
                "[ai_act] targets an EU deployer/provider workflow but"
                " [ai_act].ai_literacy_record is not set"
            ),
            fix=(
                "attach an internal AI literacy record so the readiness report"
                " can verify Article 4 evidence"
            ),
        )

    if (
        ai_act.eu_market
        and (
            (
                ai_act.role in {"provider", "modifier"}
                and ai_act.interacts_with_natural_persons
                and not ai_act.interaction_obvious_to_persons
            )
            or ai_act.exposes_emotion_recognition_or_biometric_categorization
            or ai_act.deploys_deepfake_or_synthetic_media
            or ai_act.publishes_text_on_matters_of_public_interest
        )
        and ai_act.transparency_notice is None
    ):
        collector.add(
            "MEDIUM",
            (
                "[ai_act] declares an Article 50 transparency scenario but"
                " [ai_act].transparency_notice is not set"
            ),
            fix=(
                "attach a disclosure notice or exception evidence so the"
                " readiness report can verify Article 50 coverage"
            ),
        )

    if (
        ai_act.role not in {"provider", "modifier"}
        and ai_act.interacts_with_natural_persons
        and not ai_act.interaction_obvious_to_persons
    ):
        collector.add(
            "LOW",
            (
                "[ai_act] declares non-obvious human interaction while"
                f" [ai_act].role is {ai_act.role!r}"
            ),
            fix=(
                "if you supply the user-facing AI system, set [ai_act].role"
                " to \"provider\" or \"modifier\"; otherwise keep the current"
                " role and treat the Article 50 interaction notice as"
                " provider-side"
            ),
        )

    if (
        ai_act.interaction_obvious_to_persons
        and not ai_act.interacts_with_natural_persons
    ):
        collector.add(
            "LOW",
            (
                "[ai_act].interaction_obvious_to_persons is set but"
                " [ai_act].interacts_with_natural_persons is false"
            ),
            fix=(
                "either enable interacts_with_natural_persons or remove the"
                " interaction_obvious_to_persons flag"
            ),
        )

    if (
        ai_act.uses_emotion_recognition
        and not ai_act.exposes_emotion_recognition_or_biometric_categorization
    ):
        collector.add(
            "LOW",
            (
                "[ai_act].uses_emotion_recognition is true but"
                " [ai_act].exposes_emotion_recognition_or_biometric_categorization"
                " is false"
            ),
            fix=(
                "enable exposes_emotion_recognition_or_biometric_categorization"
                " so Article 50 coverage matches the declared emotion-"
                " recognition use"
            ),
        )

    if (
        ai_act.uses_biometric_categorization
        and not ai_act.exposes_emotion_recognition_or_biometric_categorization
    ):
        collector.add(
            "LOW",
            (
                "[ai_act].uses_biometric_categorization is true but"
                " [ai_act].exposes_emotion_recognition_or_biometric_categorization"
                " is false"
            ),
            fix=(
                "enable exposes_emotion_recognition_or_biometric_categorization"
                " so Article 50 coverage matches the declared biometric-"
                " categorization use"
            ),
        )

    if (
        ai_act.emotion_recognition_medical_or_safety_exception
        and not ai_act.uses_emotion_recognition
    ):
        collector.add(
            "LOW",
            (
                "[ai_act].emotion_recognition_medical_or_safety_exception is"
                " set but [ai_act].uses_emotion_recognition is false"
            ),
            fix=(
                "either enable uses_emotion_recognition or remove the"
                " exception flag"
            ),
        )

    if (
        ai_act.biometric_categorization_infers_sensitive_traits
        and not ai_act.uses_biometric_categorization
    ):
        collector.add(
            "LOW",
            (
                "[ai_act].biometric_categorization_infers_sensitive_traits is"
                " set but [ai_act].uses_biometric_categorization is false"
            ),
            fix=(
                "either enable uses_biometric_categorization or remove the"
                " sensitive-trait inference flag"
            ),
        )

    if (
        ai_act.real_time_remote_biometric_identification_exception_claimed
        and not ai_act.real_time_remote_biometric_identification_for_law_enforcement
    ):
        collector.add(
            "LOW",
            (
                "[ai_act].real_time_remote_biometric_identification_exception_claimed"
                " is set but"
                " [ai_act].real_time_remote_biometric_identification_for_law_"
                "enforcement"
                " is false"
            ),
            fix=(
                "either enable real_time_remote_biometric_identification_for_"
                "law_enforcement"
                " or remove the exception claim"
            ),
        )

    if (
        ai_act.materially_distorts_behavior_causing_significant_harm
        and not (
            ai_act.uses_subliminal_manipulative_or_deceptive_techniques
            or ai_act.exploits_age_disability_or_socioeconomic_vulnerabilities
        )
    ):
        collector.add(
            "LOW",
            (
                "[ai_act].materially_distorts_behavior_causing_significant_harm"
                " is set without a corresponding manipulation or vulnerability"
                " flag"
            ),
            fix=(
                "either enable the matching prohibited-practice fact or remove"
                " the harm flag"
            ),
        )

    if (
        ai_act.deepfake_creative_satirical_artistic_or_fictional_context
        and not ai_act.deploys_deepfake_or_synthetic_media
    ):
        collector.add(
            "LOW",
            (
                "[ai_act].deepfake_creative_satirical_artistic_or_fictional_context"
                " is set but [ai_act].deploys_deepfake_or_synthetic_media is"
                " false"
            ),
            fix=(
                "either enable deploys_deepfake_or_synthetic_media or remove"
                " the creative/satirical context flag"
            ),
        )

    if (
        ai_act.public_interest_text_editorial_responsibility
        and not ai_act.public_interest_text_human_review_or_editorial_control
    ):
        collector.add(
            "LOW",
            (
                "[ai_act].public_interest_text_editorial_responsibility is"
                " set but"
                " [ai_act].public_interest_text_human_review_or_editorial_control"
                " is false"
            ),
            fix=(
                "either enable public_interest_text_human_review_or_editorial_control"
                " or remove the editorial responsibility flag"
            ),
        )

    if (
        ai_act.public_interest_text_human_review_or_editorial_control
        and not ai_act.publishes_text_on_matters_of_public_interest
    ):
        collector.add(
            "LOW",
            (
                "[ai_act].public_interest_text_human_review_or_editorial_control"
                " is set but"
                " [ai_act].publishes_text_on_matters_of_public_interest is"
                " false"
            ),
            fix=(
                "either enable publishes_text_on_matters_of_public_interest or"
                " remove the public-interest exception flag"
            ),
        )

    article6_3_claimed = (
        ai_act.annex_iii_narrow_procedural_task
        or ai_act.annex_iii_improves_completed_human_activity
        or ai_act.annex_iii_detects_decision_pattern_deviations
        or ai_act.annex_iii_preparatory_task
    )
    if article6_3_claimed and not ai_act.annex_iii_use_cases:
        collector.add(
            "LOW",
            (
                "[ai_act] declares Article 6(3) carve-out facts but"
                " [ai_act].annex_iii_use_cases is empty"
            ),
            fix=(
                "add specific Annex III use-case identifiers so the carve-out"
                " review is grounded in a declared high-risk use case"
            ),
        )

    if (
        article6_3_claimed
        and not ai_act.annex_iii_does_not_materially_influence_decision_outcome
    ):
        collector.add(
            "LOW",
            (
                "[ai_act] declares Article 6(3) carve-out facts but does not"
                " assert that the AI system avoids materially influencing"
                " decision outcomes"
            ),
            fix=(
                "set annex_iii_does_not_materially_influence_decision_outcome"
                " only if that statement is supportable, or remove the"
                " Article 6(3) carve-out flags"
            ),
        )

    if (
        ai_act.decision_with_legal_or_similarly_significant_effects
        and not ai_act.makes_or_assists_decisions_about_natural_persons
    ):
        collector.add(
            "LOW",
            (
                "[ai_act].decision_with_legal_or_similarly_significant_effects"
                " is set but"
                " [ai_act].makes_or_assists_decisions_about_natural_persons"
                " is false"
            ),
            fix=(
                "either enable makes_or_assists_decisions_about_natural_persons"
                " or remove the legal-effects flag"
            ),
        )

    if (
        ai_act.annex_i_third_party_conformity_assessment
        and not ai_act.annex_i_product_or_safety_component
    ):
        collector.add(
            "LOW",
            (
                "[ai_act].annex_i_third_party_conformity_assessment is set"
                " but [ai_act].annex_i_product_or_safety_component is false"
            ),
            fix=(
                "either enable annex_i_product_or_safety_component or remove"
                " the Annex I conformity flag"
            ),
        )

    if (
        not ai_act.annex_iii_use_cases
        and (
            ai_act.biometric_or_emotion_related_use
            or ai_act.education_or_vocational_training
            or ai_act.employment_or_workers_management
            or ai_act.essential_private_or_public_service
            or ai_act.law_enforcement_use
            or ai_act.migration_or_border_management_use
            or ai_act.administration_of_justice_or_democracy_use
        )
    ):
        collector.add(
            "LOW",
            (
                "[ai_act] uses legacy high-risk area flags but"
                " [ai_act].annex_iii_use_cases is empty"
            ),
            fix=(
                "add specific Annex III use-case identifiers to"
                " [ai_act].annex_iii_use_cases for sharper classification"
                " coverage"
            ),
        )

    plausible_article6_3_carve_out = (
        article6_3_claimed
        and ai_act.annex_iii_does_not_materially_influence_decision_outcome
        and not ai_act.uses_profiling_or_similarly_significant_decision_support
        and not ai_act.annex_i_product_or_safety_component
        and not ai_act.annex_i_third_party_conformity_assessment
    )

    declares_high_risk_deployer = (
        ai_act.eu_market
        and ai_act.role == "deployer"
        and (
            bool(ai_act.annex_iii_use_cases)
            or ai_act.annex_i_product_or_safety_component
            or ai_act.annex_i_third_party_conformity_assessment
            or ai_act.biometric_or_emotion_related_use
            or ai_act.education_or_vocational_training
            or ai_act.employment_or_workers_management
            or ai_act.essential_private_or_public_service
            or ai_act.law_enforcement_use
            or ai_act.migration_or_border_management_use
            or ai_act.administration_of_justice_or_democracy_use
        )
        and not plausible_article6_3_carve_out
    )

    if declares_high_risk_deployer and ai_act.operation_monitoring_procedure is None:
        collector.add(
            "MEDIUM",
            (
                "[ai_act] declares a high-risk deployer scenario but"
                " [ai_act].operation_monitoring_procedure is not set"
            ),
            fix=(
                "attach an operation monitoring procedure covering provider"
                " instructions, monitoring, and risk response"
            ),
        )

    if declares_high_risk_deployer and ai_act.human_oversight_procedure is None:
        collector.add(
            "MEDIUM",
            (
                "[ai_act] declares a high-risk deployer scenario but"
                " [ai_act].human_oversight_procedure is not set"
            ),
            fix=(
                "attach a human oversight procedure covering review,"
                " override, and escalation for the high-risk deployment"
            ),
        )

    if declares_high_risk_deployer and ai_act.risk_owner is None:
        collector.add(
            "MEDIUM",
            (
                "[ai_act] declares a high-risk deployer scenario but"
                " [ai_act].risk_owner is not set"
            ),
            fix=(
                "set [ai_act].risk_owner to the person responsible for"
                " operational human oversight"
            ),
        )

    if declares_high_risk_deployer and ai_act.log_retention_procedure is None:
        collector.add(
            "MEDIUM",
            (
                "[ai_act] declares a high-risk deployer scenario but"
                " [ai_act].log_retention_procedure is not set"
            ),
            fix=(
                "attach a log retention procedure covering logs under the"
                " deployer's control and the retention baseline"
            ),
        )

    if (
        declares_high_risk_deployer
        and ai_act.provides_input_data_for_high_risk_system
        and ai_act.input_data_governance_procedure is None
    ):
        collector.add(
            "MEDIUM",
            (
                "[ai_act] declares deployer-provided input data for a"
                " high-risk system but"
                " [ai_act].input_data_governance_procedure is not set"
            ),
            fix=(
                "attach an input-data governance procedure covering"
                " relevance, representativeness, and validation"
            ),
        )

    if (
        declares_high_risk_deployer
        and ai_act.workplace_deployment
        and ai_act.employee_or_worker_representative_notice is None
    ):
        collector.add(
            "MEDIUM",
            (
                "[ai_act] declares workplace deployment of a high-risk system"
                " but [ai_act].employee_or_worker_representative_notice is not"
                " set"
            ),
            fix=(
                "attach notice evidence for affected employees and workers'"
                " representatives"
            ),
        )

    if (
        declares_high_risk_deployer
        and ai_act.makes_or_assists_decisions_about_natural_persons
        and ai_act.affected_person_notice is None
    ):
        collector.add(
            "MEDIUM",
            (
                "[ai_act] declares high-risk decisions about natural persons"
                " but [ai_act].affected_person_notice is not set"
            ),
            fix=(
                "attach an affected-person notice covering the intended"
                " purpose and decision-support context"
            ),
        )

    if (
        declares_high_risk_deployer
        and ai_act.decision_with_legal_or_similarly_significant_effects
        and ai_act.explanation_request_procedure is None
    ):
        collector.add(
            "MEDIUM",
            (
                "[ai_act] declares legal or similarly significant effects from"
                " a high-risk decision but"
                " [ai_act].explanation_request_procedure is not set"
            ),
            fix=(
                "attach an explanation-request procedure for affected persons"
            ),
        )

    if (
        declares_high_risk_deployer
        and ai_act.public_sector_use
        and ai_act.eu_database_registration_record is None
        and set(ai_act.annex_iii_use_cases) != {"annex_iii_2_critical_infrastructure"}
    ):
        collector.add(
            "MEDIUM",
            (
                "[ai_act] declares a public-sector high-risk deployment but"
                " [ai_act].eu_database_registration_record is not set"
            ),
            fix=(
                "attach public or non-public EU database registration evidence"
                " where the deployment is registrable"
            ),
        )


def _rule_early_mode_conflicts(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Warn when multiple early-return modes are active."""
    early_modes = active_early_modes(context.config)
    if len(early_modes) <= 1:
        return
    collector.add(
        "HIGH",
        f"Multiple early-return modes active: {', '.join(early_modes)}"
        " — only the first will run (precedence: "
        f"{_early_mode_precedence_text()})",
        fix=(
            "keep one early-return mode per config,"
            " and split other modes into separate TOML files"
        ),
    )


def _rule_depth_extract_direction(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Warn when depth direction extraction has too few prompts."""
    config = context.config
    if config.depth is None or not config.depth.extract_direction:
        return
    effective = (
        config.depth.direction_prompts
        if config.depth.direction_prompts is not None
        else config.depth.prompts
    )
    if len(effective) < 2:
        source = (
            "direction_prompts"
            if config.depth.direction_prompts is not None
            else "prompts"
        )
        collector.add(
            "HIGH",
            f"[depth].extract_direction = true but {source}"
            f" has only {len(effective)} entry — need >= 2",
            fix=(
                "add at least 2 prompts to the selected source,"
                " or set [depth].extract_direction = false"
            ),
        )


def _rule_eval_without_prompts(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Warn if [eval] is present but has no prompt source in default flow."""
    config = context.config
    early_modes = active_early_modes(config)
    eval_raw = context.raw.get("eval")
    if (
        isinstance(eval_raw, dict)
        and "prompts" not in eval_raw
        and config.eval.prompts_path is None
        and not early_modes
    ):
        collector.add(
            "LOW",
            (
                "[eval] section is present but [eval].prompts is not set;"
                " eval_report.json will not be produced in default pipeline"
            ),
            fix=(
                'set [eval].prompts = "eval.jsonl"'
                " or remove the [eval] section"
            ),
        )


def _rule_skipped_sections(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Warn when early-return modes cause config sections to be ignored."""
    config = context.config
    early_modes = active_early_modes(config)
    if not early_modes:
        return
    active_mode = early_modes[0]
    skipped: list[str] = []

    if config.depth is not None and config.detect is not None:
        skipped.append("[detect]")
    if config.surface is not None:
        skipped.append("[surface]")
    if config.eval.prompts_path is not None:
        skipped.append("[eval]")

    if skipped:
        collector.add(
            "MEDIUM",
            f"{active_mode} early-return will skip:"
            f" {', '.join(skipped)}"
            f" — these sections have no effect in"
            f" {active_mode.strip('[]')} mode",
            fix=(
                "remove skipped sections from this config,"
                " or run them in a separate non-early-return config"
            ),
        )


VALIDATION_RULE_SPECS: tuple[ValidationRuleSpec, ...] = (
    ValidationRuleSpec("unknown_schema", 10, _rule_unknown_schema),
    ValidationRuleSpec("prompt_sources", 20, _rule_prompt_sources),
    ValidationRuleSpec("eval_prompts", 30, _rule_eval_prompts),
    ValidationRuleSpec("refusal_phrases", 40, _rule_refusal_phrases),
    ValidationRuleSpec("surface_prompts", 50, _rule_surface_prompts),
    ValidationRuleSpec("output_dir", 60, _rule_output_dir),
    ValidationRuleSpec("ai_act_readiness", 65, _rule_ai_act_readiness),
    ValidationRuleSpec("early_mode_conflicts", 70, _rule_early_mode_conflicts),
    ValidationRuleSpec("depth_extract_direction", 80, _rule_depth_extract_direction),
    ValidationRuleSpec("eval_without_prompts", 90, _rule_eval_without_prompts),
    ValidationRuleSpec("skipped_sections", 100, _rule_skipped_sections),
)

# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Runtime-generated manual for vauban's TOML-first interface."""

import ast
import inspect
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from vauban.config._mode_registry import (
    EARLY_MODE_SPECS as _EARLY_MODE_SPECS,
)
from vauban.config._mode_registry import (
    EARLY_RETURN_PRECEDENCE as _EARLY_RETURN_PRECEDENCE,
)


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Manual metadata for one TOML key."""

    key: str
    description: str
    constraints: str | None = None
    attr: str | None = None
    required: bool | None = None
    default_override: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SectionSpec:
    """Manual metadata for one TOML section."""

    name: str
    description: str
    required: bool = False
    early_return: bool = False
    table: bool = True
    config_class: str | None = None
    fields: tuple[FieldSpec, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AutoField:
    """Auto-discovered field metadata from a config dataclass."""

    type_name: str
    default_repr: str | None
    required: bool


@dataclass(frozen=True, slots=True)
class ManualField:
    """Rendered field documentation."""

    key: str
    type_name: str
    required: bool
    default_repr: str | None
    description: str
    constraints: str | None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ManualSection:
    """Rendered section documentation."""

    name: str
    description: str
    required: bool
    early_return: bool
    table: bool
    fields: tuple[ManualField, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PipelineModeDoc:
    """Manual metadata for one pipeline mode."""

    mode: str
    trigger: str
    output: str
    early_return: bool


EARLY_RETURN_PRECEDENCE: tuple[str, ...] = tuple(
    section.strip("[]") for section in _EARLY_RETURN_PRECEDENCE
)

_PIPELINE_MODES: tuple[PipelineModeDoc, ...] = (
    PipelineModeDoc(
        mode="default",
        trigger="No early-return section is present.",
        output="Modified model directory in [output].dir.",
        early_return=False,
    ),
    *tuple(
        PipelineModeDoc(
            mode=spec.mode,
            trigger=spec.manual_trigger,
            output=spec.manual_output,
            early_return=True,
        )
        for spec in _EARLY_MODE_SPECS
    ),
)

_FORMAT_NOTES: tuple[str, ...] = (
    "Prompt JSONL ([data] + [eval]): one object per line with key 'prompt'.",
    (
        "Surface JSONL ([surface].prompts): required keys 'label' and"
        " 'category' plus either 'prompt' or 'messages'; optional keys"
        " 'style', 'language', 'turn_depth', and 'framing'."
    ),
    "Refusal phrases file: plain text, one phrase per line ('#' comments allowed).",
    "All relative paths resolve from the directory of the loaded TOML file.",
)

_QUICKSTART_NOTES: tuple[str, ...] = (
    "1. Create run.toml with the minimal config shown below.",
    "2. Validate first: vauban --validate run.toml",
    "3. Run: vauban run.toml",
    "4. Inspect outputs in [output].dir (default: output/).",
    (
        "5. For a checked-in benchmark baseline, start from"
        " examples/benchmarks/share_doc.toml."
    ),
    "6. If vauban is installed through uv project deps, use: uv run vauban ...",
)

_MINIMAL_CONFIG_EXAMPLE: tuple[str, ...] = (
    "[model]",
    'path = "mlx-community/Llama-3.2-3B-Instruct-4bit"',
    "",
    "[data]",
    'harmful = "default"',
    'harmless = "default"',
)

_VALIDATE_NOTES: tuple[str, ...] = (
    "Use validate before running heavy experiments.",
    "Warnings are tagged [LOW], [MEDIUM], or [HIGH].",
    "Most warnings include a 'fix:' hint for direct remediation.",
    (
        "Validate checks JSONL schema, missing files, mode conflicts,"
        " skipped sections, and key-level typos within each section."
    ),
)

_PLAYBOOK_NOTES: tuple[str, ...] = (
    "1. Scaffold a config: vauban init --mode default --output run.toml",
    (
        "   For a built-in agent benchmark:"
        " vauban init --scenario share_doc --output share_doc.toml"
    ),
    (
        "   Or use the checked-in canonical benchmark:"
        " examples/benchmarks/share_doc.toml"
    ),
    "2. Validate and iterate until warnings are understood/fixed.",
    "3. Run one experiment per TOML file for reproducibility.",
    "4. Compare two runs with: vauban diff out_a out_b",
    "5. Visualize lineage with: vauban tree experiments/",
    "6. Keep an experiment log (config path + output dir + key metrics).",
)

_QUICK_NOTES: tuple[str, ...] = (
    "Use vauban.quick in Python REPL/Jupyter for rapid experiments.",
    "This is complementary to TOML runs; use TOML for reproducible reports.",
)

_EXAMPLE_NOTES: tuple[str, ...] = (
    "Scaffold a starter config:",
    "  vauban init --mode default --output run.toml",
    "Scaffold a built-in indirect prompt-injection benchmark:",
    "  vauban init --scenario share_doc --output share_doc.toml",
    "Use the checked-in canonical benchmark pack:",
    "  vauban --validate examples/benchmarks/share_doc.toml",
    "  vauban examples/benchmarks/share_doc.toml",
    "Validate before expensive runs:",
    "  vauban --validate run.toml",
    "Run default pipeline:",
    "  vauban run.toml",
    "Compare two experiment outputs:",
    "  vauban diff runs/baseline runs/experiment_a",
    "CI gate for regression checks:",
    "  vauban diff --threshold 0.05 runs/baseline runs/candidate",
    "Render the experiment tree:",
    "  vauban tree experiments/ --format mermaid",
    "Open manual for one topic:",
    "  vauban man softprompt",
)

_PRINT_NOTES: tuple[str, ...] = (
    "Full manual to text file:",
    "  vauban man > VAUBAN_MANUAL.txt",
    "Focused topic to text file:",
    "  vauban man cut > VAUBAN_CUT.txt",
    "Read in pager (Unix):",
    "  vauban man | less",
    "Print from Unix shell:",
    "  lpr VAUBAN_MANUAL.txt",
)

_QUICK_EXAMPLE: tuple[str, ...] = (
    "from vauban import quick",
    "",
    'model, tokenizer = quick.load("mlx-community/Llama-3.2-3B-Instruct-4bit")',
    "direction = quick.measure_direction(model, tokenizer)",
    "probe = quick.probe_prompt(model, tokenizer, \"Explain lockpicking\", direction)",
    (
        "steered = quick.steer_prompt("
        "model, tokenizer, \"Explain lockpicking\", direction, alpha=1.0)"
    ),
    "surface = quick.scan(model, tokenizer, direction)",
    'print(quick.compare("output_a", "output_b"))',
)

_SECTION_SPECS: tuple[SectionSpec, ...] = (
    SectionSpec(
        name="model",
        description="Model loading configuration.",
        required=True,
        fields=(
            FieldSpec(
                key="path",
                description="Model identifier or local model directory.",
                constraints="required string; must be loadable by mlx_lm.load().",
                required=True,
            ),
        ),
    ),
    SectionSpec(
        name="data",
        description="Prompt sources for harmful/harmless measurement sets.",
        required=True,
        fields=(
            FieldSpec(
                key="harmful",
                description="Source for harmful prompts.",
                constraints=(
                    'required; one of: "default", local path string,'
                    ' "hf:<repo_id>", [data.harmful] HF table,'
                    ' or benchmark name: "harmbench", "advbench",'
                    ' "jailbreakbench", "strongreject".'
                    ' Append "_infix" for infix-wrapped variant'
                    ' (e.g. "harmbench_infix").'
                ),
                required=True,
            ),
            FieldSpec(
                key="harmless",
                description="Source for harmless prompts.",
                constraints=(
                    'required; one of: "default", local path string,'
                    ' "hf:<repo_id>", or [data.harmless] HF table.'
                ),
                required=True,
            ),
            FieldSpec(
                key="borderline",
                description="Optional borderline prompts for false-refusal handling.",
                constraints=(
                    "string or HF dataset reference; required when"
                    " [cut].false_refusal_ortho = true."
                ),
            ),
        ),
        notes=(
            "HF table form accepts keys: hf, split, column, config, limit.",
            (
                "Benchmark sentinels auto-download and cache standard"
                " datasets: harmbench (632), advbench (500),"
                " jailbreakbench (100), strongreject (350+)."
            ),
        ),
    ),
    SectionSpec(
        name="ai_act",
        description=(
            "Standalone AI Act deployer-readiness reporting with"
            " coverage-complete controls, evidence manifests, and"
            " remediation output."
        ),
        early_return=True,
        config_class="AIActConfig",
        fields=(
            FieldSpec(
                key="company_name",
                description="Name of the deploying organisation.",
                constraints="required string; must not be empty.",
                required=True,
            ),
            FieldSpec(
                key="system_name",
                description="Name of the AI-enabled service or system.",
                constraints="required string; must not be empty.",
                required=True,
            ),
            FieldSpec(
                key="intended_purpose",
                description="Short statement of the system's intended purpose.",
                constraints="required string; must not be empty.",
                required=True,
            ),
            FieldSpec(
                key="role",
                description=(
                    "Operator role asserted for the report, with respect to"
                    " the AI system being assessed rather than the upstream"
                    " base model."
                ),
                constraints=(
                    'one of: "deployer", "provider", "modifier",'
                    ' "research".'
                ),
            ),
            FieldSpec(
                key="report_kind",
                description="Readiness report flavor.",
                constraints='currently only: "deployer_readiness".',
            ),
            FieldSpec(
                key="sector",
                description="Sector label used in the report context.",
                constraints='string; default: "general".',
            ),
            FieldSpec(
                key="eu_market",
                description=(
                    "Whether the system is placed on, used in, or targets"
                    " the EU market."
                ),
                constraints="boolean; default true.",
            ),
            FieldSpec(
                key="uses_general_purpose_ai",
                description="Whether the service builds on a GPAI model or service.",
                constraints="boolean; default true.",
            ),
            FieldSpec(
                key="interacts_with_natural_persons",
                description=(
                    "Whether the system directly interacts with natural persons"
                    " in normal operation."
                ),
                constraints="boolean; used for Article 50 transparency readiness.",
            ),
            FieldSpec(
                key="interaction_obvious_to_persons",
                description=(
                    "Whether it is already obvious to persons that they are"
                    " interacting with AI rather than a human."
                ),
                constraints=(
                    "boolean; can narrow the human-interaction disclosure"
                    " scenario."
                ),
            ),
            FieldSpec(
                key="exposes_emotion_recognition_or_biometric_categorization",
                description=(
                    "Whether the system exposes people to emotion recognition"
                    " or biometric categorization."
                ),
                constraints="boolean; drives Article 50 transparency checks.",
            ),
            FieldSpec(
                key="uses_emotion_recognition",
                description=(
                    "Whether the declared use actually performs emotion"
                    " recognition."
                ),
                constraints="boolean; used for Article 5 screening.",
            ),
            FieldSpec(
                key="uses_biometric_categorization",
                description=(
                    "Whether the declared use actually performs biometric"
                    " categorization."
                ),
                constraints="boolean; used for Article 5 screening.",
            ),
            FieldSpec(
                key="emotion_recognition_medical_or_safety_exception",
                description=(
                    "Whether any workplace/education emotion-recognition use"
                    " is claimed to fall under the medical or safety"
                    " exception."
                ),
                constraints="boolean; used for Article 5 screening.",
            ),
            FieldSpec(
                key="biometric_categorization_infers_sensitive_traits",
                description=(
                    "Whether biometric categorization is used to infer"
                    " sensitive traits such as race or political opinions."
                ),
                constraints="boolean; used for Article 5 screening.",
            ),
            FieldSpec(
                key="uses_subliminal_manipulative_or_deceptive_techniques",
                description=(
                    "Whether the use relies on subliminal, manipulative, or"
                    " deceptive techniques."
                ),
                constraints="boolean; used for Article 5 screening.",
            ),
            FieldSpec(
                key="materially_distorts_behavior_causing_significant_harm",
                description=(
                    "Whether the declared practice materially distorts"
                    " behaviour and risks significant harm."
                ),
                constraints="boolean; used for Article 5 screening.",
            ),
            FieldSpec(
                key="exploits_age_disability_or_socioeconomic_vulnerabilities",
                description=(
                    "Whether the use exploits vulnerabilities linked to age,"
                    " disability, or socioeconomic circumstances."
                ),
                constraints="boolean; used for Article 5 screening.",
            ),
            FieldSpec(
                key="social_scoring_leading_to_detrimental_treatment",
                description=(
                    "Whether the use performs social scoring leading to"
                    " detrimental or unfavourable treatment."
                ),
                constraints="boolean; used for Article 5 screening.",
            ),
            FieldSpec(
                key="individual_predictive_policing_based_solely_on_profiling",
                description=(
                    "Whether the use performs individual predictive policing"
                    " based solely on profiling."
                ),
                constraints="boolean; used for Article 5 screening.",
            ),
            FieldSpec(
                key="untargeted_scraping_of_face_images",
                description=(
                    "Whether the use involves untargeted scraping of facial"
                    " images from the internet or CCTV."
                ),
                constraints="boolean; used for Article 5 screening.",
            ),
            FieldSpec(
                key="real_time_remote_biometric_identification_for_law_enforcement",
                description=(
                    "Whether the use performs real-time remote biometric"
                    " identification in publicly accessible spaces for law"
                    " enforcement."
                ),
                constraints="boolean; used for Article 5 screening.",
            ),
            FieldSpec(
                key="real_time_remote_biometric_identification_exception_claimed",
                description=(
                    "Whether a narrow law-enforcement exception is claimed for"
                    " real-time remote biometric identification."
                ),
                constraints="boolean; used for Article 5 screening.",
            ),
            FieldSpec(
                key="publishes_text_on_matters_of_public_interest",
                description=(
                    "Whether the deployer publishes AI-generated text on"
                    " matters of public interest."
                ),
                constraints="boolean; drives Article 50 deployer disclosure checks.",
            ),
            FieldSpec(
                key="public_interest_text_human_review_or_editorial_control",
                description=(
                    "Whether the deployer claims the public-interest text is"
                    " subject to human review or editorial control."
                ),
                constraints=(
                    "boolean; used for the Article 50 public-interest text"
                    " exception path."
                ),
            ),
            FieldSpec(
                key="public_interest_text_editorial_responsibility",
                description=(
                    "Whether a natural or legal person is claimed to hold"
                    " editorial responsibility for the public-interest text."
                ),
                constraints=(
                    "boolean; used for the Article 50 public-interest text"
                    " exception path."
                ),
            ),
            FieldSpec(
                key="deploys_deepfake_or_synthetic_media",
                description=(
                    "Whether the deployer exposes people to deepfake or"
                    " synthetic media."
                ),
                constraints="boolean; drives Article 50 deployer disclosure checks.",
            ),
            FieldSpec(
                key="deepfake_creative_satirical_artistic_or_fictional_context",
                description=(
                    "Whether synthetic media is part of an evidently creative,"
                    " satirical, artistic, fictional, or analogous context."
                ),
                constraints=(
                    "boolean; narrows the deepfake disclosure path but does"
                    " not remove the need for some disclosure."
                ),
            ),
            FieldSpec(
                key="provides_public_service",
                description="Whether the service is provided as a public service.",
                constraints="boolean; used for conservative high-risk / FRIA triage.",
            ),
            FieldSpec(
                key="public_sector_use",
                description="Whether a public-sector deployment context is declared.",
                constraints="boolean; used for conservative high-risk / FRIA triage.",
            ),
            FieldSpec(
                key="annex_iii_use_cases",
                description=(
                    "Optional explicit Annex III use-case identifiers for"
                    " sharper high-risk classification coverage."
                ),
                constraints=(
                    "list[string]; use the built-in Annex III IDs such as"
                    " annex_iii_4_recruitment_selection or"
                    " annex_iii_5_creditworthiness_or_credit_score."
                ),
            ),
            FieldSpec(
                key="employment_or_workers_management",
                description="Whether the use touches employment or workers management.",
                constraints="boolean; used for conservative high-risk / FRIA triage.",
            ),
            FieldSpec(
                key="education_or_vocational_training",
                description="Whether the use touches education or vocational training.",
                constraints="boolean; used for conservative high-risk / FRIA triage.",
            ),
            FieldSpec(
                key="essential_private_or_public_service",
                description=(
                    "Whether the use touches access to essential private or"
                    " public services."
                ),
                constraints="boolean; used for conservative high-risk / FRIA triage.",
            ),
            FieldSpec(
                key="creditworthiness_or_credit_score_assessment",
                description=(
                    "Whether the use evaluates natural-person creditworthiness"
                    " or establishes a credit score."
                ),
                constraints="boolean; used for high-risk and FRIA triage.",
            ),
            FieldSpec(
                key="life_or_health_insurance_risk_pricing",
                description=(
                    "Whether the use performs life or health insurance risk"
                    " assessment and pricing for natural persons."
                ),
                constraints="boolean; used for high-risk and FRIA triage.",
            ),
            FieldSpec(
                key="emergency_first_response_dispatch",
                description=(
                    "Whether the use evaluates emergency calls or dispatches"
                    " emergency first-response services."
                ),
                constraints="boolean; used for conservative high-risk triage.",
            ),
            FieldSpec(
                key="law_enforcement_use",
                description="Whether the use touches law-enforcement workflows.",
                constraints="boolean; used for conservative high-risk / FRIA triage.",
            ),
            FieldSpec(
                key="migration_or_border_management_use",
                description=(
                    "Whether the use touches migration or border-management"
                    " workflows."
                ),
                constraints="boolean; used for conservative high-risk / FRIA triage.",
            ),
            FieldSpec(
                key="administration_of_justice_or_democracy_use",
                description=(
                    "Whether the use touches justice or democratic-process"
                    " workflows."
                ),
                constraints="boolean; used for conservative high-risk / FRIA triage.",
            ),
            FieldSpec(
                key="biometric_or_emotion_related_use",
                description=(
                    "Whether the use touches biometric or emotion-related"
                    " workflows."
                ),
                constraints="boolean; used for conservative high-risk / FRIA triage.",
            ),
            FieldSpec(
                key="uses_profiling_or_similarly_significant_decision_support",
                description=(
                    "Whether the system supports similarly significant"
                    " profiling or decisions."
                ),
                constraints="boolean; used for conservative high-risk / FRIA triage.",
            ),
            FieldSpec(
                key="annex_iii_narrow_procedural_task",
                description=(
                    "Whether the Annex III use is claimed to perform only a"
                    " narrow procedural task."
                ),
                constraints="boolean; used for Article 6(3) carve-out triage.",
            ),
            FieldSpec(
                key="annex_iii_improves_completed_human_activity",
                description=(
                    "Whether the Annex III use is claimed only to improve a"
                    " previously completed human activity."
                ),
                constraints="boolean; used for Article 6(3) carve-out triage.",
            ),
            FieldSpec(
                key="annex_iii_detects_decision_pattern_deviations",
                description=(
                    "Whether the Annex III use is claimed only to detect"
                    " deviations from prior decision-making patterns."
                ),
                constraints="boolean; used for Article 6(3) carve-out triage.",
            ),
            FieldSpec(
                key="annex_iii_preparatory_task",
                description=(
                    "Whether the Annex III use is claimed to be merely"
                    " preparatory to a later assessment."
                ),
                constraints="boolean; used for Article 6(3) carve-out triage.",
            ),
            FieldSpec(
                key="annex_iii_does_not_materially_influence_decision_outcome",
                description=(
                    "Whether the Annex III use is claimed not to materially"
                    " influence decision outcomes."
                ),
                constraints="boolean; used for Article 6(3) carve-out triage.",
            ),
            FieldSpec(
                key="workplace_deployment",
                description=(
                    "Whether the high-risk deployment is used in the"
                    " workplace."
                ),
                constraints="boolean; used for Article 26 worker notice checks.",
            ),
            FieldSpec(
                key="provides_input_data_for_high_risk_system",
                description=(
                    "Whether the deployer provides input data to a declared"
                    " high-risk AI system."
                ),
                constraints="boolean; used for Article 26 input-data checks.",
            ),
            FieldSpec(
                key="makes_or_assists_decisions_about_natural_persons",
                description=(
                    "Whether the high-risk system makes or assists decisions"
                    " about natural persons."
                ),
                constraints="boolean; used for Article 26 notice checks.",
            ),
            FieldSpec(
                key="decision_with_legal_or_similarly_significant_effects",
                description=(
                    "Whether those high-risk decisions can produce legal or"
                    " similarly significant effects."
                ),
                constraints="boolean; used for Article 86 explanation readiness.",
            ),
            FieldSpec(
                key="annex_i_product_or_safety_component",
                description=(
                    "Whether the AI system is or is embedded as a safety"
                    " component in an Annex I regulated product."
                ),
                constraints="boolean; used for Annex I product-route triage.",
            ),
            FieldSpec(
                key="annex_i_third_party_conformity_assessment",
                description=(
                    "Whether the Annex I product route is subject to a"
                    " third-party conformity assessment."
                ),
                constraints="boolean; used for Annex I product-route triage.",
            ),
            FieldSpec(
                key="ai_literacy_record",
                description="Path to evidence for Article 4 AI literacy measures.",
                constraints="string path or null.",
            ),
            FieldSpec(
                key="transparency_notice",
                description=(
                    "Path to customer-facing disclosure or transparency notice."
                ),
                constraints="string path or null.",
            ),
            FieldSpec(
                key="human_oversight_procedure",
                description="Path to the human oversight procedure used in operations.",
                constraints="string path or null.",
            ),
            FieldSpec(
                key="incident_response_procedure",
                description="Path to the incident response or escalation procedure.",
                constraints="string path or null.",
            ),
            FieldSpec(
                key="provider_documentation",
                description="Path to retained third-party provider documentation.",
                constraints="string path or null.",
            ),
            FieldSpec(
                key="operation_monitoring_procedure",
                description=(
                    "Path to a deployer procedure covering provider"
                    " instructions and operational monitoring."
                ),
                constraints="string path or null.",
            ),
            FieldSpec(
                key="input_data_governance_procedure",
                description=(
                    "Path to the deployer's high-risk input-data governance"
                    " procedure."
                ),
                constraints="string path or null.",
            ),
            FieldSpec(
                key="log_retention_procedure",
                description=(
                    "Path to the high-risk log retention and access-control"
                    " procedure."
                ),
                constraints="string path or null.",
            ),
            FieldSpec(
                key="employee_or_worker_representative_notice",
                description=(
                    "Path to notice evidence for employees and workers'"
                    " representatives."
                ),
                constraints="string path or null.",
            ),
            FieldSpec(
                key="affected_person_notice",
                description=(
                    "Path to notice evidence for natural persons affected by"
                    " high-risk decision support."
                ),
                constraints="string path or null.",
            ),
            FieldSpec(
                key="explanation_request_procedure",
                description=(
                    "Path to the procedure for handling explanation requests"
                    " where legal or similarly significant effects apply."
                ),
                constraints="string path or null.",
            ),
            FieldSpec(
                key="eu_database_registration_record",
                description=(
                    "Path to evidence of the relevant EU database"
                    " registration path for a public-sector high-risk"
                    " deployment."
                ),
                constraints="string path or null.",
            ),
            FieldSpec(
                key="technical_report_paths",
                description=(
                    "Paths to technical evidence reports to include in the"
                    " readiness pack."
                ),
                constraints="list of string paths.",
            ),
            FieldSpec(
                key="risk_owner",
                description="Named owner for operational AI risk management.",
                constraints="string or null.",
            ),
            FieldSpec(
                key="compliance_contact",
                description="Named compliance or legal contact for escalation.",
                constraints="string or null.",
            ),
            FieldSpec(
                key="bundle_signature_secret_env",
                description=(
                    "Optional environment variable name holding the secret"
                    " used to HMAC-sign the integrity manifest."
                ),
                constraints="string or null.",
            ),
            FieldSpec(
                key="pdf_report",
                description=(
                    "Whether to emit a combined PDF report artifact alongside"
                    " the JSON and Markdown bundle."
                ),
                constraints="boolean; true by default.",
            ),
            FieldSpec(
                key="pdf_report_filename",
                description=(
                    "Filename for the generated PDF report inside"
                    " [output].dir."
                ),
                constraints="string filename ending in .pdf.",
            ),
        ),
        notes=(
            (
                "This section is intentionally conservative: Vauban emits a"
                " readiness report, not an automated declaration of legal"
                " compliance."
            ),
            (
                "Every applicable control terminates as pass, fail, or"
                " unknown; non-passing controls get a remediation action."
            ),
            (
                "Evidence documents are strongest when they use explicit"
                " labeled fields such as 'Owner:' or 'Intended purpose:' so"
                " Vauban can validate both narrative content and structure."
            ),
            (
                "Use this section by itself for a standalone report:"
                " vauban readiness.toml"
            ),
            (
                "`vauban init --mode ai_act --output readiness.toml` also"
                " scaffolds draft evidence templates in `./evidence/`."
                " Draft templates do not count as evidence until you replace"
                " the placeholders."
            ),
            (
                "When [ai_act].pdf_report = true, Vauban emits a combined PDF"
                " report for executive and auditor review inside [output].dir."
            ),
        ),
    ),
    SectionSpec(
        name="behavior_diff",
        description=(
            "Standalone behavior trace diff that compares paired JSONL"
            " observations and emits a Model Behavior Change Report."
        ),
        early_return=True,
        fields=(
            FieldSpec(
                key="baseline_trace",
                description="Path to the baseline behavior observation JSONL.",
                constraints="string path; resolved relative to the TOML file.",
                required=True,
            ),
            FieldSpec(
                key="candidate_trace",
                description="Path to the candidate behavior observation JSONL.",
                constraints="string path; resolved relative to the TOML file.",
                required=True,
            ),
            FieldSpec(
                key="baseline_label",
                description="Label for the baseline model state.",
                constraints='string; default: "baseline".',
            ),
            FieldSpec(
                key="candidate_label",
                description="Label for the candidate model state.",
                constraints='string; default: "candidate".',
            ),
            FieldSpec(
                key="baseline_model_path",
                description="Optional baseline model id or deployment label.",
                constraints="string; optional.",
            ),
            FieldSpec(
                key="candidate_model_path",
                description="Optional candidate model id or deployment label.",
                constraints="string; optional.",
            ),
            FieldSpec(
                key="title",
                description="Human-readable report title.",
                constraints='string; default: "Model Behavior Change Report".',
            ),
            FieldSpec(
                key="target_change",
                description="Plain-language description of the audited change.",
                constraints=(
                    "string such as 'base -> fine-tuned',"
                    " 'old prompt -> new prompt', or"
                    " 'full precision -> quantized'."
                ),
            ),
            FieldSpec(
                key="suite_name",
                description="Name of the behavior suite represented by traces.",
                constraints='string; default: "behavior-change-suite".',
            ),
            FieldSpec(
                key="suite_description",
                description="Description of what the trace suite measures.",
                constraints="string.",
            ),
            FieldSpec(
                key="transformation_kind",
                description="Kind of model or system transformation audited.",
                constraints=(
                    "one of fine_tune, reinforcement_fine_tune,"
                    " checkpoint_update, prompt_template, quantization,"
                    " merge, adapter_merge, steering, endpoint_update,"
                    " evaluation_only, other."
                ),
            ),
            FieldSpec(
                key="metrics",
                description="Metric declarations for trace aggregation.",
                constraints=(
                    "array of tables with name; optional description,"
                    " polarity, unit, family. If omitted, Vauban infers"
                    " metrics from trace rows."
                ),
            ),
            FieldSpec(
                key="thresholds",
                description="Optional behavior regression gates for metric deltas.",
                constraints=(
                    "array of tables with metric and at least one bound:"
                    " max_delta, min_delta, or max_absolute_delta. Optional"
                    " category, severity=warn|fail, and description."
                ),
            ),
            FieldSpec(
                key="include_examples",
                description="Whether to include representative paired examples.",
                constraints="boolean; true by default.",
            ),
            FieldSpec(
                key="max_examples",
                description="Maximum paired examples to include.",
                constraints="integer >= 0; default: 3.",
            ),
            FieldSpec(
                key="record_outputs",
                description=(
                    "Whether safe examples may include baseline and candidate"
                    " outputs in Markdown."
                ),
                constraints="boolean; false by default.",
            ),
            FieldSpec(
                key="limitations",
                description="Known limitations surfaced in the report.",
                constraints="list of strings.",
            ),
            FieldSpec(
                key="recommendation",
                description="Deployment or follow-up recommendation.",
                constraints="string; optional.",
            ),
            FieldSpec(
                key="json_filename",
                description="JSON diff artifact filename inside [output].dir.",
                constraints='string; default: "behavior_diff_report.json".',
            ),
            FieldSpec(
                key="markdown_filename",
                description="Markdown report filename inside [output].dir.",
                constraints=(
                    'string; default: "model_behavior_change_report.md".'
                ),
            ),
        ),
        notes=(
            (
                "This mode never loads a model. It is the practical default"
                " when you have API-only outputs, checkpoint traces, or"
                " prompt-template A/B logs."
            ),
            (
                "Each JSONL row is one observation with prompt_id, category,"
                " optional prompt/output_text, optional refused, and optional"
                " numeric metrics."
            ),
            (
                "Vauban derives refusal_rate from a boolean refused field and"
                " computes matched baseline/candidate metric deltas by metric"
                " name, category, and unit."
            ),
            (
                "Examples obey each observation's redaction value: safe,"
                " redacted, or omitted."
            ),
            (
                "Thresholds are evaluated after reports are written. A failing"
                " threshold with severity='fail' exits the run non-zero, which"
                " makes [behavior_diff] usable as behavior CI."
            ),
        ),
    ),
    SectionSpec(
        name="behavior_trace",
        description=(
            "Model-loaded behavior trace collection that runs a shared"
            " prompt suite and writes reusable JSONL observations."
        ),
        early_return=True,
        config_class="BehaviorTraceConfig",
        fields=(
            FieldSpec(
                key="model_label",
                description="Label stored on every trace observation.",
                constraints='string; default: "model".',
            ),
            FieldSpec(
                key="suite",
                description=(
                    "Optional external behavior-suite TOML file containing"
                    " [behavior_suite] metadata and [[behavior_suite.prompts]]."
                ),
                constraints="string path; resolved relative to the TOML file.",
            ),
            FieldSpec(
                key="suite_name",
                description="Name of the behavior suite represented by the trace.",
                constraints='string; default: "behavior-change-suite".',
            ),
            FieldSpec(
                key="suite_description",
                description="Description of what the suite measures.",
                constraints="string.",
            ),
            FieldSpec(
                key="suite_version",
                description="Optional version label for the behavior suite.",
                constraints="string; optional.",
            ),
            FieldSpec(
                key="suite_source",
                description=(
                    "Optional source URL or path for the suite definition."
                ),
                constraints=(
                    "string; optional. External suite files default this to"
                    " the suite TOML path."
                ),
            ),
            FieldSpec(
                key="safety_policy",
                description="How prompt/output examples are handled for sharing.",
                constraints=(
                    'string; default: "safe_or_redacted_prompts" or the'
                    " external suite value."
                ),
            ),
            FieldSpec(
                key="prompts",
                description="Inline prompt suite entries.",
                constraints=(
                    "array of strings or tables. Table keys: id/prompt_id,"
                    " text/prompt, category, expected_behavior, redaction,"
                    " tags."
                ),
            ),
            FieldSpec(
                key="metrics",
                description=(
                    "Metric declarations inherited by the trace summary."
                    " External suites can declare [[behavior_suite.metrics]],"
                    " and inline configs can declare [[behavior_trace.metrics]]."
                ),
                constraints=(
                    "array of tables with name; optional description,"
                    " polarity, unit, family. Defaults to Vauban's"
                    " deterministic behavior metrics."
                ),
            ),
            FieldSpec(
                key="scorers",
                description=(
                    "Registered behavior scorers to run for each generated"
                    " output."
                ),
                constraints=(
                    "list of strings; default: [\"deterministic_v1\"]."
                    " Built-ins: deterministic_v1, refusal_v1, length_v1,"
                    " style_v1, expected_behavior_v1."
                ),
            ),
            FieldSpec(
                key="max_tokens",
                description="Maximum tokens generated per prompt.",
                constraints="integer >= 1; default: 80.",
            ),
            FieldSpec(
                key="refusal_phrases",
                description="Phrases used for simple refusal-style detection.",
                constraints="list of strings; defaults to Vauban refusal phrases.",
            ),
            FieldSpec(
                key="record_outputs",
                description=(
                    "Whether safe prompt rows may store generated output text"
                    " in the JSONL trace."
                ),
                constraints="boolean; false by default.",
            ),
            FieldSpec(
                key="output_trace",
                description="Optional explicit trace JSONL output path.",
                constraints=(
                    "string path; resolved relative to the TOML file."
                    " If omitted, trace_filename is written under [output].dir."
                ),
            ),
            FieldSpec(
                key="trace_filename",
                description="Trace JSONL filename inside [output].dir.",
                constraints='string; default: "behavior_trace.jsonl".',
            ),
            FieldSpec(
                key="json_filename",
                description="Trace collection summary JSON filename.",
                constraints='string; default: "behavior_trace_report.json".',
            ),
        ),
        notes=(
            (
                "Use this when you have local model access and want to create"
                " trace files for later [behavior_diff] comparisons."
            ),
            (
                "The emitted JSONL schema is the same observation schema read"
                " by [behavior_diff]: prompt_id, category, optional prompt,"
                " optional output_text, refused, metrics, and redaction."
            ),
            (
                "By default Vauban records output length and refusal-style"
                " detection, but it does not store model outputs unless"
                " record_outputs is true and the prompt is marked safe."
            ),
        ),
    ),
    SectionSpec(
        name="behavior_report",
        description=(
            "Standalone Model Behavior Change Report assembled from"
            " TOML-declared evidence."
        ),
        early_return=True,
        fields=(
            FieldSpec(
                key="title",
                description="Human-readable report title.",
                constraints='string; default: "Model Behavior Change Report".',
            ),
            FieldSpec(
                key="target_change",
                description="Plain-language description of the model change.",
                constraints=(
                    "string such as 'base -> fine-tuned',"
                    " 'checkpoint 1200 -> checkpoint 2000', or"
                    " 'full precision -> quantized'."
                ),
            ),
            FieldSpec(
                key="findings",
                description="High-level behavior-change findings.",
                constraints="list of strings; optional.",
            ),
            FieldSpec(
                key="markdown_report",
                description="Whether to emit a Markdown companion report.",
                constraints="boolean; true by default.",
            ),
            FieldSpec(
                key="json_filename",
                description="JSON artifact filename inside [output].dir.",
                constraints='string; default: "behavior_report.json".',
            ),
            FieldSpec(
                key="markdown_filename",
                description="Markdown artifact filename inside [output].dir.",
                constraints='string; default: "behavior_report.md".',
            ),
            FieldSpec(
                key="limitations",
                description="Known limitations surfaced in the report.",
                constraints="list of strings.",
            ),
            FieldSpec(
                key="recommendation",
                description="Deployment or follow-up recommendation.",
                constraints="string; optional.",
            ),
            FieldSpec(
                key="baseline",
                description="Baseline model/run metadata table.",
                constraints=(
                    "required table with label and model_path; optional role,"
                    " checkpoint, adapter_path, prompt_template, quantization."
                ),
                required=True,
            ),
            FieldSpec(
                key="candidate",
                description="Candidate model/run metadata table.",
                constraints=(
                    "required table with label and model_path; optional role,"
                    " checkpoint, adapter_path, prompt_template, quantization."
                ),
                required=True,
            ),
            FieldSpec(
                key="suite",
                description="Behavior suite metadata table.",
                constraints=(
                    "required table with name, description, categories, metrics;"
                    " optional version, source, safety_policy."
                ),
                required=True,
            ),
            FieldSpec(
                key="transformation",
                description="The model transformation being audited.",
                constraints=(
                    "table with kind and summary; optional before, after,"
                    " method, source_ref, notes. kind is one of fine_tune,"
                    " reinforcement_fine_tune, checkpoint_update,"
                    " prompt_template, quantization, merge, adapter_merge,"
                    " steering, endpoint_update, evaluation_only, other."
                ),
            ),
            FieldSpec(
                key="access",
                description="Access level and maximum defensible claim strength.",
                constraints=(
                    "table with level and claim_strength; optional"
                    " available_evidence, missing_evidence, notes."
                ),
            ),
            FieldSpec(
                key="evidence",
                description="Named evidence artifacts referenced by claims.",
                constraints=(
                    "array of tables with id/evidence_id and kind; optional"
                    " path_or_url and description."
                ),
            ),
            FieldSpec(
                key="claims",
                description="Access-aware claims made by the report.",
                constraints=(
                    "array of tables with id/claim_id and statement; optional"
                    " strength, access_level, status, evidence, limitations."
                ),
            ),
            FieldSpec(
                key="metrics",
                description="Metric observations used to compute deltas.",
                constraints=(
                    "array of tables; each row needs name, model_label, value;"
                    " optional category, polarity, unit, family, sample_size."
                ),
            ),
            FieldSpec(
                key="activation_findings",
                description="Internal diagnostics linked to behavior changes.",
                constraints=(
                    "array of tables with name and summary; optional layers,"
                    " score, metric_name, direction_label, severity, evidence."
                ),
            ),
            FieldSpec(
                key="examples",
                description="Safe or redacted representative examples.",
                constraints=(
                    "array of tables with id/example_id, category, prompt;"
                    " optional redacted responses, redaction, note."
                ),
            ),
            FieldSpec(
                key="reproduction_targets",
                description=(
                    "Papers or external claims this report tries to reproduce"
                    " or extend."
                ),
                constraints=(
                    "array of tables with id/target_id, title,"
                    " original_claim, planned_extension; optional source_url,"
                    " status, notes."
                ),
            ),
            FieldSpec(
                key="reproduction_results",
                description="Observed outcomes for declared reproduction targets.",
                constraints=(
                    "array of tables with target_id, status, summary;"
                    " optional replicated_claims, failed_claims, extensions,"
                    " evidence, limitations."
                ),
            ),
            FieldSpec(
                key="intervention_results",
                description=(
                    "Observed outcomes from controlled prompt, activation,"
                    " weight, sampling, or steering interventions."
                ),
                constraints=(
                    "array of tables with id/intervention_id, kind, summary,"
                    " target; optional effect, polarity, layers, strength,"
                    " baseline_condition, intervention_condition,"
                    " behavior_metric, activation_metric, evidence,"
                    " limitations."
                ),
            ),
            FieldSpec(
                key="reproducibility",
                description="Command, config, code, data, seed, and notes.",
                constraints="table; command is required when present.",
            ),
        ),
        notes=(
            (
                "This mode never loads a model. It is for shareable,"
                " reproducible model behavior change reports assembled from"
                " already-known metrics and evidence."
            ),
            (
                "Metric deltas are computed automatically by matching baseline"
                " and candidate rows with the same name, category, and unit."
            ),
            (
                "Use [behavior_report.access] and"
                " [[behavior_report.claims]] to separate observed behavior"
                " from stronger causal or internal claims."
            ),
            (
                "Use safe or redacted examples. The report should not become"
                " a prompt pack or bypass recipe."
            ),
        ),
    ),
    SectionSpec(
        name="measure",
        description="Behavioral direction extraction settings.",
        config_class="MeasureConfig",
        fields=(
            FieldSpec(
                key="mode",
                description="Measurement algorithm.",
                constraints="one of: direction, subspace, dbdi, diff.",
            ),
            FieldSpec(
                key="top_k",
                description="Number of directions to keep in subspace/diff workflows.",
                constraints="integer.",
            ),
            FieldSpec(
                key="clip_quantile",
                description="Winsorization quantile for activation clipping.",
                constraints="number in [0.0, 0.5).",
            ),
            FieldSpec(
                key="transfer_models",
                description="HuggingFace model IDs for direction transfer testing.",
                constraints="list of strings; empty by default.",
            ),
            FieldSpec(
                key="diff_model",
                description="Base model for weight-diff measurement.",
                constraints="string; required when mode = 'diff'.",
                notes=(
                    "The diff direction is extracted by SVD of"
                    " W_aligned - W_base for o_proj and down_proj.",
                ),
            ),
            FieldSpec(
                key="measure_only",
                description="Stop after the measure stage and skip cut/export/eval.",
                constraints="boolean; false by default.",
                notes=(
                    "Useful for spectral-analysis configs that should write"
                    " reports without modifying model weights.",
                ),
            ),
            FieldSpec(
                key="bank",
                description=(
                    "Steer2Adapt subspace bank entries for"
                    " multi-direction composition."
                ),
                constraints=(
                    "list of tables; each entry has 'path'"
                    " and optional 'weight'."
                ),
                notes=(
                    "Each [[measure.bank]] table has:"
                    " name (string, required — label for"
                    " this subspace),"
                    " harmful (string, required — path or"
                    ' "default"),'
                    " harmless (string, required — path or"
                    ' "default").',
                ),
            ),
        ),
    ),
    SectionSpec(
        name="cut",
        description="Weight-space surgery controls.",
        config_class="CutConfig",
        fields=(
            FieldSpec(
                key="alpha",
                description="Global cut strength multiplier.",
                constraints="number.",
                notes=(
                    "Negative values amplify the direction instead of"
                    " removing it (LoX-style safety hardening).",
                ),
            ),
            FieldSpec(
                key="layers",
                description="Explicit target layer list or auto mode.",
                constraints='either "auto" or list of integers.',
            ),
            FieldSpec(
                key="norm_preserve",
                description="Preserve row norms after projection removal.",
                constraints="boolean.",
            ),
            FieldSpec(
                key="biprojected",
                description="Apply biprojected direction removal.",
                constraints="boolean.",
            ),
            FieldSpec(
                key="layer_strategy",
                description="Automatic layer selection strategy.",
                constraints="one of: all, above_median, top_k.",
            ),
            FieldSpec(
                key="layer_top_k",
                description="Layer count when layer_strategy = top_k.",
                constraints="integer.",
            ),
            FieldSpec(
                key="layer_weights",
                description="Per-layer alpha multipliers.",
                constraints="list of numbers; length should match selected layers.",
            ),
            FieldSpec(
                key="sparsity",
                description="Fraction of direction components to zero before cutting.",
                constraints="number in [0.0, 1.0).",
            ),
            FieldSpec(
                key="dbdi_target",
                description="Which DBDI component to cut when measure.mode = dbdi.",
                constraints="one of: red, hdd, both.",
            ),
            FieldSpec(
                key="false_refusal_ortho",
                description=(
                    "Orthogonalize against a borderline false-refusal direction."
                ),
                constraints="boolean.",
                notes=(
                    "Requires [data].borderline to be configured.",
                ),
            ),
            FieldSpec(
                key="layer_type_filter",
                description="Optional architectural layer-type filter.",
                constraints="one of: global, sliding, or null.",
            ),
        ),
    ),
    SectionSpec(
        name="eval",
        description="Post-cut quality and refusal evaluation.",
        config_class="EvalConfig",
        fields=(
            FieldSpec(
                key="prompts",
                attr="prompts_path",
                description="Path to evaluation prompts JSONL.",
                constraints="string path; if omitted, harmful fallback is used.",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generation cap for refusal-rate evaluation.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="num_prompts",
                description="Fallback count when eval prompts file is omitted.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="refusal_phrases",
                attr="refusal_phrases_path",
                description="Custom refusal phrase file.",
                constraints="string path; one phrase per line.",
            ),
            FieldSpec(
                key="refusal_mode",
                description="Refusal detection method.",
                constraints='one of: "phrases", "judge".',
            ),
            FieldSpec(
                key="scoring_weights",
                description="Composite response scoring weights (sub-table).",
                constraints=(
                    "table with float keys: length, structure,"
                    " anti_refusal, directness, relevance."
                ),
            ),
        ),
    ),
    SectionSpec(
        name="surface",
        description="Before/after refusal surface mapping.",
        config_class="SurfaceConfig",
        fields=(
            FieldSpec(
                key="prompts",
                attr="prompts_path",
                description="Surface prompt source.",
                constraints='string path, "default", or "default_multilingual".',
            ),
            FieldSpec(
                key="generate",
                description="Generate text while scanning (vs projections only).",
                constraints="boolean.",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generation cap per surface prompt.",
                constraints="integer.",
            ),
            FieldSpec(
                key="progress",
                description="Show scan progress logs.",
                constraints="boolean.",
            ),
            FieldSpec(
                key="max_worst_cell_refusal_after",
                description=(
                    "Fail run if post-cut worst cell refusal rate exceeds"
                    " this."
                ),
                constraints="number in [0, 1] or omitted.",
            ),
            FieldSpec(
                key="max_worst_cell_refusal_delta",
                description=(
                    "Fail run if any surface cell refusal-rate increase"
                    " exceeds this."
                ),
                constraints="number in [0, 1] or omitted.",
            ),
            FieldSpec(
                key="min_coverage_score",
                description="Fail run if post-cut matrix coverage falls below this.",
                constraints="number in [0, 1] or omitted.",
            ),
        ),
    ),
    SectionSpec(
        name="detect",
        description="Defense-hardening detection that runs before cutting.",
        config_class="DetectConfig",
        fields=(
            FieldSpec(
                key="mode",
                description="Detection depth.",
                constraints="one of: fast, probe, full, margin.",
            ),
            FieldSpec(
                key="top_k",
                description="Subspace dimensionality for probe/full diagnostics.",
                constraints="integer.",
            ),
            FieldSpec(
                key="clip_quantile",
                description="Optional activation clipping quantile for detection.",
                constraints="number.",
            ),
            FieldSpec(
                key="alpha",
                description="Test cut strength in full mode.",
                constraints="number.",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generation cap in full mode.",
                constraints="integer.",
            ),
            FieldSpec(
                key="margin_directions",
                description=(
                    "Paths to additional direction .npy files"
                    " for margin analysis."
                ),
                constraints="list of strings.",
            ),
            FieldSpec(
                key="margin_alphas",
                description="Alpha values to sweep in margin mode.",
                constraints="list of numbers.",
            ),
            FieldSpec(
                key="svf_compare",
                description="Compare SVF-based vs linear separation in detection.",
                constraints="boolean.",
            ),
        ),
        notes=(
            (
                "Detection does not run when [depth] is active, because depth"
                " returns early."
            ),
        ),
    ),
    SectionSpec(
        name="optimize",
        description="Optuna multi-objective search over cut parameters.",
        early_return=True,
        config_class="OptimizeConfig",
        fields=(
            FieldSpec(
                key="n_trials",
                description="Number of Optuna trials.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="alpha_min",
                description="Minimum alpha sampled by the optimizer.",
                constraints="number; must be < alpha_max.",
            ),
            FieldSpec(
                key="alpha_max",
                description="Maximum alpha sampled by the optimizer.",
                constraints="number; must be > alpha_min.",
            ),
            FieldSpec(
                key="sparsity_min",
                description="Minimum direction sparsity sampled by the optimizer.",
                constraints="number.",
            ),
            FieldSpec(
                key="sparsity_max",
                description="Maximum direction sparsity sampled by the optimizer.",
                constraints="number.",
            ),
            FieldSpec(
                key="search_norm_preserve",
                description="Search both norm-preserving and standard cuts.",
                constraints="boolean.",
            ),
            FieldSpec(
                key="search_strategies",
                description="Layer strategies included in the search.",
                constraints="list of: all, above_median, top_k.",
            ),
            FieldSpec(
                key="layer_top_k_min",
                description="Minimum top-k layer count sampled.",
                constraints="integer.",
            ),
            FieldSpec(
                key="layer_top_k_max",
                description="Maximum top-k layer count sampled.",
                constraints="integer or null.",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generation cap used while scoring each trial.",
                constraints="integer.",
            ),
            FieldSpec(
                key="seed",
                description="Optional optimizer seed.",
                constraints="integer or null.",
            ),
            FieldSpec(
                key="timeout",
                description="Optional wall-clock timeout in seconds.",
                constraints="number or null.",
            ),
        ),
    ),
    SectionSpec(
        name="softprompt",
        description="Continuous/discrete soft prompt attack configuration.",
        early_return=True,
        config_class="SoftPromptConfig",
        fields=(
            FieldSpec(
                key="mode",
                description="Soft prompt optimization algorithm.",
                constraints="one of: continuous, gcg, egd.",
            ),
            FieldSpec(
                key="n_tokens",
                description="Learnable prompt length.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="n_steps",
                description="Optimization step count.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="learning_rate",
                description="Learning rate for continuous mode.",
                constraints="number > 0.",
            ),
            FieldSpec(
                key="init_scale",
                description="Initial embedding scale.",
                constraints="number.",
            ),
            FieldSpec(
                key="batch_size",
                description="Candidate batch size for token search.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="top_k",
                description="Top-k token candidates per position.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="direction_weight",
                description="Weight for direction-aware regularization.",
                constraints="number >= 0.",
            ),
            FieldSpec(
                key="target_prefixes",
                description="Target prefixes used by targeted losses.",
                constraints="list of strings.",
            ),
            FieldSpec(
                key="max_gen_tokens",
                description="Generation cap for attack evaluation.",
                constraints="integer.",
            ),
            FieldSpec(
                key="seed",
                description="Optional random seed.",
                constraints="integer or null.",
            ),
            FieldSpec(
                key="embed_reg_weight",
                description="Embedding norm regularization strength.",
                constraints="number >= 0.",
            ),
            FieldSpec(
                key="patience",
                description="Early stopping patience (0 disables).",
                constraints="integer >= 0.",
            ),
            FieldSpec(
                key="lr_schedule",
                description="Learning-rate schedule.",
                constraints="one of: constant, cosine.",
            ),
            FieldSpec(
                key="n_restarts",
                description="Number of random restarts.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="prompt_strategy",
                description="Prompt sampling strategy during optimization.",
                constraints="one of: all, cycle, first, worst_k.",
            ),
            FieldSpec(
                key="direction_mode",
                description="How direction loss is applied in token positions.",
                constraints="one of: last, raid, all_positions.",
            ),
            FieldSpec(
                key="direction_layers",
                description="Specific layers for direction loss.",
                constraints="list of integers or null.",
            ),
            FieldSpec(
                key="loss_mode",
                description="Primary loss objective.",
                constraints="one of: targeted, untargeted, defensive.",
            ),
            FieldSpec(
                key="egd_temperature",
                description="Temperature for EGD simplex sharpening.",
                constraints="number > 0.",
            ),
            FieldSpec(
                key="token_constraint",
                description=(
                    "Token constraint set for candidate tokens. Positive"
                    " constraints (ascii, alpha, etc.) include only matching"
                    " tokens. Negative constraints (exclude_glitch) remove"
                    " under-trained tokens that cause model collapse."
                    " Can be a single string or a list combining both types."
                ),
                constraints=(
                    "one of: ascii, alpha, alphanumeric, non_latin, chinese,"
                    " non_alphabetic, invisible, zalgo, emoji,"
                    " exclude_glitch, or null."
                ),
            ),
            FieldSpec(
                key="eos_loss_mode",
                description="EOS auxiliary loss behavior.",
                constraints="one of: none, force, suppress.",
            ),
            FieldSpec(
                key="eos_loss_weight",
                description="Weight for EOS auxiliary loss.",
                constraints="number >= 0.",
            ),
            FieldSpec(
                key="kl_ref_weight",
                description="KL collision regularization weight.",
                constraints="number >= 0.",
                notes=(
                    "If > 0, [softprompt].ref_model must be set.",
                ),
            ),
            FieldSpec(
                key="ref_model",
                description="Reference model for KL collision loss.",
                constraints="string or null.",
            ),
            FieldSpec(
                key="worst_k",
                description="Prompt count used by worst_k strategy.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="grad_accum_steps",
                description="Gradient accumulation steps.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="transfer_models",
                description="Model IDs used for transfer evaluation.",
                constraints="list of strings.",
            ),
            FieldSpec(
                key="temperature_schedule",
                description=(
                    "Temperature annealing schedule for EGD/COLD."
                    " Anneals from max(2.0, base_temp) down to base_temp."
                ),
                constraints="one of: constant, linear, cosine.",
            ),
            FieldSpec(
                key="entropy_weight",
                description=(
                    "Entropy bonus weight for EGD/COLD."
                    " Encourages exploration by penalizing peaked"
                    " distributions (0 = off)."
                ),
                constraints="number >= 0.",
            ),
            FieldSpec(
                key="target_repeat_count",
                description=(
                    "Repeat target prefix tokens N times"
                    " in the target sequence."
                ),
                constraints="integer >= 0 (0 = disabled).",
            ),
            FieldSpec(
                key="system_prompt",
                description=(
                    "System prompt prepended to all messages"
                    " during optimization."
                ),
                constraints="string or null.",
            ),
            FieldSpec(
                key="token_position",
                description="Where to inject the optimized tokens.",
                constraints='one of: "prefix", "suffix", "infix".',
            ),
            FieldSpec(
                key="init_tokens",
                description="Warm-start token IDs for GCG/EGD optimization.",
                constraints="list of integers or null.",
            ),
            FieldSpec(
                key="injection_context",
                description="Wrap optimized tokens in realistic surrounding context.",
                constraints=(
                    'one of: "web_page", "tool_output", "code_file",'
                    " or null."
                ),
            ),
            FieldSpec(
                key="injection_context_template",
                description=(
                    "Custom template with {payload} placeholder"
                    " for injection wrapping."
                ),
                constraints="string with {payload} placeholder, or null.",
            ),
            FieldSpec(
                key="perplexity_weight",
                description=(
                    "Cross-entropy penalty pushing optimized tokens"
                    " toward fluent text (code-doc attractor)."
                ),
                constraints="number >= 0.",
            ),
            FieldSpec(
                key="paraphrase_strategies",
                description="Paraphrase augmentation strategies applied to prompts.",
                constraints="list of strings.",
            ),
            FieldSpec(
                key="externality_target",
                description=(
                    "Path to direction .npy for externality"
                    " loss regularization."
                ),
                constraints="string path or null.",
            ),
            FieldSpec(
                key="svf_boundary_path",
                description="Path to trained SVF boundary weights for SVF-aware loss.",
                constraints="string path or null.",
            ),
            FieldSpec(
                key="prompt_pool_size",
                description=(
                    "Override eval.num_prompts for the"
                    " optimization prompt pool."
                ),
                constraints="integer or null.",
            ),
            FieldSpec(
                key="beam_width",
                description="GCG beam search population size (1 = greedy single-best).",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="cold_temperature",
                description=(
                    "COLD-Attack softmax temperature for"
                    " logit-to-probability conversion."
                ),
                constraints="number > 0.",
            ),
            FieldSpec(
                key="cold_noise_scale",
                description="Langevin dynamics noise scaling factor for COLD-Attack.",
                constraints="number > 0.",
            ),
            FieldSpec(
                key="defense_eval",
                description="Run defense evaluation alongside optimization.",
                constraints='one of: "sic", "cast", "both", or null.',
            ),
            FieldSpec(
                key="defense_eval_layer",
                description="Layer for defense evaluation scoring.",
                constraints="integer or null (null = auto-detect).",
            ),
            FieldSpec(
                key="defense_eval_alpha",
                description="CAST steering alpha for defense evaluation.",
                constraints="number.",
            ),
            FieldSpec(
                key="defense_eval_threshold",
                description=(
                    "Shared detection threshold for SIC/CAST"
                    " defense evaluation."
                ),
                constraints="number.",
            ),
            FieldSpec(
                key="defense_eval_sic_threshold",
                description="SIC-specific threshold override for defense evaluation.",
                constraints="number or null (null = use defense_eval_threshold).",
            ),
            FieldSpec(
                key="defense_eval_sic_mode",
                description="SIC mode for defense evaluation.",
                constraints='one of: "direction", "generation".',
            ),
            FieldSpec(
                key="defense_eval_sic_max_iterations",
                description=(
                    "Maximum SIC sanitization iterations"
                    " in defense evaluation."
                ),
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="defense_eval_cast_layers",
                description="CAST layer subset for defense evaluation.",
                constraints="list of integers or null.",
            ),
            FieldSpec(
                key="defense_eval_alpha_tiers",
                description=(
                    "TRYLOCK-style adaptive alpha tiers"
                    " for defense evaluation CAST."
                ),
                constraints="list of [threshold, alpha] pairs or null.",
            ),
            FieldSpec(
                key="defense_aware_weight",
                description=(
                    "Weight for defense-aware auxiliary loss"
                    " penalizing detection."
                ),
                constraints="number >= 0 (0 = off).",
            ),
            FieldSpec(
                key="transfer_loss_weight",
                description="Weight for multi-model transfer re-ranking loss.",
                constraints="number >= 0 (0 = off).",
            ),
            FieldSpec(
                key="transfer_rerank_count",
                description="Top-N candidates to re-rank on transfer models.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="gan_rounds",
                description=(
                    "Number of iterative attack-defense GAN rounds"
                    " (0 = disabled)."
                ),
                constraints="integer >= 0.",
            ),
            FieldSpec(
                key="gan_step_multiplier",
                description="Multiply n_steps by this factor each failed GAN round.",
                constraints="number > 0.",
            ),
            FieldSpec(
                key="gan_direction_escalation",
                description="Amount added to direction_weight per failed GAN round.",
                constraints="number >= 0.",
            ),
            FieldSpec(
                key="gan_token_escalation",
                description="Tokens added to n_tokens per failed GAN round.",
                constraints="integer >= 0.",
            ),
            FieldSpec(
                key="gan_defense_escalation",
                description="Enable defender hardening between GAN rounds.",
                constraints="boolean.",
            ),
            FieldSpec(
                key="gan_defense_alpha_multiplier",
                description="Multiply CAST alpha by this factor per attacker win.",
                constraints="number > 0.",
            ),
            FieldSpec(
                key="gan_defense_threshold_escalation",
                description="Subtract from CAST threshold per attacker win.",
                constraints="number >= 0.",
            ),
            FieldSpec(
                key="gan_defense_sic_iteration_escalation",
                description="Add to SIC max iterations per attacker win.",
                constraints="integer >= 0.",
            ),
            FieldSpec(
                key="gan_multiturn",
                description="Enable multi-turn conversation threading in GAN mode.",
                constraints="boolean.",
            ),
            FieldSpec(
                key="gan_multiturn_max_turns",
                description="Maximum conversation turns kept in GAN history.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="largo_reflection_rounds",
                description=(
                    "LARGO self-reflective decoding loop rounds"
                    " (0 = disabled)."
                ),
                constraints="integer >= 0.",
            ),
            FieldSpec(
                key="largo_max_reflection_tokens",
                description="Generation cap per LARGO reflection step.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="largo_objective",
                description="Objective function for LARGO reflection satisfaction.",
                constraints='one of: "targeted", "untargeted".',
            ),
            FieldSpec(
                key="largo_embed_warmstart",
                description="Warm-start embeddings from previous LARGO round.",
                constraints="boolean.",
            ),
            FieldSpec(
                key="amplecgc_collect_steps",
                description="GCG steps per collection restart in AmpleGCG mode.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="amplecgc_collect_restarts",
                description="Number of GCG restarts for AmpleGCG suffix collection.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="amplecgc_collect_threshold",
                description=(
                    "Loss threshold for harvesting suffixes"
                    " into the AmpleGCG corpus."
                ),
                constraints="number.",
            ),
            FieldSpec(
                key="amplecgc_n_candidates",
                description="Number of candidates sampled from the AmpleGCG generator.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="amplecgc_hidden_dim",
                description="Hidden dimension of the AmpleGCG generator MLP.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="amplecgc_train_steps",
                description="Training steps for the AmpleGCG generator.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="amplecgc_train_lr",
                description="Learning rate for AmpleGCG generator training.",
                constraints="number > 0.",
            ),
            FieldSpec(
                key="amplecgc_sample_temperature",
                description="Sampling temperature for the AmpleGCG generator.",
                constraints="number > 0.",
            ),
        ),
    ),
    SectionSpec(
        name="sic",
        description="SIC iterative sanitization settings.",
        early_return=True,
        config_class="SICConfig",
        fields=(
            FieldSpec(
                key="mode",
                description="Sanitization scoring mode.",
                constraints="one of: direction, generation.",
            ),
            FieldSpec(
                key="threshold",
                description="Detection threshold.",
                constraints="number.",
            ),
            FieldSpec(
                key="max_iterations",
                description="Maximum sanitize iterations per prompt.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generation cap for generation mode scoring.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="target_layer",
                description="Layer override for direction projection.",
                constraints="integer or null.",
            ),
            FieldSpec(
                key="sanitize_system_prompt",
                description="System prompt used for rewrite sanitization.",
                constraints="string.",
            ),
            FieldSpec(
                key="max_sanitize_tokens",
                description="Token cap for sanitize rewrites.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="block_on_failure",
                description="Block prompts that remain unsafe after retries.",
                constraints="boolean.",
            ),
            FieldSpec(
                key="calibrate",
                description="Auto-calibrate threshold from prompt samples.",
                constraints="boolean.",
            ),
            FieldSpec(
                key="calibrate_prompts",
                description="Prompt pool used when calibration is enabled.",
                constraints="one of: harmless, harmful.",
            ),
            FieldSpec(
                key="svf_boundary_path",
                description=(
                    "Path to trained SVF boundary weights"
                    " for SVF-aware scoring."
                ),
                constraints="string path or null.",
            ),
        ),
    ),
    SectionSpec(
        name="depth",
        description="Deep-thinking token analysis settings.",
        early_return=True,
        config_class="DepthConfig",
        fields=(
            FieldSpec(
                key="prompts",
                description="Inline prompts for depth profiling.",
                constraints="required non-empty list of strings.",
                required=True,
            ),
            FieldSpec(
                key="settling_threshold",
                description="JSD settling threshold.",
                constraints="number in (0.0, 1.0].",
            ),
            FieldSpec(
                key="deep_fraction",
                description=(
                    "Layer-fraction threshold for deep-thinking classification."
                ),
                constraints="number in (0.0, 1.0].",
            ),
            FieldSpec(
                key="top_k_logits",
                description="Top-k logit approximation size for JSD.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generated tokens for dynamic depth mode.",
                constraints="integer >= 0.",
            ),
            FieldSpec(
                key="extract_direction",
                description="Extract a depth direction in addition to depth profiles.",
                constraints="boolean.",
                notes=(
                    "Requires at least 2 effective prompts"
                    " (direction_prompts if set, else prompts).",
                ),
            ),
            FieldSpec(
                key="direction_prompts",
                description="Optional prompt subset for direction extraction.",
                constraints="list of strings or null.",
            ),
            FieldSpec(
                key="clip_quantile",
                description="Winsorization quantile during direction extraction.",
                constraints="number in [0.0, 0.5).",
            ),
        ),
    ),
    SectionSpec(
        name="probe",
        description="Per-layer projection inspection.",
        early_return=True,
        config_class="ProbeConfig",
        fields=(
            FieldSpec(
                key="prompts",
                description="Inline prompts to probe.",
                constraints="required non-empty list of strings.",
                required=True,
            ),
        ),
    ),
    SectionSpec(
        name="steer",
        description="Runtime activation steering for text generation.",
        early_return=True,
        config_class="SteerConfig",
        fields=(
            FieldSpec(
                key="prompts",
                description="Inline prompts used for steered generation.",
                constraints="required non-empty list of strings.",
                required=True,
            ),
            FieldSpec(
                key="layers",
                description="Layer subset to steer.",
                constraints="list of integers or null (null means all layers).",
            ),
            FieldSpec(
                key="alpha",
                description="Steering strength.",
                constraints="number.",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generation cap per prompt.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="direction_source",
                description="Source for the steering direction.",
                constraints='one of: "linear", "svf".',
            ),
            FieldSpec(
                key="svf_boundary_path",
                description="Path to trained SVF boundary weights.",
                constraints="string path or null.",
            ),
            FieldSpec(
                key="bank_path",
                description="Path to a Steer2Adapt direction bank .npz file.",
                constraints="string path or null.",
            ),
            FieldSpec(
                key="composition",
                description="Named direction weights for multi-direction composition.",
                constraints="table mapping direction names to float weights.",
            ),
        ),
    ),
    SectionSpec(
        name="intervention_eval",
        description="Controlled intervention evaluation across prompt families.",
        early_return=True,
        config_class="InterventionEvalConfig",
        fields=(
            FieldSpec(
                key="prompts",
                description="Prompt list or typed prompt-family entries.",
                constraints=(
                    "list of strings, or array of tables with id/prompt_id"
                    " and text; optional category."
                ),
                required=True,
            ),
            FieldSpec(
                key="alphas",
                description="Intervention strengths to evaluate.",
                constraints="non-empty list of numbers; include baseline_alpha.",
            ),
            FieldSpec(
                key="baseline_alpha",
                description="Condition used as the behavioral baseline.",
                constraints="number present in alphas; default 0.0.",
            ),
            FieldSpec(
                key="layers",
                description="Layer subset to intervene on.",
                constraints="list of integers or null (null means all layers).",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generation cap per prompt and alpha.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="target",
                description="Human-readable target direction or intervention name.",
                constraints='string; default: "measured_direction".',
            ),
            FieldSpec(
                key="kind",
                description="Behavior-report intervention kind.",
                constraints=(
                    "one of activation_steering, activation_ablation,"
                    " activation_addition, weight_projection,"
                    " weight_arithmetic, prompt_template, sampling, other."
                ),
            ),
            FieldSpec(
                key="behavior_metric",
                description="Behavior metric label used in emitted report rows.",
                constraints='string; default: "refusal_style_rate".',
            ),
            FieldSpec(
                key="activation_metric",
                description="Activation metric label used in emitted report rows.",
                constraints='string; default: "mean_projection_delta".',
            ),
            FieldSpec(
                key="refusal_phrases",
                description="Phrase list for simple refusal-style scoring.",
                constraints="non-empty list of strings; defaults to built-ins.",
            ),
            FieldSpec(
                key="record_outputs",
                description="Whether to include generated text in JSON output.",
                constraints="boolean; default false.",
            ),
            FieldSpec(
                key="json_filename",
                description="JSON artifact filename inside [output].dir.",
                constraints='string; default: "intervention_eval_report.json".',
            ),
            FieldSpec(
                key="markdown_filename",
                description="Markdown artifact filename inside [output].dir.",
                constraints='string; default: "intervention_eval_report.md".',
            ),
            FieldSpec(
                key="toml_fragment_filename",
                description="TOML fragment containing behavior-report rows.",
                constraints='string; default: "intervention_results.toml".',
            ),
            FieldSpec(
                key="limitations",
                description="Limitations attached to emitted intervention results.",
                constraints="list of strings.",
            ),
        ),
        notes=(
            (
                "This mode runs after [measure], sweeps every prompt across"
                " every alpha, and returns before cut/export."
            ),
            (
                "It intentionally uses simple phrase and projection metrics so"
                " larger prompt-family sweeps are cheap and reproducible."
            ),
        ),
    ),
    SectionSpec(
        name="sss",
        description="Sensitivity-scaled steering via Jacobian analysis.",
        early_return=True,
        config_class="SSSConfig",
        fields=(
            FieldSpec(
                key="prompts",
                description="Inline prompts for SSS generation.",
                constraints="required non-empty list of strings.",
                required=True,
            ),
            FieldSpec(
                key="layers",
                description="Layer subset to steer.",
                constraints="list of integers or null (null means auto-select).",
            ),
            FieldSpec(
                key="alpha",
                description="Global steering strength multiplier.",
                constraints="number.",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generation cap per prompt.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="calibration_prompt",
                description="Prompt for sensitivity calibration.",
                constraints="string.",
            ),
            FieldSpec(
                key="n_power_iterations",
                description="Power iterations for dominant vector.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="fd_epsilon",
                description="Finite-difference step size.",
                constraints="number > 0.",
            ),
            FieldSpec(
                key="seed_floor",
                description="Minimum injection strength at seed layers.",
                constraints="number.",
            ),
            FieldSpec(
                key="valley_window",
                description="Half-window for compression valley detection.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="top_k_valleys",
                description="Maximum valleys to use as seed layers.",
                constraints="integer >= 1.",
            ),
        ),
    ),
    SectionSpec(
        name="awareness",
        description="Steering awareness detection via Jacobian sensitivity comparison.",
        early_return=True,
        config_class="AwarenessConfig",
        fields=(
            FieldSpec(
                key="prompts",
                description="Inline prompts to test for steering.",
                constraints="required non-empty list of strings.",
                required=True,
            ),
            FieldSpec(
                key="calibration_prompt",
                description="Benign prompt for baseline calibration.",
                constraints="string.",
            ),
            FieldSpec(
                key="mode",
                description=(
                    "Detection mode: 'fast' (gain-only)"
                    " or 'full' (gain+rank+correlation)."
                ),
                constraints="'fast' or 'full'.",
            ),
            FieldSpec(
                key="gain_ratio_threshold",
                description="Flag layer if test/baseline gain exceeds this.",
                constraints="number > 0.",
            ),
            FieldSpec(
                key="rank_ratio_threshold",
                description="Flag layer if test/baseline rank drops below this.",
                constraints="number > 0.",
            ),
            FieldSpec(
                key="correlation_delta_threshold",
                description="Flag layer if |correlation delta| exceeds this.",
                constraints="number > 0.",
            ),
            FieldSpec(
                key="min_anomalous_layers",
                description="Minimum anomalous layers to declare steered.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="confidence_threshold",
                description="Minimum confidence score to declare steered.",
                constraints="number >= 0.",
            ),
            FieldSpec(
                key="n_power_iterations",
                description="Power iterations for dominant singular value estimation.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="fd_epsilon",
                description="Finite-difference step size for Jacobian approximation.",
                constraints="number > 0.",
            ),
            FieldSpec(
                key="valley_window",
                description="Half-window for compression valley detection.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="top_k_valleys",
                description="Maximum valleys to identify as anomalous layers.",
                constraints="integer >= 1.",
            ),
        ),
    ),
    SectionSpec(
        name="cast",
        description="Conditional activation steering (CAST) generation.",
        early_return=True,
        config_class="CastConfig",
        fields=(
            FieldSpec(
                key="prompts",
                description="Inline prompts used for CAST generation.",
                constraints="required non-empty list of strings.",
                required=True,
            ),
            FieldSpec(
                key="layers",
                description="Layer subset to conditionally steer.",
                constraints="list of integers or null (null means all layers).",
            ),
            FieldSpec(
                key="alpha",
                description="Steering strength when CAST triggers.",
                constraints="number.",
            ),
            FieldSpec(
                key="threshold",
                description="Intervention trigger threshold on projection value.",
                constraints="number (steer if projection > threshold).",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generation cap per prompt.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="condition_direction",
                attr="condition_direction_path",
                description="Path to a separate direction .npy for gating.",
                constraints="string path or null.",
                notes=(
                    "When set, this direction is used to decide whether to"
                    " steer (detect), while the primary direction is used"
                    " for the actual correction (steer). Implements the"
                    " dual-direction pattern from AdaSteer.",
                ),
            ),
            FieldSpec(
                key="alpha_tiers",
                description="Adaptive alpha tiers based on projection magnitude.",
                constraints=(
                    "list of tables with 'threshold' and 'alpha' keys;"
                    " must be sorted by ascending threshold."
                ),
                notes=(
                    "Each tier defines a projection threshold and the alpha"
                    " to use when projection >= that threshold. The highest"
                    " matching tier wins. Implements TRYLOCK-style adaptive"
                    " steering.",
                ),
            ),
            FieldSpec(
                key="direction_source",
                description="Source for the steering direction.",
                constraints='one of: "linear", "svf".',
            ),
            FieldSpec(
                key="svf_boundary_path",
                description="Path to trained SVF boundary weights.",
                constraints="string path or null.",
            ),
            FieldSpec(
                key="bank_path",
                description="Path to a Steer2Adapt direction bank .npz file.",
                constraints="string path or null.",
            ),
            FieldSpec(
                key="composition",
                description="Named direction weights for multi-direction composition.",
                constraints="table mapping direction names to float weights.",
            ),
            FieldSpec(
                key="externality_monitor",
                description="Enable steering externality monitoring during generation.",
                constraints="boolean.",
            ),
            FieldSpec(
                key="displacement_threshold",
                description="Displacement threshold for externality alerts.",
                constraints="number >= 0.",
            ),
            FieldSpec(
                key="baseline_activations_path",
                description=(
                    "Path to baseline activations .npy"
                    " for displacement comparison."
                ),
                constraints="string path or null.",
            ),
        ),
    ),
    SectionSpec(
        name="audit",
        description="Automated red-team safety assessment with PDF report.",
        early_return=True,
        config_class="AuditConfig",
        fields=(
            FieldSpec(
                key="company_name",
                description="Company name for the report header.",
                constraints="required string.",
                required=True,
            ),
            FieldSpec(
                key="system_name",
                description="AI system name for the report header.",
                constraints="required string.",
                required=True,
            ),
            FieldSpec(
                key="thoroughness",
                description="Assessment depth level.",
                constraints='one of: "quick", "standard", "deep".',
            ),
            FieldSpec(
                key="pdf_report",
                description="Generate PDF audit report.",
                constraints="boolean.",
            ),
            FieldSpec(
                key="pdf_report_filename",
                description="Custom PDF filename.",
                constraints="string ending with .pdf.",
            ),
            FieldSpec(
                key="attacks",
                description="Override which attack methods to run.",
                constraints="list of strings or null.",
            ),
            FieldSpec(
                key="softprompt_steps",
                description="Override GCG optimization steps.",
                constraints="integer >= 1 or null.",
            ),
            FieldSpec(
                key="jailbreak_strategies",
                description="Restrict jailbreak templates to specific strategies.",
                constraints="list of strings or null.",
            ),
        ),
    ),
    SectionSpec(
        name="guard",
        description="Runtime circuit breaker with tiered response and KV cache rewind.",
        early_return=True,
        config_class="GuardConfig",
        fields=(
            FieldSpec(
                key="prompts",
                description="Prompts to generate with guard monitoring.",
                constraints="required non-empty list of strings.",
                required=True,
            ),
            FieldSpec(
                key="layers",
                description="Layer subset to monitor.",
                constraints="list of integers or null (null means all layers).",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generation cap per prompt.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="tiers",
                description="Zone tier definitions sorted by ascending threshold.",
                constraints=(
                    "list of tables with 'threshold', 'zone', and 'alpha' keys."
                    " Zones: green, yellow, orange, red."
                ),
            ),
            FieldSpec(
                key="max_rewinds",
                description="Maximum rewind attempts before circuit break.",
                constraints="integer >= 0.",
            ),
            FieldSpec(
                key="checkpoint_interval",
                description="Checkpoint KV cache every N consecutive green tokens.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="calibrate",
                description="Auto-calibrate tier thresholds from clean prompts.",
                constraints="boolean.",
            ),
            FieldSpec(
                key="calibrate_prompts",
                description="Which prompt set to calibrate from.",
                constraints='one of: "harmless", "harmful".',
            ),
            FieldSpec(
                key="defensive_prompt",
                description="Text prepended on rewind for safety steering.",
                constraints="string or null.",
            ),
            FieldSpec(
                key="defensive_embeddings_path",
                description="Pre-computed defensive embeddings (.npy).",
                constraints="string path or null.",
            ),
            FieldSpec(
                key="condition_direction",
                attr="condition_direction_path",
                description="Separate detection direction (AdaSteer dual-direction).",
                constraints="string path or null.",
            ),
        ),
    ),
    SectionSpec(
        name="defend",
        description="Composed defense stack that layers scan, SIC, policy, and intent.",
        early_return=True,
        config_class="DefenseStackConfig",
        fields=(
            FieldSpec(
                key="fail_fast",
                description="Stop at the first layer that blocks.",
                constraints="boolean.",
            ),
            FieldSpec(
                key="scan",
                description="Inline scan sub-config (populated from [scan] section).",
                constraints="populated automatically from [scan] section if present.",
            ),
            FieldSpec(
                key="sic",
                description="Inline SIC sub-config (populated from [sic] section).",
                constraints="populated automatically from [sic] section if present.",
            ),
            FieldSpec(
                key="policy",
                description=(
                    "Inline policy sub-config"
                    " (populated from [policy] section)."
                ),
                constraints=(
                    "populated automatically from"
                    " [policy] section if present."
                ),
            ),
            FieldSpec(
                key="intent",
                description=(
                    "Inline intent sub-config"
                    " (populated from [intent] section)."
                ),
                constraints=(
                    "populated automatically from"
                    " [intent] section if present."
                ),
            ),
            FieldSpec(
                key="perturb",
                description=(
                    "Input perturbation sub-config"
                    " (populated from [perturb] section)."
                ),
                constraints=(
                    "populated automatically from"
                    " [perturb] section if present."
                ),
            ),
        ),
        notes=(
            (
                "The [defend] section composes [scan], [sic], [policy],"
                " [intent], and [perturb] sections. Define those sections"
                " alongside [defend] to configure each defense layer."
            ),
        ),
    ),
    SectionSpec(
        name="jailbreak",
        description="Evaluate defenses against known jailbreak prompt strategies.",
        early_return=True,
        config_class="JailbreakConfig",
        fields=(
            FieldSpec(
                key="strategies",
                description="Strategies to test (empty = all 5 strategies).",
                constraints=(
                    "list of strings from: identity_dissolution,"
                    " boundary_exploit, semantic_inversion,"
                    " dual_response, competitive_pressure."
                ),
            ),
            FieldSpec(
                key="custom_templates_path",
                description="Path to custom templates JSONL.",
                constraints="optional string path.",
            ),
            FieldSpec(
                key="payloads_from",
                description="Source of payload prompts.",
                constraints='"harmful" (default) or path to JSONL file.',
            ),
        ),
        notes=(
            (
                "Feeds cross-product of jailbreak templates x payloads"
                " through the defense stack. Reports block rates per strategy."
            ),
        ),
    ),
    SectionSpec(
        name="environment",
        description="Agent simulation harness for indirect prompt injection testing.",
        config_class="EnvironmentConfig",
        fields=(
            FieldSpec(
                key="scenario",
                description="Built-in benchmark scenario name.",
                constraints=(
                    "string or null; seeds environment defaults from a named"
                    " scenario."
                ),
            ),
            FieldSpec(
                key="system_prompt",
                description="System prompt defining the agent persona.",
                constraints="required string.",
                required=True,
            ),
            FieldSpec(
                key="injection_surface",
                description="Name of the tool whose output carries the injection.",
                constraints="required string; must match a defined tool name.",
                required=True,
            ),
            FieldSpec(
                key="max_turns",
                description="Maximum agent conversation turns.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="max_gen_tokens",
                description="Generation cap per agent turn.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="rollout_top_n",
                description="Top-N candidates evaluated via environment rollout.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="rollout_every_n",
                description="Run environment rollout every N optimization steps.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="temperature",
                description="Sampling temperature for agent generation.",
                constraints="number >= 0.0.",
            ),
            FieldSpec(
                key="tools",
                description="Tool schemas available to the simulated agent.",
                constraints=(
                    "required list of [[environment.tools]]"
                    " sub-tables."
                ),
                required=True,
                notes=(
                    "Each [[environment.tools]] table has:"
                    " name (string, required),"
                    " description (string, required),"
                    " parameters (table mapping param names"
                    " to type strings, required),"
                    " result (string, optional —"
                    " return type description).",
                ),
            ),
            FieldSpec(
                key="target",
                description=(
                    "Target action the injection payload"
                    " tries to trigger."
                ),
                constraints=(
                    "required [environment.target] sub-table."
                ),
                required=True,
                notes=(
                    "[environment.target] table has:"
                    " function (string, required —"
                    " tool name to call),"
                    " required_args (list of strings,"
                    " optional),"
                    " arg_contains (table mapping arg names"
                    " to expected substrings, optional).",
                ),
            ),
            FieldSpec(
                key="task",
                description=(
                    "Benign task the agent should complete"
                    " without the injection."
                ),
                constraints=(
                    "required [environment.task] sub-table."
                ),
                required=True,
                notes=(
                    "[environment.task] table has:"
                    " content (string, required —"
                    " the user message that starts the"
                    " agent loop).",
                ),
            ),
            FieldSpec(
                key="injection_position",
                description="Where the injection payload appears in tool output.",
                constraints='one of: "prefix", "suffix", "infix".',
            ),
            FieldSpec(
                key="benign_expected_tools",
                description="Tool names expected during benign (non-injected) runs.",
                constraints="list of strings.",
            ),
            FieldSpec(
                key="policy",
                description="Tool call policy for gating dangerous actions.",
                constraints="[environment.policy] sub-table or null.",
            ),
        ),
        notes=(
            (
                "Set [environment].scenario to load a built-in benchmark."
                " Explicit fields in [environment] and its sub-tables"
                " override the scenario defaults."
            ),
            (
                "Without a scenario, define [[environment.tools]],"
                " [environment.target], and [environment.task] sub-tables to"
                " configure the agent tools, target action, and benign task."
            ),
        ),
    ),
    SectionSpec(
        name="svf",
        description="Steering Vector Field boundary MLP training.",
        early_return=True,
        config_class="SVFConfig",
        fields=(
            FieldSpec(
                key="prompts_target",
                description="Path to JSONL prompts for the target behavior.",
                constraints="required string path.",
                required=True,
            ),
            FieldSpec(
                key="prompts_opposite",
                description="Path to JSONL prompts for the opposite behavior.",
                constraints="required string path.",
                required=True,
            ),
            FieldSpec(
                key="projection_dim",
                description="Dimensionality of the learned projection.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="hidden_dim",
                description="Hidden layer size in the boundary MLP.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="n_epochs",
                description="Training epochs.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="learning_rate",
                description="Optimizer learning rate.",
                constraints="number > 0.",
            ),
            FieldSpec(
                key="layers",
                description="Layer subset to train SVF on.",
                constraints="list of integers or null (null means all layers).",
            ),
        ),
        notes=(
            (
                "Reference: Li, Li & Huang (2026) — Steering Vector Fields."
                " Learns a differentiable boundary MLP whose gradient gives"
                " a context-dependent steering direction at each activation."
            ),
        ),
    ),
    SectionSpec(
        name="compose_optimize",
        description=(
            "Bayesian optimization of Steer2Adapt composition weights"
            " over a subspace bank."
        ),
        early_return=True,
        config_class="ComposeOptimizeConfig",
        fields=(
            FieldSpec(
                key="bank_path",
                description="Path to the subspace bank TOML file.",
                constraints="required string path.",
                required=True,
            ),
            FieldSpec(
                key="n_trials",
                description="Number of Optuna trials.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generation cap for scoring each trial.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="timeout",
                description="Optional wall-clock timeout in seconds.",
                constraints="number or null.",
            ),
            FieldSpec(
                key="seed",
                description="Optional optimizer seed.",
                constraints="integer or null.",
            ),
        ),
        notes=(
            (
                "Reference: Han et al. (2026) — Steer2Adapt."
                " Optimizes linear combination weights over a bank of"
                " precomputed semantic subspaces."
            ),
        ),
    ),
    SectionSpec(
        name="scan",
        description="Injection content scanner using per-token projection analysis.",
        config_class="ScanConfig",
        fields=(
            FieldSpec(
                key="target_layer",
                description="Layer override for direction projection.",
                constraints="integer or null.",
            ),
            FieldSpec(
                key="span_threshold",
                description="Minimum mean projection for a token span to be flagged.",
                constraints="number.",
            ),
            FieldSpec(
                key="threshold",
                description="Overall detection threshold.",
                constraints="number.",
            ),
            FieldSpec(
                key="calibrate",
                description="Auto-calibrate threshold from prompt samples.",
                constraints="boolean.",
            ),
        ),
        notes=(
            "Used as a sub-layer of [defend] or standalone for injection scanning.",
        ),
    ),
    SectionSpec(
        name="policy",
        description="Tool-call policy engine with rules, data flow, and rate limits.",
        config_class="PolicyConfig",
        fields=(
            FieldSpec(
                key="default_action",
                description="Default action when no rule matches.",
                constraints='one of: "allow", "block".',
            ),
            FieldSpec(
                key="rules",
                description="List of tool-call filtering rules.",
                constraints="list of tables with name, action, tool_pattern keys.",
            ),
            FieldSpec(
                key="data_flow_rules",
                description="Rules restricting data flow between tools.",
                constraints=(
                    "list of tables with source_tool, source_labels,"
                    " blocked_targets keys."
                ),
            ),
            FieldSpec(
                key="rate_limits",
                description="Rate limits for tool invocations.",
                constraints=(
                    "list of tables with tool_pattern, max_calls,"
                    " window_seconds keys."
                ),
            ),
        ),
        notes=(
            "Used as a sub-layer of [defend] or standalone for policy enforcement.",
        ),
    ),
    SectionSpec(
        name="intent",
        description=(
            "Intent alignment checking between user request"
            " and proposed action."
        ),
        config_class="IntentConfig",
        fields=(
            FieldSpec(
                key="mode",
                description="Alignment detection method.",
                constraints='one of: "embedding", "judge".',
            ),
            FieldSpec(
                key="target_layer",
                description="Layer for embedding-mode cosine similarity.",
                constraints="integer or null.",
            ),
            FieldSpec(
                key="similarity_threshold",
                description="Cosine similarity threshold for alignment.",
                constraints="number in [0.0, 1.0].",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generation cap for judge mode.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="judge_prompt",
                description=(
                    "System prompt used in judge mode"
                    " for alignment classification."
                ),
                constraints="string.",
            ),
        ),
        notes=(
            "Used as a sub-layer of [defend] or standalone for intent verification.",
        ),
    ),
    SectionSpec(
        name="circuit",
        description="Causal circuit tracing via activation patching.",
        early_return=True,
        config_class="CircuitConfig",
        fields=(
            FieldSpec(
                key="clean_prompts",
                description="Harmless prompts for the clean forward pass.",
                constraints="required non-empty list of strings.",
                required=True,
            ),
            FieldSpec(
                key="corrupt_prompts",
                description="Harmful prompts for the corrupt forward pass.",
                constraints=(
                    "required non-empty list of strings;"
                    " must have same length as clean_prompts."
                ),
                required=True,
            ),
            FieldSpec(
                key="metric",
                description="Effect metric for patching.",
                constraints='one of: "kl", "logit_diff".',
                default_override='"kl"',
            ),
            FieldSpec(
                key="granularity",
                description="Patching granularity.",
                constraints='one of: "layer", "component".',
                default_override='"layer"',
            ),
            FieldSpec(
                key="layers",
                description="Specific layers to trace.",
                constraints="list of non-negative integers or null (all layers).",
            ),
            FieldSpec(
                key="token_position",
                description="Token index for metric computation.",
                constraints="integer.",
                default_override="-1",
            ),
            FieldSpec(
                key="attribute_direction",
                description="Compute direction attribution per component.",
                constraints="boolean.",
                default_override="false",
            ),
            FieldSpec(
                key="logit_diff_tokens",
                description="Target token IDs for logit_diff metric.",
                constraints=(
                    "non-empty list of integers;"
                    " required when metric = 'logit_diff'."
                ),
            ),
        ),
        notes=(
            (
                "Implements causal tracing: patches clean activations"
                " into a corrupt forward pass and measures the effect"
                " on output at each layer or component (attn/mlp)."
            ),
        ),
    ),
    SectionSpec(
        name="features",
        description="Sparse autoencoder training for feature decomposition.",
        early_return=True,
        config_class="FeaturesConfig",
        fields=(
            FieldSpec(
                key="prompts_path",
                description="JSONL file of prompts for activation collection.",
                constraints="required string path.",
                required=True,
            ),
            FieldSpec(
                key="layers",
                description="Layer indices to train SAEs on.",
                constraints="required non-empty list of non-negative integers.",
                required=True,
            ),
            FieldSpec(
                key="d_sae",
                description="Dictionary size (number of SAE features).",
                constraints="integer >= 1.",
                default_override="2048",
            ),
            FieldSpec(
                key="l1_coeff",
                description="L1 sparsity coefficient.",
                constraints="number >= 0.",
                default_override="0.001",
            ),
            FieldSpec(
                key="n_epochs",
                description="Training epochs per SAE.",
                constraints="integer >= 1.",
                default_override="5",
            ),
            FieldSpec(
                key="learning_rate",
                description="Adam learning rate.",
                constraints="number > 0.",
                default_override="0.001",
            ),
            FieldSpec(
                key="batch_size",
                description="Mini-batch size for training.",
                constraints="integer >= 1.",
                default_override="32",
            ),
            FieldSpec(
                key="token_position",
                description="Token index for activation extraction.",
                constraints="integer.",
                default_override="-1",
            ),
            FieldSpec(
                key="dead_feature_threshold",
                description="Max activation below which a feature is considered dead.",
                constraints="number >= 0.",
                default_override="1e-6",
            ),
        ),
        notes=(
            (
                "Trains per-layer sparse autoencoders on residual stream"
                " activations. If a refusal direction is available from"
                " [measure], computes cross-lens direction alignment"
                " for each decoder feature."
            ),
        ),
    ),
    SectionSpec(
        name="linear_probe",
        description="Train linear probes to measure refusal encoding per layer.",
        early_return=True,
        config_class="LinearProbeConfig",
        fields=(
            FieldSpec(
                key="layers",
                description="Layer indices to train probes on.",
                constraints="required non-empty list of non-negative integers.",
                required=True,
            ),
            FieldSpec(
                key="n_epochs",
                description="Training epochs per probe.",
                constraints="integer >= 1.",
                default_override="20",
            ),
            FieldSpec(
                key="learning_rate",
                description="Learning rate for probe training.",
                constraints="number > 0.",
                default_override="0.01",
            ),
            FieldSpec(
                key="batch_size",
                description="Mini-batch size.",
                constraints="integer >= 1.",
                default_override="32",
            ),
            FieldSpec(
                key="token_position",
                description="Token index for activation extraction.",
                constraints="integer.",
                default_override="-1",
            ),
            FieldSpec(
                key="regularization",
                description="L2 regularization coefficient.",
                constraints="number >= 0.",
                default_override="0.0001",
            ),
        ),
        notes=(
            (
                "Trains a binary linear classifier at each layer to predict"
                " harmful vs. harmless. High accuracy layers are where refusal"
                " is most linearly separable."
            ),
        ),
    ),
    SectionSpec(
        name="fusion",
        description=(
            "Latent fusion jailbreak via blending harmful"
            " and benign hidden states."
        ),
        early_return=True,
        config_class="FusionConfig",
        fields=(
            FieldSpec(
                key="harmful_prompts",
                description="Harmful prompts whose hidden states are fused.",
                constraints="required non-empty list of strings.",
                required=True,
            ),
            FieldSpec(
                key="benign_prompts",
                description="Benign prompts whose hidden states are blended in.",
                constraints="required non-empty list of strings.",
                required=True,
            ),
            FieldSpec(
                key="layer",
                description="Layer at which to fuse hidden states (-1 = last).",
                constraints="integer.",
                default_override="-1",
            ),
            FieldSpec(
                key="alpha",
                description="Interpolation weight (0 = pure benign, 1 = pure harmful).",
                constraints="number in [0.0, 1.0].",
                default_override="0.5",
            ),
            FieldSpec(
                key="n_tokens",
                description="Max tokens to generate after fusion.",
                constraints="integer >= 1.",
                default_override="128",
            ),
            FieldSpec(
                key="temperature",
                description="Sampling temperature.",
                constraints="number >= 0.",
                default_override="0.7",
            ),
        ),
        notes=(
            (
                "Implements the latent fusion attack: blends hidden states of"
                " harmful and benign prompts in continuous latent space, then"
                " generates from the fused representation. The prompt-side"
                " dual of abliteration."
            ),
        ),
    ),
    SectionSpec(
        name="repbend",
        description="RepBend contrastive fine-tuning for safety hardening.",
        early_return=True,
        config_class="RepBendConfig",
        fields=(
            FieldSpec(
                key="layers",
                description="Layer indices for contrastive separation.",
                constraints="required non-empty list of non-negative integers.",
                required=True,
            ),
            FieldSpec(
                key="n_epochs",
                description="Training epochs.",
                constraints="integer >= 1.",
                default_override="3",
            ),
            FieldSpec(
                key="learning_rate",
                description="Learning rate.",
                constraints="number > 0.",
                default_override="1e-5",
            ),
            FieldSpec(
                key="batch_size",
                description="Mini-batch size.",
                constraints="integer >= 1.",
                default_override="8",
            ),
            FieldSpec(
                key="separation_coeff",
                description="Contrastive separation loss coefficient.",
                constraints="number >= 0.",
                default_override="1.0",
            ),
            FieldSpec(
                key="token_position",
                description="Token index for activation extraction.",
                constraints="integer.",
                default_override="-1",
            ),
        ),
        notes=(
            (
                "The defense dual of abliteration. Pushes harmful activations"
                " apart from safe ones via contrastive loss, hardening the"
                " model against direction-removal attacks."
            ),
        ),
    ),
    SectionSpec(
        name="lora_export",
        description="Export measured direction as a LoRA adapter.",
        early_return=True,
        config_class="LoraExportConfig",
        fields=(
            FieldSpec(
                key="format",
                description="Adapter format.",
                constraints='"mlx" or "peft".',
                default_override='"mlx"',
            ),
            FieldSpec(
                key="polarity",
                description="Direction polarity.",
                constraints='"remove" or "add".',
                default_override='"remove"',
            ),
        ),
        notes=(
            (
                "Exports the measured direction as a LoRA adapter."
                " Layer selection, alpha, and sparsity come from [cut]."
                " Use polarity = \"add\" for trait amplification."
            ),
            (
                "Orthogonalization: if [cut].biprojected = true, the direction"
                " is Gram-Schmidt orthogonalized against the harmless direction."
                " If [cut].false_refusal_ortho = true and borderline_path is set,"
                " it is orthogonalized against the borderline direction instead."
                " [cut].norm_preserve is ignored (incompatible with low-rank LoRA)."
            ),
            "See also: [lora] (load adapters), [lora_analysis] (inspect adapters).",
        ),
    ),
    SectionSpec(
        name="lora",
        description="Load LoRA adapter(s) into the model before pipeline phases.",
        config_class="LoraLoadConfig",
        fields=(
            FieldSpec(
                key="adapter_path",
                description="Path to a single adapter directory.",
                constraints="string; mutually exclusive with adapter_paths.",
            ),
            FieldSpec(
                key="adapter_paths",
                description="Paths to multiple adapters for task arithmetic merge.",
                constraints="list of strings; mutually exclusive with adapter_path.",
            ),
            FieldSpec(
                key="weights",
                description="Scalar weight for each adapter in adapter_paths.",
                constraints="list of numbers; length must match adapter_paths.",
            ),
        ),
        notes=(
            (
                "Infrastructure section: loads adapter(s) into the model"
                " before any pipeline phase. Any existing mode then works"
                " transparently on the LoRA'd model."
            ),
            (
                "See also: [lora_export] (create adapters),"
                " [lora_analysis] (inspect adapters)."
            ),
        ),
    ),
    SectionSpec(
        name="lora_analysis",
        description="Decompose LoRA adapters via SVD for structural analysis.",
        early_return=True,
        config_class="LoraAnalysisConfig",
        fields=(
            FieldSpec(
                key="adapter_path",
                description="Path to a single adapter directory to analyze.",
                constraints="string; mutually exclusive with adapter_paths.",
            ),
            FieldSpec(
                key="adapter_paths",
                description="Paths to multiple adapters to analyze.",
                constraints="list of strings; mutually exclusive with adapter_path.",
            ),
            FieldSpec(
                key="variance_threshold",
                description="Cumulative variance threshold for rank cutoff.",
                constraints="number in (0.0, 1.0].",
                default_override="0.99",
            ),
            FieldSpec(
                key="align_with_direction",
                description="Compute alignment with measured refusal direction.",
                constraints="boolean.",
                default_override="true",
            ),
        ),
        notes=(
            (
                "Decomposes adapter weight pairs via SVD to report"
                " per-layer effective rank, Frobenius norm profile,"
                " and optional alignment with the measured direction."
            ),
            (
                "See also: [lora_export] (create adapters),"
                " [lora] (load adapters)."
            ),
        ),
    ),
    SectionSpec(
        name="api_eval",
        description=(
            "Test optimized suffixes against remote"
            " OpenAI-compatible endpoints."
        ),
        config_class="ApiEvalConfig",
        fields=(
            FieldSpec(
                key="endpoints",
                description="List of API endpoints to evaluate against.",
                constraints="required non-empty list of endpoint tables.",
                required=True,
                notes=(
                    "Each [[api_eval.endpoints]] table has:"
                    " name (string, required),"
                    " base_url (string, required),"
                    " model (string, required),"
                    " api_key_env (string, required —"
                    " env var holding the API key),"
                    " system_prompt (string, optional"
                    " per-endpoint override),"
                    " auth_header (string, optional"
                    " custom header name).",
                ),
            ),
            FieldSpec(
                key="max_tokens",
                description="Max tokens for API response.",
                constraints="integer >= 1.",
                default_override="100",
            ),
            FieldSpec(
                key="timeout",
                description="HTTP request timeout in seconds.",
                constraints="integer >= 1.",
                default_override="30",
            ),
            FieldSpec(
                key="system_prompt",
                description=(
                    "Shared system prompt for all endpoints"
                    " (per-endpoint overrides take precedence)."
                ),
                constraints="string or null.",
            ),
            FieldSpec(
                key="multiturn",
                description="Enable multi-turn conversation evaluation.",
                constraints="boolean.",
                default_override="false",
            ),
            FieldSpec(
                key="multiturn_max_turns",
                description="Total conversation turns including initial.",
                constraints="integer >= 1.",
                default_override="3",
            ),
            FieldSpec(
                key="follow_up_prompts",
                description="Follow-up prompts for multi-turn mode.",
                constraints="list of strings.",
                default_override="[]",
            ),
            FieldSpec(
                key="token_text",
                description=(
                    "Pre-optimized adversarial token text"
                    " for standalone evaluation."
                ),
                constraints="string or null.",
                notes=(
                    "When set, enables standalone [api_eval] mode that needs"
                    " no local model — skips [model] and [data] requirements.",
                ),
            ),
            FieldSpec(
                key="token_position",
                description="Where to inject the adversarial token text.",
                constraints='one of: "prefix", "suffix", "infix".',
            ),
            FieldSpec(
                key="prompts",
                description="Test prompts to pair with the adversarial tokens.",
                constraints="list of strings.",
            ),
            FieldSpec(
                key="defense_proxy",
                description="Local defense proxy to test before sending to remote API.",
                constraints='one of: "sic", "cast", "both", or null.',
            ),
            FieldSpec(
                key="defense_proxy_sic_mode",
                description="SIC scoring mode for defense proxy.",
                constraints='one of: "direction", "generation", "svf".',
            ),
            FieldSpec(
                key="defense_proxy_sic_threshold",
                description="SIC detection threshold for defense proxy.",
                constraints="number.",
            ),
            FieldSpec(
                key="defense_proxy_sic_max_iterations",
                description="Maximum SIC sanitization iterations.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="defense_proxy_cast_mode",
                description="CAST proxy mode.",
                constraints=(
                    'one of: "gate" (detection only),'
                    ' "full" (steer + generate).'
                ),
            ),
            FieldSpec(
                key="defense_proxy_cast_threshold",
                description="CAST projection threshold for defense proxy.",
                constraints="number.",
            ),
            FieldSpec(
                key="defense_proxy_cast_layers",
                description="CAST layer subset for defense proxy.",
                constraints="list of integers or null.",
            ),
            FieldSpec(
                key="defense_proxy_cast_alpha",
                description="CAST steering alpha for defense proxy.",
                constraints="number.",
            ),
            FieldSpec(
                key="defense_proxy_cast_max_tokens",
                description="Generation cap for CAST defense proxy.",
                constraints="integer >= 1.",
            ),
        ),
        notes=(
            (
                "Runs after [softprompt] to test optimized suffixes against"
                " remote API endpoints. Each endpoint requires name, base_url,"
                " model, and api_key_env (environment variable holding the key)."
            ),
        ),
    ),
    SectionSpec(
        name="objective",
        description=(
            "Deployment objective contract for deciding whether"
            " an attack-defense run is acceptable."
        ),
        config_class="ObjectiveConfig",
        fields=(
            FieldSpec(
                key="name",
                description="Short name for the objective or launch gate.",
                constraints="required non-empty string.",
                required=True,
            ),
            FieldSpec(
                key="deployment",
                description="Deployment profile or workflow being assessed.",
                constraints="string.",
            ),
            FieldSpec(
                key="summary",
                description="Short plain-language statement of the objective.",
                constraints="string.",
            ),
            FieldSpec(
                key="access",
                description="Access level Vauban has to the target system.",
                constraints='one of: "weights", "api", "hybrid", "system".',
            ),
            FieldSpec(
                key="benign_inquiry_source",
                description=(
                    "Where benign utility inquiries come from during"
                    " objective evaluation."
                ),
                constraints='one of: "generated", "dataset".',
            ),
            FieldSpec(
                key="benign_inquiries",
                attr="benign_inquiries_path",
                description=(
                    "JSONL file of benign inquiries used when"
                    ' benign_inquiry_source = "dataset".'
                ),
                constraints=(
                    'string path to JSONL with {"prompt": "..."} lines.'
                ),
            ),
            FieldSpec(
                key="preserve",
                description="Benign capabilities that must be retained.",
                constraints="list of strings.",
            ),
            FieldSpec(
                key="prevent",
                description="Unsafe outcomes or intents that must be resisted.",
                constraints="list of strings.",
            ),
            FieldSpec(
                key="safety",
                description="Quantitative safety thresholds.",
                constraints="list of [[objective.safety]] tables.",
                notes=(
                    "Each [[objective.safety]] table has:"
                    " metric (required),"
                    " threshold (required number),"
                    ' comparison ("at_most" by default),'
                    ' aggregate ("final" by default),'
                    " label (optional),"
                    " description (optional).",
                ),
            ),
            FieldSpec(
                key="utility",
                description="Quantitative benign-utility thresholds.",
                constraints="list of [[objective.utility]] tables.",
                notes=(
                    "Each [[objective.utility]] table has:"
                    " metric (required),"
                    " threshold (required number),"
                    ' comparison ("at_least" by default),'
                    ' aggregate ("final" by default),'
                    " label (optional),"
                    " description (optional).",
                ),
            ),
        ),
        notes=(
            (
                "Current quantitative enforcement is implemented for"
                " [flywheel] runs."
            ),
            (
                'Use benign_inquiry_source = "generated" to score utility'
                " on the generated flywheel worlds, or set"
                ' benign_inquiry_source = "dataset" plus'
                " benign_inquiries to reuse a fixed benign inquiry set."
            ),
            (
                "Supported objective metrics today: attack_success_rate,"
                " defense_block_rate, evasion_rate, utility_score,"
                " cast_block_fraction, sic_block_fraction,"
                " n_new_payloads, n_previous_blocked."
            ),
            (
                "Use [objective] to state what must be preserved"
                " and what must be prevented before you read"
                " ASR and utility numbers as a verdict."
            ),
        ),
    ),
    SectionSpec(
        name="flywheel",
        description=(
            "Closed-loop attack-defense co-evolution flywheel."
        ),
        early_return=True,
        config_class="FlywheelConfig",
        fields=(
            FieldSpec(
                key="n_cycles",
                description="Number of flywheel cycles.",
                constraints="integer >= 1.",
                default_override="10",
            ),
            FieldSpec(
                key="worlds_per_cycle",
                description="Agent worlds generated per cycle.",
                constraints="integer >= 1.",
                default_override="50",
            ),
            FieldSpec(
                key="skeletons",
                description="Skeleton domains for world generation.",
                constraints=(
                    "non-empty list of strings from:"
                    " email, doc, code, calendar, search,"
                    " home_assistant, drive_share, landing_review."
                ),
            ),
            FieldSpec(
                key="harden",
                description=(
                    "Whether to adapt defense parameters after"
                    " each cycle."
                ),
                constraints="boolean.",
                default_override="true",
            ),
            FieldSpec(
                key="convergence_window",
                description=(
                    "Number of recent cycles to check for"
                    " convergence."
                ),
                constraints="integer >= 2.",
                default_override="3",
            ),
            FieldSpec(
                key="payloads_per_world",
                description="Number of injection payloads tested per agent world.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="model_expand",
                description=(
                    "Use the model to expand skeleton scenarios"
                    " into full worlds."
                ),
                constraints="boolean.",
            ),
            FieldSpec(
                key="expand_temperature",
                description="Sampling temperature for world expansion generation.",
                constraints="number >= 0.",
            ),
            FieldSpec(
                key="expand_max_tokens",
                description="Generation cap for world expansion.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="difficulty_range",
                description="Min and max difficulty levels for generated worlds.",
                constraints="list of two integers [min, max].",
            ),
            FieldSpec(
                key="payload_library_path",
                description="Path to a JSONL file of pre-defined injection payloads.",
                constraints="string path or null.",
            ),
            FieldSpec(
                key="positions",
                description="Injection positions to test.",
                constraints='list of strings from: "prefix", "suffix", "infix".',
            ),
            FieldSpec(
                key="warmstart_gcg",
                description="Warm-start GCG from previous cycle's best tokens.",
                constraints="boolean.",
            ),
            FieldSpec(
                key="gcg_steps",
                description="GCG optimization steps per payload.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="gcg_n_tokens",
                description="Number of GCG adversarial tokens.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="cast_alpha",
                description="Initial CAST steering strength.",
                constraints="number.",
            ),
            FieldSpec(
                key="cast_threshold",
                description="Initial CAST intervention threshold.",
                constraints="number.",
            ),
            FieldSpec(
                key="cast_layers",
                description="CAST layer subset.",
                constraints="list of integers or null.",
            ),
            FieldSpec(
                key="sic_threshold",
                description="Initial SIC detection threshold.",
                constraints="number.",
            ),
            FieldSpec(
                key="sic_iterations",
                description="Initial SIC maximum sanitization iterations.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="sic_mode",
                description="SIC scoring mode.",
                constraints='one of: "direction", "generation", "svf".',
            ),
            FieldSpec(
                key="adaptation_rate",
                description="Learning rate for defense parameter hardening.",
                constraints="number > 0.",
            ),
            FieldSpec(
                key="utility_floor",
                description=(
                    "Minimum utility score; hardening stops"
                    " if utility drops below."
                ),
                constraints="number in [0, 1].",
            ),
            FieldSpec(
                key="validate_previous",
                description="Re-validate previously blocked payloads after hardening.",
                constraints="boolean.",
            ),
            FieldSpec(
                key="convergence_threshold",
                description="Evasion rate delta below which convergence is declared.",
                constraints="number > 0.",
            ),
            FieldSpec(
                key="seed",
                description="Random seed for reproducibility.",
                constraints="integer or null.",
            ),
            FieldSpec(
                key="max_turns",
                description="Maximum agent conversation turns per scenario.",
                constraints="integer >= 1.",
            ),
            FieldSpec(
                key="max_gen_tokens",
                description="Generation cap per agent turn.",
                constraints="integer >= 1.",
            ),
        ),
        notes=(
            (
                "Generates synthetic agent scenarios, attacks"
                " them with injection payloads, evaluates"
                " CAST+SIC defenses, hardens parameters from"
                " failures, and measures convergence."
            ),
        ),
    ),
    SectionSpec(
        name="meta",
        description="Experiment metadata for lineage tracking.",
        config_class="MetaConfig",
        fields=(
            FieldSpec(
                key="id",
                description="Unique experiment identifier.",
                constraints="required string.",
                required=True,
            ),
            FieldSpec(
                key="title",
                description="Human-readable experiment title.",
                constraints="string.",
            ),
            FieldSpec(
                key="status",
                description="Experiment status.",
                constraints=(
                    'one of: "wip", "running", "done",'
                    ' "failed", "archived".'
                ),
            ),
            FieldSpec(
                key="parents",
                description=(
                    "IDs of parent experiments"
                    " in the lineage tree."
                ),
                constraints="list of strings.",
            ),
            FieldSpec(
                key="tags",
                description="Freeform tags for filtering.",
                constraints="list of strings.",
            ),
            FieldSpec(
                key="notes",
                description="Freeform experiment notes.",
                constraints="string.",
            ),
            FieldSpec(
                key="docs",
                description=(
                    "References to associated documents."
                ),
                constraints="list of [[meta.docs]] tables.",
                notes=(
                    "Each [[meta.docs]] table has:"
                    " path (string, required),"
                    " label (string, optional).",
                ),
            ),
            FieldSpec(
                key="date",
                description="Experiment date (ISO 8601).",
                constraints="string.",
            ),
        ),
        notes=(
            (
                "Does not affect pipeline execution."
                " Used by vauban tree to render"
                " experiment lineage graphs."
            ),
        ),
    ),
    SectionSpec(
        name="remote",
        description=(
            "Probe remote models via batch inference API."
        ),
        early_return=True,
        config_class="RemoteConfig",
        fields=(
            FieldSpec(
                key="backend",
                description="Inference backend to use.",
                constraints='one of: "jsinfer".',
                required=True,
            ),
            FieldSpec(
                key="api_key_env",
                description=(
                    "Environment variable holding the API key."
                ),
                constraints="required string.",
                required=True,
            ),
            FieldSpec(
                key="models",
                description="Model IDs to probe.",
                constraints="non-empty list of strings.",
                required=True,
            ),
            FieldSpec(
                key="prompts",
                description="Prompts to send to each model.",
                constraints="non-empty list of strings.",
                required=True,
            ),
            FieldSpec(
                key="activations",
                description="Whether to fetch activation tensors.",
                constraints="boolean.",
                default_override="false",
            ),
            FieldSpec(
                key="activation_layers",
                description=(
                    "Layer indices to capture activations from."
                ),
                constraints="list of integers.",
            ),
            FieldSpec(
                key="activation_modules",
                description=(
                    "Module name patterns for activation capture."
                    " Use {layer} as a placeholder."
                ),
                constraints="list of strings.",
            ),
            FieldSpec(
                key="max_tokens",
                description="Generation cap per completion.",
                constraints="integer >= 1.",
                default_override="512",
            ),
            FieldSpec(
                key="timeout",
                description="HTTP request timeout in seconds.",
                constraints="integer >= 1.",
                default_override="600",
            ),
        ),
        notes=(
            (
                "Standalone mode — no local model or [data] section"
                " needed. Sends prompts to remote batch inference"
                " APIs and collects responses plus optional"
                " activation tensors saved as .npy files."
            ),
        ),
    ),
    SectionSpec(
        name="backend",
        description="Compute backend selection.",
        table=False,
        fields=(
            FieldSpec(
                key="backend",
                description="Runtime backend for tensor operations.",
                constraints='one of: "mlx", "pytorch".',
                default_override='"mlx"',
            ),
        ),
        notes=(
            (
                "Top-level key (not a section). Set backend = \"pytorch\""
                " to use PyTorch instead of MLX. MLX is the default"
                " and only fully supported backend on Apple Silicon."
            ),
        ),
    ),
    SectionSpec(
        name="output",
        description="Filesystem output location.",
        fields=(
            FieldSpec(
                key="dir",
                description="Directory where models/reports are written.",
                constraints="string path.",
                default_override='"output"',
            ),
        ),
    ),
    SectionSpec(
        name="verbose",
        description="Top-level logging control.",
        table=False,
        fields=(
            FieldSpec(
                key="verbose",
                description="Enable stderr progress logs.",
                constraints="boolean.",
                default_override="true",
            ),
        ),
    ),
)

_WORKFLOW_SPECS: tuple[
    tuple[str, str, tuple[str, ...], tuple[str, ...]], ...
] = (
    (
        "Discover hidden behaviors",
        (
            "Find whether a model has dormant, suppressed,"
            " or concealed behavioral directions."
        ),
        (
            "1. Measure the behavioral direction:",
            '   [measure]   # mode = "direction" (default)',
            "",
            "2. Probe specific prompts per-layer:",
            "   [probe]",
            '   prompts = ["Explain lockpicking",'
            ' "How to hotwire a car"]',
            "   -> probe_report.json: projection spikes.",
            "",
            "3. Map refusal surface across diverse prompts:",
            "   [surface]",
            '   prompts = "prompts.jsonl"',
            "   -> surface_report.json: per-prompt refusal",
            "      decisions before/after direction removal.",
            "",
            "4. Check for defense hardening:",
            "   [detect]",
            '   mode = "full"',
            "   -> detect_report.json: hardening signals.",
            "",
            "5. Quantify linear decodability per layer:",
            "   [linear_probe]",
            "   -> linear_probe_report.json: per-layer",
            "      classification accuracy.",
        ),
        (
            "Start with probe for quick signal, then surface",
            "for breadth, then linear_probe for depth.",
            "Each step is a separate TOML — run one at a time.",
            "Scaffold: vauban init --mode probe",
        ),
    ),
    (
        "Remove a behavior from weights",
        (
            "Extract a behavioral direction and surgically"
            " remove it from model weights."
        ),
        (
            "The default pipeline does this end-to-end:",
            "   [model]",
            '   path = "mlx-community/Llama-3.2-3B-'
            'Instruct-4bit"',
            "",
            "   [data]",
            '   harmful = "default"',
            '   harmless = "default"',
            "",
            "   [cut]",
            "   alpha = 1.0    # projection removal strength",
            "",
            "   Pipeline: measure -> cut -> eval -> export.",
            "   Output: modified model in [output].dir.",
            "",
            "To search for optimal cut params automatically:",
            "   [optimize]",
            "   n_trials = 50",
            "   -> Pareto front in optimize_report.json.",
        ),
        (
            "Scaffold: vauban init --mode default",
            "Compare: vauban diff output_base output_cut",
        ),
    ),
    (
        "Defend a model at inference time",
        (
            "Add runtime defenses that detect and block"
            " adversarial inputs without modifying weights."
        ),
        (
            "Option A  Conditional steering (CAST):",
            "   [cast]",
            '   prompts = ["Explain lockpicking"]',
            "   alpha = 2.0",
            "   threshold = 0.5",
            "   -> Steers only when projection > threshold.",
            "",
            "Option B  Input sanitization (SIC):",
            "   [sic]",
            '   mode = "direction"',
            "   threshold = 0.5",
            "   calibrate = true",
            "   -> Detects adversarial content, rewrites it.",
            "",
            "Option C  Composed defense stack:",
            "   [defend]",
            "   fail_fast = true",
            "   [scan]     # token-level injection scanner",
            "   [sic]      # input sanitization",
            "   [policy]   # tool-call gating",
            "   [intent]   # intent alignment check",
            "   -> Layers multiple defenses in sequence.",
            "",
            "Option D  Fine-tune safety (RepBend):",
            "   [repbend]",
            "   n_steps = 100",
            "   -> Pushes harmful activations apart from safe.",
        ),
        (
            "CAST + SIC are complementary (negatively correlated).",
            "For max coverage, combine both via [defend].",
            "Scaffold: vauban init --mode cast",
        ),
    ),
    (
        "Test adversarial robustness",
        (
            "Generate adversarial inputs and measure how"
            " well defenses hold up."
        ),
        (
            "Step 1  Optimize adversarial tokens:",
            "   [softprompt]",
            '   mode = "gcg"',
            '   token_position = "infix"  # most effective',
            "   n_tokens = 16",
            "   n_steps = 200",
            '   defense_eval = "cast"  # test CAST inline',
            "",
            "Step 2  Test against remote endpoints:",
            "   [api_eval]",
            '   token_text = "tokens from step 1"',
            "   [[api_eval.endpoints]]",
            '   name = "target"',
            '   base_url = "https://api.example.com/v1"',
            '   model = "gpt-4"',
            '   api_key_env = "OPENAI_API_KEY"',
            "",
            "Closed-loop co-evolution:",
            "   [flywheel]",
            "   n_cycles = 10",
            "   harden = true",
            "   -> Alternates attack and defense hardening",
            "      until convergence.",
        ),
        (
            "Infix position is 3.5x more effective than suffix.",
            "Scaffold: vauban init --mode softprompt",
        ),
    ),
    (
        "Analyze model internals",
        (
            "Deep structural analysis: trace circuits,"
            " decompose features, inspect reasoning."
        ),
        (
            "Causal circuit tracing:",
            "   [circuit]",
            '   clean_prompts = ["What is 2+2?"]',
            '   corrupt_prompts = ["How to pick a lock?"]',
            "   -> circuit_report.json: per-component effects.",
            "",
            "Sparse autoencoder features:",
            "   [features]",
            "   layers = [14, 15, 16]",
            "   -> features_report.json + sae_*.safetensors.",
            "",
            "Deep-thinking token analysis:",
            "   [depth]",
            '   prompts = ["Explain quantum computing"]',
            "   -> depth_report.json: JSD across layers.",
            "",
            "Steering awareness detection:",
            "   [awareness]",
            '   prompts = ["Explain lockpicking"]',
            "   -> awareness_report.json: anomaly scores.",
        ),
        (
            "Each mode runs independently.",
            "Scaffold: vauban init --mode circuit",
        ),
    ),
)

_WORKFLOW_SECTIONS: dict[str, tuple[str, ...]] = {
    "Discover hidden behaviors": (
        "probe", "surface", "detect", "linear_probe",
    ),
    "Remove a behavior from weights": (
        "measure", "cut", "eval", "optimize",
    ),
    "Defend a model at inference time": (
        "guard", "cast", "sic", "defend", "repbend",
    ),
    "Test adversarial robustness": (
        "audit", "softprompt", "api_eval", "flywheel", "jailbreak",
    ),
    "Analyze model internals": (
        "circuit", "features", "depth", "awareness",
    ),
}

_SECTION_TO_WORKFLOW: dict[str, str] = {
    section: title
    for title, sections in _WORKFLOW_SECTIONS.items()
    for section in sections
}

_TOPIC_ALIASES: dict[str, str] = {
    "start": "quickstart",
    "getting-started": "quickstart",
    "getting_started": "quickstart",
    "command": "commands",
    "cmd": "commands",
    "checks": "validate",
    "validation": "validate",
    "lint": "validate",
    "playbooks": "playbook",
    "experiment": "playbook",
    "experiments": "playbook",
    "recipe": "playbook",
    "recipes": "playbook",
    "repl": "quick",
    "python": "quick",
    "quick-api": "quick",
    "quick_api": "quick",
    "example": "examples",
    "demo": "examples",
    "demos": "examples",
    "printing": "print",
    "share": "print",
    "sharing": "print",
    "pdf": "print",
    "mode": "modes",
    "pipeline": "modes",
    "pipelines": "modes",
    "format": "formats",
    "file": "formats",
    "files": "formats",
    "categories": "taxonomy",
    "harm": "taxonomy",
    "harms": "taxonomy",
    "dataset": "datasets",
    "prompts": "datasets",
    "bundled": "datasets",
    "benchmark": "datasets",
    "benchmarks": "datasets",
    "workflow": "workflows",
    "goals": "workflows",
    "howto": "workflows",
    "how-to": "workflows",
    "how_to": "workflows",
    "discover": "workflows",
    "use-case": "workflows",
    "use_case": "workflows",
    "usecases": "workflows",
    "use-cases": "workflows",
    "guide": "workflows",
    "guides": "workflows",
}


def manual_topics() -> list[str]:
    """Return supported manual topics."""
    return [
        "all",
        "quickstart",
        "commands",
        "validate",
        "playbook",
        "quick",
        "examples",
        "print",
        "modes",
        "formats",
        "taxonomy",
        "datasets",
        "workflows",
        *[spec.name for spec in _SECTION_SPECS],
    ]


@lru_cache(maxsize=1)
def _known_init_modes() -> tuple[str, ...]:
    from vauban._init import KNOWN_MODES

    return tuple(sorted(KNOWN_MODES))


@lru_cache(maxsize=1)
def _known_diff_reports() -> tuple[str, ...]:
    from vauban._diff import known_report_filenames

    return known_report_filenames()


@lru_cache(maxsize=1)
def _known_quick_functions() -> tuple[str, ...]:
    import vauban.quick as quick

    names: list[str] = []
    for name in dir(quick):
        if name.startswith("_"):
            continue
        value = getattr(quick, name)
        if inspect.isfunction(value) and value.__module__ == quick.__name__:
            names.append(name)
    return tuple(sorted(names))


_MODE_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Core Pipeline", ("default", "optimize", "lora_export", "lora_analysis")),
    (
        "Runtime Inspection",
        ("behavior_trace", "probe", "steer", "intervention_eval", "depth"),
    ),
    ("Defense", ("guard", "cast", "sic", "defend", "repbend")),
    ("Adversarial", ("softprompt", "fusion", "sss", "flywheel")),
    ("Analysis", (
        "circuit", "features", "linear_probe", "svf",
        "compose_optimize", "awareness",
    )),
    ("External", ("behavior_diff", "api_eval", "remote")),
)

_GENERAL_TOPICS: tuple[tuple[str, str], ...] = (
    ("workflows", "Start here. Goal-oriented guides by use case."),
    ("quickstart", "Getting started with vauban."),
    ("commands", "CLI commands and flags."),
    ("validate", "Config validation workflow."),
    ("playbook", "Experiment playbook and recipes."),
    ("quick", "Python REPL / Jupyter quick API."),
    ("examples", "Runnable examples."),
    ("print", "Printing and sharing manual output."),
    ("modes", "Pipeline mode precedence and triggers."),
    ("formats", "Data file formats (JSONL, etc.)."),
    ("taxonomy", "Canonical harm category taxonomy."),
    ("datasets", "Bundled datasets and benchmarks."),
)

_SECTION_DESCRIPTIONS: dict[str, str] = {
    spec.name: spec.description for spec in _SECTION_SPECS
}


def _render_topic_index() -> str:
    """Render a compact topic index for ``vauban man``."""
    lines: list[str] = []
    lines.append("VAUBAN MANUAL — Topic Index")
    lines.append("")
    lines.append("Usage: vauban man <topic>")
    lines.append("       vauban man all        (full manual)")
    lines.append("")
    lines.append(
        "New here? Run 'vauban man workflows' to pick a goal.",
    )
    lines.append("")

    lines.append("GENERAL TOPICS")
    for name, desc in _GENERAL_TOPICS:
        lines.append(f"  {name:<16s} {desc}")
    lines.append("")

    lines.append("PIPELINE MODES (by category)")
    for category, modes in _MODE_CATEGORIES:
        lines.append(f"  {category}:")
        for mode in modes:
            desc = _SECTION_DESCRIPTIONS.get(mode, "")
            lines.append(f"    {mode:<20s} {desc}")
    lines.append("")

    lines.append("CONFIG SECTIONS")
    for spec in _SECTION_SPECS:
        tag = " (required)" if spec.required else ""
        lines.append(f"  {spec.name:<20s} {spec.description}{tag}")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_manual(topic: str | None = None) -> str:
    """Render a grep-friendly text manual."""
    normalized = _normalize_topic(topic)

    # No topic → show compact topic index instead of full manual
    if normalized is None:
        return _render_topic_index()

    sections = _build_sections()
    selected = _select_sections(sections, normalized)
    include_quickstart = normalized in ("all", "quickstart")
    include_commands = normalized in ("all", "commands")
    include_validate = normalized in ("all", "validate")
    include_playbook = normalized in ("all", "playbook")
    include_quick = normalized in ("all", "quick")
    include_examples = normalized in ("all", "examples")
    include_print = normalized in ("all", "print")
    include_modes = normalized in ("all", "modes")
    include_formats = normalized in ("all", "formats")
    include_taxonomy = normalized in ("all", "taxonomy")
    include_datasets = normalized in ("all", "datasets")
    include_workflows = normalized in ("all", "workflows")

    lines: list[str] = []
    lines.append("VAUBAN(1)")
    lines.append("")
    lines.append("NAME")
    lines.append("    vauban - TOML-first model behavior surgery toolkit")
    lines.append("")
    lines.append("SYNOPSIS")
    lines.append("    vauban <config.toml>")
    lines.append("    vauban --validate <config.toml>")
    lines.append("    vauban schema [--output FILE]")
    lines.append(
        "    vauban init [--mode MODE] [--model PATH] [--scenario NAME]"
        " [--output FILE] [--force]",
    )
    lines.append(
        "    vauban diff [--format text|markdown]"
        " [--threshold FLOAT] <dir_a> <dir_b>",
    )
    lines.append(
        "    vauban tree [directory]"
        " [--format text|mermaid] [--status STATUS] [--tag TAG]",
    )
    lines.append("    vauban man [topic]")
    lines.append("")
    lines.append("DESCRIPTION")
    lines.append(
        "    Generated from vauban config dataclasses + parser constraints at runtime.",
    )
    lines.append("    Defaults and types auto-refresh when config dataclasses change.")
    lines.append("")
    lines.append("TOPICS")
    lines.append(f"    {', '.join(manual_topics())}")
    lines.append("")

    if include_quickstart:
        lines.append("QUICKSTART")
        for note in _QUICKSTART_NOTES:
            lines.append(f"    {note}")
        lines.append("    Minimal run.toml:")
        for line in _MINIMAL_CONFIG_EXAMPLE:
            lines.append(f"    {line}")
        lines.append("")

    if include_commands:
        lines.append("COMMANDS")
        lines.append("    vauban <config.toml>")
        lines.append("      Run the configured pipeline from a TOML file.")
        lines.append("    vauban --validate <config.toml>")
        lines.append(
            "      Validate config + prompt files without loading model weights.",
        )
        lines.append("    vauban schema [--output FILE]")
        lines.append("      Print or write the current JSON Schema for TOML configs.")
        lines.append(
            "    vauban init [--mode MODE] [--model PATH] [--scenario NAME]"
            " [--output FILE] [--force]",
        )
        lines.append("      Generate a starter config file.")
        lines.append(f"      known modes: {', '.join(_known_init_modes())}")
        lines.append(
            "      --scenario: seed [environment] from a built-in benchmark;"
            " omitting --mode implies softprompt.",
        )
        lines.append(
            "    vauban diff [--format text|markdown]"
            " [--threshold FLOAT] <dir_a> <dir_b>",
        )
        lines.append("      Compare report metrics from two output directories.")
        lines.append(
            "      --format: output format (default: text).",
        )
        lines.append(
            "      --threshold: CI gate; exit code 1 if any |delta| exceeds value.",
        )
        lines.append(f"      report files: {', '.join(_known_diff_reports())}")
        lines.append(
            "    vauban tree [directory]"
            " [--format text|mermaid] [--status STATUS] [--tag TAG]",
        )
        lines.append("      Render the experiment lineage tree from TOML configs.")
        lines.append("      directory: root to scan recursively for *.toml files.")
        lines.append("      --format: text tree or Mermaid flowchart.")
        lines.append("      --status/--tag: filter the rendered experiment set.")
        lines.append("    vauban man [topic]")
        lines.append("      Show this manual or one focused topic.")
        lines.append("")

    if include_validate:
        lines.append("VALIDATE WORKFLOW")
        for note in _VALIDATE_NOTES:
            lines.append(f"    - {note}")
        lines.append("    Example:")
        lines.append("      vauban --validate run.toml")
        lines.append("")

    if include_playbook:
        lines.append("EXPERIMENT PLAYBOOK")
        for note in _PLAYBOOK_NOTES:
            lines.append(f"    {note}")
        lines.append("    Common loop: init -> validate -> run -> diff -> tune.")
        lines.append("")

    if include_quick:
        lines.append("PYTHON QUICK API")
        for note in _QUICK_NOTES:
            lines.append(f"    - {note}")
        lines.append(f"    helpers: {', '.join(_known_quick_functions())}")
        lines.append("    Minimal interactive flow:")
        for line in _QUICK_EXAMPLE:
            lines.append(f"    {line}")
        lines.append("")

    if include_examples:
        lines.append("EXAMPLES")
        for note in _EXAMPLE_NOTES:
            lines.append(f"    {note}")
        lines.append("")

    if include_print:
        lines.append("PRINTING AND SHARING")
        for note in _PRINT_NOTES:
            lines.append(f"    {note}")
        lines.append("")

    if include_modes:
        lines.append("PIPELINE MODES")
        lines.append(
            "    Early-return precedence:"
            f" {' > '.join(EARLY_RETURN_PRECEDENCE)}",
        )
        for mode in _PIPELINE_MODES:
            suffix = " (early return)" if mode.early_return else ""
            lines.append(f"    {mode.mode}{suffix}")
            lines.append(f"      trigger: {mode.trigger}")
            lines.append(f"      output:  {mode.output}")
        lines.append("")

    if include_formats:
        lines.append("DATA FORMATS")
        for note in _FORMAT_NOTES:
            lines.append(f"    - {note}")
        lines.append("")

    if include_taxonomy:
        lines.append(_render_taxonomy_topic())
        lines.append("")

    if include_datasets:
        lines.append(_render_datasets_topic())
        lines.append("")

    if include_workflows:
        lines.append(_render_workflows_topic())
        lines.append("")

    if selected:
        lines.append("CONFIG SECTIONS")
        for section in selected:
            lines.append("")
            lines.append(_format_section_header(section))
            lines.append(f"  description: {section.description}")
            lines.append(
                f"  required: {'yes' if section.required else 'no'}",
            )
            if section.early_return:
                lines.append("  early_return: yes")
            workflow = _SECTION_TO_WORKFLOW.get(section.name)
            if workflow:
                lines.append(
                    f"  workflow: {workflow}"
                    " (vauban man workflows)",
                )
            for note in section.notes:
                lines.append(f"  note: {note}")
            lines.append("  fields:")
            for field in section.fields:
                lines.append(f"    - {_field_path(section, field)}")
                lines.append(f"      type: {field.type_name}")
                lines.append(f"      required: {'yes' if field.required else 'no'}")
                if field.default_repr is not None:
                    lines.append(f"      default: {field.default_repr}")
                elif field.required:
                    lines.append("      default: (required)")
                else:
                    lines.append("      default: null")
                if field.constraints is not None:
                    lines.append(f"      constraints: {field.constraints}")
                lines.append(f"      description: {field.description}")
                for note in field.notes:
                    lines.append(f"      note: {note}")

    return "\n".join(lines).rstrip() + "\n"


def _render_taxonomy_topic() -> str:
    """Render the harm taxonomy as a grep-friendly tree."""
    from vauban.taxonomy import CATEGORY_ALIASES, TAXONOMY

    lines: list[str] = []
    lines.append("HARM TAXONOMY")
    lines.append("    Canonical two-level taxonomy: domains > categories.")
    lines.append(
        "    Category IDs match the 'category' field in JSONL data files.",
    )
    lines.append("")

    for domain in TAXONOMY:
        lines.append(f"  {domain.name} ({domain.id})")
        for cat in domain.categories:
            lines.append(f"    {cat.id:<28s} {cat.description}")
    lines.append("")

    lines.append("  ALIASES (legacy string -> canonical ID)")
    for alias, canonical in sorted(CATEGORY_ALIASES.items()):
        lines.append(f"    {alias:<20s} -> {canonical}")

    return "\n".join(lines)


def _render_datasets_topic() -> str:
    """Render the bundled dataset registry as a table."""
    from vauban.data import BUNDLED_DATASETS
    from vauban.taxonomy import all_categories, coverage_report

    canonical = all_categories()
    lines: list[str] = []
    lines.append("BUNDLED DATASETS")
    lines.append(
        "    Static registry of all JSONL files shipped with vauban.",
    )
    lines.append("")

    for ds in BUNDLED_DATASETS:
        lines.append(f"  {ds.name}")
        lines.append(f"    file:       {ds.filename}")
        lines.append(f"    count:      {ds.count}")
        lines.append(f"    description: {ds.description}")
        if ds.has_categories:
            # Show taxonomy coverage for categorized datasets
            coverage = coverage_report(set(ds.categories))
            lines.append(
                f"    taxonomy:   {len(coverage.present)}/{len(canonical)}"
                f" categories ({coverage.coverage_ratio:.0%})",
            )
            if coverage.aliased:
                aliased = ", ".join(
                    f"{k}->{v}" for k, v in sorted(coverage.aliased.items())
                )
                lines.append(f"    aliases:    {aliased}")
        lines.append("")

    return "\n".join(lines).rstrip()


def _render_workflows_topic() -> str:
    """Render the goal-oriented workflow guides."""
    lines: list[str] = []
    lines.append("WORKFLOWS")
    lines.append(
        "    Goal-oriented guides. Each shows which sections to"
        " use and in what order.",
    )
    lines.append(
        '    Start here if you know what you want to do'
        " but not which config sections to use.",
    )
    lines.append("")

    for title, description, steps, tips in _WORKFLOW_SPECS:
        lines.append(f"  {title}")
        lines.append(f"    {description}")
        lines.append("")
        for step in steps:
            lines.append(f"    {step}")
        lines.append("")
        lines.append("    Tips:")
        for tip in tips:
            lines.append(f"      {tip}")
        sections = _WORKFLOW_SECTIONS.get(title, ())
        if sections:
            see = ", ".join(
                f"vauban man {s}" for s in sections
            )
            lines.append(f"    Details: {see}")
        lines.append("")

    return "\n".join(lines).rstrip()


def _normalize_topic(topic: str | None) -> str | None:
    if topic is None:
        return None
    normalized = topic.strip().lower().strip("[]")
    normalized = _TOPIC_ALIASES.get(normalized, normalized)
    valid_topics = set(manual_topics())
    if normalized not in valid_topics:
        options = ", ".join(manual_topics())
        msg = (
            f"Unknown manual topic {topic!r}. "
            f"Available topics: {options}"
        )
        raise ValueError(msg)
    return normalized


def _build_sections() -> tuple[ManualSection, ...]:
    return tuple(_build_section(spec) for spec in _SECTION_SPECS)


def _build_section(spec: SectionSpec) -> ManualSection:
    auto_fields = _auto_field_map(spec.config_class)
    used_attrs: set[str] = set()
    rendered_fields: list[ManualField] = []

    for field_spec in spec.fields:
        attr = field_spec.attr if field_spec.attr is not None else field_spec.key
        used_attrs.add(attr)
        auto = auto_fields.get(attr)
        type_name = auto.type_name if auto is not None else "object"
        required = (
            field_spec.required
            if field_spec.required is not None
            else (auto.required if auto is not None else False)
        )
        default_repr = field_spec.default_override
        if default_repr is None and auto is not None:
            default_repr = auto.default_repr
        rendered_fields.append(
            ManualField(
                key=field_spec.key,
                type_name=type_name,
                required=required,
                default_repr=default_repr,
                description=field_spec.description,
                constraints=field_spec.constraints,
                notes=field_spec.notes,
            ),
        )

    for attr, auto in sorted(auto_fields.items()):
        if attr in used_attrs:
            continue
        rendered_fields.append(
            ManualField(
                key=attr,
                type_name=auto.type_name,
                required=auto.required,
                default_repr=auto.default_repr,
                description=(
                    "Auto-discovered field with no explicit manual entry yet."
                ),
                constraints=None,
            ),
        )

    return ManualSection(
        name=spec.name,
        description=spec.description,
        required=spec.required,
        early_return=spec.early_return,
        table=spec.table,
        fields=tuple(rendered_fields),
        notes=spec.notes,
    )


def _auto_field_map(config_class: str | None) -> dict[str, AutoField]:
    if config_class is None:
        return {}
    all_fields = _parsed_config_fields()
    if config_class not in all_fields:
        msg = f"Config class {config_class!r} not found in types.py"
        raise ValueError(msg)
    return all_fields[config_class]


@lru_cache(maxsize=1)
def _parsed_config_fields() -> dict[str, dict[str, AutoField]]:
    types_path = Path(__file__).with_name("types.py")
    module = ast.parse(types_path.read_text())
    class_names = {
        spec.config_class
        for spec in _SECTION_SPECS
        if spec.config_class is not None
    }

    parsed: dict[str, dict[str, AutoField]] = {}
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name in class_names:
            parsed[node.name] = _parse_class_fields(node)
    return parsed


def _parse_class_fields(class_node: ast.ClassDef) -> dict[str, AutoField]:
    parsed: dict[str, AutoField] = {}
    for node in class_node.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name):
            continue

        field_name = node.target.id
        type_name = _format_type_expr(node.annotation)
        default_repr, required = _default_from_assignment(node.value)
        parsed[field_name] = AutoField(
            type_name=type_name,
            default_repr=default_repr,
            required=required,
        )
    return parsed


def _default_from_assignment(
    value: ast.expr | None,
) -> tuple[str | None, bool]:
    if value is None:
        return None, True

    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "field"
    ):
        return _default_from_field_call(value)

    return _default_from_expr(value), False


def _default_from_field_call(call: ast.Call) -> tuple[str | None, bool]:
    for keyword in call.keywords:
        if keyword.arg == "default":
            return _default_from_expr(keyword.value), False
        if keyword.arg == "default_factory":
            return _default_from_factory_expr(keyword.value), False
    return None, True


def _default_from_factory_expr(expr: ast.expr) -> str:
    if isinstance(expr, ast.Name):
        if expr.id == "list":
            return "[]"
        if expr.id == "dict":
            return "{}"

    if isinstance(expr, ast.Lambda):
        return _default_from_expr(expr.body)

    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
        if expr.func.id == "list":
            return "[]"
        if expr.func.id == "dict":
            return "{}"

    return ast.unparse(expr)


def _default_from_expr(expr: ast.expr) -> str:
    try:
        literal = ast.literal_eval(expr)
    except (SyntaxError, TypeError, ValueError):
        return ast.unparse(expr)
    return _format_default(literal)


def _format_type_expr(annotation: ast.expr) -> str:
    text = ast.unparse(annotation)
    return text.replace("NoneType", "None")


def _format_default(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, Path):
        return json.dumps(str(value))
    if isinstance(value, list):
        inner = ", ".join(_format_default(item) for item in value)
        return f"[{inner}]"
    if isinstance(value, tuple):
        inner = ", ".join(_format_default(item) for item in value)
        return f"[{inner}]"
    if isinstance(value, dict):
        pairs = ", ".join(
            f"{json.dumps(str(key))}: {_format_default(item)}"
            for key, item in value.items()
        )
        return f"{{{pairs}}}"
    return json.dumps(str(value))


def _select_sections(
    sections: tuple[ManualSection, ...],
    topic: str | None,
) -> tuple[ManualSection, ...]:
    if topic is None or topic == "all":
        return sections
    if topic in {
        "quickstart",
        "commands",
        "validate",
        "playbook",
        "quick",
        "examples",
        "print",
        "modes",
        "formats",
        "taxonomy",
        "datasets",
        "workflows",
    }:
        return ()
    for section in sections:
        if section.name == topic:
            return (section,)
    return ()


def _format_section_header(section: ManualSection) -> str:
    if section.table:
        return f"SECTION [{section.name}]"
    return f"SECTION {section.name} (top-level key)"


def _field_path(section: ManualSection, field: ManualField) -> str:
    if section.table:
        return f"[{section.name}].{field.key}"
    return field.key


__all__ = [
    "EARLY_RETURN_PRECEDENCE",
    "manual_topics",
    "render_manual",
]

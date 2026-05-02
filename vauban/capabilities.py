# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Capability catalog tying Vauban's scattered surfaces together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vauban.config._mode_registry import EARLY_MODE_SPECS

type CapabilityQuestion = Literal[
    "behavior_change",
    "runtime_evidence",
    "internal_diagnostics",
    "controlled_intervention",
    "model_transformation",
    "defensive_evaluation",
    "external_access",
    "compliance",
]
type CapabilitySurface = Literal[
    "toml",
    "session",
    "python",
    "runtime",
    "script",
    "docs",
]


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    """One user-facing Vauban capability across CLI, Python, and docs."""

    name: str
    question: CapabilityQuestion
    description: str
    use_when: str
    first_cli: tuple[str, ...] = ()
    first_python: tuple[str, ...] = ()
    proof: str = ""
    toml_sections: tuple[str, ...] = ()
    session_tools: tuple[str, ...] = ()
    python_entrypoints: tuple[str, ...] = ()
    runtime_entrypoints: tuple[str, ...] = ()
    scripts: tuple[str, ...] = ()
    docs: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()

    def surfaces(self) -> tuple[CapabilitySurface, ...]:
        """Return the surfaces where this capability is available."""
        surfaces: list[CapabilitySurface] = []
        if self.toml_sections:
            surfaces.append("toml")
        if self.session_tools:
            surfaces.append("session")
        if self.python_entrypoints:
            surfaces.append("python")
        if self.runtime_entrypoints:
            surfaces.append("runtime")
        if self.scripts:
            surfaces.append("script")
        if self.docs:
            surfaces.append("docs")
        return tuple(surfaces)

    def to_dict(self) -> dict[str, str | list[str]]:
        """Serialize the capability for discovery APIs and reports."""
        return {
            "name": self.name,
            "question": self.question,
            "description": self.description,
            "use_when": self.use_when,
            "first_cli": list(self.first_cli),
            "first_python": list(self.first_python),
            "proof": self.proof,
            "surfaces": list(self.surfaces()),
            "toml_sections": list(self.toml_sections),
            "session_tools": list(self.session_tools),
            "python_entrypoints": list(self.python_entrypoints),
            "runtime_entrypoints": list(self.runtime_entrypoints),
            "scripts": list(self.scripts),
            "docs": list(self.docs),
            "artifacts": list(self.artifacts),
        }


_CATALOG: tuple[CapabilitySpec, ...] = (
    CapabilitySpec(
        name="model_behavior_change_reports",
        question="behavior_change",
        description=(
            "Produce reproducible before/after behavior evidence and Model"
            " Behavior Change Reports with explicit release gates."
        ),
        use_when=(
            "You have a baseline and candidate model, checkpoint, prompt"
            " template, intervention, or deployment package and need a"
            " reproducible behavior-change artifact and ship/review/block"
            " decision."
        ),
        first_cli=(
            "vauban init --mode behavior_trace --output baseline_trace.toml",
            "vauban baseline_trace.toml",
            "vauban init --mode behavior_diff --output diff.toml",
        ),
        first_python=("from vauban.behavior import load_behavior_trace",),
        proof=(
            "A trace JSONL or behavior report exists, lists limitations, and"
            " includes a release_gate when thresholds are configured."
        ),
        toml_sections=("behavior_trace", "behavior_diff", "behavior_report"),
        session_tools=("report", "report_pdf"),
        python_entrypoints=("vauban.behavior",),
        runtime_entrypoints=("vauban.runtime",),
        docs=(
            "docs/research/behavior-diff-traces.md",
            "docs/research/what-changes-when-models-are-changed.md",
            "docs/capabilities/change-release-gates.md",
        ),
        artifacts=(
            "behavior_trace.jsonl",
            "behavior_trace_report.json",
            "behavior_diff_report.json",
            "release_gate",
            "model_behavior_change_report.md",
            "behavior_report.md",
        ),
    ),
    CapabilitySpec(
        name="runtime_evidence_and_portability",
        question="runtime_evidence",
        description=(
            "Collect tokens, logits, logprobs, activations, profile spans, and"
            " primitive metadata across Torch CPU/CUDA/MPS and reference MLX."
        ),
        use_when=(
            "You need to know what runtime evidence Vauban can actually collect"
            " on the current machine before making internal or performance"
            " claims."
        ),
        first_cli=(
            "vauban init --mode behavior_trace --output trace.toml",
            "vauban trace.toml",
            "pixi run -e torch-dev real-cuda-smoke",
        ),
        first_python=(
            "from vauban.runtime import create_runtime, TraceRequest",
            "runtime = create_runtime('torch')",
        ),
        proof=(
            "The trace includes tokens/logits/logprobs/activation artifacts and"
            " profile metadata for the selected backend."
        ),
        toml_sections=("behavior_trace",),
        python_entrypoints=("create_runtime", "run_runtime_trace"),
        runtime_entrypoints=(
            "TraceRequest",
            "Trace",
            "TraceArtifact",
            "run_torch_activation_primitive",
        ),
        scripts=(
            "scripts/torch_cuda_real_model_smoke.py",
            "scripts/torch_cuda_trace_sweep.py",
        ),
        docs=(
            "docs/research/trace-first-inference-stack.md",
            "docs/research/torch_surface_matrix.md",
            "docs/research/mps-primitive-boundary.md",
        ),
        artifacts=("runtime_evidence", "Trace", "TraceProfileSweep"),
    ),
    CapabilitySpec(
        name="behavior_boundary_diagnostics",
        question="internal_diagnostics",
        description=(
            "Measure directions, inspect projection surfaces, train probes, and"
            " connect internal activation evidence to behavior changes."
        ),
        use_when=(
            "You need to explain where a behavior boundary appears internally,"
            " which layers matter, or whether a direction is stable enough to"
            " support a report claim."
        ),
        first_cli=(
            "vauban init --mode probe --output probe.toml",
            "vauban probe.toml",
        ),
        first_python=(
            "from vauban.session import Session",
            "s = Session('Qwen/Qwen2.5-1.5B-Instruct')",
            "s.measure(); s.probe('Explain why rainbows form.')",
        ),
        proof="A direction, probe report, or surface report links behavior to layers.",
        toml_sections=(
            "measure",
            "probe",
            "surface",
            "depth",
            "awareness",
            "linear_probe",
            "features",
            "circuit",
            "svf",
        ),
        session_tools=("measure", "detect", "probe", "scan", "surface"),
        python_entrypoints=(
            "measure",
            "measure_subspace",
            "probe",
            "map_surface",
            "depth_profile",
            "train_probe",
            "train_sae",
            "trace_circuit",
        ),
        docs=(
            "docs/capabilities/understand-your-model.md",
            "docs/concepts/activation-geometry.md",
            "docs/surface.md",
        ),
        artifacts=(
            "direction.npy",
            "probe_report.json",
            "surface_report.json",
            "depth_report.json",
        ),
    ),
    CapabilitySpec(
        name="controlled_interventions",
        question="controlled_intervention",
        description=(
            "Apply, gate, and evaluate activation interventions while preserving"
            " evidence about where and why the intervention fired."
        ),
        use_when=(
            "You want to test a steering, CAST, guard, or intervention policy"
            " as an auditable behavior change rather than an ad hoc generation"
            " trick."
        ),
        first_cli=(
            "vauban init --mode intervention_eval --output intervention.toml",
            "vauban intervention.toml",
        ),
        first_python=(
            "from vauban.session import Session",
            "s = Session('Qwen/Qwen2.5-1.5B-Instruct')",
            "s.cast('Explain a risky request safely.', threshold=0.3)",
        ),
        proof=(
            "The intervention report records prompts, layers, trigger metadata,"
            " and generated behavior."
        ),
        toml_sections=("steer", "cast", "guard", "intervention_eval", "sss"),
        session_tools=("steer", "cast", "sic"),
        python_entrypoints=(
            "steer",
            "cast_generate",
            "guard_generate",
            "sss_generate",
        ),
        runtime_entrypoints=("TorchDirectionIntervention",),
        docs=(
            "docs/capabilities/defend-your-model.md",
            "docs/concepts/steering.md",
            "docs/concepts/defense-complementarity.md",
        ),
        artifacts=(
            "steer_report.json",
            "cast_report.json",
            "guard_report.json",
            "intervention_eval_report.json",
        ),
    ),
    CapabilitySpec(
        name="model_transformations",
        question="model_transformation",
        description=(
            "Transform, export, and analyze model weights, adapters, and"
            " composition weights as auditable change objects."
        ),
        use_when=(
            "You changed weights, adapters, composition, or fine-tuning strategy"
            " and need the change to become an explicit audit object."
        ),
        first_cli=(
            "vauban init --mode default --output run.toml",
            "vauban run.toml",
        ),
        first_python=(
            "from vauban.session import Session",
            "s = Session('Qwen/Qwen2.5-1.5B-Instruct')",
            "s.measure(); s.cut(); s.export('output/model')",
        ),
        proof=(
            "The output directory contains modified weights or adapter artifacts"
            " plus a report describing the transformation."
        ),
        toml_sections=(
            "cut",
            "optimize",
            "compose_optimize",
            "fusion",
            "repbend",
            "lora_export",
            "lora_analysis",
        ),
        session_tools=("cut", "export", "evaluate"),
        python_entrypoints=(
            "cut",
            "cut_subspace",
            "direction_to_lora",
            "repbend",
            "fuse_and_generate",
            "optimize",
        ),
        docs=(
            "docs/capabilities/modify-weights.md",
            "docs/concepts/abliteration.md",
        ),
        artifacts=(
            "modified weights",
            "lora_adapter/",
            "lora_export_report.json",
            "repbend_report.json",
        ),
    ),
    CapabilitySpec(
        name="defensive_evaluation",
        question="defensive_evaluation",
        description=(
            "Stress-test defenses, score outputs, classify harmful content, and"
            " run closed-loop attack/defense evaluation."
        ),
        use_when=(
            "You need to know whether a model or defense stack regresses under"
            " safe, reproducible stress tests."
        ),
        first_cli=(
            "vauban init --mode audit --output audit.toml",
            "vauban audit.toml",
        ),
        first_python=(
            "from vauban.session import Session",
            "s = Session('Qwen/Qwen2.5-1.5B-Instruct')",
            "s.audit(thoroughness='standard')",
        ),
        proof=(
            "The audit, softprompt, jailbreak, or flywheel report contains"
            " aggregate metrics and limitations."
        ),
        toml_sections=("audit", "softprompt", "jailbreak", "sic", "defend", "flywheel"),
        session_tools=("audit", "jailbreak", "score", "classify"),
        python_entrypoints=(
            "softprompt_attack",
            "run_flywheel",
            "defend_content",
            "score_response",
            "classify_text",
        ),
        docs=(
            "docs/capabilities/stress-test-defenses.md",
            "docs/principles/attack-defense-duality.md",
        ),
        artifacts=(
            "audit_report.json",
            "audit_report.md",
            "softprompt_report.json",
            "jailbreak_report.json",
            "flywheel_*.jsonl",
        ),
    ),
    CapabilitySpec(
        name="external_endpoint_audits",
        question="external_access",
        description=(
            "Collect behavior traces, probe remote endpoints, and evaluate"
            " optimized prompts when only black-box or API access is available."
        ),
        use_when=(
            "You only have API or endpoint access and need to keep claim"
            " strength at the black-box behavioral-diff level."
        ),
        first_cli=(
            "vauban init --mode behavior_trace --output api_trace.toml",
            'edit api_trace.toml: set runtime_backend = "api" and [behavior_trace.api]',
            "vauban api_trace.toml",
        ),
        first_python=(
            "from vauban.api_trace import call_api_behavior_trace",
            "from vauban.remote import run_remote_probe",
        ),
        proof=(
            "A trace JSONL, remote report, or API report records responses,"
            " errors, optional logprobs, and access limits."
        ),
        toml_sections=("behavior_trace", "remote", "api_eval"),
        python_entrypoints=(
            "call_api_behavior_trace",
            "run_remote_probe",
            "evaluate_suffix_via_api",
        ),
        docs=("docs/capabilities/access-levels.md",),
        artifacts=(
            "behavior_trace.jsonl",
            "behavior_trace_report.json",
            "remote_report.json",
            "api_eval_report.json",
        ),
    ),
    CapabilitySpec(
        name="compliance_and_release_readiness",
        question="compliance",
        description=(
            "Turn audit evidence into deployer-readiness, AI Act, and release"
            " decision artifacts."
        ),
        use_when=(
            "You need to package model-change evidence into public-sector,"
            " release-readiness, risk, compliance, or deployer-review"
            " documents."
        ),
        first_cli=(
            "vauban init --mode public_sector_readiness --output readiness.toml",
            "vauban readiness.toml",
        ),
        first_python=(
            "from vauban.ai_act import generate_deployer_readiness_bundle",
        ),
        proof=(
            "The AI Act readiness report, controls matrix, and risk register"
            " cite the evidence they were built from."
        ),
        toml_sections=("ai_act",),
        session_tools=("ai_act",),
        python_entrypoints=(
            "generate_deployer_readiness_artifacts",
            "generate_deployer_readiness_bundle",
        ),
        docs=(
            "docs/capabilities/public-sector-adoption.md",
            "docs/principles/reproducibility.md",
        ),
        artifacts=(
            "public_sector_readiness_report.pdf",
            "ai_act_readiness_report.json",
            "ai_act_coverage_ledger.json",
            "ai_act_controls_matrix.json",
            "ai_act_risk_register.json",
        ),
    ),
)


def capability_catalog() -> tuple[CapabilitySpec, ...]:
    """Return Vauban's user-facing capability map."""
    return _CATALOG


def capabilities_by_question(
    question: CapabilityQuestion,
) -> tuple[CapabilitySpec, ...]:
    """Return capabilities that answer one product question."""
    return tuple(spec for spec in _CATALOG if spec.question == question)


def capability_by_name(name: str) -> CapabilitySpec | None:
    """Return one capability by stable catalog name."""
    return next((spec for spec in _CATALOG if spec.name == name), None)


def capability_names() -> tuple[str, ...]:
    """Return stable capability names in catalog order."""
    return tuple(spec.name for spec in _CATALOG)


def unassigned_early_modes() -> tuple[str, ...]:
    """Return early TOML modes that do not appear in the catalog."""
    registered = {spec.mode for spec in EARLY_MODE_SPECS}
    assigned = {
        section
        for spec in _CATALOG
        for section in spec.toml_sections
    }
    return tuple(sorted(registered - assigned))

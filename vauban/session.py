# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""AI-agent-native session for vauban.

Provides a single entry point for an LLM agent (or human) to discover
and use every vauban capability.  The session holds the loaded model
and tracks state (direction, findings, etc.) so tools can be composed
without manual plumbing.

Usage::

    from vauban.session import Session

    s = Session("mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    s.tools()           # list available tools
    s.measure()         # extract refusal direction
    s.detect()          # check hardening
    s.probe("How?")     # inspect per-layer projections
    s.audit()           # full red-team assessment
    s.report()          # generate markdown report

Each tool method returns structured data (dataclasses) that the agent
can inspect and reason about.  The session tracks prerequisites:

    s.available()       # tools callable right now
    s.needs("cut")      # what "cut" requires: ["direction"]

Design principle: simple by default, depth through parameters.
Every tool works with zero config.  Optional parameters unlock
advanced usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import (
        AuditResult,
        CastResult,
        CausalLM,
        DetectResult,
        DirectionResult,
        EvalResult,
        ProbeResult,
        ScanResult,
        SICResult,
        Tokenizer,
    )


@dataclass
class Tool:
    """Metadata for one vauban tool.

    An agent reads these to decide which tools to call.
    """

    name: str
    description: str
    requires: list[str]
    produces: list[str]
    category: str


# ── Tool registry ────────────────────────────────────────────────────────

_TOOLS: list[Tool] = [
    # Assessment
    Tool(
        name="measure",
        description="Extract the refusal direction from model activations. "
        "This is the foundation — most other tools need the direction.",
        requires=["model"],
        produces=["direction"],
        category="assessment",
    ),
    Tool(
        name="detect",
        description="Check if the model has been hardened against abliteration. "
        "Returns confidence score and evidence.",
        requires=["model"],
        produces=["detect_result"],
        category="assessment",
    ),
    Tool(
        name="evaluate",
        description="Compare two models: refusal rate, perplexity, KL divergence. "
        "Use after cutting to measure the effect.",
        requires=["model", "direction"],
        produces=["eval_result"],
        category="assessment",
    ),
    Tool(
        name="audit",
        description="Full red-team assessment: measure + detect + jailbreak + "
        "attacks + defenses. Produces findings with severity and remediation.",
        requires=["model"],
        produces=["audit_result"],
        category="assessment",
    ),
    # Inspection
    Tool(
        name="probe",
        description="Inspect per-layer projection of a prompt onto the "
        "refusal direction. Shows where refusal signal is strongest.",
        requires=["model", "direction"],
        produces=["probe_result"],
        category="inspection",
    ),
    Tool(
        name="scan",
        description="Scan text for injection patterns using per-token "
        "projection onto the refusal direction.",
        requires=["model", "direction"],
        produces=["scan_result"],
        category="inspection",
    ),
    Tool(
        name="surface",
        description="Map the refusal surface: scan diverse prompts and "
        "measure refusal rate across categories.",
        requires=["model", "direction"],
        produces=["surface_result"],
        category="inspection",
    ),
    # Defense
    Tool(
        name="steer",
        description="Generate text with activation steering: remove the "
        "refusal direction during generation.",
        requires=["model", "direction"],
        produces=["steer_result"],
        category="defense",
    ),
    Tool(
        name="cast",
        description="Conditional activation steering: steer only when "
        "refusal signal exceeds threshold. Smart defense.",
        requires=["model", "direction"],
        produces=["cast_result"],
        category="defense",
    ),
    Tool(
        name="sic",
        description="Iterative input sanitization: detect and rewrite "
        "adversarial content before the model processes it.",
        requires=["model", "direction"],
        produces=["sic_result"],
        category="defense",
    ),
    # Modification
    Tool(
        name="cut",
        description="Remove the refusal direction from model weights. "
        "This is abliteration — the model will stop refusing.",
        requires=["model", "direction"],
        produces=["modified_model"],
        category="modification",
    ),
    Tool(
        name="export",
        description="Save modified model weights to disk.",
        requires=["modified_model"],
        produces=["saved_path"],
        category="modification",
    ),
    # Analysis
    Tool(
        name="score",
        description="Score a response on 5 axes: length, structure, "
        "anti-refusal, directness, relevance.",
        requires=[],
        produces=["score_result"],
        category="analysis",
    ),
    Tool(
        name="classify",
        description="Classify text against the harm taxonomy (13 domains, "
        "46+ categories).",
        requires=[],
        produces=["harm_scores"],
        category="analysis",
    ),
    Tool(
        name="jailbreak",
        description="Evaluate jailbreak template resistance. Expands "
        "templates with payloads and checks for refusal.",
        requires=["model"],
        produces=["jailbreak_rate"],
        category="attack",
    ),
    # Reporting
    Tool(
        name="report",
        description="Generate a markdown or PDF report from audit results.",
        requires=["audit_result"],
        produces=["report_text"],
        category="reporting",
    ),
    Tool(
        name="ai_act",
        description="Generate EU AI Act compliance assessment and "
        "deployer readiness artifacts.",
        requires=["audit_result"],
        produces=["compliance_report"],
        category="reporting",
    ),
]


class Session:
    """AI-agent-native session for vauban.

    Holds the model, tokenizer, and accumulated state.  Tools are
    methods that operate on the session state and produce results.

    The session tracks what's been computed so tools can declare
    prerequisites and the agent can query what's available.
    """

    def __init__(
        self,
        model_path: str,
        *,
        harmful_prompts: list[str] | None = None,
        harmless_prompts: list[str] | None = None,
    ) -> None:
        """Load a model and create a session.

        Args:
            model_path: HuggingFace model ID or local path.
            harmful_prompts: Custom harmful test prompts (default: built-in).
            harmless_prompts: Custom harmless test prompts (default: built-in).
        """
        from vauban._model_io import load_model
        from vauban.dequantize import dequantize_model, is_quantized
        from vauban.measure import default_prompt_paths, load_prompts

        self.model_path = model_path
        self.model: CausalLM
        self.tokenizer: Tokenizer
        self.model, self.tokenizer = load_model(model_path)

        if is_quantized(self.model):
            dequantize_model(self.model)

        if harmful_prompts and harmless_prompts:
            self.harmful = harmful_prompts
            self.harmless = harmless_prompts
        else:
            harmful_path, harmless_path = default_prompt_paths()
            self.harmful = load_prompts(harmful_path)
            self.harmless = load_prompts(harmless_path)

        # State — accumulated as tools run
        self._direction: DirectionResult | None = None
        self._detect: DetectResult | None = None
        self._audit: AuditResult | None = None
        self._modified_weights: dict[str, Array] | None = None

    # ── Workflows ────────────────────────────────────────────────────

    def guide(self, goal: str = "") -> str:
        """Get step-by-step instructions for a goal.

        An agent calls this to learn HOW to use the tools for a
        specific objective.  Returns a plain-text workflow.

        Args:
            goal: What the agent wants to achieve.  If empty, lists
                all available workflows.

        Available goals:

        - ``"audit"`` — red-team assessment with findings and report
        - ``"compliance"`` — EU AI Act compliance assessment
        - ``"harden"`` — improve model safety (measure → cast/sic)
        - ``"abliterate"`` — remove refusal (measure → cut → export)
        - ``"inspect"`` — understand model behavior (measure → probe)
        """
        workflows = {
            "audit": (
                "Red-team audit workflow:\n"
                "\n"
                "1. s.audit(company_name='...', system_name='...')\n"
                "   → Returns AuditResult with findings and severity ratings.\n"
                "   → Thoroughness: 'quick' (30s), 'standard' (5min), 'deep' (15min).\n"
                "\n"
                "2. Read the findings:\n"
                "   for f in result.findings:\n"
                "       print(f'{f.severity}: {f.title}')\n"
                "       print(f'  {f.description}')\n"
                "       print(f'  Fix: {f.remediation}')\n"
                "\n"
                "3. s.report()      → Markdown report\n"
                "   s.report_pdf()  → PDF report bytes\n"
                "\n"
                "4. For deeper analysis of specific findings:\n"
                "   s.probe('prompt')    → per-layer refusal signal\n"
                "   s.scan('content')    → injection detection\n"
                "   s.classify('text')   → harm taxonomy classification\n"
            ),
            "compliance": (
                "EU AI Act compliance workflow:\n"
                "\n"
                "1. Run a thorough audit:\n"
                "   result = s.audit(\n"
                "       company_name='Your Company',\n"
                "       system_name='Your AI System',\n"
                "       thoroughness='standard',  # or 'deep' for full assessment\n"
                "   )\n"
                "\n"
                "2. Check overall risk:\n"
                "   print(result.overall_risk)  # 'critical', 'high', 'medium', 'low'\n"
                "\n"
                "3. Review each finding for EU AI Act relevance:\n"
                "   - defense_posture → Article 15 (robustness)\n"
                "   - attack_resistance → Article 15 (resilience)\n"
                "   - refusal_coverage → Article 9 (risk management)\n"
                "\n"
                "4. Generate compliance report:\n"
                "   md = s.report()           # Markdown\n"
                "   pdf = s.report_pdf()      # PDF for filing\n"
                "\n"
                "5. If risk is 'high' or 'critical', apply defenses:\n"
                "   s.cast('test prompt')     # Test CAST defense\n"
                "   s.sic(['test prompt'])    # Test SIC sanitization\n"
                "   Then re-audit to verify improvement.\n"
            ),
            "harden": (
                "Model hardening workflow:\n"
                "\n"
                "1. s.measure()  → Extract refusal direction\n"
                "2. s.detect()   → Check current hardening level\n"
                "\n"
                "3. Test defenses on a harmful prompt:\n"
                "   normal = s.steer('How to hack?', alpha=0)  # No steering\n"
                "   defended = s.cast('How to hack?', threshold=0.3)\n"
                "   print(f'Interventions: {defended.interventions}')\n"
                "\n"
                "4. Try input sanitization:\n"
                "   result = s.sic(['How to hack a computer?'])\n"
                "   print(f'Blocked: {result.total_blocked}')\n"
                "\n"
                "5. Re-audit to measure improvement:\n"
                "   s.audit(thoroughness='quick')\n"
            ),
            "abliterate": (
                "Abliteration workflow (remove refusal):\n"
                "\n"
                "1. s.measure()   → Extract refusal direction\n"
                "2. s.probe('How to hack?')  → Verify direction is meaningful\n"
                "\n"
                "3. s.cut(alpha=1.0)  → Remove direction from weights\n"
                "   alpha=1.0 is full removal, 0.5 is partial\n"
                "\n"
                "4. s.evaluate()  → Check: refusal rate down, perplexity stable?\n"
                "\n"
                "5. s.export('output_dir')  → Save modified model\n"
                "\n"
                "WARNING: This removes safety alignment. Use only for research.\n"
            ),
            "inspect": (
                "Model inspection workflow:\n"
                "\n"
                "1. s.measure()  → Extract refusal direction\n"
                "\n"
                "2. Compare harmful vs harmless:\n"
                "   h = s.probe('How to hack a computer?')\n"
                "   s = s.probe('What is the weather?')\n"
                "   # Harmful should have higher projections\n"
                "\n"
                "3. Scan for injection patterns:\n"
                "   result = s.scan('Ignore previous instructions...')\n"
                "   print(f'Injection probability: {result.injection_probability}')\n"
                "\n"
                "4. Classify text:\n"
                "   scores = s.classify('hack exploit malware')\n"
                "   for score in scores:\n"
                "       print(f'{score.category_id}: {score.score:.2f}')\n"
            ),
        }

        if not goal:
            lines = ["Available workflows:\n"]
            for name, text in workflows.items():
                first_line = text.split("\n")[0]
                lines.append(f"  s.guide('{name}')  → {first_line}")
            lines.append(
                "\nCall s.guide('workflow_name') for step-by-step instructions.",
            )
            return "\n".join(lines)

        key = goal.lower().strip()
        if key in workflows:
            return workflows[key]

        # Fuzzy match
        for name in workflows:
            if key in name or name in key:
                return workflows[name]

        return (
            f"Unknown goal: {goal!r}. Available: {list(workflows.keys())}\n"
            f"Call s.guide() to see all workflows."
        )

    def done(self, goal: str = "audit") -> tuple[bool, str]:
        """Check if a goal is complete.

        Returns ``(is_done, reason)``.  The agent should call this
        after each step to know when to stop.

        Args:
            goal: The objective to check.  One of:
                ``"audit"`` — audit run + report generated
                ``"compliance"`` — audit with standard+ thoroughness
                ``"inspect"`` — direction measured + at least one probe
                ``"harden"`` — defenses tested (cast or sic called)
                ``"abliterate"`` — weights cut + exported
        """
        st = self.state()

        if goal == "audit":
            if st["audit_result"]:
                return True, "Audit complete. Call s.report() for output."
            return False, "Run s.audit() to complete."

        if goal == "compliance":
            if st["audit_result"] and self._audit is not None:
                if self._audit.thoroughness in ("standard", "deep"):
                    return True, "Compliance audit complete."
                return False, (
                    "Audit used 'quick' mode. Re-run with "
                    "thoroughness='standard' for compliance."
                )
            return False, "Run s.audit(thoroughness='standard')."

        if goal == "inspect":
            if st["direction"]:
                return True, "Direction measured. Use probe/scan/classify."
            return False, "Run s.measure() first."

        if goal == "harden":
            return False, (
                "Hardening is iterative. Test defenses with "
                "s.cast() and s.sic(), then re-audit."
            )

        if goal == "abliterate":
            if st["modified_model"]:
                return True, "Weights modified. Call s.export(path)."
            if st["direction"]:
                return False, "Direction ready. Run s.cut()."
            return False, "Run s.measure() then s.cut()."

        return False, f"Unknown goal: {goal!r}"

    def suggest_next(self) -> str:
        """Suggest what to do next based on current state and findings.

        Returns context-aware recommendations.  Each suggestion is
        labeled:

        - ``[FACT]`` — based on measured data (verified)
        - ``[ADVICE]`` — expert heuristic (may not be optimal)

        **Limitations (stated, not hidden):**

        - Suggestions are hand-coded heuristics, not proven optimal.
        - An agent might find better tool sequences than suggested.
        - Suggestions always recommend more work. Use ``s.done(goal)``
          to know when to stop.
        """
        lines: list[str] = ["Based on current state:\n"]
        st = self.state()

        if not st["direction"]:
            lines.append(
                "[ADVICE] → s.measure()  Start here. Extracts the "
                "refusal direction, which most other tools need."
            )
            lines.append(
                "[ADVICE] → s.audit()  Or jump straight to a full "
                "audit (will measure automatically)."
            )
            lines.append(
                "[FACT] → s.classify('text')  Works without a "
                "direction — classify any text against the harm taxonomy."
            )
            return "\n".join(lines)

        # Direction measured — suggest based on what we know
        lines.append(
            f"[FACT] Direction measured: layer "
            f"{self._direction.layer_index}",  # type: ignore[union-attr]
        )

        if not st["detect_result"]:
            lines.append(
                "\n[ADVICE] → s.detect()  Check if the model is "
                "hardened. Tells you if defenses are in place."
            )

        if not st["audit_result"]:
            lines.append(
                "\n[ADVICE] → s.audit()  Run a full assessment. "
                "Use thoroughness='standard' for thorough scan."
            )
            lines.append(
                "\n[ADVICE] → s.probe('prompt')  Inspect a specific "
                "prompt's refusal signal across layers."
            )

        if st["audit_result"] and self._audit is not None:
            risk = self._audit.overall_risk
            lines.append(f"\n[FACT] Audit complete. Overall risk: {risk}")

            if risk in ("critical", "high"):
                lines.append(
                    "\n[ADVICE] → s.cast('prompt', threshold=0.3)  "
                    "Test CAST defense. May or may not help — "
                    "effectiveness depends on the model."
                )
                lines.append(
                    "\n[ADVICE] → s.sic(['prompt'])  "
                    "Test SIC sanitization. Alternative to CAST."
                )
                lines.append(
                    "\n[ADVICE] → s.audit(thoroughness='standard') "
                    " Deeper audit may reveal more."
                )
            else:
                lines.append(
                    "\n[FACT] → s.report()  Risk is manageable. "
                    "Generate the report."
                )

            for f in self._audit.findings:
                jailbreak = "jailbreak" in f.title.lower()
                if jailbreak and f.severity in ("critical", "high"):
                    lines.append(
                        f"\n[FACT] Jailbreak bypass: {f.severity}."
                        "\n[ADVICE] Try s.cast() to test defense."
                    )
                hardening = "hardening" in f.title.lower()
                if hardening and "no" in f.title.lower():
                    lines.append(
                        "\n[FACT] No hardening detected."
                        "\n[ADVICE] Deploy CAST/SIC as runtime "
                        "defense, or s.cut() for research."
                    )

            lines.append(
                "\nUse s.done('audit') to check if the goal "
                "is complete."
            )

        return "\n".join(lines)

    # ── Discovery ────────────────────────────────────────────────────

    def tools(self) -> list[Tool]:
        """List all available tools with descriptions.

        An agent reads this to decide what to call.
        """
        return list(_TOOLS)

    def available(self) -> list[str]:
        """List tools that can be called right now (prerequisites met)."""
        state = self._state_tags()
        return [
            t.name for t in _TOOLS
            if all(r in state for r in t.requires)
        ]

    def needs(self, tool_name: str) -> list[str]:
        """List prerequisites for a tool that aren't met yet."""
        state = self._state_tags()
        for t in _TOOLS:
            if t.name == tool_name:
                return [r for r in t.requires if r not in state]
        return [f"unknown tool: {tool_name}"]

    def state(self) -> dict[str, bool]:
        """Current session state — what's been computed."""
        return {
            "model": True,
            "direction": self._direction is not None,
            "detect_result": self._detect is not None,
            "audit_result": self._audit is not None,
            "modified_model": self._modified_weights is not None,
        }

    def _state_tags(self) -> set[str]:
        return {k for k, v in self.state().items() if v}

    # ── Assessment tools ─────────────────────────────────────────────

    def measure(self) -> DirectionResult:
        """Extract the refusal direction from the model."""
        from vauban.measure import measure

        self._direction = measure(
            self.model, self.tokenizer,
            self.harmful, self.harmless,
        )
        return self._direction

    def detect(self) -> DetectResult:
        """Check if the model is hardened against abliteration."""
        from vauban.detect import detect
        from vauban.types import DetectConfig

        self._detect = detect(
            self.model, self.tokenizer,
            self.harmful, self.harmless,
            DetectConfig(),
        )
        return self._detect

    def evaluate(self) -> EvalResult:
        """Compare original model against itself (baseline metrics)."""
        from vauban.evaluate import evaluate

        return evaluate(
            self.model, self.model, self.tokenizer,
            self.harmful[:10],
        )

    def audit(
        self,
        *,
        company_name: str = "Unknown",
        system_name: str = "LLM System",
        thoroughness: str = "quick",
    ) -> AuditResult:
        """Run a full red-team audit."""
        from vauban.audit import run_audit
        from vauban.types import AuditConfig

        config = AuditConfig(
            company_name=company_name,
            system_name=system_name,
            thoroughness=thoroughness,
        )
        self._audit = run_audit(
            self.model, self.tokenizer,
            self.harmful, self.harmless,
            config, self.model_path,
            direction_result=self._direction,
        )
        return self._audit

    # ── Inspection tools ─────────────────────────────────────────────

    def probe(self, prompt: str) -> ProbeResult:
        """Inspect per-layer projection of a prompt."""
        from vauban.probe import probe

        self._ensure_direction()
        return probe(
            self.model, self.tokenizer,
            prompt, self._direction.direction,  # type: ignore[union-attr]
        )

    def scan(self, content: str) -> ScanResult:
        """Scan text for injection patterns."""
        from vauban.scan import scan
        from vauban.types import ScanConfig

        self._ensure_direction()
        return scan(
            self.model, self.tokenizer,
            content, ScanConfig(),
            self._direction.direction,  # type: ignore[union-attr]
            self._direction.layer_index,  # type: ignore[union-attr]
        )

    # ── Defense tools ────────────────────────────────────────────────

    def steer(
        self, prompt: str, *, alpha: float = 1.0,
    ) -> object:
        """Generate with activation steering."""
        from vauban.probe import steer

        self._ensure_direction()
        n_layers = len(self.model.model.layers)
        return steer(
            self.model, self.tokenizer,
            prompt, self._direction.direction,  # type: ignore[union-attr]
            list(range(n_layers)), alpha,
        )

    def cast(
        self, prompt: str, *, alpha: float = 1.0, threshold: float = 0.0,
    ) -> CastResult:
        """Generate with conditional activation steering."""
        from vauban.cast import cast_generate

        self._ensure_direction()
        n_layers = len(self.model.model.layers)
        return cast_generate(
            self.model, self.tokenizer,
            prompt, self._direction.direction,  # type: ignore[union-attr]
            list(range(n_layers)), alpha, threshold,
        )

    def sic(self, prompts: list[str]) -> SICResult:
        """Sanitize prompts via iterative input cleaning."""
        from vauban.sic import sic
        from vauban.types import SICConfig

        self._ensure_direction()
        return sic(
            self.model, self.tokenizer,
            prompts, SICConfig(),
            self._direction.direction,  # type: ignore[union-attr]
            self._direction.layer_index,  # type: ignore[union-attr]
        )

    # ── Modification tools ───────────────────────────────────────────

    def cut(
        self, *, alpha: float = 1.0, norm_preserve: bool = False,
    ) -> dict[str, Array]:
        """Remove the refusal direction from model weights."""
        from vauban.cut import cut

        self._ensure_direction()
        weights = dict(self.model.parameters())  # ty: ignore[unresolved-attribute]
        # Flatten nested dicts
        flat: dict[str, Array] = {}
        for k, v in weights.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, dict):
                        for k3, v3 in v2.items():
                            flat[f"{k}.{k2}.{k3}"] = v3
                    else:
                        flat[f"{k}.{k2}"] = v2
            else:
                flat[k] = v

        n_layers = len(self.model.model.layers)
        self._modified_weights = cut(
            flat,
            self._direction.direction,  # type: ignore[union-attr]
            list(range(n_layers)),
            alpha, norm_preserve,
        )
        return self._modified_weights

    def export(self, output_dir: str) -> str:
        """Save modified weights to disk."""
        from vauban.export import export_model

        if self._modified_weights is None:
            msg = "No modified weights. Run cut() first."
            raise RuntimeError(msg)
        export_model(
            self.model_path,
            self._modified_weights,
            output_dir,
        )
        return output_dir

    # ── Analysis tools (no model needed) ─────────────────────────────

    @staticmethod
    def score(prompt: str, response: str) -> object:
        """Score a response on 5 axes."""
        from vauban.scoring import score_response

        return score_response(prompt, response)

    @staticmethod
    def classify(text: str) -> object:
        """Classify text against the harm taxonomy."""
        from vauban.taxonomy import score_text

        return score_text(text)

    # ── Reporting tools ──────────────────────────────────────────────

    def report(self, fmt: str = "markdown") -> str:
        """Generate a report from the last audit."""
        if self._audit is None:
            msg = "No audit results. Run audit() first."
            raise RuntimeError(msg)
        if fmt == "markdown":
            from vauban.audit import audit_result_to_markdown

            return audit_result_to_markdown(self._audit)
        if fmt == "dict":
            import json

            from vauban.audit import audit_result_to_dict

            return json.dumps(audit_result_to_dict(self._audit), indent=2)
        msg = f"Unknown format: {fmt!r}. Use 'markdown' or 'dict'."
        raise ValueError(msg)

    def report_pdf(self) -> bytes:
        """Generate a PDF report from the last audit."""
        if self._audit is None:
            msg = "No audit results. Run audit() first."
            raise RuntimeError(msg)
        from vauban.audit_pdf import render_audit_report_pdf

        return render_audit_report_pdf(self._audit)

    # ── Internal ─────────────────────────────────────────────────────

    def _ensure_direction(self) -> None:
        """Lazily measure direction if not already done."""
        if self._direction is None:
            self.measure()

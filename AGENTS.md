<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Agents Guidelines

Vauban is a local-first behavioral auditing toolkit for open language models.
Its motto is: "What changes when models change?"

Model transformations are the object. Access-aware auditing is the method.
Vauban Reports are the artifact.

Treat it as infrastructure for model behavior change reports: it should help a
researcher or engineer answer what changed in model behavior, under which
conditions, where that change appears internally, and whether the finding is
reproducible.

The flagship artifact is the Model Behavior Change Report: a reproducible report
that compares behavior across prompts, checkpoints, fine-tunes, steering
interventions, quantization variants, or post-training runs. Every major feature
should make that report more accurate, more reproducible, or easier to interpret.

## Core Working Rules

1. Justify decisions explicitly.
Every meaningful design or implementation choice must include a short reason.
If a tradeoff is unclear, state the competing options and why one was chosen.

2. Do not assume; verify.
Before making claims about behavior, performance, correctness, or architecture:
- inspect the local code
- run the relevant tests or checks when possible
- measure behavior instead of inferring it
- distinguish verified facts from hypotheses

3. Reproduce before fixing.
When debugging, reproduce the issue first with the smallest available command,
test, config, or fixture. If reproduction is impossible, say what evidence is
missing and keep the fix conservative.

4. Validate or debunk.
If an optimization, safety claim, behavior claim, or architecture claim is
proposed, try to confirm or falsify it with a focused benchmark, a minimal
behavior suite, a profiler trace, or direct code inspection with cited evidence.

5. Surface uncertainty.
If something has not been verified, say so plainly and describe what would
verify it.

## Product North Star

Build Vauban around behavioral auditing, not around bypassing, exploiting, or
showcasing model manipulation. Activation steering, direction discovery, cut,
guard, SIC, soft prompts, model-diffing methods, and surface mapping are
supporting instruments. The product surface is the behavioral diff and the model
behavior change report.

Center model transformations:
- fine-tuning and post-training
- checkpoint updates
- prompt-template or system-wrapper changes
- quantization and deployment packaging
- model merges, soups, and adapter composition
- controlled steering or runtime interventions

Use "model diffing" as the technical neighborhood, not the primary brand.
Vauban should lead with behavioral diffs, model behavior regression testing, and
Model Behavior Change Reports. Internal model-diffing methods matter when they
improve the human-facing report.

The explicit questions Vauban should help answer are:
- We fine-tuned a model. Did it become weird?
- We changed a prompt template. Did safety regress?
- We quantized this model. Did behavior drift?
- This new checkpoint benchmarks better. What did it sacrifice?
- Can we ship this model update without surprises?

The conceptual center is:

```bash
vauban diff model_before model_after --suite behavior_suite.toml --report report.md
```

The CLI remains TOML-first, but every TOML report workflow should be shaped
around that same comparison: before, after, suite, evidence, findings,
recommendation, and reproducibility.

Every feature should answer at least one of these questions:
- What behavior changed?
- Under what conditions did it change?
- Where is there internal evidence of that change?
- Is the change stable across prompts, prompt families, models, or runs?
- Is the change desirable, undesirable, or ambiguous?
- Can the finding be reproduced from config, data, code version, and outputs?

If a feature does not support those questions, defer it.

## Access-Aware Claims

Claim strength depends on access. Do not imply causal or internal conclusions
from black-box outputs alone.

Use this ladder when designing features and writing reports:
- One model or endpoint snapshot supports a behavioral profile.
- Two output traces or run reports support a black-box behavioral diff.
- Logprobs support a distributional diff when they are available.
- Local weights and activations support activation diagnostics.
- Base plus transformed weights support the strongest model-change audit.

When base weights, training data, checkpoints, logits, or activations are
missing, say that explicitly in the report and narrow the conclusion. The
no-base-model problem is a reporting constraint, not the whole project.

## Responsible Framing

Use behavioral auditing language by default:
- "behavioral diff", "model behavior change report", "behavior report",
  "refusal boundary", "over-refusal", "activation diagnostic",
  "controlled intervention", "defensive analysis"

Avoid leading with adversarial or bypass framing:
- do not present Vauban as a jailbreak toolkit
- do not present Vauban as a safety-bypass toolkit
- do not optimize public examples around unredacted harmful prompts
- do not add exploit prompt packs or model-specific bypass recipes

Adversarial modules are allowed when they serve defensive auditing,
reproducibility, or side-effect measurement. Public reports should prefer
aggregate metrics, redacted examples, benign stand-ins, limitations, and safety
notes.

## Architecture Principles

### TOML-first configuration

The primary CLI is:

```bash
vauban <config.toml>
```

All pipeline behavior belongs in the TOML file. Utility commands exist for
scaffolding, comparison, validation, schema generation, lineage viewing, and
reference:

```bash
vauban init --mode <mode>
vauban --validate <config.toml>
vauban schema
vauban diff <dir_a> <dir_b>
vauban tree <experiments_dir>
vauban man [topic]
```

Do not add custom subcommands for ordinary pipeline behavior. Add a TOML section
and wire it through the config and mode system.

### Report-first workflows

TOML-driven report generation is the first-class product surface. Prefer a
shareable config such as `[behavior_report]` run through `vauban <config.toml>`
over adding new subcommands for product workflows. Utility commands such as
`vauban diff` may exist for quick comparisons, but the durable artifact should
come from TOML.

When adding or changing a mode, ask what report artifact it emits and how that
artifact can be compared later.

A Model Behavior Change Report should include, when relevant:
- model metadata
- target change
- suite metadata
- config and code provenance
- high-level findings
- behavioral metrics
- activation-space diagnostics
- before/after or A/B comparison
- representative safe or redacted examples
- recommendation
- limitations and uncertainty
- reproducibility details

Markdown and JSON are the default report targets. HTML and PDF are secondary
formats and should not be required for the core workflow.

### Unix-style modules

Build modular, composable components. Each module should do one thing well.
Components should be easy to pipe, compose, test, and replace.

Keep orchestration separate from:
- config parsing and validation
- model loading and backend concerns
- activation collection
- behavior metrics
- activation metrics
- interventions
- report rendering
- external API evaluation

Prefer small, focused functions and classes over monolithic objects. Interfaces
between components should be explicit and typed: dataclasses, protocols, and
small result types.

### Code organization

The default should be many small files with narrow responsibilities, not large
multi-purpose modules.

- Prefer folders with focused modules over a single large file.
- Keep one main concept per file.
- Use descriptive names that reveal responsibility.
- Avoid `utils` catch-all modules unless the functions are truly generic.
- Prefer composition over deep inheritance or hidden control flow.

Soft limits, unless a clear reason justifies otherwise:
- files should usually stay below roughly 200-300 lines
- functions should usually stay below roughly 40-60 lines
- complex logic should be split into named helpers instead of long branches

When a file grows, split it by responsibility, not arbitrarily.

## TOML Integration Checklist

Every new pipeline module must be wired into TOML before it is considered
complete. No feature ships as Python-only behavior hidden from config.

For a new mode or pipeline section, verify this checklist:

1. Add a frozen, slotted config dataclass in `vauban/types.py`.
2. Add a parser in `vauban/config/_parse_{name}.py`.
3. Register the parser in `vauban/config/_registry.py`.
4. Add validation rules or schema coverage where needed.
5. Add an `EarlyModeSpec` in `vauban/config/_mode_registry.py` with the correct
   phase and `requires_direction` value.
6. Add a mode runner in `vauban/_pipeline/_mode_{name}.py`.
7. Wire the runner through `vauban/_pipeline/_modes.py`.
8. Re-export public types or functions from `vauban/__init__.py` only when they
   are intended as public API.
9. Add loader, registry, mode, validation, and behavior tests.

Use existing tests such as `tests/test_config_registry.py`,
`tests/test_mode_registry.py`, `tests/test_config_loader.py`, and
`tests/test_validation_registry.py` as wiring references.

## Behavior Suites

Behavior suites should be small, explicit, versioned, and safe to publish.
They exist to make behavior claims reproducible, not to maximize shock value.

A suite should define:
- name and description
- categories under test
- metrics to compute
- prompt provenance
- redaction or safety policy for examples
- known limitations

Do not publish exploitative prompt lists. Prefer harmless stand-ins, redacted
examples, public benchmarks with clear provenance, or local private suites that
are not committed.

## Activation Diagnostics

Activation-space work must stay connected to behavior. Direction discovery,
subspaces, DBDI, probes, CAST, guard, SIC, steering, and cut are useful when
they help explain, predict, or test a behavioral change.

Useful diagnostics include:
- layer-wise activation collection
- direction discovery and stability
- similarity between runs or prompt families
- intervention sensitivity near behavioral boundaries
- correlation between activation metrics and observed behavior
- side effects such as over-refusal, verbosity drift, uncertainty drift, or
  capability regression

Do not add activation plots or internal metrics without a clear behavioral
question they help answer.

## Research Workflow

Use prior papers and tools as calibration targets, not as claims of priority.
The strongest Vauban research loop is:

1. A paper or tool claims a model-behavior result.
2. Reproduce the core claim as faithfully as practical.
3. Run the experiment through Vauban.
4. Add a Vauban-native behavior diff or side-effect diagnostic.
5. Report what replicated, what failed, what changed, and what remains uncertain.

When building on a paper, cite it clearly and separate the original method from
Vauban's extension. Keep long bibliographies in docs or report references, not
in this agent guide.

## PyTorch Runtime Rules

PyTorch is the primary portable runtime. Vauban should preserve the properties
that make behavioral auditing reproducible across CPU, CUDA, and MPS targets:
- explicit tensor operations
- direct access to layer activations
- explicit projection and intervention points
- direct safetensors and HuggingFace model I/O where possible
- primitive-level device handling instead of framework-specific product logic

MLX may remain as an optional legacy or reference backend, but it must not define
Vauban's public surface. If a PyTorch/MPS path needs lower-level performance
work, prefer a small custom kernel behind a Vauban primitive instead of moving
product logic back into an MLX-centered design.

Avoid hidden framework magic. If activations are captured, steered, projected,
or compared, make the data flow explicit and testable.

## Validation Workflow

When changing behavior, reports, metrics, modes, or performance-sensitive code:

1. State the claim.
Example: "This metric detects over-refusal drift between checkpoints."

2. Identify the evidence needed.
Example: "Run a small suite with known benign and ambiguous prompts before and
after the change."

3. Run the smallest check that can validate or debunk the claim.

4. Report the result precisely.
If the result is ambiguous, say that it is ambiguous.

Preferred commands before handing off code:

```bash
uv run ruff check .
uv run ty check
uv run pytest
```

For config or CLI changes, also run the narrow command that exercises the
changed path, such as:

```bash
uv run vauban --validate <config.toml>
uv run vauban man <topic>
uv run vauban <behavior_report.toml>
```

If full validation is too expensive or blocked by local model availability,
run the narrowest relevant test subset and state what was not run.

## Testing Expectations

- Add unit tests for logic with clear correctness conditions.
- Add regression tests for config parsing, schema, and mode routing.
- Keep simple reference paths for mathematically sensitive code when practical.
- For optimized paths, compare against the reference path on representative
  inputs.
- Do not merge performance claims without measurements.
- Do not treat a single noisy benchmark run as decision-quality evidence when
  variance is material.
- Behavior reports should be testable as artifacts: stable keys, stable metric
  names, and deterministic formatting where practical.

## Typing Rules

All code must be fully typed.

- Every function parameter must have an explicit type.
- Every function return must have an explicit type.
- Important local variables should be annotated when inference is not obvious.
- `Any` is prohibited. Do not import or use `typing.Any`.
- `None` types must be explicit: use `X | None`, not `Optional[X]`.
- Collections must be typed: use `list[str]`, `dict[str, int]`, and similar.
- Use modern Python 3.12 typing syntax: `X | Y`, `list[X]`, `dict[K, V]`.

All public functions and classes must have docstrings. Imports must stay sorted
according to the configured Ruff rules.

## Communication

When working in this repository:

- explain what you are doing before large edits
- explain why the change is justified
- cite files, tests, configs, measurements, or sources when making claims
- label assumptions, verified facts, and open questions
- keep explanations short, direct, and evidence-based

Do not present guesses as facts. Do not hide uncertainty behind confident prose.

## What To Avoid

- grab-bag features that do not support behavior diffing or reports
- public exploit packs or bypass-oriented examples
- architecture claims without checking the code
- behavior claims without a suite, fixture, report, or reproduction
- performance claims without measurement
- large opaque files
- hidden global state or silent fallbacks
- premature abstraction that obscures data flow
- mixing prototype code, benchmark code, and production code in the same module

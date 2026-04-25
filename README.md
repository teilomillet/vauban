<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Vauban

What changes when models change?

Open-source behavioral diffing for language models.

Vauban helps researchers and engineers audit model transformations:
fine-tunes, checkpoint updates, merges, prompt wrappers, steering interventions,
quantization variants, and post-training runs. It asks what changed
behaviorally, what evidence supports that claim, and how strong the claim can be
given the access available.

Model transformations are the object. Access-aware auditing is the method.
Vauban Reports are the artifact.

Named after [Sébastien Le Prestre de Vauban](https://en.wikipedia.org/wiki/Vauban),
the military engineer who worked both siege and fortification. Vauban does the
same for model behavior: map the boundary, test the change, inspect the
evidence, and produce a report.

## What it does

Vauban is a TOML-first CLI for model behavior change reports:

- compare behavior across model states, prompts, and interventions
- measure refusal, over-refusal, uncertainty, compliance, and side effects
- inspect activation-space evidence when it helps explain a behavioral change
- state the access level behind each claim instead of over-reading the evidence
- produce JSON and Markdown Model Behavior Change Reports that can be shared and rerun
- keep controlled interventions, defenses, and stress tests as supporting tools

The primary interface is not a pile of subcommands. It is:

```bash
vauban <config.toml>
```

All pipeline behavior lives in the TOML file.

Vauban is related to model diffing, but the product surface is narrower and more
practical: behavioral diffs and model behavior change reports. Model-diffing
methods are useful when they help answer what changed, why it matters, and what
evidence supports the claim.

The core question is practical model-change management:

- We fine-tuned a model. Did it become weird?
- We changed a prompt template. Did safety regress?
- We quantized this model. Did behavior drift?
- This new checkpoint benchmarks better. What did it sacrifice?
- Can we ship this model update without surprises?

Conceptually, the center is:

```bash
vauban diff model_before model_after --suite behavior_suite.toml --report report.md
```

The durable interface remains TOML-first: configs should encode the same
comparison, suite, evidence, limitations, and output report in a shareable file.

## Access-aware auditing

Vauban should not make the same claim from every evidence source. The report
language depends on what you can observe:

| Access | What Vauban can support | Claim strength |
|---|---|---|
| One model or endpoint snapshot | Behavioral profile | "This is what the model did under this suite." |
| Two output traces or run reports | Behavioral diff | "Behavior changed across these observed snapshots." |
| Endpoint with logprobs | Distributional diff | "Token probabilities shifted in these cases." |
| Local weights and activations | Activation diagnostics | "This internal signal correlates with the behavior." |
| Base plus transformed model | Model-change audit | "This transformation changed behavior and internals this way." |

The no-base-model problem is not the product. It is the discipline: if the base
model, training data, checkpoints, logits, or activations are unavailable, the
report must say so and narrow its conclusions.

The longer thesis is in
[What Changes When Models Are Changed?](docs/research/what-changes-when-models-are-changed.md).

## What Vauban is not

Vauban is not a jailbreak toolkit, not a safety-bypass toolkit, and not a claim
that a model is safe. It is intended for responsible model behavior auditing,
defensive analysis, and reproducible research on model changes.

## Requirements

- Apple Silicon Mac
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Install

Install the released CLI:

```bash
uv tool install vauban
```

If your shell cannot find `vauban`, update your shell config once:

```bash
uv tool update-shell
```

Then open a new shell and check the command:

```bash
vauban --help
vauban man workflows
```

For development from this repo:

```bash
uv tool install --editable .
```

Contributor workflow and repo policy live in [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick Start

Pick a goal — the manual will tell you which sections to use:

```bash
vauban man workflows
```

Then scaffold a config and go:

```bash
vauban man quickstart
```

Scaffold a starter config:

```bash
vauban init --mode default --output run.toml
```

Scaffold a named indirect prompt-injection benchmark:

```bash
vauban init --scenario share_doc --output share_doc.toml
```

Or run one of the checked-in canonical benchmarks directly:

```bash
vauban --validate examples/benchmarks/share_doc.toml
vauban examples/benchmarks/share_doc.toml
```

Or keep the benchmark choice directly in TOML and override only what you need:

```toml
[environment]
scenario = "share_doc"
max_turns = 5
```

If you want Vauban to judge a flywheel run against an explicit deployment goal,
declare that goal in TOML too:

```toml
[objective]
name = "customer_support_gate"
deployment = "customer_support"
summary = "Preserve support quality while resisting refund abuse."
access = "api"
preserve = ["explain refund policy", "escalate billing issues"]
prevent = ["unauthorized refund", "PII disclosure"]

[[objective.safety]]
metric = "evasion_rate"
threshold = 0.05

[[objective.utility]]
metric = "utility_score"
threshold = 0.90
```

Today, `[objective]` is enforced quantitatively by `[flywheel]` runs and recorded
in `flywheel_report.json` as an explicit pass/fail assessment.

If preserved utility should come from a fixed benign inquiry set instead of the
generated flywheel worlds, add:

```toml
benign_inquiry_source = "dataset"
benign_inquiries = "data/customer_support_benign.jsonl"
```

All pipeline modes can be scaffolded. See the full list with:

```bash
vauban init --help
```

The full checked-in benchmark pack lives under `examples/benchmarks/`.

Validate before a real run:

```bash
vauban --validate run.toml
```

Then run the pipeline:

```bash
vauban run.toml
```

By default, output goes to `output/` relative to the TOML file.

## Model Behavior Change Reports

The flagship artifact is a Model Behavior Change Report: a reproducible behavioral
change report for a model update, fine-tune, prompt/template change, controlled
intervention, quantization variant, or post-training run.

Reports are TOML-driven. A minimal standalone report does not need to load a
model; it assembles declared evidence into JSON and Markdown:

```toml
[behavior_report]
title = "Refusal Boundary Model Behavior Change Report"
target_change = "base -> instruction-tuned"
findings = [
  "Target-task performance improved.",
  "Refusal behavior changed.",
  "Over-refusal increased in ambiguous benign cases.",
  "Uncertainty expression decreased.",
]
recommendation = "Do not deploy without additional benign-request regression testing."
limitations = [
  "Small suite; not a safety certification.",
  "Representative examples are safe or redacted.",
]

[behavior_report.baseline]
label = "base"
model_path = "mlx-community/example-base"
role = "baseline"

[behavior_report.candidate]
label = "instruct"
model_path = "mlx-community/example-instruct"
role = "candidate"

[behavior_report.suite]
name = "refusal-boundary"
description = "Measures refusal, over-refusal, and ambiguous compliance."
categories = ["safety_refusal", "benign_request", "ambiguous_request"]
metrics = ["safety_refusal_rate", "over_refusal_rate"]

[[behavior_report.metrics]]
name = "safety_refusal_rate"
model_label = "base"
category = "safety_refusal"
value = 0.52
polarity = "higher_is_better"

[[behavior_report.metrics]]
name = "safety_refusal_rate"
model_label = "instruct"
category = "safety_refusal"
value = 0.84
polarity = "higher_is_better"
```

Run it like any other Vauban config:

```bash
vauban --validate examples/behavior_report.toml
vauban examples/behavior_report.toml
```

The output is `behavior_report.json` plus `behavior_report.md` in `[output].dir`.
See [examples/behavior_report.toml](examples/behavior_report.toml) for a fuller
safe/redacted example.

## Minimal TOML

This is the minimal config the code accepts for the default pipeline:

```toml
[model]
path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

[data]
harmful = "default"
harmless = "default"
```

`[model].path` is required for model-loading runs. Standalone report modes such
as `[behavior_report]` do not need `[model]`.

`[data].harmful` and `[data].harmless` are required for most runs. `"default"` uses Vauban's bundled prompt sets.

You can also choose the output directory explicitly:

```toml
[output]
dir = "runs/baseline"
```

## Experiment Tech Tree

Vauban has built-in experiment lineage tracking through an optional `[meta]` section. This metadata does not change pipeline execution. It exists so you can organize runs as a tech tree.

Minimal example:

```toml
[meta]
id = "cut-alpha-1"
title = "Baseline cut, alpha 1.0"
status = "baseline"
parents = ["measure-v1"]
tags = ["cut", "baseline"]
date = 2026-03-02
notes = "First stable reference run."
```

Verified status values are:

```text
archived, baseline, dead_end, promising, superseded, wip
```

If `[meta].id` is omitted, Vauban uses the TOML filename stem.

Render the tree from a directory of TOML configs:

```bash
vauban tree experiments/
vauban tree experiments/ --format mermaid
vauban tree experiments/ --status promising
vauban tree experiments/ --tag gcg
```

Each run also appends an `experiment_log.jsonl` file inside the configured output directory with the resolved config path, pipeline mode, report files, metrics, and selected `[meta]` fields.

## How TOML Drives Vauban

`vauban <config.toml>` loads one TOML file and decides what to do from the sections you include.

The default path is:

1. measure
2. cut
3. export

You extend that run by adding more sections. Common examples:

- `[behavior_report]` creates a standalone model behavior change report from declared evidence.
- `[eval]` adds post-cut evaluation reports.
- `[surface]` adds before/after refusal-surface mapping.
- `[detect]` adds hardening detection during measurement.
- some sections switch Vauban into dedicated mode-specific runs instead of the default pipeline.

If you know what you want to do but not which sections to use:

```bash
vauban man workflows
```

For field-level reference on any section:

```bash
vauban man cast
vauban man softprompt
vauban man measure
```

For mode precedence:

```bash
vauban man modes
```

## Commands You Will Actually Use

Inspect the manual:

```bash
vauban man workflows
vauban man quickstart
vauban man cast
vauban man all
```

Scaffold configs:

```bash
vauban init --help
vauban init --mode default --output run.toml
vauban init --mode probe --output probe.toml
```

Validate config and prompt files without loading model weights:

```bash
vauban --validate run.toml
```

Export the current JSON Schema for editor tooling:

```bash
vauban schema
vauban schema --output vauban.schema.json
```

Compare two run directories as a utility check:

```bash
vauban diff run_a run_b
vauban diff --format markdown run_a run_b
vauban diff --threshold 0.05 run_a run_b
```

`vauban diff --threshold ...` is a CI gate: it exits non-zero if any absolute metric delta crosses the threshold.
For durable behavior-change artifacts, prefer a TOML `[behavior_report]` config
so the comparison, evidence, examples, limitations, and provenance travel
together.

Render the experiment lineage tree:

```bash
vauban tree experiments/
vauban tree experiments/ --format mermaid
vauban tree experiments/ --status promising
```

## Data Formats

Verified by the generated manual:

- prompt JSONL for `[data]` and `[eval]`: one JSON object per line with a `prompt` key
- surface JSONL for `[surface].prompts`: requires `label` and `category`, plus either `prompt` or `messages`
- refusal phrase files: plain text, one phrase per line
- relative paths resolve from the TOML file's directory

Minimal prompt dataset example:

```jsonl
{"prompt":"What is the capital of France?"}
{"prompt":"Write a haiku about rain."}
```

## Notes On Verification

This README is aligned to the code in this repo:

- package name: `vauban`
- console script: `vauban = vauban.__main__:main`
- verified commands: `vauban <config.toml>`, `--validate`, `schema`, `init`, `diff`, `tree`, `man`
- verified standalone report mode: `[behavior_report]`
- verified manual topics and scaffolded modes were checked against the live CLI help and generated manual

The current README previously had some stale mode/output claims; this version removes those and points readers to `vauban man ...` for the parts generated directly from code.

## Python API (Session)

For programmatic use, the `Session` class wraps a loaded model with tool discovery, prerequisite tracking, and structured results.

```python
from vauban.session import Session

s = Session("mlx-community/Qwen2.5-1.5B-Instruct-bf16")
s.tools()           # discover all capabilities
s.guide("audit")    # step-by-step workflow
s.describe("cast")  # detailed tool info with current status
s.catalog()         # all tools grouped by category
```

### Tools

| Method | Returns | What it does |
|--------|---------|-------------|
| `s.measure()` | `DirectionResult` | Extract refusal direction from activations |
| `s.detect()` | `DetectResult` | Check if model is hardened against abliteration |
| `s.audit(thoroughness=...)` | `AuditResult` | Full red-team: jailbreak + softprompt + surface + guard |
| `s.evaluate()` | `EvalResult` | Refusal rate + perplexity + KL divergence |
| `s.probe("prompt")` | `ProbeResult` | Per-layer projection onto refusal direction |
| `s.scan("text")` | `ScanResult` | Per-token injection detection |
| `s.surface()` | `SurfaceResult` | Map refusal boundary across prompt categories |
| `s.cast("prompt", threshold=0.3)` | `CastResult` | Conditional activation steering (defense) |
| `s.sic(["prompt", ...])` | `SICResult` | Iterative input sanitization (defense) |
| `s.steer("prompt", alpha=-1.0)` | `str` | Unconditional activation steering |
| `s.cut(alpha=1.0)` | `dict[str, Array]` | Remove refusal direction from weights |
| `s.export("output/")` | `str` | Save modified model to disk |
| `s.classify("text")` | harm scores | Score against 13-domain harm taxonomy |
| `s.score("prompt", "response")` | score result | 5-axis quality assessment |
| `s.report()` | `str` | Markdown report from audit findings |

### Result Types

**DirectionResult** (from `measure`): `direction` (Array, shape d_model), `layer_index` (best layer), `cosine_scores` (per-layer separation), `d_model`, `model_path`.

**CastResult** (from `cast`): `text` (generated output), `interventions` (tokens where CAST steered, 0 = defense didn't engage), `considered` (total tokens), `projections_before`/`projections_after` (per-layer).

**SICResult** (from `sic`): `prompts_clean` (sanitized text), `prompts_blocked` (bool per prompt), `initial_scores`/`final_scores` (direction projection), `total_blocked`/`total_sanitized`/`total_clean`.

**DetectResult** (from `detect`): `hardened` (bool), `confidence` (0.0-1.0), `effective_rank` (>1.5 suggests hardening), `evidence` (list of strings).

**AuditResult** (from `audit`): `overall_risk` ("critical"/"high"/"medium"/"low"), `findings` (list of AuditFinding), `jailbreak_success_rate`, `softprompt_success_rate`, `surface_refusal_rate`.

### Decision Guide

| I want to... | Use |
|--------------|-----|
| Understand what a model refuses | `measure()` then `surface()` |
| Check if a model is hardened | `detect()` |
| Full safety audit | `audit()` then `report()` |
| Defend against adversarial inputs | `measure()` then `sic()` + `cast()` |
| Remove refusal permanently | `measure()` then `cut()` then `export()` |
| Score response quality | `score("prompt", "response")` (no model needed) |

### Prerequisites

```
model (loaded at Session init)
  ├── measure() → direction
  │     ├── probe(), scan(), surface()
  │     ├── steer(), cast(), sic()
  │     ├── evaluate()
  │     └── cut() → modified_model → export()
  ├── detect(), audit() → report()
  └── jailbreak()
classify(), score() → no prerequisites
```

## Documentation

Full docs: [docs.vauban.dev](https://docs.vauban.dev/)

| Resource | Description |
|----------|-------------|
| [Concepts](https://docs.vauban.dev/concepts/activation-geometry/) | Domain knowledge: activation geometry, refusal directions, measurement, steering |
| [Capabilities](https://docs.vauban.dev/capabilities/understand-your-model/) | What you can do: understand, defend, stress-test, modify |
| [Principles](https://docs.vauban.dev/principles/attack-defense-duality/) | Design philosophy: duality, composability, reproducibility |
| [Spinning Up in Abliteration](https://docs.vauban.dev/class/) | Eight-part progressive curriculum |
| [Configuration Reference](https://docs.vauban.dev/config/) | TOML field reference |
| [`examples/config.toml`](examples/config.toml) | Annotated example config |

## License

Apache-2.0

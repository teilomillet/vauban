<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# vauban

An MLX-native toolkit for understanding and reshaping how language models behave on Apple Silicon.

Named after [Sébastien Le Prestre de Vauban](https://en.wikipedia.org/wiki/Vauban), the military engineer who worked both siege and fortification. Vauban does the same for model behavior: measure it, cut it, probe it, steer it, or harden it.

## What it does

Vauban is a TOML-first CLI for workflows built around activation-space geometry:

- measure behavioral directions from model activations
- cut or sparsify those directions in weights
- probe and steer models at runtime
- map refusal surfaces before and after intervention
- run defense, sanitization, optimization, and attack loops

The primary interface is not a pile of subcommands. It is:

```bash
vauban <config.toml>
```

All pipeline behavior lives in the TOML file.

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

## Minimal TOML

This is the minimal config the code accepts for the default pipeline:

```toml
[model]
path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

[data]
harmful = "default"
harmless = "default"
```

`[model].path` is required.

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

Compare two run directories:

```bash
vauban diff run_a run_b
vauban diff --format markdown run_a run_b
vauban diff --threshold 0.05 run_a run_b
```

`vauban diff --threshold ...` is a CI gate: it exits non-zero if any absolute metric delta crosses the threshold.

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

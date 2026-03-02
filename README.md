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
vauban man quickstart
vauban tree --help
```

For development from this repo:

```bash
uv tool install --editable .
```

## Quick Start

Start with the built-in manual:

```bash
vauban man
vauban man quickstart
vauban man commands
```

Scaffold a starter config:

```bash
vauban init --mode default --output run.toml
```

The verified scaffolded modes are:

```text
cast, circuit, default, depth, detect, features, optimize, probe, sic, softprompt, steer, surface
```

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

Mode precedence is not trivial and changes with the code, so rely on the generated manual instead of memorizing it:

```bash
vauban man modes
```

For field-level reference, use:

```bash
vauban man model
vauban man data
vauban man measure
vauban man cut
vauban man eval
vauban man surface
vauban man output
```

## Commands You Will Actually Use

Inspect the manual:

```bash
vauban man
vauban man quickstart
vauban man commands
vauban man formats
vauban man output
vauban tree --help
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

## Documentation

Full docs: [vauban.readthedocs.io](https://vauban.readthedocs.io/)

| Resource | Description |
|----------|-------------|
| [Spinning Up in Abliteration](https://vauban.readthedocs.io/class/index/) | Seven-part progressive curriculum |
| [Getting Started](https://vauban.readthedocs.io/getting-started/) | Guided walkthrough |
| [Configuration Reference](https://vauban.readthedocs.io/config/) | TOML field reference |
| [Surface Mapping](https://vauban.readthedocs.io/surface/) | Surface mapping reference |
| [`examples/config.toml`](examples/config.toml) | Annotated example config |

## License

Apache-2.0

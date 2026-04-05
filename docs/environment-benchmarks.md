---
title: "Environment Benchmarks"
description: "Canonical scenario-backed environment benchmark configs for Vauban."
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Environment Benchmarks

Vauban ships a checked-in benchmark pack for the built-in scenario-backed
environment harness. These are not just scaffolds: they are canonical TOML
configs you can validate, run, diff, and keep under version control.

The benchmark pack lives under `examples/benchmarks/`.

## What The Pack Gives You

Each benchmark config includes:

- a named `[environment].scenario`
- a shared GCG softprompt baseline
- baseline `[meta]` metadata for experiment lineage
- an output path under `output/benchmarks/<scenario>`

That makes the scenarios useful both as runnable examples and as stable
regression fixtures.

## Run One

Validate the config first:

```bash
vauban --validate examples/benchmarks/share_doc.toml
```

Run it:

```bash
vauban examples/benchmarks/share_doc.toml
```

If you want a fresh copy to modify, scaffold the same benchmark shape:

```bash
vauban init --scenario share_doc --output share_doc.toml
```

## Included Benchmarks

| Scenario | Example file | Purpose |
|----------|--------------|---------|
| `data_exfil` | `examples/benchmarks/data_exfil.toml` | Browser-style inbox exfiltration via malicious summarization. |
| `fedex_phishing` | `examples/benchmarks/fedex_phishing.toml` | Hidden phishing approval inside a landing-page review. |
| `garage_door` | `examples/benchmarks/garage_door.toml` | Home-assistant device activation during benign email summarization. |
| `ignore_email` | `examples/benchmarks/ignore_email.toml` | Suppressing a critical deadline email from a workspace summary. |
| `salesforce_admin` | `examples/benchmarks/salesforce_admin.toml` | Unauthorized privileged navigation inside a training review task. |
| `share_doc` | `examples/benchmarks/share_doc.toml` | External sharing of a sensitive drive file during document summarization. |

## How To Customize

Keep the `scenario = "<name>"` line and override only the fields you need.
For example:

```toml
[environment]
scenario = "share_doc"
max_turns = 7
rollout_every_n = 2
```

Explicit TOML values override the built-in scenario defaults while preserving
the benchmark identity.

## Why Check In TOMLs

Checked-in benchmark configs are useful because they give you:

- repeatable regression cases
- exact file paths for docs and CI
- stable experiment metadata
- baseline configs that can be diffed over time

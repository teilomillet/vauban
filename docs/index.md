<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Vauban

An MLX-native toolkit for understanding and reshaping how language models behave on Apple Silicon.

Named after [Sébastien Le Prestre de Vauban](https://en.wikipedia.org/wiki/Vauban) — the military engineer who mastered both siege and fortification. Vauban works both sides: break a model's safety alignment, or harden it against attacks.

## What it does

Refusal in language models is mediated by a single direction in activation space ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)). Vauban operates directly on this geometry:

- **Measure** a behavioral direction from the model's activations
- **Cut** it from the weights (abliteration)
- **Probe** per-layer projections to see what the model encodes
- **Steer** generation at runtime by modifying activations mid-forward-pass
- **CAST** runtime generation with conditional activation steering rules
- **Map** the full refusal surface across diverse prompts
- **Optimize** cut parameters automatically (Optuna search)
- **Soft-prompt** — optimize learnable prefixes in embedding space (GCG, continuous, EGD)
- **Sanitize** inputs iteratively before they reach the model (SIC)
- **Detect** whether a model has been hardened against abliteration

Everything runs natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx) — no CUDA, no Docker, no hooks. All configuration lives in TOML files.

## Install

```bash
pip install vauban
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install vauban
```

## Quick start

Generate a starter config and run:

```bash
vauban init --mode default --output run.toml
vauban --validate run.toml
vauban run.toml
```

See the [Getting Started](getting-started.md) guide for a full walkthrough.

## Learn

The [Spinning Up in Abliteration](class/index.md) course is a seven-part progressive curriculum — from geometric intuition to production pipelines. Start there if you want to understand what abliteration is and how it works.

## Pipeline modes

| Section | What it does | Output |
|---------|-------------|--------|
| *(default)* | Measure refusal direction, cut it, export modified model | model directory |
| `[surface]` | Map the refusal landscape before and after | `surface_report.json` |
| `[eval]` | Refusal rate, perplexity, KL divergence | `eval_report.json` |
| `[detect]` | Check if a model has been hardened against abliteration | `detect_report.json` |
| `[depth]` | Deep-thinking token analysis | `depth_report.json` |
| `[probe]` | Per-layer projection inspection | `probe_report.json` |
| `[steer]` | Runtime steered generation | `steer_report.json` |
| `[cast]` | Conditional activation steering generation | `cast_report.json` |
| `[optimize]` | Optuna search for best cut parameters | `optimize_report.json` |
| `[softprompt]` | Optimize learnable prefixes in embedding space | `softprompt_report.json` |
| `[sic]` | Iterative input sanitization (SIC) | `sic_report.json` |
| `[api_eval]` | Remote API suffix evaluation | `api_eval_report.json` |
| `[meta]` | Experiment metadata (no pipeline effect) | `python -m vauban.tree` |

## License

Apache-2.0

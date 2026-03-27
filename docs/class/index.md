<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Spinning Up in Abliteration

A progressive, hands-on curriculum for understanding and applying refusal-direction manipulation in large language models — built on [vauban](https://github.com/teilomillet/vauban) and Apple Silicon via MLX.

## What This Is

This is an eight-part educational series that takes you from first intuition to production workflows in **abliteration** — the discovery that safety refusal in language models is mediated by a single direction in activation space, and the techniques to measure, remove, and defend against that direction.

Each part follows a **Define → Derive → Code → Extend** progression. Theory is conversational but formally correct; every code example calls the real vauban API (no pseudocode). By the end, you will be able to measure refusal directions, perform weight surgery, map refusal surfaces, run soft prompt attacks, deploy defenses, and optimize production pipelines — all from a single Mac.

## Who This Is For

- **ML researchers** studying alignment, interpretability, or adversarial robustness
- **Security engineers** evaluating LLM safety boundaries
- **Graduate students** looking for a hands-on entry point to mechanistic interpretability
- **Engineers** building evaluation or red-teaming pipelines

You should be comfortable reading and writing Python. You do not need prior experience with abliteration, MLX, or mechanistic interpretability — the series introduces everything it uses.

## Prerequisites

**Linear algebra.** You need projections, dot products, cosine similarity, SVD, and rank-1 updates. If you can explain what $\langle a, d \rangle \cdot d$ does to a vector $a$, you are ready.

**Transformer basics.** You should know what a residual stream is, how attention and MLP layers write into it, and what "last-token position" means in a causal language model. Familiarity with `o_proj` and `down_proj` weight matrices is helpful but not required — Part 3 derives everything from scratch.

**Python fluency.** All examples use Python 3.12+ with type annotations. We use `mlx` and `mlx-lm` for model loading and array operations.

## How to Read

**Parts 1–3 are the sequential core.** Read them in order — each builds directly on the previous:

1. **Part 1** builds geometric intuition (no code).
2. **Part 2** runs a full abliteration before you understand every detail.
3. **Part 3** opens the hood on every step.

**Parts 4–7 are independent modules.** After finishing the core, read them in any order based on your interest:

- **Part 4** if you care about coverage and evaluation rigor.
- **Part 5** if you want to go deeper into geometry and detection.
- **Part 6** if you are interested in attacks and defenses.
- **Part 7** if you are building production pipelines.
- **Part 8** if you want weight-diff directions, enhanced CAST, or safety hardening.

## Table of Contents

| Part | Title | Focus |
|------|-------|-------|
| [Part 1](part1_what_is_abliteration.md) | What is Abliteration? | Theory and geometric intuition — no code |
| [Part 2](part2_your_first_abliteration.md) | Your First Abliteration | Hands-on quickstart with the `quick` API |
| [Part 3](part3_under_the_hood.md) | Under the Hood | Step-by-step deep dive into measure, cut, evaluate |
| [Part 4](part4_the_refusal_surface.md) | The Refusal Surface | Surface mapping, coverage scores, quality gates |
| [Part 5](part5_going_deeper.md) | Going Deeper | Depth analysis, subspaces, DBDI, detection, transfer |
| [Part 6](part6_attacks_and_defenses.md) | Attacks and Defenses | Soft prompt attacks and SIC defense |
| [Part 7](part7_production_workflows.md) | Production Workflows | TOML pipelines, optimization, experiment management |
| [Part 8](part8_model_diffing_and_defense.md) | Model Diffing and Enhanced Defense | Weight-diff directions, dual-direction CAST, LoX amplification |

Supporting materials: [References](references.md) · [Glossary](glossary.md)

## Environment Setup

**Hardware.** Apple Silicon Mac (M1 or later). Unified memory means no VRAM ceiling — a 96 GB machine can hold 70B fp16 weights with zero copies.

**Software.**

```bash
# Python 3.12+
python3 --version

# Install vauban (pulls mlx and mlx-lm automatically)
pip install vauban

# Verify
python3 -c "from vauban import quick; print('Ready')"
```

Models are downloaded automatically from HuggingFace on first use. The default model (`mlx-community/Llama-3.2-3B-Instruct-4bit`) is ~2 GB.

## Notation Conventions

Throughout the series, we use:

| Symbol | Meaning |
|--------|---------|
| $W$ | Weight matrix |
| $d$, $\hat{d}$ | Direction vector, unit direction vector |
| $h$, $a$ | Hidden state / activation vector |
| $\alpha$ | Alpha (scaling factor for projection removal) |
| $l$ (superscript) | Layer index |
| $p$ (subscript) | Prompt index |
| $H$, $B$ | Set of harmful / harmless (benign) prompts |
| $d_{\text{model}}$ | Hidden dimension of the model |
| $L$ | Total number of layers |

## Acknowledgements

This series would not exist without:

- **Arditi et al.** for discovering that refusal is mediated by a single direction (arXiv:2406.11717)
- The **MLX team** at Apple for making transformer internals accessible on consumer hardware
- OpenAI's **Spinning Up in Deep RL** for the pedagogical template — accessible entry, rigorous content, tight theory-code coupling
- The **NousResearch** and **Heretic** communities for pushing abliteration techniques forward

All citations are consolidated in the [References](references.md) page.

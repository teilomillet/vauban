---
title: "Measuring Behavioral Directions — Mean-Diff, SVD, DBDI, Weight-Diff"
description: "Four methods to extract behavioral directions from LLM activations: difference-in-means, SVD subspace, DBDI decomposition, and weight-diff between aligned and base models."
keywords: "LLM direction measurement, difference in means, SVD subspace, DBDI, weight diff, behavioral direction extraction"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Measurement

How behavioral directions are extracted from a model's activations or weights.

## The problem

To [abliterate](abliteration.md), [steer](steering.md), or [defend](defense-complementarity.md), you need a direction. Measurement is the process of finding it. Vauban provides four measurement modes, each a different lens on the same underlying [geometry](activation-geometry.md).

> **What are "harmful" and "harmless" prompts?** --- Harmful prompts are inputs that a safety-aligned model should refuse (e.g., instructions for dangerous activities). Harmless prompts are benign inputs (e.g., "explain photosynthesis"). Measurement compares the model's internal response to each set to find what differs --- the [refusal direction](refusal-direction.md). The prompts themselves are dataset files loaded from TOML config.

> **What does "collect activations" mean?** --- Run each prompt through the model and record the residual stream vector at the last token position for every layer. This gives you one vector per (prompt, layer) pair. No generation happens --- just a single forward pass per prompt.

## Difference-in-means (mode: `direction`)

The default. Collect last-token activations on harmful ($H$) and harmless ($B$) prompt sets at each layer $l$, compute the mean difference, and normalize:

$$\hat{d}_l = \frac{\mu_H^l - \mu_B^l}{\|\mu_H^l - \mu_B^l\|}$$

This produces **one direction per layer**. Each direction is a unit vector in $\mathbb{R}^{d_{\text{model}}}$.

To select which layer matters most, Vauban computes cosine separation:

$$s_l = \langle \mu_H^l, \hat{d}_l \rangle - \langle \mu_B^l, \hat{d}_l \rangle$$

The layer with the highest $s_l$ is chosen. In practice, this is typically a middle-to-upper layer where the refusal decision has crystallized but the model has not yet committed to specific output tokens.

**When to use:** Most cases. Difference-in-means is fast, robust, and sufficient for standard abliteration and CAST.

## Subspace (mode: `subspace`)

Instead of collapsing to a single direction, extract the top-$k$ singular vectors of the activation difference matrix via SVD.

Given centered activation matrices $A_H$ and $A_B$, the combined difference matrix $\Delta$ is decomposed:

$$\Delta = U \Sigma V^\top$$

The first $k$ columns of $U$ span the behavioral subspace. This captures structure that a single direction misses --- behaviors that are rank-2 or higher.

**When to use:** When rank-1 abliteration damages model quality (perplexity spikes), suggesting the behavior is not perfectly rank-1. Also useful for hardened models where the refusal signal has been deliberately spread across multiple dimensions (effective rank > 1).

## DBDI (mode: `dbdi`)

Decomposed Behavioral Direction Intervention separates refusal into two components:

- **HDD** (Harm Detection Direction) --- extracted at the **instruction-final** token position. This is where the model recognizes that a request is harmful.
- **RED** (Refusal Execution Direction) --- extracted at the **sequence-final** token position. This is where the model decides to refuse.

Each is extracted via the same difference-in-means procedure, but at different token positions within the prompt.

The critical insight: HDD and RED are often nearly orthogonal. Cutting RED while preserving HDD yields a model that **recognizes** harmful content but does not **refuse** it. This is useful when you want to understand a model's harm-detection capabilities separately from its refusal behavior.

**When to use:** When you need to selectively remove refusal execution while preserving harm detection. Also for research into the internal structure of safety alignment.

## Weight-diff (mode: `diff`)

Instead of collecting activations on prompt datasets, compare the weight matrices of an aligned model against its base (pre-alignment) counterpart. The difference encodes what alignment changed:

$$\Delta W^l = W^l_{\text{aligned}} - W^l_{\text{base}}$$

SVD of $\Delta W^l$ extracts the directions along which alignment modified the weights. The top singular vectors correspond to safety-relevant directions.

This approach comes from the Task Arithmetic (Ilharco et al., 2023) and LoX (Perin et al., 2025) literature. Key advantages:

- **No prompt dataset needed.** The directions come from weights alone.
- **Captures distributed effects.** Weight-diff captures safety modifications across the entire layer, not just the direction visible at one token position.
- **Negative application.** Applying $-\Delta W$ amplifies safety (hardening) rather than removing it.

**When to use:** When you have access to both the aligned and base model weights. Particularly valuable when the aligned model has undergone complex fine-tuning (DPO, RLHF) that may not be fully captured by activation-based measurement.

## Comparison

| Mode | Input | Output | Captures | Cost |
|---|---|---|---|---|
| `direction` | Prompt sets | 1 direction / layer | Rank-1 refusal | 2N forward passes |
| `subspace` | Prompt sets | k directions / layer | Rank-k behavior | 2N forward passes + SVD |
| `dbdi` | Prompt sets | HDD + RED / layer | Detection vs. execution | 2N forward passes |
| `diff` | Two model checkpoints | k directions / layer | Weight-level safety changes | Weight load + SVD |

All four modes produce directions (or subspaces) in $\mathbb{R}^{d_{\text{model}}}$ that can be passed to `cut`, `steer`, `cast`, `probe`, or `sic`. The measurement mode determines what aspect of the model's behavior you are observing; the downstream modules are agnostic to how the direction was found.

---
title: "Abliteration — Removing Refusal from LLM Weights"
description: "Abliteration removes the refusal direction from model weights via rank-1 projection. Learn about alpha scaling, norm-preserve, biprojected cuts, and the tradeoff between refusal removal and quality."
keywords: "abliteration, LLM abliteration, remove refusal, weight modification, rank-1 projection, model surgery"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Abliteration

Removing a behavioral direction from model weights — permanently altering what the model can express.

## What abliteration is

A portmanteau of "ablation" and "obliteration." Abliteration removes a [direction](refusal-direction.md) from the weight matrices that write into the [residual stream](activation-geometry.md), so the model can no longer produce activations with a component along that direction.

The core operation is a rank-1 projection removal applied to every targeted weight matrix.

> **What is a "projection"?** — Projecting a vector onto a direction extracts the component along that direction, like measuring how far north you are regardless of your east-west position. Removing a projection means zeroing out that component — after removal, the vector has no north-south component at all. In abliteration, we remove the refusal component from weight matrices so they cannot write refusal into the residual stream.

> **What does "rank-1" mean?** — A rank-1 matrix has the form $u v^\top$ — the outer product of two vectors. It maps the entire input space onto a single direction. A rank-1 projection removal modifies the weight matrix along exactly one axis, leaving all other axes untouched. This is why abliteration is surgical: it changes one dimension out of thousands.

## The cut operation

Given a [measured](measurement.md) direction $\hat{d}$ and a weight matrix $W$, the cut is:

$$W' = W - \alpha \cdot \hat{d} \, \hat{d}^\top W$$

The term $\hat{d} \, \hat{d}^\top$ is the projection matrix onto $\hat{d}$. Multiplying by $W$ gives the component of each row of $W$ along $\hat{d}$. Subtracting removes that component.

This is applied to the two matrix types that write into the residual stream:

- **`o_proj`** — attention output projection. Each attention head's contribution flows through this matrix.
- **`down_proj`** — MLP output projection. The feedforward block's contribution flows through this matrix.

Other weight matrices (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`) are not cut because they do not write directly into the residual stream.

## Alpha: controlling strength

The parameter $\alpha$ controls how aggressively the direction is removed:

| Alpha | Effect |
|---|---|
| $\alpha = 0$ | No change |
| $\alpha < 1$ | Partial removal — reduces refusal but does not eliminate it |
| $\alpha = 1$ | Full projection removal — zeroes out the refusal component exactly |
| $\alpha > 1$ | Overshoot — removes *more* than the projection, pushing the model actively away from refusal |

Higher $\alpha$ means less refusal but more collateral damage. The trade-off is measured by evaluation metrics: refusal rate should drop while perplexity and KL divergence should not spike.

> **Perplexity** — a measure of how "surprised" the model is by text. Lower perplexity means the model finds the text more predictable and natural. If cutting raises perplexity sharply, the model's language ability has been damaged — it is struggling to produce coherent text.

> **KL divergence** — a measure of how differently two probability distributions behave. After cutting, if the KL divergence between the original and modified model is large, the cut changed too much about the model's behavior beyond just refusal.

In Vauban, $\alpha$ can be set globally or per-layer, allowing finer control over which layers receive aggressive cuts and which are left gentle.

## Layer selection

Not all layers carry the refusal signal equally. Vauban supports three strategies:

- **`all`** — cut every layer. Simple, aggressive.
- **`above_median`** — cut layers whose cosine separation score exceeds the median. Focuses on layers where the refusal signal is strong.
- **`top_k`** — cut only the $k$ layers with the highest separation scores. Most selective.

Layer selection interacts with alpha: cutting fewer layers at higher alpha can produce a different quality/refusal trade-off than cutting many layers at lower alpha.

## Variants

### Norm-preserve

Standard projection removal shrinks the norm of weight rows (because removing a component makes the vector shorter). Norm-preserve rescales each modified row to its original norm:

$$w'_i \leftarrow w'_i \cdot \frac{\|w_i\|}{\|w'_i\|}$$

This prevents the systematic norm reduction that can degrade model quality, especially at high $\alpha$.

### Biprojected

The refusal direction may have a nonzero component along the harmless direction. Standard cut removes the refusal direction, but this also removes part of the harmless direction, damaging benign capabilities.

Biprojected cut orthogonalizes the refusal direction against the harmless direction via Gram-Schmidt before projecting.

> **Gram-Schmidt orthogonalization** — a procedure that adjusts one vector so it becomes perpendicular (orthogonal) to another. Imagine two arrows that partially overlap: Gram-Schmidt removes the overlapping part so the arrows point in completely independent directions. Here, it ensures the refusal direction has no component along the harmless direction before we remove it.

$$\hat{d}_{\perp} = \hat{d} - \langle \hat{d}, \hat{d}_B \rangle \hat{d}_B, \quad \hat{d}_{\perp} \leftarrow \frac{\hat{d}_{\perp}}{\|\hat{d}_{\perp}\|}$$

The projection removal then uses $\hat{d}_{\perp}$ instead of $\hat{d}$, preserving the harmless-direction variance.

### Sparsity

Instead of cutting all targeted weight matrices at full strength, apply cut only to a random subset of rows (controlled by a sparsity parameter). This reduces the total perturbation to the weight matrix while still removing most of the refusal signal.

## Abliteration vs. steering

Abliteration and [steering](steering.md) both manipulate the refusal direction, but at different points:

| | Abliteration | Steering |
|---|---|---|
| **What changes** | Weight matrices | Activations |
| **When** | Once, offline | Every generation, at runtime |
| **Reversible** | No — weights are permanently modified | Yes — no weight changes |
| **Scope** | All inputs | Per-generation (can be conditional) |
| **Output** | A new model checkpoint | Same model, different behavior |

Abliteration is the right choice when you want a permanent model variant. Steering (and CAST) is the right choice when you want runtime control without modifying weights.

!!! warning "Abliteration is permanent"
    The cut modifies weights in-place (or produces a new checkpoint). There is no undo operation. Always evaluate after cutting, and keep the original model weights.

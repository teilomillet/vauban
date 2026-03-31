---
title: "Refusal Direction — The Single Vector That Controls LLM Safety"
description: "A single direction in activation space mediates whether language models refuse requests. Learn how it is extracted via difference-in-means and used for abliteration, steering, and defense."
keywords: "refusal direction, LLM refusal, safety alignment direction, abliteration direction, Arditi et al, activation projection"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Refusal Direction

The single direction in activation space that controls whether a language model refuses.

## One direction mediates refusal

Arditi et al. (2024) demonstrated a remarkable finding: safety refusal in instruction-tuned language models is mediated by a **single direction** in activation space. This is not a distributed, entangled property learned during RLHF --- it is a vector that can be isolated, measured, and manipulated with linear algebra.

The refusal direction $\hat{d}$ has a simple signature:

- Harmful prompts produce activations with a **large positive** projection onto $\hat{d}$.
- Harmless prompts produce activations with a **small or negative** projection.
- The magnitude of the projection predicts whether the model will refuse.

This direction is the foundation for every module in Vauban: [abliteration](abliteration.md) removes it, [CAST](steering.md) monitors it, [SIC](defense-complementarity.md) uses it for detection, and [probing](measurement.md) inspects it layer by layer.

## Extraction via difference-in-means

The refusal direction is extracted by comparing activations from harmful and harmless prompts. Given sets $H$ (harmful) and $B$ (harmless), collect last-token activations at layer $l$ and compute:

$$d_l = \frac{1}{|H|} \sum_{p \in H} a_l(p) \;-\; \frac{1}{|B|} \sum_{p \in B} a_l(p)$$

Then L2-normalize:

$$\hat{d}_l = \frac{d_l}{\|d_l\|}$$

This is the simplest form of [measurement](measurement.md) --- it produces one direction per layer. More sophisticated modes (subspace, DBDI, weight-diff) exist for cases where a single direction is insufficient.

> **What is "a direction in high-dimensional space"?** --- Think of a compass needle. It does not describe a location --- it describes an *orientation*. In 3D, a direction is a unit vector like $(0.6, 0.8, 0)$. In 1536 dimensions, it works the same way: a unit vector that picks out one axis of variation from the space. You cannot visualize 1536 dimensions, but the math is identical. Projecting an activation onto this direction gives a single number --- positive means "refusal-like," negative means "compliance-like."

## Layer selection

Not all layers carry the refusal signal equally. Vauban scores each layer by **cosine separation** --- the gap between the mean projection of harmful activations and the mean projection of harmless activations onto the direction extracted at that layer:

$$s_l = \langle \mu_H^l, \hat{d}_l \rangle - \langle \mu_B^l, \hat{d}_l \rangle$$

The layer with the highest $s_l$ is the most discriminative --- it separates harmful from harmless most cleanly. This layer is selected for cutting and monitoring.

In practice, the refusal signal tends to concentrate in the middle-to-upper layers. Early layers encode mostly syntactic and positional information; the refusal decision crystallizes later.

## What projection tells you

Given an activation $a$ and the refusal direction $\hat{d}$, the scalar projection is:

$$\text{proj} = \langle a, \hat{d} \rangle$$

This number is Vauban's primary observable:

| Projection | Interpretation |
|---|---|
| Large positive | Model is about to refuse --- high refusal activation |
| Near zero | Ambiguous --- model could go either way |
| Negative | Model is about to comply --- refusal suppressed |

[CAST](steering.md) uses this value for zone classification (green/yellow/orange/red). [Surface mapping](surface-mapping.md) records it per-prompt to map where refusal is strong or weak.

## The direction is not the behavior

An important subtlety: the refusal direction does not *cause* refusal. It is a **linear readout** of an internal decision that the model has already made (or is making). The causal structure flows through weights --- attention and MLP layers write refusal-correlated signal into the [residual stream](activation-geometry.md). The direction is how we observe and measure that signal.

This distinction matters for intervention design:

- **Abliteration** modifies weights to prevent the signal from being written. Permanent, affects all inputs.
- **Steering** modifies activations after the signal is written. Temporary, per-generation.
- **CAST** reads the signal and conditionally steers. Adaptive, only intervenes when needed.

## Beyond a single direction

The refusal direction is rank-1 --- a single vector captures nearly all the variance. But some behaviors are richer:

- **Subspace measurement** extracts a $k$-dimensional subspace via SVD, capturing more structure.
- **DBDI** decomposes refusal into two directions: HDD (harm detection, at instruction-final token) and RED (refusal execution, at sequence-final token). Cutting RED while preserving HDD lets the model recognize harm without refusing.
- **Weight-diff** extracts directions from the difference between aligned and base model weights, without needing prompt datasets at all.

These are all covered in [Measurement](measurement.md). The single refusal direction remains the default starting point --- it is the simplest, fastest, and sufficient for most use cases.

!!! note "Shrivastava & Holtzman (2025)"
    Refused knowledge remains linearly decodable from hidden states even when the model refuses to output it. This confirms that refusal is a **gate**, not erasure --- the model knows the answer but chooses not to say it. The refusal direction controls that gate.

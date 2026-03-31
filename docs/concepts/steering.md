---
title: "Activation Steering and CAST — Runtime LLM Behavior Control"
description: "Modify LLM behavior at runtime without changing weights. CAST conditionally steers activations based on refusal direction projection, with tiered alpha and dual-direction detection."
keywords: "activation steering, CAST, conditional activation steering, LLM steering, runtime defense, representation engineering"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Steering

Modifying model behavior at runtime by adding or subtracting directions from activations --- without changing weights.

## The idea

Instead of permanently removing a direction from weights ([abliteration](abliteration.md)), steering modifies activations *during generation*. At a chosen layer, the [refusal direction](refusal-direction.md) is scaled and added to or subtracted from the residual stream:

$$a' = a + \beta \cdot \hat{d}$$

- **Positive $\beta$**: amplify the direction (increase refusal)
- **Negative $\beta$**: suppress the direction (decrease refusal)

> **What is the difference between changing weights and changing activations?** --- Weights are the model's permanent knowledge --- they define the function the model computes. Activations are the intermediate values produced when the model processes a specific input. Changing weights alters behavior for *all* inputs, permanently. Changing activations alters behavior for *this generation only*, and the model returns to its original state afterward. Steering changes activations; abliteration changes weights.

Steering is reversible, per-generation, and requires no modified checkpoint. The original model is never touched.

## Plain steering (steer mode)

The simplest form: apply a fixed $\beta$ at a fixed layer for every token generated. This is useful for exploration --- quickly testing what happens when refusal is amplified or suppressed --- but it is blunt. The same intervention is applied regardless of whether the input is harmful, harmless, or ambiguous.

The activation modification happens mid-forward-pass, between transformer layers:

```
layer l-1 output  -->  add beta * d  -->  layer l input
```

Because MLX uses eager execution, this is a single array operation inside the generation loop. No hooks, no framework registration.

## CAST: Conditional Activation Steering

CAST (Programming Refusal with Conditional Activation Steering, IBM, ICLR 2025) adds a critical refinement: **only steer when the activation signal indicates it is necessary**.

### Zone classification

CAST computes the projection of the current activation onto the refusal direction and classifies it into zones:

| Zone | Projection | Action |
|---|---|---|
| **Green** | Below low threshold | No intervention --- model is complying normally |
| **Yellow** | Between low and mid threshold | Monitor --- light or no steering |
| **Orange** | Between mid and high threshold | Moderate steering applied |
| **Red** | Above high threshold | Full steering applied |

The thresholds are calibrated from a set of harmful and harmless prompts. This means CAST only intervenes when the model's own activations indicate refusal-relevant content --- benign prompts pass through untouched.

### Why conditionality matters

Fixed steering has a cost: applying $-\beta \cdot \hat{d}$ to suppress refusal on every token degrades output quality on harmless prompts (the model is being pushed away from refusal even when it was not going to refuse). CAST avoids this by checking the projection first. On harmless prompts, the projection falls in the green zone and no modification occurs.

This is the defense application of steering: the model runs normally unless its activations indicate it is about to refuse on something it should not refuse on (false positive), or about to comply on something it should refuse (attack).

## Dual-direction steering (AdaSteer)

Standard CAST uses the same direction for both detection (checking the projection) and steering (modifying the activation). AdaSteer separates these:

- **Detect direction**: used to compute the projection and classify the zone.
- **Steer direction**: used for the actual activation modification.

This matters when the optimal detection signal differs from the optimal intervention vector. For example, harm detection might be best captured at the instruction-final token while refusal execution is best steered at a later position. Separating the directions allows more precise gating.

## Adaptive alpha (TRYLOCK)

Fixed steering strength ($\beta$) can overshoot. TRYLOCK identifies a **non-monotonic danger zone**: there exists an intermediate range of projection values where increasing $\beta$ actually makes the model *more* likely to produce harmful content (it overcorrects past compliance into a mode that bypasses safety).

TRYLOCK addresses this with tiered alpha schedules:

| Activation zone | Alpha behavior |
|---|---|
| Low projection | No steering |
| Moderate projection | Gentle alpha |
| Danger zone | Reduced alpha (avoid overshoot) |
| High projection | Full alpha |

The tiers are calibrated empirically. The key insight is that the relationship between steering strength and safety is not monotonic --- more is not always better.

## Steering as defense

In Vauban's [defense stack](defense-complementarity.md), CAST operates as the middle layer:

1. **SIC** sanitizes input *before* generation
2. **CAST** steers activations *during* generation
3. **Guard** monitors output and can rewind *after* generation

CAST's role is real-time: it reads the model's internal state token-by-token and intervenes when the projection crosses a threshold. This catches attacks that SIC missed (because the adversarial content only manifests in activation space, not in the surface-level prompt).

## Steering as exploration

Outside defense, steering is a research tool. Varying $\beta$ across a range maps out how the model responds to amplification and suppression of a direction:

- What happens when you *amplify* refusal on harmless prompts? (Does the model become overly cautious?)
- What happens when you *suppress* refusal on harmful prompts? (Does the model comply? How coherently?)
- Where is the tipping point between compliance and refusal?

This exploration produces behavioral data that [surface mapping](surface-mapping.md) can visualize and quantify.

!!! note "No weight changes"
    Steering never modifies model weights. Every intervention is applied to activations during the forward pass and discarded afterward. The model file on disk is unchanged.

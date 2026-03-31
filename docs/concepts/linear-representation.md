---
title: "Linear Representation Hypothesis — Why LLM Concepts Are Directions"
description: "High-level concepts like refusal, sentiment, and honesty are encoded as linear directions in activation space. This hypothesis underpins abliteration, steering, and probing."
keywords: "linear representation hypothesis, linear probing, concept directions, LLM interpretability, activation space geometry"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Linear Representation Hypothesis

The theoretical foundation for why activation-space interventions work.

## The hypothesis

High-level concepts in neural networks --- refusal, sentiment, truthfulness, style, toxicity --- are represented as **linear directions** in activation space. A concept is not smeared across the network in an inscrutable way; it corresponds to a vector, and the strength of that concept in a given activation is the scalar projection onto that vector.

Formally, for a concept $c$ with associated direction $\hat{d}_c \in \mathbb{R}^{d_{\text{model}}}$ and an activation $a$:

$$\text{strength}(c, a) = \langle a, \hat{d}_c \rangle$$

This is a strong claim. It does not say that *everything* is linear --- only that many behaviorally important properties are. Refusal turns out to be one of the cleanest examples, nearly perfectly rank-1.

> **What does "linear" mean here?** --- Think of a compass needle. No matter how complex the terrain, the compass gives you a single reading: how much you are aligned with north. A linear direction works the same way --- it reduces a high-dimensional activation to a single number (the projection) that tells you "how much of concept X is present." The concept itself may emerge from complex interactions across layers, but its *readout* is a dot product. A non-linear concept would be more like a winding mountain path --- you cannot summarize your position with one number. Refusal is a compass; some behaviors are paths.

## Evidence

The evidence for linear representation comes from multiple independent lines of work:

**Probing classifiers.** Train a linear classifier on activations to predict a concept label (e.g., "is this prompt harmful?"). If a linear probe achieves high accuracy, the concept is linearly decodable. This has been shown for sentiment, factuality, part-of-speech, entity type, and many other properties.

**Abliteration.** Arditi et al. (2024) showed that removing a single direction from weight matrices eliminates refusal behavior. If refusal were non-linear, a rank-1 projection would not suffice --- yet it does, cleanly and reliably.

**Linearly decodable refused knowledge.** Shrivastava & Holtzman (2025) demonstrated that even when a model refuses to answer, the correct answer is linearly decodable from its hidden states. Simple linear probes extract the refused content. This confirms the [refusal direction](refusal-direction.md) is a gate, not an eraser.

**Steering vectors.** Adding or subtracting a direction from activations reliably shifts model behavior in the expected direction. If the representation were non-linear, linear perturbations would produce unpredictable results.

## Three operations

If a concept is linear, three operations become possible:

### Read: projection

Compute $\langle a, \hat{d}_c \rangle$ to measure the concept's strength in an activation. This is what Vauban's `probe` mode does --- it runs a forward pass and reports per-layer projections onto a direction.

### Remove: orthogonal projection

Project the direction out of weight matrices:

$$W' = W - \alpha \cdot \hat{d} \cdot \hat{d}^\top W$$

The modified weight matrix $W'$ cannot write the removed direction into the [residual stream](activation-geometry.md). This is [abliteration](abliteration.md).

### Add: vector addition

During generation, add a scaled direction to activations:

$$a' = a + \beta \cdot \hat{d}$$

Positive $\beta$ amplifies the concept; negative $\beta$ suppresses it. This is [steering](steering.md).

All three operations are linear algebra on vectors in $\mathbb{R}^{d_{\text{model}}}$. No gradient computation, no fine-tuning, no retraining.

## Limitations

Not everything is linear. The hypothesis has known boundaries:

**Subspace, not direction.** Some behaviors are better described by a $k$-dimensional subspace than a single direction. Vauban's subspace [measurement](measurement.md) mode extracts the top-$k$ singular vectors to capture richer structure. If a concept requires $k > 1$ dimensions, a single direction misses the off-axis components.

**Detection vs. execution.** DBDI shows that "refusal" is really two things: harm *detection* (HDD) and refusal *execution* (RED). These correspond to different directions at different token positions. Treating refusal as a single direction conflates them. Separating them allows finer-grained intervention --- cut RED while preserving HDD.

**Distributed effects across layers.** A direction extracted at one layer may not capture contributions from other layers. Weight-diff measurement (LoX, Lermen et al.) operates on weight matrices directly, capturing distributed safety effects that activation-based measurement may miss.

**Non-linear interactions.** Compositional behaviors --- "be helpful *unless* the request is harmful" --- involve conditional logic that a single linear direction cannot represent. The linear representation hypothesis describes the components; the interactions between components may be non-linear.

!!! tip "Practical implication"
    When a rank-1 direction does not fully capture a behavior (e.g., cut reduces refusal but also damages quality), try subspace or DBDI measurement. If the behavior is not well-approximated by *any* low-rank subspace, activation-space intervention may not be the right tool.

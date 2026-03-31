---
title: "Glossary — LLM Safety and Abliteration Terms Defined"
description: "Definitions for abliteration, CAST, DBDI, GCG, refusal direction, steering, SIC, and 30+ terms used in LLM safety research and activation-space geometry."
keywords: "LLM glossary, abliteration glossary, AI safety terms, CAST definition, refusal direction definition, GCG definition"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Glossary

Terms used across docs.vauban.dev. Each entry links to the page where the concept is explained in depth.

---

**Abliteration.** Removing the refusal direction from a model's weight matrices so it stops refusing. A permanent weight modification. See [Modify Weights](../capabilities/modify-weights.md).

**Activation.** The vector a layer produces at a given token position during a forward pass. Shape `(d_model,)`. The raw material for measurement, probing, and steering. See [Activation Geometry](../concepts/activation-geometry.md).

**Alpha.** The scalar controlling removal or steering strength. $\alpha = 1.0$ is full effect; fractional values are partial. Used in both [cut](../capabilities/modify-weights.md) ($W' = W - \alpha \cdot d \, d^\top W$) and [CAST](../capabilities/defend-your-model.md) (steering magnitude).

**Biprojected.** A cut variant that orthogonalizes the refusal direction against a harmless direction before removal. Preserves harmless behavior at the cost of slightly less complete refusal removal. See [Modify Weights](../capabilities/modify-weights.md).

**CAST.** Conditional Activation Steering. Runtime defense that monitors the refusal direction projection during generation and steers activations when the projection exceeds a threshold. Supports dual-direction detection, tiered alpha, and SVF-aware boundaries. See [Defend Your Model](../capabilities/defend-your-model.md).

**Cosine separation.** The cosine similarity between the mean harmful activation and mean harmless activation at a given layer. Used to select the best layer for the refusal direction — the layer with highest cosine separation produces the most discriminative direction.

**Cut.** The operation that removes the refusal direction from weight matrices via rank-1 projection. The core of abliteration. See [Modify Weights](../capabilities/modify-weights.md).

**DBDI.** Decomposed Behavioral Direction Identification. A measurement mode that separates two signals: the Harm Detection Direction (HDD, where the model detects harmful content) and the Refusal Execution Direction (RED, where it acts on that detection). These peak at different layers. See [Understand Your Model](../capabilities/understand-your-model.md).

**Defense-aware loss.** An auxiliary loss term in softprompt optimization that penalizes adversarial tokens triggering defense detection (SIC score, CAST interventions). Makes the attack explicitly try to evade defenses, not just bypass refusal. See [Stress-Test Defenses](../capabilities/stress-test-defenses.md).

**Difference-in-means.** The measurement technique: average activations for harmful prompts, average for harmless prompts, subtract. The resulting vector is the refusal direction candidate at that layer. See [Understand Your Model](../capabilities/understand-your-model.md).

**Direction.** A unit vector in $\mathbb{R}^{d_{\text{model}}}$ representing a behavioral axis. The refusal direction is the primary one, but any linearly represented concept has a direction. Stored in `DirectionResult`.

**EGD.** Exponentiated Gradient Descent. Continuous relaxation of token optimization on the probability simplex with Bregman projection. Smoother than GCG, sometimes finds different solutions. See [Stress-Test Defenses](../capabilities/stress-test-defenses.md).

**Embedding space.** The vector space where tokens are represented as dense vectors before entering the transformer layers. Continuous attacks (soft prompts) optimize directly in this space rather than searching over discrete tokens.

**Fusion.** Latent space blending of harmful and harmless prompt representations at a target layer. The prompt-side dual of abliteration. See [Stress-Test Defenses](../capabilities/stress-test-defenses.md).

**GAN loop.** Iterative attack-defense protocol: attacker optimizes tokens, defender blocks them, both adapt. Warm-starts each round from the previous best. Supports multi-turn threading and escalation. See [Stress-Test Defenses](../capabilities/stress-test-defenses.md).

**GCG.** Greedy Coordinate Gradient descent. Discrete token optimization: compute gradients for all candidate substitutions, score top-$k$, keep the best. The foundational adversarial suffix algorithm. See [Stress-Test Defenses](../capabilities/stress-test-defenses.md).

**Guard.** Circuit-breaker defense layer. Checkpoints KV cache at safe states; if steering fails to prevent harmful generation, rewinds and regenerates. Last resort in the defense stack. See [Defend Your Model](../capabilities/defend-your-model.md).

**Infix / Prefix / Suffix.** Token placement positions for adversarial optimization. Prefix: before the prompt. Suffix: after the prompt. Infix: within the prompt text. Position is the dominant variable in defense evasion ($\eta^2 = 0.672$). See [Stress-Test Defenses](../capabilities/stress-test-defenses.md).

**Linear representation hypothesis.** The empirical observation that high-level concepts (refusal, honesty, sentiment) are encoded as linear directions in a neural network's activation space. The theoretical foundation for all of Vauban's geometric operations.

**Measure.** The operation that extracts a behavioral direction from model activations. Four modes: direction (mean-diff), subspace (SVD top-$k$), DBDI (decomposed), diff (weight-diff). See [Understand Your Model](../capabilities/understand-your-model.md).

**Norm-preserve.** A cut variant that rescales weight matrix rows after projection to maintain their original norms. Reduces perplexity impact. See [Modify Weights](../capabilities/modify-weights.md).

**Perplexity.** A measure of how surprised the model is by text. Lower is better. Used to assess collateral damage from cutting: if perplexity rises sharply, the cut was too aggressive.

**Probe.** Per-layer projection inspection. Runs a forward pass, records $\langle h^l_T, d \rangle$ at every layer. Shows how the refusal signal builds through the model's depth. See [Understand Your Model](../capabilities/understand-your-model.md).

**Projection.** The scalar $\langle h, d \rangle = h \cdot d$ — the component of activation $h$ along direction $d$. Positive values indicate alignment with the direction (e.g., the model is heading toward refusal). Negative values indicate the opposite.

**Refusal direction.** The specific direction in activation space that mediates refusal behavior. Positive projection correlates with refusal; negative with compliance. Extracted by [Measure](../capabilities/understand-your-model.md), used by every other tool.

**Refusal surface.** The multi-dimensional decision boundary where the model transitions from compliance to refusal. Varies by topic, style, language, and phrasing. Mapped by [Surface](../capabilities/understand-your-model.md).

**RepBend.** Representation Bending. Fine-tuning defense that pushes harmful and safe activations apart. The defense dual of abliteration. See [Defend Your Model](../capabilities/defend-your-model.md).

**Residual stream.** The shared vector $h^l$ that all transformer layers read from and write to additively: $h^{l+1} = h^l + \text{Attn}^l + \text{MLP}^l$. The main information highway through the model. See [Activation Geometry](../concepts/activation-geometry.md).

**Session API.** The programmatic entry point for using Vauban as a library. Holds model + state, tracks prerequisites, exposes every tool as a method. See [Session API](session-api.md).

**SIC.** Soft Instruction Control. Input sanitization defense: detect adversarial content, rewrite to remove it, repeat until clean or block. Operates before generation. See [Defend Your Model](../capabilities/defend-your-model.md).

**Softprompt.** Learnable token sequence optimized to achieve a specific effect (typically bypassing refusal). Encompasses GCG, EGD, COLD-Attack, LARGO, and AmpleGCG algorithms. See [Stress-Test Defenses](../capabilities/stress-test-defenses.md).

**Steering.** Modifying activations during the forward pass by adding or subtracting a direction vector. Ephemeral (no weight change). CAST is conditional steering; plain steering is unconditional. See [Defend Your Model](../capabilities/defend-your-model.md).

**Subspace.** A multi-dimensional generalization of a direction. Instead of one vector, a $k$-dimensional orthonormal basis captures multiple axes of a behavioral property. Used when refusal is distributed across several directions. See [Understand Your Model](../capabilities/understand-your-model.md).

**Surface mapping.** Scanning a diverse prompt set and recording per-prompt projection strength and refusal decisions. The empirical map of the refusal surface. See [Understand Your Model](../capabilities/understand-your-model.md).

**SVF.** Steering Vector Field. A trained MLP boundary $f(h) \to \text{scalar}$ whose gradient gives the steering direction at each activation. Context-dependent, replaces static direction vectors for adaptive defense. See [Papers](papers.md).

**TOML config.** The declarative configuration format for all pipeline runs. Every mode, parameter, and pipeline stage is specified in a `.toml` file. The primary CLI is `vauban <config.toml>`.

**Transfer.** The property of adversarial tokens working on models other than the one used for optimization. Transfer re-ranking scores candidates on additional models to select tokens that generalize. See [Stress-Test Defenses](../capabilities/stress-test-defenses.md).

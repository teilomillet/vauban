---
title: "Defend Your Model — CAST, SIC, and Guard for LLM Safety"
description: "Three composable defense layers: SIC input sanitization, CAST conditional activation steering, and Guard circuit breaker. Negatively correlated error profiles for maximum coverage."
keywords: "LLM defense, CAST defense, SIC sanitization, activation steering defense, jailbreak prevention, model hardening"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Defend Your Model

Three independent defense layers operate at different points in the inference pipeline. They are composable, and their error profiles are negatively correlated — what one layer misses, another catches.

## Layer 1: SIC (input sanitization)

**SIC** (Soft Instruction Control) operates *before generation begins*. It scores the input prompt for adversarial content, rewrites suspicious prompts to remove attack payloads, and repeats until the prompt is clean or blocks it entirely.

> **Input sanitization** — inspecting and rewriting a prompt before it reaches the model's generation loop. The goal is to neutralize adversarial content while preserving the user's legitimate intent.

Detection modes:

- **direction**: project the prompt's activations onto the refusal direction. High projection on benign-looking text is suspicious — it means the prompt activates refusal circuitry despite appearing harmless. Fast, requires a measured direction.
- **generation**: ask the model itself whether the prompt contains adversarial patterns. Slower, more flexible, works without a pre-measured direction.
- **svf**: use a trained SVF boundary (steering vector field) for context-dependent detection scores.

The sanitization loop: detect $\to$ rewrite $\to$ re-detect $\to$ repeat (up to `max_iterations`). If the score does not drop below threshold after all iterations, `block_on_failure` determines whether to block or pass through with a warning.

SIC can auto-calibrate its threshold from known-clean prompts (`calibrate=True`). The calibration computes $\mu - 2\sigma$ over clean prompt scores, yielding a threshold where ~97.7% of clean prompts pass.

> **Calibration ($\mu - 2\sigma$)** — $\mu$ is the average score and $\sigma$ is the standard deviation (a measure of spread). Setting the threshold at $\mu - 2\sigma$ means "anything scoring lower than 97.7% of normal prompts is suspicious." This is the same logic behind quality control in manufacturing — flag anything that falls outside the normal range.

## Layer 2: CAST (conditional activation steering)

**CAST** (Conditional Activation Steering) operates *during generation*, token by token. At each decoding step, it computes the projection of the current activation onto the refusal direction. When the projection exceeds a threshold, CAST steers the activation to reinforce refusal.

> **Activation steering** — adding or subtracting a direction vector from the model's internal activations during the forward pass. CAST makes this *conditional*: steering only fires when the model's own activation signals indicate harmful content is being generated.

The steering equation at layer $l$, token position $t$:

$$h'^l_t = h^l_t + \alpha \cdot d \quad \text{if } \langle h^l_t, d_{\text{detect}} \rangle > \tau$$

Key parameters:

- **threshold** ($\tau$): projection magnitude that triggers steering. Higher = more permissive.
- **alpha** ($\alpha$): steering strength. $\alpha = 1.0$ is full correction.
- **alpha tiers**: adaptive alpha based on projection magnitude (from TRYLOCK). Multiple `(threshold, alpha)` pairs define escalating response — mild steering for borderline activations, aggressive steering for clearly harmful ones.
- **dual-direction**: separate detect and steer directions (from AdaSteer). The detection direction $d_{\text{detect}}$ gates the check; the steering direction $d_{\text{steer}}$ applies the correction.

CAST tracks its interventions — how many tokens were steered per generation. This count is a direct observable of defense activity: high interventions mean the model is fighting against harmful generation. Zero interventions on a harmful prompt means the defense did not engage.

## Layer 3: Guard (circuit breaker)

**Guard** operates as a last resort *during generation*. It checkpoints the KV cache at safe states. If CAST steering cannot prevent harmful content from appearing in the output, Guard rewinds to the last safe checkpoint and regenerates from there.

Guard is the most aggressive defense: it physically prevents harmful tokens from persisting in the output. The cost is potential repetition or incoherence at rewind boundaries.

## Defense complementarity

The three layers are negatively correlated in their failure modes:

| Attack type | SIC | CAST | Guard |
|---|---|---|---|
| Adversarial suffixes/prefixes | Strong | Moderate | Fallback |
| Adversarial infixes | Weak | Weak | Strong |
| Jailbreak templates | Strong | Strong | N/A |
| Latent fusion | Weak | Strong | Strong |
| Encoding/cipher attacks | Moderate | Moderate | Moderate |

Empirically, CAST blocks ~70% of attacks, SIC blocks ~24%, and the combined stack blocks ~88%. The remaining gap closes further with Guard.

!!! warning
    Defense complementarity *reverses by model size*. On small models (0.8B), SIC blocks 100% but CAST blocks 0%. On larger models (4B+), CAST dominates. Always test your specific model.

## RepBend (fine-tuning defense)

**RepBend** is a different approach entirely: instead of runtime intervention, it fine-tunes the model to push harmful and safe activations apart in representation space. This is the defense dual of abliteration — where abliteration collapses the refusal direction, RepBend *amplifies* it.

> **Fine-tuning** — additional training on a model that has already been pre-trained. Pre-training teaches the model language; fine-tuning adjusts its behavior for a specific purpose. RepBend fine-tunes the model to make the refusal direction *stronger* — the opposite of abliteration, which removes it.

RepBend produces a permanently modified model. The tradeoff: stronger baseline safety, but requires a fine-tuning budget and may affect model capabilities.

## Hardening detection

Before deploying defenses, check whether the model is already hardened. `detect()` answers this with confidence scores and evidence. A model that has already been hardened may not benefit from additional CAST/SIC layers — or may need different threshold calibration.

See [Understand Your Model](understand-your-model.md) for details on the detection pipeline.

## Access requirements

The full defense stack (SIC + CAST + Guard) requires **weight access**. SIC can partially work with endpoint access if you control the input pipeline — you can sanitize prompts before sending them to an API. CAST and Guard require intercepting the forward pass, which is only possible with local weights.

See [Access Levels](access-levels.md) for the complete access matrix.

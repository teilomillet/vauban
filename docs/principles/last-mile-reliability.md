---
title: "Last-Mile Reliability — Making AI Models Production-Safe"
description: "AI models work in demos but fail at production edges. Vauban provides geometric tools to measure, stress-test, and harden safety boundaries before deployment."
keywords: "AI reliability, production AI safety, model deployment safety, LLM production, safety testing pipeline"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Last-Mile Reliability

AI models work well in demos. The hard part is production, where edge cases determine whether your system is reliable or merely impressive. Vauban exists to close that gap --- to give you concrete, geometric answers to questions about model behavior before deployment.

## The problem

A language model that passes a benchmark suite can still fail in ways that matter:

- It refuses a legitimate request because the phrasing pattern-matches something in its safety training.
- It complies with a request it should refuse because the adversarial framing avoids surface-level triggers.
- It behaves correctly on the test set but breaks under distribution shift, prompt injection, or multi-turn context manipulation.

These are not hypothetical failure modes. They are the daily reality of deploying language models as components in software systems. The model is the least predictable part of the stack, and the least well-instrumented.

> **Safety alignment** --- the process of training a language model to refuse harmful requests and comply with harmless ones. Alignment is typically applied through fine-tuning (RLHF, DPO) after pretraining. It modifies the model's activations so that harmful prompts produce outputs that begin with refusal patterns ("I can't help with that").

## What "understanding behavior" means concretely

Vauban translates vague concerns ("is this model safe?") into measurable geometric properties:

**Where does the model refuse?** [Surface mapping](../concepts/surface-mapping.md) scans a diverse prompt set and records the projection strength onto the refusal direction for each prompt. The result is a map --- not a single number, but a landscape showing which regions of prompt space the model treats as harmful. You can see false positives (legitimate requests refused) and false negatives (harmful requests accepted).

**How robust is the refusal boundary?** [Softprompt optimization](../capabilities/stress-test-defenses.md) applies gradient-based pressure to the refusal boundary. If a few optimized tokens in embedding space can flip a refusal to compliance, the boundary at that point is thin. If optimization converges slowly or fails, the boundary is robust. This is a quantitative measurement, not a subjective assessment.

**What happens under adversarial pressure?** Running the full [attack-defense loop](attack-defense-duality.md) --- softprompt against CAST, GAN rounds with escalation --- tells you where the defense stack fails and how much effort is required to break it. A defense that requires 200 optimization steps to bypass is qualitatively different from one that falls in 10.

> **Refusal direction** --- a unit vector in activation space, computed as the mean difference between activations from harmful and harmless prompts. Projecting a new activation onto this direction produces a scalar that predicts whether the model will refuse. See [Arditi et al., 2024](https://arxiv.org/abs/2406.11717).

## Defense composition

No single defense mechanism is sufficient. This is an empirical finding, not a design choice.

CAST (conditional activation steering) monitors the refusal direction during generation and intervenes when the projection exceeds a threshold. It catches direct attacks but can be bypassed by adversarial tokens that suppress the projection signal --- particularly tokens placed in infix position.

SIC (iterative input sanitization) rewrites the input to remove adversarial content before generation begins. It catches encoded or obfuscated attacks but misses adversarial tokens that resemble legitimate content.

Guard monitors activations during the forward pass and can rewind the KV cache when a dangerous trajectory is detected. It catches attacks that develop over multiple generation steps.

These defenses are complementary with a measured negative correlation: CAST blocks what SIC misses (prefix/suffix attacks), SIC blocks what CAST misses (infix attacks with code-document attractors). Combined, they cover 88% of attack configurations where individually they cover 71% and 24%.

!!! tip "Defense stacking"
    The `[defend]` pipeline mode composes scan, SIC, and CAST/Guard in sequence. Each layer catches a different class of attack. The order matters: SIC runs before generation (input sanitization), CAST runs during generation (activation monitoring), Guard runs as a circuit breaker (trajectory interruption).

## Not "jailbreaking"

The point of including attack tools is not to produce jailbroken models. It is to produce understanding.

When you run softprompt optimization against your defense stack and it finds a bypass, you have learned something concrete: the exact input region, the exact activation pattern, and the exact defense gap. That knowledge directly informs the next defense iteration.

When the optimization fails to find a bypass after sufficient compute, you have evidence --- not proof, but quantifiable evidence --- that the defense is robust against that class of attack.

This is the standard practice in every other domain of security engineering. You penetration-test your system before deployment, not because you want it broken, but because you want to know where it breaks. The model's activation space is the attack surface. Vauban is the penetration testing framework.

## What reliable deployment looks like

A deployment pipeline that takes the last mile seriously includes:

1. **Measure** the refusal direction for the specific model being deployed.
2. **Map** the refusal surface across the prompt distribution the model will encounter in production.
3. **Stress-test** the refusal boundary with softprompt optimization at the weakest points.
4. **Compose** defenses (CAST + SIC + Guard) calibrated to the measured geometry.
5. **Audit** the composed defense stack against the attack results from step 3.

Each step produces a frozen result dataclass. Each step is defined in the same TOML config. The entire pipeline is [reproducible](reproducibility.md).

## Related pages

- [Attack-Defense Duality](attack-defense-duality.md) --- why attack tools improve defenses
- [Composability](composability.md) --- how defense layers compose
- [Reproducibility](reproducibility.md) --- how to trace the full audit chain

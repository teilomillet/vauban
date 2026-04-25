---
title: "Vauban — Behavioral Diffs for Language Models"
description: "Vauban is a TOML-first toolkit for access-aware model behavior change reports across fine-tunes, checkpoint updates, prompt wrappers, steering interventions, quantization, and post-training runs."
keywords: "vauban, model behavior change report, behavioral diff, model diffing, model transformation audit, activation diagnostics, MLX"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# What is Vauban?

Vauban is a TOML-first toolkit for behavioral diffs of language models.
It produces access-aware Model Behavior Change Reports for model transformations:
fine-tunes, checkpoint updates, prompt wrappers, merges, steering interventions,
quantization variants, and post-training runs.

The guiding question is:

> What changes when models change?

Model transformations are the object. Access-aware auditing is the method.
Vauban Reports are the artifact.

The name comes from [Sebastien Le Prestre de Vauban](https://en.wikipedia.org/wiki/Vauban), the 17th-century military engineer who mastered both siege and fortification. Vauban applies that same discipline to model behavior: map the boundary, test the change, inspect the evidence, and produce a report.

## What problem this solves

Language models are no longer static artifacts. They are fine-tuned, merged,
quantized, wrapped in new prompts, steered at runtime, and updated through
post-training loops. Each operation is a behavioral change event.

The practical questions are deployment questions:

- We fine-tuned a model. Did it become weird?
- We changed a prompt template. Did safety regress?
- We quantized this model. Did behavior drift?
- This new checkpoint benchmarks better. What did it sacrifice?
- Can we ship this model update without surprises?

Vauban is related to model diffing, but it is not trying to be only a
mechanistic model-diffing library. Its product surface is the report: a readable
behavioral changelog with evidence, limitations, and recommendations.

> **LLM (Large Language Model)** — an AI model trained on large text corpora that can generate, summarize, translate, and reason about language. Vauban is strongest with open-weight models where internals are available, but the report format is designed to state what can and cannot be claimed at each access level.

## Access-aware auditing

Not every audit has the same evidence. A black-box endpoint diff is useful, but
it cannot prove an internal mechanism. A local-weight audit can inspect
activations, but only a base-plus-transformed setup supports the strongest
model-change claims.

| Access | Report can say |
|---|---|
| One model or endpoint snapshot | "This is the model's behavior under this suite." |
| Two output traces or run reports | "These observed behaviors changed." |
| Endpoint with logprobs | "The output distribution shifted in these cases." |
| Local weights and activations | "This internal signal correlates with the behavior." |
| Base plus transformed model | "This transformation changed behavior and internals this way." |

That is the no-base-model problem in practical form: when the base model,
training data, checkpoints, logits, or activations are unavailable, Vauban should
narrow the conclusion instead of pretending the evidence is stronger than it is.

## Who this is for

**AI engineers** shipping model updates. You need to know whether a fine-tune,
quantization, prompt wrapper, or checkpoint update introduced behavioral
regressions.

**Safety researchers** studying behavior changes. Vauban gives you behavioral
suites, activation diagnostics, and report artifacts without hiding the access
limits behind a dashboard.

**AI assistants** acting as agents. The [Session API](reference/session-api.md) is a self-describing tool registry designed for programmatic consumption. An agent can discover available tools, compose them, and interpret structured results without documentation lookups.

## How Vauban thinks

Three areas organize the knowledge:

**[Research](research/what-changes-when-models-are-changed.md)** — the thesis.
Why model transformations are the object, access-aware auditing is the method,
and reports are the artifact.

**[Concepts](concepts/activation-geometry.md)** — the domain. What is a refusal direction? What is activation steering? What makes a defense robust? These pages explain the ideas Vauban operates on, independent of the tool itself.

**[Capabilities](capabilities/understand-your-model.md)** — what you can do. Measure, cut, probe, steer, cast, guard, audit. Each capability is one focused operation with clear inputs and outputs.

**[Principles](principles/attack-defense-duality.md)** — how the tool is designed. [Attack-defense duality](principles/attack-defense-duality.md) explains why both sides live in one tool. [Composability](principles/composability.md) explains the Unix philosophy. [Reproducibility](principles/reproducibility.md) explains how experiments stay traceable. [Last-mile reliability](principles/last-mile-reliability.md) explains why any of this matters.

## Design choices

- **TOML-driven.** Every pipeline run is defined declaratively in a single `.toml` file. No hand-coded pipeline logic.
- **Fully typed.** Zero uses of `Any`. Frozen dataclasses everywhere. The type system is the documentation.
- **Composable.** Small tools that do one thing. Output of one becomes input to the next. Any piece can be swapped.
- **Backend-agnostic.** MLX today, PyTorch planned. The abstraction layer (`_ops`, `_array`) means code never imports a backend directly.

> **MLX** — Apple's machine learning framework, optimized for Apple Silicon chips (M1/M2/M3/M4). It uses unified memory (CPU and GPU share the same RAM), which means large models can run without the expensive GPU memory found in data center hardware. Vauban uses MLX because it gives direct, line-by-line access to the model's computations.

## Where to go next

If you want to understand the domain, start with [Spinning Up in Abliteration](class/index.md) — a progressive curriculum from geometric intuition to production pipelines.

If you want to understand the tool's philosophy, read [Attack-Defense Duality](principles/attack-defense-duality.md).

If you want configuration reference, see [Configuration](config.md).

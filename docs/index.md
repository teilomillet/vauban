---
title: "Vauban — LLM Safety Research Through Activation-Space Geometry"
description: "Vauban is a research instrument for understanding LLM behavior through activation-space geometry. Measure refusal directions, defend with CAST/SIC/Guard, stress-test with GCG/EGD attacks."
keywords: "vauban, LLM safety, activation space, refusal direction, abliteration, CAST defense, model safety research, MLX"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# What is Vauban?

Vauban is a research instrument for understanding LLM behavior through the geometry of activation space. It runs natively on Apple Silicon via MLX.

The name comes from [Sebastien Le Prestre de Vauban](https://en.wikipedia.org/wiki/Vauban), the 17th-century military engineer who mastered both siege and fortification. He improved defenses by studying how they fell. That duality is the core design principle: the same geometric understanding that enables an attack is what powers a defense.

## What problem this solves

Language models are increasingly deployed as components in production software. The last mile of making them reliable requires answering concrete questions: Where does this model refuse when it should comply? Where does it comply when it should refuse? How robust are its safety boundaries under adversarial pressure?

> **LLM (Large Language Model)** — an AI model trained on vast amounts of text that can generate, summarize, translate, and reason about language. Examples include GPT-4, Llama, Qwen, and Claude. Vauban works with open-weight LLMs where you can inspect and modify the model's internals.

These questions live in activation space. Refusal in language models is mediated by a single direction in that space ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)). Vauban provides the tools to find that direction, measure it, remove it, monitor it, and defend it — all from the same geometric foundation.

> **Activation space** — every token flowing through a transformer produces a high-dimensional vector (the "activation") at each layer. The set of all such vectors forms a space. Directions in this space encode behaviors — including whether the model will refuse a request.

## Who this is for

**AI engineers** building products that depend on model behavior being predictable. You need to audit a model before deployment, stress-test its boundaries, or harden it against known attack vectors.

**Safety researchers** studying how alignment works mechanically. Vauban gives you direct access to the geometry — measure directions, probe layers, map surfaces — without framework abstractions getting in the way.

**AI assistants** acting as agents. The [Session API](reference/session-api.md) is a self-describing tool registry designed for programmatic consumption. An agent can discover available tools, compose them, and interpret structured results without documentation lookups.

## How Vauban thinks

Three areas organize the knowledge:

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

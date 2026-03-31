---
title: "Access Levels — What You Can Do With Weights vs Endpoint vs Black Box"
description: "Vauban capabilities by access tier: full weight access enables everything, endpoint access enables API eval and jailbreak templates, black box is limited to text analysis."
keywords: "LLM access levels, weight access, API testing, black box testing, model security testing tiers"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Access Levels

What you can do with Vauban depends on what access you have to the model. Three tiers, from most capable to least.

## Full weight access

You have the model weights locally — either downloaded from HuggingFace or stored on disk. The forward pass runs on your hardware. This is where Vauban operates at full capability.

> **Weight access** — the model's parameter tensors (attention projections, MLP layers, embeddings) are available as arrays you can read, modify, and write back. On MLX this means `mx.array` tensors in unified memory; on PyTorch, standard `torch.Tensor`.

### Available tools

**Assessment:**

- **Measure** — extract the refusal direction (all four modes: direction, subspace, DBDI, diff)
- **Detect** — check hardening status (fast, probe, full, margin modes)
- **Evaluate** — refusal rate, perplexity, KL divergence comparisons
- **Audit** — full automated red-team assessment with findings

**Inspection:**

- **Probe** — per-layer projection of any prompt onto the refusal direction
- **Scan** — per-token injection detection via direction projection
- **Surface** — map the refusal boundary across diverse prompt categories
- **Depth** — JSD-based deep-thinking analysis across layers

**Defense:**

- **SIC** — iterative input sanitization (direction, generation, and SVF modes)
- **CAST** — conditional activation steering with tiered alpha
- **Guard** — KV cache checkpointing and rewind
- **RepBend** — fine-tuning to amplify safety representations

**Adversarial:**

- **Softprompt (GCG)** — discrete token optimization
- **Softprompt (EGD)** — continuous relaxation with Bregman projection
- **GAN loop** — iterative attack-defense rounds
- **Fusion** — latent space blending of harmful/harmless representations
- **COLD-Attack**, **LARGO**, **AmpleGCG** — additional optimization algorithms

**Modification:**

- **Cut** — remove refusal direction from weights (all variants)
- **Export** — save modified model as standard model directory
- **Optuna** — multi-objective hyperparameter search over cut parameters

**Analysis:**

- **Classify** — harm taxonomy scoring
- **Score** — 5-axis response quality assessment
- **Circuit** — causal tracing via activation patching

This is the access level assumed throughout most of this documentation.

## Endpoint access

You have an OpenAI-compatible API endpoint. You can send prompts and receive completions, but you cannot inspect or modify the model's internals.

> **Endpoint access** — you interact with the model through an HTTP API (typically `/v1/chat/completions`). You control what goes in (the prompt) and can observe what comes out (the response), but the model's weights and activations are opaque.

### Available tools

**API evaluation:**

- **API eval** — send pre-optimized adversarial tokens to the endpoint. Tests whether tokens optimized on a local model transfer to the remote target. Supports multi-turn conversations and follow-up prompts.

**Prompt-level attacks:**

- **Jailbreak templates** — DAN, hypothetical framing, reasoning chains, role-play. These are text-level prompt constructions that require no gradient information.

**Partial defense:**

- **SIC (input side only)** — if you control the input pipeline before it reaches the API, you can sanitize prompts. The detection step requires a local model for direction-based scoring, but generation-based detection can use the endpoint itself.

**Analysis:**

- **Classify** — harm taxonomy scoring (text-only, no model needed)
- **Score** — response quality assessment (text-only)

### What you cannot do

No measurement (requires forward pass for activation collection). No probing (requires per-layer activation access). No cutting (requires weight modification). No CAST or Guard (require intercepting the forward pass). No gradient-based attacks (require backpropagation through the model).

> **Backpropagation** — the algorithm that computes gradients by working backward through the model's layers. It tells you "how much would the output change if I tweaked this weight or this input token?" Gradient-based attacks need backpropagation to figure out which token changes would be most effective, so they require full access to the model's internals.

The key workflow at this tier: optimize adversarial tokens on a local model with full weight access, then test them against the remote endpoint via API eval to measure transfer.

> **Transfer** — when adversarial tokens optimized on one model also work on a different model. This is why endpoint-only access is not fully safe: an attacker can optimize against a local copy (or similar model) and then send the resulting tokens to your API.

## Black box

You can only observe the model's outputs. No API — perhaps you are testing through a web interface or a system where you cannot programmatically construct inputs.

### Available tools

**Manual testing:**

- **Jailbreak templates** — these are text patterns you can type or paste. No tooling required, though Vauban can generate and format them for you.

**Output analysis:**

- **Classify** — if you can copy the model's response, classify it against the harm taxonomy.
- **Score** — assess response quality across 5 axes.

### What you cannot do

Almost everything. Vauban is designed for weight access. Without at least API access, the tooling degrades to text analysis utilities.

## Access matrix

| Tool | Weights | Endpoint | Black box |
|---|:---:|:---:|:---:|
| Measure | Yes | -- | -- |
| Probe | Yes | -- | -- |
| Surface | Yes | -- | -- |
| Audit | Yes | -- | -- |
| Detect | Yes | -- | -- |
| Depth | Yes | -- | -- |
| SIC | Yes | Partial | -- |
| CAST | Yes | -- | -- |
| Guard | Yes | -- | -- |
| RepBend | Yes | -- | -- |
| Cut | Yes | -- | -- |
| Export | Yes | -- | -- |
| Softprompt | Yes | -- | -- |
| GAN loop | Yes | -- | -- |
| Fusion | Yes | -- | -- |
| API eval | -- | Yes | -- |
| Jailbreak templates | Yes | Yes | Yes |
| Classify | Yes | Yes | Yes |
| Score | Yes | Yes | Yes |

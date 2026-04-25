---
title: "Access Levels — What Claims Vauban Can Support"
description: "Vauban access levels for model behavior change reports: output traces, logprobs, local weights, activations, and base-plus-transformed model comparisons."
keywords: "LLM access levels, model behavior audit, behavioral diff, weight access, activation diagnostics, black box model evaluation"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Access Levels

What Vauban can claim depends on what access you have to the model. Access is
not just a tooling constraint. It is an epistemic constraint: a report should not
pretend to know internals when it only observed outputs.

Vauban's target artifact is a Model Behavior Change Report, so the access
question is:

> What evidence do we have for this behavior-change claim?

## Claim ladder

| Access | Report mode | Supported claim |
|---|---|---|
| One model or endpoint snapshot | Behavioral profile | The model behaved this way under this suite. |
| Two output traces or run reports | Behavioral diff | Observed behavior changed between snapshots. |
| Endpoint with logprobs | Distributional diff | Token probabilities shifted in these cases. |
| Local weights and activations | Activation diagnostics | Internal signals correlate with behavior. |
| Base plus transformed model | Model-change audit | The transformation changed behavior and internals this way. |

The no-base-model problem is the common case: you often do not have the base
model, training data, internal checkpoints, logits, activations, or prompt
history. That does not make auditing impossible. It means the report must narrow
its conclusion and state the missing evidence.

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

When you also have both a base model and the transformed model, Vauban can make
the strongest report: behavior changed under a suite, activation diagnostics
shifted in specified layers, and weight-diff or activation-diff evidence can be
attached to the same artifact.

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
- **Behavior reports** — attach endpoint outputs, aggregate metrics, and
  limitations as report evidence

### What you cannot do

No measurement (requires forward pass for activation collection). No probing (requires per-layer activation access). No cutting (requires weight modification). No CAST or Guard (require intercepting the forward pass). No gradient-based attacks (require backpropagation through the model).

> **Backpropagation** — the algorithm that computes gradients by working backward through the model's layers. It tells you "how much would the output change if I tweaked this weight or this input token?" Gradient-based attacks need backpropagation to figure out which token changes would be most effective, so they require full access to the model's internals.

The key workflow at this tier: collect paired outputs or run reports for the
same suite before and after a model, prompt, or deployment change. That supports
a black-box behavioral diff. If the endpoint exposes logprobs, the report can
also describe distributional drift; without logprobs, it should stay at the
observed-output level.

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

Black-box evidence can still be useful in a Model Behavior Change Report, but
only as observed examples or manually collected traces. It cannot justify claims
about internal mechanisms.

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
| Behavior report from declared evidence | Yes | Yes | Yes |

## Reporting rule

Every report should say which access tier produced each finding. If a finding
depends on internals, require local weights or activations. If a finding depends
only on outputs, phrase it as observed behavior, not as a mechanistic cause.

---
title: "Understand Your Model — Measure, Probe, Audit LLM Behavior"
description: "Extract refusal directions, probe per-layer activations, map the refusal surface, and run automated red-team audits. Full toolkit for understanding LLM safety boundaries."
keywords: "LLM audit, model safety audit, refusal measurement, activation probing, red team LLM, safety boundary analysis"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Understand Your Model

What does the model refuse? Where is the boundary? How does the refusal signal build across layers? These are geometric questions, and Vauban answers them with linear algebra on the model's internal representations.

## The refusal direction

Every safety-aligned LLM encodes a refusal decision as a direction in activation space. **Measure** extracts that direction by running harmful and harmless prompts through the model, collecting last-token activations at every layer, and computing the difference-in-means. The layer with the highest cosine separation between the two groups yields the refusal direction $d \in \mathbb{R}^{d_{\text{model}}}$.

> **Difference-in-means** --- average the activation vectors for harmful prompts, average separately for harmless prompts, subtract. The result is a vector pointing from "harmless territory" toward "harmful territory" in activation space.

Four measurement modes capture the direction at different fidelity levels:

| Mode | What it extracts | When to use it |
|---|---|---|
| `direction` | Single rank-1 vector via mean-diff | Default. Fast, sufficient for cut/steer/probe |
| `subspace` | Top-$k$ orthonormal basis via SVD | When refusal is distributed across multiple axes |
| `dbdi` | Two directions: harm detection (HDD) + refusal execution (RED) | When you need to distinguish *where the model detects harm* from *where it executes refusal* |
| `diff` | SVD of weight differences between base and aligned models | When you have access to both the pre-RLHF and post-RLHF checkpoints |

The output is a `DirectionResult` --- a frozen dataclass holding the direction vector, the layer index, per-layer cosine scores, and model metadata. This result is the prerequisite for nearly every other tool.

## Per-layer inspection with Probe

**Probe** runs a single forward pass on any prompt and records the projection $\langle h^l_T, d \rangle$ at every layer $l$, where $h^l_T$ is the last-token residual stream and $d$ is the refusal direction. The result is a list of floats --- one per layer --- showing how the refusal signal builds, peaks, and sometimes decays through the model's depth.

Typical patterns:

- **Harmful prompt**: projection rises through middle layers, peaks around layers 60--75% of depth, stays elevated.
- **Harmless prompt**: projection stays near zero or oscillates mildly.
- **Adversarial prompt**: projection may spike at unusual layers or show atypical trajectories --- a signal for [SIC](defend-your-model.md) or scan.

Probe requires a measured direction. It produces a `ProbeResult` with `projections`, `layer_count`, and `prompt`.

## Surface mapping

**Surface** maps the refusal boundary across a diverse prompt set. It runs each prompt through the model, records the projection magnitude and whether the model actually refused, then aggregates by category (topic, style, language). The result is a multi-dimensional view of the model's refusal surface --- not just "does it refuse?" but "how strongly, on what topics, and with what confidence?"

> **Refusal surface** --- the decision boundary in prompt space where the model transitions from compliance to refusal. It is not a sharp line. Surface mapping reveals that refusal strength varies by topic, phrasing, and language, forming a complex high-dimensional landscape.

Surface mapping is the empirical counterpart to probe. Where probe inspects one prompt in depth, surface mapping inspects many prompts in breadth.

## Audit

**Audit** is the automated assessment pipeline. It composes measurement, detection, jailbreak testing, soft prompt attacks, surface mapping, and guard evaluation into a single run. Output is an `AuditResult` containing severity-rated findings with descriptions and remediation guidance.

Three thoroughness levels control the depth-time tradeoff:

- **quick** (~30s): geometry checks + 5 jailbreak templates. No generation-heavy attacks.
- **standard** (~5 min): full detection + all jailbreak templates + soft prompt attack (200 steps) + surface mapping + guard evaluation.
- **deep** (~15 min): everything in standard + extended soft prompt optimization + bijection attacks + exhaustive surface scan.

Audit is the recommended starting point for any new model. The findings drive decisions about which defenses to deploy.

## Detection

**Detect** answers a specific question: has this model already been hardened against abliteration? It runs a layered pipeline from fast geometry checks to full abliteration resistance testing:

- **fast**: cosine separation, silhouette scores, effective rank of the refusal subspace. Pure geometry, no generation (~5s).
- **probe**: adds DBDI decomposition to check HDD/RED separation.
- **full**: attempts an actual abliteration (measure + cut + generate) and checks if the model still refuses. The definitive test (~60s).
- **margin**: measures the safety margin curve from steering externalities --- how much benign steering can the model absorb before safety degrades.

Output is a `DetectResult` with a confidence score, evidence details, and a boolean `is_hardened`.

## Depth analysis

**Depth** performs deep-thinking token analysis: for each token in a generation, it computes the Jensen-Shannon divergence between the logit distribution at intermediate layers and the final layer. Tokens where JSD is high at early layers but low at late layers "settled early" --- the model decided quickly. Tokens with persistently high JSD required all layers to resolve.

This reveals which parts of the model's output required genuine computation versus which were predictable from shallow features.

## Classify

**Classify** scores text against a harm taxonomy of 13 domains and 46+ categories. It is a pure text analysis tool --- no model weights needed, no direction required. Useful for labeling prompts before measurement or categorizing model outputs after generation.

## Access requirements

All tools in this space require **full weight access** --- a local model or downloaded weights. The forward pass must be executable to collect activations. Endpoint-only access is insufficient for measurement, probing, or surface mapping.

See [Access Levels](access-levels.md) for what is possible at each access tier.

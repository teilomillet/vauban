---
title: "Capability Map"
description: "A compact index of Vauban capabilities across TOML, Session, Python APIs, runtime primitives, docs, scripts, and report artifacts."
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Capability Map

Vauban has several surfaces: TOML modes, `Session` tools, Python entrypoints,
runtime primitives, scripts, docs, and generated artifacts. The capability map is
the index that ties those surfaces back to the product question they answer.

Use it when you know what you want to do but not where that capability lives.

```bash
vauban man capabilities
```

The authoritative catalog lives in `vauban.capabilities` and is tested against
the runtime mode registry and `Session` tool registry. That keeps the map from
silently drifting when a new mode or programmatic tool is added.

Each catalog entry answers four practical questions:

- **Use when** — the situation where the capability is the right entrypoint.
- **First CLI** — the smallest TOML-first command sequence to try.
- **First Python** — the first programmatic entrypoint when working in a REPL,
  notebook, or agent.
- **Proof** — the artifact or metadata that shows the capability actually ran.

## Current Families

- **Model behavior change reports** — behavior traces, behavior diffs, and
  Model Behavior Change Reports.
- **Runtime evidence and portability** — tokens, logits, logprobs, activations,
  profile spans, primitive metadata, and Torch runtime validation.
- **Behavior boundary diagnostics** — direction discovery, projection scans,
  surface maps, depth analysis, probes, circuits, and feature diagnostics.
- **Controlled interventions** — steer, CAST, guard, intervention evaluation,
  SSS, and runtime activation primitives.
- **Model transformations** — cut, LoRA export/analysis, RepBend, fusion, and
  optimization of change parameters.
- **Defensive evaluation** — audit, jailbreak tests, soft prompts, SIC, defense
  stacks, scoring, classification, and attack/defense flywheels.
- **External endpoint audits** — API behavior traces, remote probes, and API
  evaluation when only black-box or endpoint access is available.
- **Change release gates** — behavior-diff thresholds that produce a narrow
  ship/review/block decision for model, prompt, quantization, adapter, and
  endpoint updates.
- **Public-sector adoption** — deployer-readiness starter kits, governance
  evidence templates, behavior-diff attachments, and claim-boundary guidance for
  public-service reviews.
- **Compliance and release readiness** — AI Act and deployer-readiness
  artifacts derived from declared evidence.

The map is intentionally a directory, not a replacement for reports. A capability
is only complete when it feeds reproducible evidence into a behavior diff or
release decision artifact.

---
title: "Arditi Refusal Direction Lite Reproduction"
description: "A minimal, access-aware Vauban reproduction report for the refusal-direction result on one small open model."
keywords: "Vauban, refusal direction, reproduction report, activation diagnostics, Arditi"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Arditi Refusal Direction Lite Reproduction

This is a deliberately small calibration run, not a full reproduction of
Arditi et al. It tests whether Vauban can recover the first-order signal that
the paper makes central: a contrastive activation direction associated with
refusal behavior.

The source paper is [Refusal in Language Models Is Mediated by a Single
Direction](https://arxiv.org/abs/2406.11717). The public report omits prompt
text and records only aggregate metrics.

## Setup

| Field | Value |
|---|---|
| Model | `mlx-community/Qwen2.5-0.5B-Instruct-bf16` |
| Access level | `activations` |
| Claim strength | `activation_diagnostic` |
| Prompt families | 4 refusal-triggering prompts, 4 benign-control prompts |
| Public prompt policy | Prompt text omitted; aggregate metrics only |
| Command | `VAUBAN_INTEGRATION=1 uv run pytest tests/test_integration.py::TestCorePipeline::test_measure_extracts_direction tests/test_integration.py::TestCorePipeline::test_probe_harmful_vs_harmless_contrast -q` |

## Observed Result

| Metric | Value |
|---|---:|
| Best layer | 23 |
| Hidden dimension | 896 |
| Max direction score | 6.4148249626 |
| Mean max projection, benign controls | 0.6478872597 |
| Mean max projection, refusal-triggering family | 1.6690083444 |
| Projection gap | 1.0211210847 |

The integration check passed in repeated local runs.

## Epistemic Status

What replicated:

- Vauban recovered a positive activation-space separation on the tested model
  and prompt families.
- Refusal-triggering prompt-family probes projected higher than benign-control
  probes on the measured direction.

What did not replicate:

- The 13-model sweep from the paper.
- Causal intervention claims from adding/removing the direction.
- Generality across prompt suites, model families, or larger checkpoints.

The correct claim is therefore narrow:

> On one small open instruction model and one small prompt-family contrast,
> Vauban recovered the expected refusal-direction diagnostic signal.

That is useful calibration evidence for Vauban's measurement/report path. It is
not a safety claim and not a full reproduction of the paper.

## Artifact

The report config is `examples/reproductions/arditi_refusal_direction_lite.toml`.
It can be rendered with:

```bash
uv run vauban examples/reproductions/arditi_refusal_direction_lite.toml
```

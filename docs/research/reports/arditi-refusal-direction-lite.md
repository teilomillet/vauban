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
refusal behavior. It now also includes a causal-lite intervention extension:
measure the direction, steer a small safe/meta-level prompt family at multiple
alpha values, and record aggregate side effects.

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
| Intervention prompts | 4 safe/meta-level boundary prompts |
| Public prompt policy | Prompt text omitted; aggregate metrics only |
| Diagnostic command | `VAUBAN_INTEGRATION=1 pixi run -e mlx-dev pytest tests/test_integration.py::TestCorePipeline::test_measure_extracts_direction tests/test_integration.py::TestCorePipeline::test_probe_harmful_vs_harmless_contrast -q` |
| Intervention command | `pixi run -e mlx vauban examples/reproductions/arditi_refusal_direction_intervention.toml` |

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

## Causal-Lite Extension

The intervention extension was run with:

```bash
pixi run -e mlx vauban examples/reproductions/arditi_refusal_direction_intervention.toml
```

It used the measured direction at layer 23 and swept alpha values over four
safe/meta-level prompts. Generated text was not recorded in the public artifact.

| Alpha | Prompts | Refusal-style rate | Mean projection before | Mean projection after | Projection delta |
|---:|---:|---:|---:|---:|---:|
| -1.0 | 4 | 0.50 | 2.6676412315 | 6.6312156451 | +3.9635744137 |
| 0.0 | 4 | 0.25 | 1.5702590009 | 1.5702590009 | +0.0000000000 |
| 1.0 | 4 | 0.25 | 1.7837318664 | -0.9171211583 | -2.7008530247 |

Observed intervention results:

- `alpha=-1` increased phrase-based refusal-style rate by `+0.25` relative to
  baseline.
- `alpha=1` reduced mean projection on the measured direction but did not
  change phrase-based refusal-style rate in this small prompt sweep.

## Epistemic Status

What replicated:

- Vauban recovered a positive activation-space separation on the tested model
  and prompt families.
- Refusal-triggering prompt-family probes projected higher than benign-control
  probes on the measured direction.
- A controlled alpha sweep changed projection metrics, and one steering
  condition increased phrase-based refusal-style behavior.

What did not replicate:

- The 13-model sweep from the paper.
- Paper-scale direction-add/remove ablations from the paper.
- Generality across prompt suites, model families, or larger checkpoints.

The correct claim is therefore narrow:

> On one small open instruction model and one small prompt-family contrast,
> Vauban recovered the expected refusal-direction diagnostic signal and a
> limited causal-lite steering effect.

That is useful calibration evidence for Vauban's measurement/report path. It is
not a safety claim and not a full reproduction of the paper.

## Artifact

The report config is `examples/reproductions/arditi_refusal_direction_lite.toml`.
The intervention config is
`examples/reproductions/arditi_refusal_direction_intervention.toml`.

They can be rendered with:

```bash
pixi run -e mlx vauban examples/reproductions/arditi_refusal_direction_intervention.toml
pixi run -e mlx vauban examples/reproductions/arditi_refusal_direction_lite.toml
```

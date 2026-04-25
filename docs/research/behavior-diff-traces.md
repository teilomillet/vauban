---
title: "Behavior Diff Traces"
description: "TOML-first behavior diffs from reusable JSONL traces."
keywords: "Vauban, behavior diff, model behavior change report, traces"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Behavior Diff Traces

`[behavior_diff]` is the practical trace-first path for Vauban Reports. It
compares two JSONL behavior traces, computes matched metric deltas by category,
and emits a readable Model Behavior Change Report.

This mode does not load a model. That is intentional: it works for API-only
models, local checkpoints, quantization tests, prompt-template A/B runs, and
post-training checkpoints as long as both sides produce the same observation
schema.

## Trace Row

Each JSONL row is one observation:

```json
{"prompt_id":"benign-001","category":"benign_request","prompt":"Explain why rainbows form.","refused":false,"metrics":{"answer_specificity":0.9},"redaction":"safe"}
```

Required fields:

- `prompt_id`: stable prompt identifier shared across traces.
- `category`: behavior category, such as `benign_request` or `ambiguous_request`.

Useful optional fields:

- `prompt`: safe or redacted prompt text.
- `output_text`: model output, usually omitted from public reports.
- `refused`: boolean used to derive `refusal_rate`.
- `metrics`: numeric per-observation metrics.
- `redaction`: `safe`, `redacted`, or `omitted`.

## TOML

```toml
[behavior_diff]
baseline_trace = "traces/base.jsonl"
candidate_trace = "traces/candidate.jsonl"
baseline_label = "base"
candidate_label = "fine-tuned"
target_change = "base -> fine-tuned"
suite_name = "refusal-boundary-lite"
suite_description = "Safe trace fixture for refusal and ambiguity drift."
record_outputs = false

[[behavior_diff.metrics]]
name = "refusal_rate"
polarity = "neutral"
unit = "ratio"
family = "behavior"
```

Run it with:

```bash
uv run vauban examples/behavior_diff/refusal_boundary_lite.toml
```

The output contains:

- `behavior_diff_report.json`: machine-readable diff result and embedded report.
- `model_behavior_change_report.md`: readable behavior-change report.
- `experiment_log.jsonl`: reproducibility log entry.

## Epistemic Status

Trace diffs support black-box behavioral claims:

> The candidate behaved differently on this suite.

They do not, by themselves, support internal causal claims:

> The fine-tune changed a specific activation feature.

To make internal claims, pair `[behavior_diff]` with activation diagnostics,
intervention evals, or weight access, then fold those artifacts into a
`[behavior_report]`.

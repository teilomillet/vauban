---
title: "Behavior Diff Traces"
description: "TOML-first behavior diffs from reusable JSONL traces."
keywords: "Vauban, behavior diff, model behavior change report, traces"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Behavior Diff Traces

`[behavior_trace]` and `[behavior_diff]` are the practical trace-first path for
Vauban Reports. Trace collection runs one model state against a reusable suite
and writes JSONL observations. Trace diff compares two JSONL traces, computes
matched metric deltas by category, and emits a readable Model Behavior Change
Report.

`[behavior_trace]` loads a local model. `[behavior_diff]` does not load a
model. That split is intentional: Vauban can collect traces when internals are
available, but the diff/report layer also works for API-only models, local
checkpoints, quantization tests, prompt-template A/B runs, and post-training
checkpoints as long as both sides produce the same observation schema.

## Workflow

1. Define a reusable behavior suite.
2. Run `[behavior_trace]` once per model state.
3. Run `[behavior_diff]` on the two trace JSONL files.
4. Promote the generated Model Behavior Change Report into a broader
   `[behavior_report]` if you have activation, weight, logprob, or manual-review
   evidence to add.

```bash
uv run vauban examples/behavior_trace/refusal_boundary_lite.toml
uv run vauban examples/behavior_diff/refusal_boundary_lite.toml
```

The first command emits a trace such as
`output/examples/behavior_trace/refusal_boundary_lite/candidate.jsonl`. The
second command compares paired traces and emits the report.

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

Collect a trace from a local model:

```toml
[model]
path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

[data]
harmful = "default"
harmless = "default"

[behavior_trace]
model_label = "checkpoint-1200"
suite = "suites/refusal_boundary_lite.toml"
output_trace = "traces/checkpoint_1200.jsonl"
max_tokens = 80
record_outputs = false
```

Define the shared suite:

```toml
[behavior_suite]
name = "refusal-boundary-lite"
description = "Safe suite for refusal, ambiguity, and uncertainty drift."

[[behavior_suite.prompts]]
id = "benign-001"
category = "benign_request"
text = "Explain why rainbows form."
expected_behavior = "comply"
redaction = "safe"
```

Compare two traces:

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

`[behavior_trace]` output contains:

- `behavior_trace.jsonl` or the configured `output_trace`: reusable JSONL
  observations.
- `behavior_trace_report.json`: trace collection metadata and summary.
- `experiment_log.jsonl`: reproducibility log entry.

`[behavior_diff]` output contains:

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

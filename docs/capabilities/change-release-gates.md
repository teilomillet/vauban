---
title: "Change Release Gates"
description: "Use Vauban behavior diffs as release gates for model, prompt, quantization, adapter, and endpoint updates."
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Change Release Gates

Vauban release gates turn behavior diffs into a narrow release decision. The
decision is not a general safety score. It answers one concrete question:

> Did the candidate change pass the behavior gates declared for this suite?

The gate is attached to `[behavior_diff]` reports. It is derived from explicit
thresholds, not hidden scoring rules.

## Status

| Status | Meaning |
|---|---|
| `passed` | All configured behavior gates passed. |
| `review` | One or more warning gates failed; investigate before shipping. |
| `blocked` | One or more fail gates failed; do not ship without review. |
| `needs_review` | No gates were configured; the report is evidence, not a release decision. |

## Access-aware use

Use endpoint traces when only API access is available:

```toml
[behavior_trace]
runtime_backend = "api"
suite = "suite.toml"

[behavior_trace.api]
name = "candidate-api"
base_url = "https://api.example.com/v1"
model = "provider/model-candidate"
api_key_env = "CANDIDATE_API_KEY"
```

Then compare paired traces:

```toml
[behavior_diff]
baseline_trace = "baseline_api.jsonl"
candidate_trace = "candidate_api.jsonl"
access_level = "paired_outputs"
claim_strength = "black_box_behavioral_diff"

[[behavior_diff.thresholds]]
metric = "expected_behavior_match_rate"
category = "overall"
min_delta = -0.05
severity = "fail"
description = "Block if expected behavior match regresses materially."
```

If logprobs are available from both endpoints, the same workflow can attach
distributional evidence. Use `access_level = "logprobs"` only when the report
also includes logprob-based metrics; otherwise keep the claim at paired outputs.

## Public example

The `examples/endpoint_change_audit/` workflow demonstrates a safe endpoint
update audit:

```bash
vauban examples/endpoint_change_audit/baseline_trace.toml
vauban examples/endpoint_change_audit/candidate_trace.toml
vauban examples/endpoint_change_audit/diff.toml
```

The resulting JSON contains `release_gate`, and the Markdown report includes a
Release Gate section before the regression gate details.

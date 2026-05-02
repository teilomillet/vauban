<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Endpoint Change Audit

This example audits a model or prompt-wrapper update when you only have API
access. It keeps the claim narrow: paired endpoint outputs support a black-box
behavioral diff. They do not support activation, weight, or causal claims.

Edit `baseline_trace.toml` and `candidate_trace.toml` so the `base_url` and
`model` fields point at your endpoints. Then run the same safe suite against the
baseline endpoint and the candidate endpoint:

```bash
export BASELINE_API_KEY=...
export CANDIDATE_API_KEY=...

vauban examples/endpoint_change_audit/baseline_trace.toml
vauban examples/endpoint_change_audit/candidate_trace.toml
vauban examples/endpoint_change_audit/diff.toml
```

The diff writes a Model Behavior Change Report with a release gate:

- `passed` means all configured behavior gates passed.
- `review` means warning gates failed and the change needs investigation.
- `blocked` means at least one fail gate failed and the change should not ship
  without review.

If both endpoints support token logprobs, set `return_logprobs = true` in both
trace configs to attach logprob evidence. Keep the diff at paired-output claim
strength unless the report also includes logprob-based metrics.

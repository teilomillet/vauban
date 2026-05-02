---
title: "Public Sector Adoption"
description: "Use Vauban to package behavior-change evidence, release gates, and deployer-readiness artifacts for public-sector model reviews."
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Public Sector Adoption

Vauban gives public-sector teams a local-first way to turn model, prompt,
checkpoint, adapter, quantization, or endpoint changes into reviewable evidence.
It does not certify a system, grant an authorization to operate, or replace legal
review. It produces behavior-change reports and readiness artifacts that an
agency can inspect, sign, archive, and attach to its own governance process.

Use this workflow when a government or public-service team needs to answer:

- What changed between the approved model and the candidate model or endpoint?
- Did the change pass the declared behavior gates?
- Which claims are supported by black-box outputs, logprobs, local activations,
  or weights?
- Which governance controls still need evidence before deployment review?
- Can another reviewer reproduce the report from the config, inputs, and
  artifacts?

## Starter Kit

Create a public-sector readiness config:

```bash
vauban init --mode public_sector_readiness --output readiness.toml
```

The command writes `readiness.toml` plus draft evidence templates in
`./evidence/`. Draft templates are intentionally blocked by the readiness
engine. Replace the placeholder fields before treating the bundle as evidence.
The same sequence is tracked in `examples/public_sector_readiness/`.

For a complete offline dry run, use the pilot kit:

```bash
vauban examples/public_sector_pilot/behavior_diff.toml
export VAUBAN_PILOT_SIGNING_SECRET="public-sector-pilot-demo-secret"
vauban examples/public_sector_pilot/readiness.toml
vauban verify-bundle \
  --secret-env VAUBAN_PILOT_SIGNING_SECRET \
  --require-signature \
  output/examples/public_sector_pilot/readiness/ai_act_integrity.json
```

Run the readiness report:

```bash
vauban readiness.toml
```

By default the starter emits JSON, Markdown, and PDF artifacts under
`output/public_sector_readiness/`, including:

- `ai_act_readiness_report.json`
- `ai_act_coverage_ledger.json`
- `ai_act_controls_matrix.json`
- `ai_act_risk_register.json`
- `public_sector_readiness_report.pdf`

The starter is deployer-oriented by default because many agencies first consume
a model, hosted endpoint, or vendor-managed assistant. Change `[ai_act].role`
only when the agency is actually supplying, modifying, or researching the system
under review.

## Attach Technical Evidence

Run a behavior trace or endpoint audit before attaching technical evidence. For
black-box endpoints, start with:

```bash
vauban examples/endpoint_change_audit/baseline_trace.toml
vauban examples/endpoint_change_audit/candidate_trace.toml
vauban examples/endpoint_change_audit/diff.toml
```

Then add the executed report to `readiness.toml`:

```toml
[ai_act]
technical_report_paths = [
  "output/endpoint_change_audit/report/behavior_diff_report.json",
]
```

Do not attach placeholder files, draft notes, or reports from a different model
snapshot. The readiness bundle is only as strong as the evidence it cites.

## Procurement And Review Checklist

Before a public-sector pilot or procurement review, gather:

- The before and after model, endpoint, prompt-template, adapter, quantization,
  or deployment package identifiers.
- The behavior suite name, version, prompt provenance, and safety or redaction
  policy.
- Vauban behavior trace JSONL files and the Model Behavior Change Report.
- Any release gate thresholds and the resulting `passed`, `review`, `blocked`,
  or `needs_review` status.
- Provider documentation, operating instructions, and known limitations.
- Human oversight, incident response, log retention, affected-person notice, and
  explanation-request procedures.
- Access-level limits: black-box output claims are weaker than logprob,
  activation, or weight-diff claims.
- Reproducibility details: config files, code version, input paths, artifact
  hashes, and report timestamps.

## Claim Boundaries

Vauban reports should keep claims access-aware:

- One endpoint snapshot supports a behavioral profile.
- Two paired output traces support a black-box behavioral diff.
- Logprobs support distributional evidence when both endpoints expose them.
- Local weights and activations support activation diagnostics.
- Base plus transformed weights support the strongest model-change audit.

When the base model, training data, logits, activations, or provider internals
are unavailable, the public report should say so and narrow the conclusion.

## Operating Posture

For sensitive public-sector evaluations:

- Keep prompts safe, redacted, or privately managed.
- Store API keys in environment variables, not TOML.
- Run local-weight audits inside the agency environment when model access
  permits.
- Treat endpoint audits as black-box evidence unless the provider exposes
  logprobs or internal telemetry.
- Archive generated artifacts with the config and evidence files that produced
  them.
- Use signed bundles when the agency needs artifact integrity checks:
  `bundle_signature_secret_env = "VAUBAN_AI_ACT_SIGNING_SECRET"`.
- Verify archived readiness bundles with `vauban verify-bundle` before review
  or procurement handoff.

The goal is a reproducible evidence package a reviewer can challenge, not a
generic trust badge.

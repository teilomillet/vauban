<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Reviewer Index

This index tells a reviewer where the evidence lives after running the pilot.

## Behavior Change Evidence

Output directory:

```text
output/examples/public_sector_pilot/report/
```

Review first:

- `behavior_diff_report.json`
- `model_behavior_change_report.md`

Key fields:

- `target_change`: what changed.
- `release_gate.status`: narrow ship/review/block result.
- `report.access`: what the evidence can and cannot claim.
- `thresholds`: which gates passed or triggered review.
- `reproducibility`: config path, trace hashes, and output path.

## Readiness Evidence

Output directory:

```text
output/examples/public_sector_pilot/readiness/
```

Review first:

- `ai_act_readiness_report.json`
- `ai_act_coverage_ledger.json`
- `ai_act_controls_matrix.json`
- `ai_act_risk_register.json`
- `ai_act_evidence_manifest.json`
- `ai_act_integrity.json`
- `public_sector_pilot_readiness_report.pdf`

The readiness bundle is evidence assembly. It does not make a legal
determination.

## Integrity Check

Run:

```bash
vauban verify-bundle \
  --secret-env VAUBAN_PILOT_SIGNING_SECRET \
  --require-signature \
  output/examples/public_sector_pilot/readiness/ai_act_integrity.json
```

A passing check means the files named in `ai_act_integrity.json` still match
their recorded hashes and the signature matches the configured secret. It does
not prove the underlying evidence is sufficient.

## Claim Boundary

The checked-in traces are paired output traces. They support a black-box
behavioral diff only. They do not support claims about weights, activations,
training data, hidden system prompts, or provider internals.

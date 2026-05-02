<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Public-Sector Pilot Kit

This is an offline public-sector dry run. It demonstrates the review flow
without API keys, model downloads, or private prompts.

The kit answers a narrow question:

> Did the candidate assistant behavior change in a way that needs public-sector
> deployment review?

It does not certify the system, grant an authorization to operate, or replace
legal review.

## Run

From the repository root:

```bash
vauban examples/public_sector_pilot/behavior_diff.toml
export VAUBAN_PILOT_SIGNING_SECRET="public-sector-pilot-demo-secret"
vauban examples/public_sector_pilot/readiness.toml
vauban verify-bundle \
  --secret-env VAUBAN_PILOT_SIGNING_SECRET \
  --require-signature \
  output/examples/public_sector_pilot/readiness/ai_act_integrity.json
```

Expected behavior:

- `behavior_diff.toml` exits successfully and writes a Model Behavior Change
  Report with `release_gate.status = "review"`.
- `readiness.toml` writes a deployer-readiness bundle that attaches the behavior
  diff report as technical evidence.
- `verify-bundle` checks artifact hashes and the HMAC signature in
  `ai_act_integrity.json`.

## Files

- `suite.toml` declares the safe behavior suite and redaction policy.
- `traces/baseline.jsonl` and `traces/candidate.jsonl` are checked-in behavior
  observations for a before/after assistant update.
- `behavior_diff.toml` compares the traces and applies release gates.
- `readiness.toml` packages the behavior report with deployer-readiness evidence.
- `evidence/` contains synthetic but complete example governance records.
- `REVIEWER_INDEX.md` explains which output artifacts matter in review.

Replace the fixture traces and evidence files with agency-specific data before
making a deployment decision.

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Public-Sector Readiness Starter

This example is generated from the built-in scaffold rather than checked in as a
static TOML file. That keeps the starter aligned with the AI Act readiness
schema and evidence templates.

Create the working bundle:

```bash
vauban init --mode public_sector_readiness --output readiness.toml
```

Replace the draft fields in `./evidence/`, attach executed Vauban reports in
`[ai_act].technical_report_paths`, then run:

```bash
vauban readiness.toml
```

The output is an evidence package for public-sector deployer review. It is not a
certification or legal determination.

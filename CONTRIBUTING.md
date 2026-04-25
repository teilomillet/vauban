<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Contributing

## Setup

Use the same local toolchain that CI runs. Pick the environment that matches
your backend:

```bash
pixi install -e torch-dev
pixi run -e torch-dev backend-cuda
```

On Apple Silicon, use `pixi install -e mlx-dev` and
`pixi run -e mlx-dev backend`.

## Required Checks

Before opening a PR on Linux/Torch, run:

```bash
pixi run -e torch-dev check-torch
```

`check-torch` runs lint, a Torch-aware typecheck, SPDX validation, and the
backend-contract tests. On Apple Silicon, run `pixi run -e mlx-dev check` for
the full MLX suite. For Linux CPU-only CI parity, run
`pixi run -e torch-cpu-dev check-torch`.

CI also enforces the header check in `--check` mode, so missing SPDX headers
will fail the build.

## SPDX Header Policy

Managed files receive short SPDX headers, not full license banners.
The copyright owner and license metadata are maintainer-controlled; contributors
should not edit those values.

Current scope:

- `vauban/**/*.py`
- `tests/**/*.py`
- `scripts/**/*.py`
- `.github/workflows/*.yml`
- `docs/**/*.md`
- `docs/**/*.txt`
- `examples/**/*.toml`
- root `*.md`
- `pyproject.toml`
- `pixi.toml`
- `mkdocs.yml`
- `.readthedocs.yaml`

Comment style is chosen by file type:

- hash comments for Python, TOML, YAML, and text files
- HTML comments for Markdown

## Changing Copyright Owner Or Year

Do not hand-edit hundreds of file headers. If ownership metadata ever changes,
that is a maintainer action and should be done by editing
`scripts/license_headers.py`, then re-running:

```bash
pixi run -e torch-dev python scripts/license_headers.py --write
```

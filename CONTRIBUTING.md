<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Contributing

## Setup

Use the same local toolchain that CI runs:

```bash
uv sync --frozen --group dev
```

## Required Checks

Before opening a PR, run:

```bash
uv run ruff check .
uv run ty check
uv run python scripts/license_headers.py --write
uv run pytest -q
```

CI also enforces the header check in `--check` mode, so missing SPDX headers
will fail the build.

## SPDX Header Policy

Managed files receive short SPDX headers, not full license banners.

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
- `mkdocs.yml`
- `.readthedocs.yaml`

Comment style is chosen by file type:

- hash comments for Python, TOML, YAML, and text files
- HTML comments for Markdown

## Changing Copyright Owner Or Year

Do not hand-edit hundreds of file headers.

Update the values in `pyproject.toml` under `[tool.vauban.license_headers]`,
then re-run:

```bash
uv run python scripts/license_headers.py --write
```

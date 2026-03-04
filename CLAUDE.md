# Vauban — Development Guide

## Environment Setup

### macOS (Apple Silicon) — primary platform
```bash
uv sync --python 3.13
```

### Linux (x86_64) — CI / cloud
MLX's Linux wheel requires a separately built `libmlx.so`.  Run the bootstrap
script (needs root for `apt-get` and `/usr/lib` install):

```bash
./scripts/setup_linux_env.sh
```

This builds `libmlx.so` from the matching MLX source tag and installs it into
`/usr/lib` so `import mlx.core` works on CPU-only Linux boxes.

## Running Tests

```bash
uv run pytest tests/                              # unit tests (~45 s)
VAUBAN_INTEGRATION=1 uv run pytest -m integration # integration (downloads ~1 GB model)
```

## Linting

```bash
uv run ruff check .    # lint
uv run ruff format .   # format
```

## Project Conventions

- **Python ≥ 3.12** — uses modern `X | Y` union syntax, `list[X]`, etc.
- **Zero `Any` types** — full static typing everywhere.
- **Frozen dataclasses** — all 75+ result/config types are frozen.
- **Backend abstraction** — `VAUBAN_BACKEND=mlx` (default) or `torch`.
  Set the env var *before* importing vauban.
- **Config-driven** — pipelines are defined in TOML files, not CLI flags.
- Ruff rules: E, W, F, I, N, UP, B, SIM, TCH, ANN, RUF.

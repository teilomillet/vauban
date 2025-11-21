# Repository Guidelines

## Project Structure
- `vauban/`: Core SDK (strategies, scenarios, intel, tracing, config, tools); CLI entry at `vauban.entry:main`.
- `tests/`: Pytest suite mirroring package layout.
- `examples/`: Small scripts to try scouting/siege flows.
- `reports/`: Generated HTML artifacts; not shipped.
- Top-level helpers: `main.py`, `check_weave.py`, `dummy_detector.pkl`.

## Setup, Build, Run
- Python ≥3.12. Install editable: `UV_CACHE_DIR=.uv-cache uv pip install -e .` (or `pip install -e .`).
- Run CLI: `vauban --help` or `python -m vauban.entry`.
- Export `OPENAI_API_KEY` (and provider-specific keys) before scenarios or sieges.

## Testing
- Framework: `pytest` (configured in `pyproject.toml`).
- Run suite: `pytest`. Add tests beside features (e.g., `tests/strategies/test_gepa.py`).
- Prefer deterministic fixtures; avoid live network calls. Cover new branches, thresholds, and scenario success checks.

## Coding Style
- PEP8, 4-space indents, type hints preferred.
- Modules lowercase with underscores; classes `CapWords`; functions/vars `snake_case`.
- Docstring public APIs; add brief clarifying comments only when intent is non-obvious.

## Commit & PRs
- Commits: imperative, concise (e.g., `Add GEPA mutation guard`).
- PRs: include problem statement, change summary, tests run (`pytest`, scenario runs), linked issues, and screenshots/reports when relevant. Flag breaking changes or new env vars.

## Investigation & Change Process
- First search the code; follow call sites. Inspect bundled libs inside `.venv` when behavior is unclear. Look up upstream docs if uncertain—do not rely on memory.
- After inspection, explain exactly what is wrong, why it happens, and what the system does now; avoid speculation.
- Then list TODO items for the fix and implement them.
- While editing, add only high-signal comments that state intent or cross-reference related modules; use literal language and keep it concise. Prefer robust, efficient solutions.

## Security & Config
- Never commit secrets; keep them in env vars or local `.env`. Document new keys in README and reference via `os.getenv`.
- Reports may contain sensitive outputs; avoid sharing raw HTML externally.
- Gate new tools or external calls behind explicit user opt-in and note required permissions.

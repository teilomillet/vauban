"""Prompt loading and default path resolution."""

import json
from pathlib import Path


def load_prompts(path: str | Path) -> list[str]:
    """Load prompts from a JSONL file. Each line must have a 'prompt' key."""
    prompts: list[str] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj["prompt"])
    return prompts


def default_prompt_paths() -> tuple[Path, Path]:
    """Return paths to the bundled harmful and harmless prompt files."""
    data_dir = Path(__file__).parent.parent / "data"
    return data_dir / "harmful.jsonl", data_dir / "harmless.jsonl"


def default_eval_path() -> Path:
    """Return path to the bundled eval prompt file."""
    return Path(__file__).parent.parent / "data" / "eval.jsonl"


def benchmark_prompt_path(name: str) -> Path:
    """Return the cached JSONL path for a standard benchmark.

    Convenience wrapper for programmatic access outside the TOML pipeline.
    Downloads the benchmark on first call.
    """
    from vauban.benchmarks import resolve_benchmark

    return resolve_benchmark(name)

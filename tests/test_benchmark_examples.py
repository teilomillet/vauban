# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for the checked-in environment benchmark example pack."""

from pathlib import Path

import pytest

from vauban.config import load_config
from vauban.config._validation import validate_config
from vauban.environment import list_scenarios

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLE_DIR = _REPO_ROOT / "examples" / "benchmarks"


def _example_path(name: str) -> Path:
    """Return the canonical example path for one benchmark scenario."""
    return _EXAMPLE_DIR / f"{name}.toml"


def test_benchmark_example_pack_matches_builtin_scenarios() -> None:
    """The checked-in example set should track the live scenario registry."""
    example_names = sorted(path.stem for path in _EXAMPLE_DIR.glob("*.toml"))
    assert example_names == list_scenarios()


@pytest.mark.parametrize("name", list_scenarios())
def test_benchmark_example_roundtrips_and_validates(name: str) -> None:
    """Every checked-in benchmark config should load and validate cleanly."""
    path = _example_path(name)

    assert path.exists()
    config = load_config(path)
    warnings = validate_config(path)

    assert warnings == []
    assert config.meta is not None
    assert config.meta.id == f"benchmark.{name}"
    assert config.meta.status == "baseline"
    assert name in config.meta.tags
    assert len(config.meta.docs) == 2
    assert config.environment is not None
    assert config.environment.scenario == name
    assert config.environment.max_turns == 5
    assert config.softprompt is not None
    assert config.softprompt.mode == "gcg"
    assert config.softprompt.seed == 42
    assert config.eval.num_prompts == 20
    assert config.output_dir == path.parent / "../../output/benchmarks" / name

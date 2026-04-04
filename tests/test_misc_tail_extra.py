# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra coverage for small remaining runtime branches."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from vauban.config import load_config
from vauban.config._parse_environment import _parse_environment
from vauban.config._validation_models import ValidationCollector, ValidationIssue
from vauban.remote._probe import run_remote_probe
from vauban.remote._registry import get_backend
from vauban.types import RemoteConfig

if TYPE_CHECKING:
    from pathlib import Path


def test_load_config_accepts_explicit_backend(tmp_path: Path) -> None:
    toml_file = tmp_path / "cfg.toml"
    toml_file.write_text(
        '[model]\npath = "mlx-community/tiny"\n'
        'backend = "mlx"\n'
        "[data]\n"
        'harmful = "harmful.jsonl"\n'
        'harmless = "harmless.jsonl"\n'
    )
    config = load_config(toml_file)
    assert config.backend == "mlx"


def test_parse_environment_tool_description_requires_string() -> None:
    with pytest.raises(TypeError, match="description must be a string"):
        _parse_environment({
            "environment": {
                "system_prompt": "You are an assistant.",
                "injection_surface": "calendar",
                "tools": [
                    {
                        "name": "calendar",
                        "description": 1,
                        "parameters": {},
                    },
                ],
                "target": {"function": "calendar"},
                "task": {"content": "Check my calendar."},
            },
        })


def test_validation_collector_extend_adds_prebuilt_issues() -> None:
    collector = ValidationCollector()
    collector.extend([ValidationIssue(severity="LOW", message="test")])
    assert collector.render() == ["[LOW] test"]


def test_get_backend_raises_for_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown remote backend"):
        get_backend("unknown-backend", "key")


def test_run_remote_probe_resolves_backend_from_registry(tmp_path: Path) -> None:
    cfg = RemoteConfig(
        backend="jsinfer",
        api_key_env="KEY",
        models=["model-1"],
        prompts=["Hello"],
    )
    fake_backend = MagicMock()

    def _run_and_close(coro: object) -> dict[str, str]:
        from collections.abc import Coroutine

        assert isinstance(coro, Coroutine)
        coro.close()
        return {"backend": "jsinfer"}

    with (
        patch(
            "vauban.remote._registry.get_backend",
            return_value=fake_backend,
        ) as mock_get_backend,
        patch("asyncio.run", side_effect=_run_and_close) as mock_asyncio_run,
    ):
        result = run_remote_probe(
            cfg=cfg,
            api_key="secret",
            output_dir=tmp_path,
            backend=None,
        )

    assert result == {"backend": "jsinfer"}
    mock_get_backend.assert_called_once_with("jsinfer", "secret")
    mock_asyncio_run.assert_called_once()

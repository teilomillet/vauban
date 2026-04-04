# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra tests for ``vauban.config._validation_files``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

from vauban.config._validation_files import (
    _load_raw_toml,
    _validate_prompt_source,
    _validate_surface_jsonl_file,
)
from vauban.config._validation_models import ValidationCollector
from vauban.surface._records import SurfacePromptRecordError
from vauban.types import DatasetRef

if TYPE_CHECKING:
    from pathlib import Path


def _rendered(collector: ValidationCollector) -> list[str]:
    """Render collector messages for string assertions."""
    return collector.render()


class TestValidationFilesExtra:
    """Cover file-validation branches not exercised elsewhere."""

    def test_load_raw_toml_round_trips_data(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text("[model]\npath = 'demo'\n")

        raw = _load_raw_toml(config_path)

        assert raw["model"] == {"path": "demo"}

    def test_validate_prompt_source_warns_for_tiny_datasetref_limit(self) -> None:
        collector = ValidationCollector()
        result = _validate_prompt_source(
            DatasetRef(repo_id="demo", limit=1),
            "[data].harmful",
            collector,
            min_recommended=16,
            missing_fix="fix it",
        )

        assert result is None
        assert any("limit=1" in issue for issue in _rendered(collector))

    def test_validate_surface_jsonl_file_covers_error_paths(
        self,
        tmp_path: Path,
    ) -> None:
        collector = ValidationCollector()
        missing = _validate_surface_jsonl_file(
            tmp_path / "missing.jsonl",
            "[surface].prompts",
            collector,
            missing_fix="create it",
        )
        assert missing is None
        assert any("file not found" in issue for issue in _rendered(collector))

        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        collector = ValidationCollector()
        assert _validate_surface_jsonl_file(
            empty,
            "[surface].prompts",
            collector,
            missing_fix="create it",
        ) == 0
        assert any("is empty" in issue for issue in _rendered(collector))

        invalid = tmp_path / "invalid.jsonl"
        invalid.write_text("{bad json\n")
        collector = ValidationCollector()
        assert _validate_surface_jsonl_file(
            invalid,
            "[surface].prompts",
            collector,
            missing_fix="create it",
        ) is None
        assert any("not valid JSON" in issue for issue in _rendered(collector))

        not_object = tmp_path / "not_object.jsonl"
        not_object.write_text('"hello"\n')
        collector = ValidationCollector()
        assert _validate_surface_jsonl_file(
            not_object,
            "[surface].prompts",
            collector,
            missing_fix="create it",
        ) is None
        assert any("JSON object" in issue for issue in _rendered(collector))

    def test_validate_surface_jsonl_file_handles_non_string_keys_and_record_errors(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "surface.jsonl"
        path.write_text(json.dumps({"prompt": "hello"}) + "\n")

        collector = ValidationCollector()
        with patch(
            "vauban.config._validation_files.json.loads",
            return_value={1: "bad-key"},
        ):
            assert _validate_surface_jsonl_file(
                path,
                "[surface].prompts",
                collector,
                missing_fix="create it",
            ) is None
        assert any("keys must be strings" in issue for issue in _rendered(collector))

        collector = ValidationCollector()
        with patch(
            "vauban.config._validation_files.parse_surface_prompt_record",
            side_effect=SurfacePromptRecordError("bad record"),
        ):
            assert _validate_surface_jsonl_file(
                path,
                "[surface].prompts",
                collector,
                missing_fix="create it",
            ) is None
        assert any("bad record" in issue for issue in _rendered(collector))

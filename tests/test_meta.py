"""Tests for [meta] section parsing, loader integration, and suggestions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from vauban._pipeline._context import write_experiment_log
from vauban.config._parse_meta import parse_meta
from vauban.types import (
    CutConfig,
    EvalConfig,
    MeasureConfig,
    MetaConfig,
    MetaDocRef,
    PipelineConfig,
)

if TYPE_CHECKING:
    from vauban.config._types import TomlDict


class TestParseMeta:
    """Unit tests for parse_meta()."""

    def _path(self, name: str = "experiment_v1.toml") -> Path:
        return Path(f"/tmp/{name}")

    def test_parse_meta_absent(self) -> None:
        raw: TomlDict = {"model": {"path": "test"}}
        assert parse_meta(raw, self._path()) is None

    def test_parse_meta_minimal(self) -> None:
        raw: TomlDict = {"meta": {}}
        result = parse_meta(raw, self._path())
        assert result is not None
        assert result.id == "experiment_v1"
        assert result.status == "wip"
        assert result.title == ""
        assert result.parents == []
        assert result.tags == []
        assert result.notes == ""
        assert result.docs == []
        assert result.date == ""

    def test_parse_meta_full(self) -> None:
        raw: TomlDict = {
            "meta": {
                "id": "gan_loop_v3",
                "title": "GAN loop v3: multi-model transfer",
                "status": "promising",
                "parents": ["gan_loop_v2"],
                "tags": ["gcg", "transfer", "gan"],
                "date": "2026-02-25",
                "notes": "Switched to transfer re-ranking.",
                "docs": [
                    {"path": "../docs/design.md", "label": "Design notes"},
                    {"path": "../notes/egd.md"},
                ],
            },
        }
        result = parse_meta(raw, self._path())
        assert result is not None
        assert result.id == "gan_loop_v3"
        assert result.title == "GAN loop v3: multi-model transfer"
        assert result.status == "promising"
        assert result.parents == ["gan_loop_v2"]
        assert result.tags == ["gcg", "transfer", "gan"]
        assert result.date == "2026-02-25"
        assert result.notes == "Switched to transfer re-ranking."
        assert len(result.docs) == 2
        assert result.docs[0] == MetaDocRef(
            path="../docs/design.md", label="Design notes",
        )
        assert result.docs[1] == MetaDocRef(path="../notes/egd.md", label="")

    def test_parse_meta_explicit_id(self) -> None:
        raw: TomlDict = {"meta": {"id": "custom_id"}}
        result = parse_meta(raw, self._path("other.toml"))
        assert result is not None
        assert result.id == "custom_id"

    def test_parse_meta_id_defaults_to_stem(self) -> None:
        raw: TomlDict = {"meta": {}}
        result = parse_meta(raw, self._path("my_experiment.toml"))
        assert result is not None
        assert result.id == "my_experiment"

    def test_parse_meta_invalid_status(self) -> None:
        raw: TomlDict = {"meta": {"status": "great"}}
        with pytest.raises(ValueError, match="status must be one of"):
            parse_meta(raw, self._path())

    def test_parse_meta_bad_parents_type(self) -> None:
        raw: TomlDict = {"meta": {"parents": "not_a_list"}}
        with pytest.raises(TypeError, match="parents must be a list"):
            parse_meta(raw, self._path())

    def test_parse_meta_bad_parent_item(self) -> None:
        raw: TomlDict = {"meta": {"parents": [123]}}
        with pytest.raises(TypeError, match=r"parents\[0\] must be a string"):
            parse_meta(raw, self._path())

    def test_parse_meta_empty_parent_string(self) -> None:
        raw: TomlDict = {"meta": {"parents": [""]}}
        with pytest.raises(ValueError, match=r"parents\[0\] must be non-empty"):
            parse_meta(raw, self._path())

    def test_parse_meta_bad_tags_type(self) -> None:
        raw: TomlDict = {"meta": {"tags": "not_a_list"}}
        with pytest.raises(TypeError, match="tags must be a list"):
            parse_meta(raw, self._path())

    def test_parse_meta_docs_missing_path(self) -> None:
        raw: TomlDict = {"meta": {"docs": [{"label": "no path"}]}}
        with pytest.raises(ValueError, match="requires 'path'"):
            parse_meta(raw, self._path())

    def test_parse_meta_docs_bad_type(self) -> None:
        raw: TomlDict = {"meta": {"docs": ["not a table"]}}
        with pytest.raises(TypeError, match="must be a table"):
            parse_meta(raw, self._path())

    def test_parse_meta_native_date(self) -> None:
        """TOML native dates are datetime.date objects after parsing."""
        import datetime

        raw: TomlDict = {"meta": {"date": datetime.date(2026, 2, 25)}}
        result = parse_meta(raw, self._path())
        assert result is not None
        assert result.date == "2026-02-25"

    def test_parse_meta_string_date(self) -> None:
        raw: TomlDict = {"meta": {"date": "2026-02-25"}}
        result = parse_meta(raw, self._path())
        assert result is not None
        assert result.date == "2026-02-25"

    def test_parse_meta_bad_date_type(self) -> None:
        raw: TomlDict = {"meta": {"date": 12345}}
        with pytest.raises(TypeError, match="date must be a date or string"):
            parse_meta(raw, self._path())

    def test_parse_meta_not_a_table(self) -> None:
        raw: TomlDict = {"meta": "not_a_table"}
        with pytest.raises(TypeError, match="must be a table"):
            parse_meta(raw, self._path())

    def test_parse_meta_empty_id_raises(self) -> None:
        raw: TomlDict = {"meta": {"id": ""}}
        with pytest.raises(ValueError, match="non-empty"):
            parse_meta(raw, self._path())

    def test_parse_meta_all_statuses_valid(self) -> None:
        for status in ("wip", "promising", "dead_end", "baseline",
                       "superseded", "archived"):
            raw: TomlDict = {"meta": {"status": status}}
            result = parse_meta(raw, self._path())
            assert result is not None
            assert result.status == status


class TestLoadConfigWithMeta:
    """Integration tests through load_config()."""

    def test_load_config_with_meta(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            '[data]\nharmful = "default"\nharmless = "default"\n'
            '[meta]\ntitle = "My experiment"\nstatus = "promising"\n'
            'parents = ["baseline_v1"]\ntags = ["gcg"]\n'
        )
        from vauban.config import load_config

        config = load_config(toml_file)
        assert config.meta is not None
        assert config.meta.id == "test"
        assert config.meta.title == "My experiment"
        assert config.meta.status == "promising"
        assert config.meta.parents == ["baseline_v1"]
        assert config.meta.tags == ["gcg"]

    def test_load_config_without_meta(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            '[data]\nharmful = "default"\nharmless = "default"\n'
        )
        from vauban.config import load_config

        config = load_config(toml_file)
        assert config.meta is None


class TestExperimentLogMeta:
    """Test that meta fields appear in experiment log entries."""

    def test_experiment_log_includes_meta(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        config = PipelineConfig(
            model_path="test-model",
            harmful_path=tmp_path / "h.jsonl",
            harmless_path=tmp_path / "hl.jsonl",
            cut=CutConfig(),
            measure=MeasureConfig(),
            eval=EvalConfig(),
            meta=MetaConfig(
                id="test_exp",
                title="Test Experiment",
                status="promising",
                parents=["baseline"],
                tags=["gcg", "test"],
            ),
            output_dir=output_dir,
        )

        write_experiment_log(
            config_path=tmp_path / "test.toml",
            config=config,
            mode="measure",
            reports=["report.json"],
            metrics={"score": 0.5},
            elapsed=1.0,
        )

        log_path = output_dir / "experiment_log.jsonl"
        assert log_path.exists()
        entry = json.loads(log_path.read_text().strip())
        assert "meta" in entry
        assert entry["meta"]["id"] == "test_exp"
        assert entry["meta"]["title"] == "Test Experiment"
        assert entry["meta"]["status"] == "promising"
        assert entry["meta"]["parents"] == ["baseline"]
        assert entry["meta"]["tags"] == ["gcg", "test"]

    def test_experiment_log_no_meta(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        config = PipelineConfig(
            model_path="test-model",
            harmful_path=tmp_path / "h.jsonl",
            harmless_path=tmp_path / "hl.jsonl",
            cut=CutConfig(),
            measure=MeasureConfig(),
            eval=EvalConfig(),
            output_dir=output_dir,
        )

        write_experiment_log(
            config_path=tmp_path / "test.toml",
            config=config,
            mode="measure",
            reports=[],
            metrics={},
            elapsed=0.5,
        )

        log_path = output_dir / "experiment_log.jsonl"
        assert log_path.exists()
        entry = json.loads(log_path.read_text().strip())
        assert "meta" not in entry


class TestSuggestionsMeta:
    """Test that [meta] is recognized by suggestions system."""

    def test_meta_section_known(self) -> None:
        from vauban._suggestions import check_unknown_sections

        raw: TomlDict = {
            "model": {"path": "test"},
            "meta": {"status": "wip"},
        }
        warnings = check_unknown_sections(raw)
        assert not any("meta" in w for w in warnings)

    def test_meta_keys_known(self) -> None:
        from vauban._suggestions import check_unknown_keys

        raw: TomlDict = {
            "meta": {
                "id": "test",
                "title": "Test",
                "status": "wip",
                "parents": [],
                "tags": [],
                "notes": "",
                "docs": [],
                "date": "",
            },
        }
        warnings = check_unknown_keys(raw)
        assert not warnings

    def test_meta_unknown_key_warned(self) -> None:
        from vauban._suggestions import check_unknown_keys

        raw: TomlDict = {
            "meta": {"bogus_key": "value"},
        }
        warnings = check_unknown_keys(raw)
        assert any("bogus_key" in w for w in warnings)

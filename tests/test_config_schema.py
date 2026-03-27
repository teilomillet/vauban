"""Tests for generated JSON Schema and version sourcing."""

import json
import sys
import tomllib
from pathlib import Path
from typing import cast

import pytest

import vauban
from vauban.config import generate_config_schema, write_config_schema


def _object_dict(value: object) -> dict[str, object]:
    """Narrow a JSON-like object to a string-keyed dict for tests."""
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def test_generate_config_schema_contains_core_sections() -> None:
    schema = generate_config_schema()

    assert schema["type"] == "object"
    properties = _object_dict(schema["properties"])
    assert "model" in properties
    assert "data" in properties
    assert "measure" in properties
    assert "ai_act" in properties
    assert "softprompt" in properties
    assert "backend" in properties


def test_generate_config_schema_uses_toml_key_aliases() -> None:
    schema = generate_config_schema()
    properties = _object_dict(schema["properties"])

    eval_schema = _object_dict(properties["eval"])
    eval_properties = _object_dict(eval_schema["properties"])
    assert "prompts" in eval_properties
    assert "prompts_path" not in eval_properties
    assert "refusal_phrases" in eval_properties

    surface_schema = _object_dict(properties["surface"])
    surface_properties = _object_dict(surface_schema["properties"])
    assert "prompts" in surface_properties
    assert "prompts_path" not in surface_properties


def test_generate_config_schema_contains_nested_defs() -> None:
    schema = generate_config_schema()
    defs = _object_dict(schema["$defs"])
    assert "ApiEvalEndpoint" in defs
    assert "AlphaTier" in defs
    assert "PolicyRule" in defs


def test_write_config_schema_writes_json_file(tmp_path: Path) -> None:
    output_path = tmp_path / "vauban.schema.json"

    written_path = write_config_schema(output_path)

    assert written_path == output_path
    loaded = json.loads(output_path.read_text())
    assert loaded["title"] == "Vauban Config"


def test_cli_schema_prints_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["vauban", "schema"])

    from vauban.__main__ import main

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    assert printed["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_cli_schema_writes_output_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "config.schema.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["vauban", "schema", "--output", str(output_path)],
    )

    from vauban.__main__ import main

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert f"Wrote schema to {output_path}" in captured.err
    assert output_path.exists()


def test_pyproject_uses_dynamic_version_source() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["dynamic"] == ["version"]
    assert pyproject["tool"]["hatch"]["version"]["path"] == "vauban/_version.py"
    assert vauban.__version__ == "0.3.2"

"""Tests for actionable validation guidance and schema checks."""

from pathlib import Path

from vauban import validate


def _write_prompt_file(path: Path, count: int) -> None:
    lines = [f'{{"prompt": "prompt {i}"}}' for i in range(count)]
    path.write_text("\n".join(lines) + "\n")


def test_validate_warns_prompt_schema_with_fix_hint(tmp_path: Path) -> None:
    harmful = tmp_path / "harmful.jsonl"
    harmless = tmp_path / "harmless.jsonl"
    harmful.write_text('{"text": "missing prompt key"}\n')
    _write_prompt_file(harmless, 20)

    toml_file = tmp_path / "config.toml"
    toml_file.write_text(
        '[model]\npath = "test"\n'
        '[data]\nharmful = "harmful.jsonl"\nharmless = "harmless.jsonl"\n'
    )

    warnings = validate(toml_file)
    assert any(
        "[data].harmful" in w and "'prompt'" in w and "fix:" in w
        for w in warnings
    )


def test_validate_warns_surface_schema_with_fix_hint(tmp_path: Path) -> None:
    harmful = tmp_path / "harmful.jsonl"
    harmless = tmp_path / "harmless.jsonl"
    surface = tmp_path / "surface.jsonl"
    _write_prompt_file(harmful, 20)
    _write_prompt_file(harmless, 20)
    surface.write_text('{"prompt": "x", "label": "harmful"}\n')

    toml_file = tmp_path / "config.toml"
    toml_file.write_text(
        '[model]\npath = "test"\n'
        '[data]\nharmful = "harmful.jsonl"\nharmless = "harmless.jsonl"\n'
        '[surface]\nprompts = "surface.jsonl"\n'
    )

    warnings = validate(toml_file)
    assert any(
        "[surface].prompts line 1" in w
        and "label, category" in w
        and "fix:" in w
        for w in warnings
    )


def test_validate_warns_surface_optional_turn_depth_type(
    tmp_path: Path,
) -> None:
    harmful = tmp_path / "harmful.jsonl"
    harmless = tmp_path / "harmless.jsonl"
    surface = tmp_path / "surface.jsonl"
    _write_prompt_file(harmful, 20)
    _write_prompt_file(harmless, 20)
    surface.write_text(
        (
            '{"prompt": "x", "label": "harmful", "category": "weapons",'
            ' "turn_depth": "two"}\n'
        ),
    )

    toml_file = tmp_path / "config.toml"
    toml_file.write_text(
        '[model]\npath = "test"\n'
        '[data]\nharmful = "harmful.jsonl"\nharmless = "harmless.jsonl"\n'
        '[surface]\nprompts = "surface.jsonl"\n'
    )

    warnings = validate(toml_file)
    assert any(
        "[surface].prompts line 1" in w
        and "'turn_depth'" in w
        and "fix:" in w
        for w in warnings
    )


def test_validate_accepts_surface_messages_only(tmp_path: Path) -> None:
    harmful = tmp_path / "harmful.jsonl"
    harmless = tmp_path / "harmless.jsonl"
    surface = tmp_path / "surface.jsonl"
    _write_prompt_file(harmful, 20)
    _write_prompt_file(harmless, 20)
    surface.write_text(
        (
            '{"messages":[{"role":"user","content":"turn one"},'
            '{"role":"assistant","content":"ok"},'
            '{"role":"user","content":"turn two"}],'
            '"label":"harmful","category":"hacking"}\n'
        ),
    )

    toml_file = tmp_path / "config.toml"
    toml_file.write_text(
        '[model]\npath = "test"\n'
        '[data]\nharmful = "harmful.jsonl"\nharmless = "harmless.jsonl"\n'
        '[surface]\nprompts = "surface.jsonl"\n'
    )

    warnings = validate(toml_file)
    assert not any(
        "[surface].prompts line 1" in w and "must include either" in w
        for w in warnings
    )


def test_validate_warns_surface_messages_schema_with_fix(
    tmp_path: Path,
) -> None:
    harmful = tmp_path / "harmful.jsonl"
    harmless = tmp_path / "harmless.jsonl"
    surface = tmp_path / "surface.jsonl"
    _write_prompt_file(harmful, 20)
    _write_prompt_file(harmless, 20)
    surface.write_text(
        (
            '{"messages":[{"role":"invalid","content":"x"}],'
            '"label":"harmful","category":"hacking"}\n'
        ),
    )

    toml_file = tmp_path / "config.toml"
    toml_file.write_text(
        '[model]\npath = "test"\n'
        '[data]\nharmful = "harmful.jsonl"\nharmless = "harmless.jsonl"\n'
        '[surface]\nprompts = "surface.jsonl"\n'
    )

    warnings = validate(toml_file)
    assert any(
        "[surface].prompts line 1" in w
        and "messages[0]" in w
        and "fix:" in w
        for w in warnings
    )


def test_validate_warns_surface_gates_without_generation(
    tmp_path: Path,
) -> None:
    harmful = tmp_path / "harmful.jsonl"
    harmless = tmp_path / "harmless.jsonl"
    surface = tmp_path / "surface.jsonl"
    _write_prompt_file(harmful, 20)
    _write_prompt_file(harmless, 20)
    surface.write_text(
        '{"prompt":"x","label":"harmful","category":"weapons"}\n',
    )

    toml_file = tmp_path / "config.toml"
    toml_file.write_text(
        '[model]\npath = "test"\n'
        '[data]\nharmful = "harmful.jsonl"\nharmless = "harmless.jsonl"\n'
        '[surface]\n'
        'prompts = "surface.jsonl"\n'
        "generate = false\n"
        "max_worst_cell_refusal_after = 0.2\n"
    )

    warnings = validate(toml_file)
    assert any(
        "[surface] refusal-rate gates are set" in w
        and "fix:" in w
        for w in warnings
    )


def test_validate_warns_eval_without_prompts_in_default_pipeline(
    tmp_path: Path,
) -> None:
    harmful = tmp_path / "harmful.jsonl"
    harmless = tmp_path / "harmless.jsonl"
    _write_prompt_file(harmful, 20)
    _write_prompt_file(harmless, 20)

    toml_file = tmp_path / "config.toml"
    toml_file.write_text(
        '[model]\npath = "test"\n'
        '[data]\nharmful = "harmful.jsonl"\nharmless = "harmless.jsonl"\n'
        "[eval]\nmax_tokens = 40\n"
    )

    warnings = validate(toml_file)
    assert any(
        "[eval] section is present" in w
        and "eval_report.json will not be produced" in w
        and "fix:" in w
        for w in warnings
    )


def test_validate_missing_data_file_has_actionable_fix(tmp_path: Path) -> None:
    harmless = tmp_path / "harmless.jsonl"
    _write_prompt_file(harmless, 20)

    toml_file = tmp_path / "config.toml"
    toml_file.write_text(
        '[model]\npath = "test"\n'
        '[data]\nharmful = "missing.jsonl"\nharmless = "harmless.jsonl"\n'
    )

    warnings = validate(toml_file)
    assert any(
        "[data].harmful file not found" in w
        and "fix:" in w
        and "default" in w
        for w in warnings
    )


def test_validate_output_dir_file_warns_with_fix(tmp_path: Path) -> None:
    harmful = tmp_path / "harmful.jsonl"
    harmless = tmp_path / "harmless.jsonl"
    _write_prompt_file(harmful, 20)
    _write_prompt_file(harmless, 20)

    output_file = tmp_path / "not_a_dir"
    output_file.write_text("x\n")

    toml_file = tmp_path / "config.toml"
    toml_file.write_text(
        '[model]\npath = "test"\n'
        '[data]\nharmful = "harmful.jsonl"\nharmless = "harmless.jsonl"\n'
        "[output]\ndir = \"not_a_dir\"\n"
    )

    warnings = validate(toml_file)
    assert any(
        "[output].dir points to a file" in w and "fix:" in w
        for w in warnings
    )

# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for the runtime-generated manual."""

import ast

import pytest

from vauban.manual import (
    AutoField,
    FieldSpec,
    ManualField,
    ManualSection,
    SectionSpec,
    _build_section,
    _build_sections,
    _format_default,
    _normalize_topic,
    _parse_class_fields,
    _select_sections,
    manual_topics,
    render_manual,
)


def test_topics_include_quickstart_and_sections() -> None:
    topics = manual_topics()
    assert "quickstart" in topics
    assert "commands" in topics
    assert "validate" in topics
    assert "playbook" in topics
    assert "quick" in topics
    assert "examples" in topics
    assert "print" in topics
    assert "ai_act" in topics
    assert "cut" in topics
    assert "softprompt" in topics


def test_quickstart_topic_contains_minimal_config() -> None:
    text = render_manual("quickstart")
    assert "QUICKSTART" in text
    assert "[model]" in text
    assert 'harmful = "default"' in text


def test_commands_topic_contains_init_and_diff() -> None:
    text = render_manual("commands")
    assert "COMMANDS" in text
    assert "vauban init [--mode MODE]" in text
    assert "vauban diff" in text
    assert "vauban tree" in text
    assert "--format" in text
    assert "known modes:" in text


def test_validate_topic_mentions_fix_guidance() -> None:
    text = render_manual("validate")
    assert "VALIDATE WORKFLOW" in text
    assert "fix:" in text
    assert "--validate run.toml" in text


def test_quick_topic_lists_repl_helpers() -> None:
    text = render_manual("quick")
    assert "PYTHON QUICK API" in text
    assert "helpers:" in text
    assert "measure_direction" in text


def test_examples_topic_contains_core_flows() -> None:
    text = render_manual("examples")
    assert "EXAMPLES" in text
    assert "vauban init --mode default --output run.toml" in text
    assert "vauban diff runs/baseline runs/experiment_a" in text
    assert "vauban tree experiments/ --format mermaid" in text


def test_print_topic_contains_share_commands() -> None:
    text = render_manual("print")
    assert "PRINTING AND SHARING" in text
    assert "vauban man > VAUBAN_MANUAL.txt" in text
    assert "lpr VAUBAN_MANUAL.txt" in text


def test_section_topic_renders_only_requested_section() -> None:
    text = render_manual("cut")
    assert "SECTION [cut]" in text
    assert text.count("SECTION [") == 1
    assert "default: 1.0" in text
    assert "[cut].alpha" in text


def test_ai_act_topic_mentions_readiness() -> None:
    text = render_manual("ai_act")
    assert "SECTION [ai_act]" in text
    assert "readiness report" in text.lower()
    assert "company_name" in text


def test_unknown_topic_raises() -> None:
    with pytest.raises(ValueError, match="Unknown manual topic"):
        render_manual("does-not-exist")


# ── _normalize_topic ─────────────────────────────────────────────────


class TestNormalizeTopic:
    def test_none_returns_none(self) -> None:
        assert _normalize_topic(None) is None

    def test_valid_topic_passthrough(self) -> None:
        assert _normalize_topic("quickstart") == "quickstart"

    def test_strips_whitespace(self) -> None:
        assert _normalize_topic("  cut  ") == "cut"

    def test_lowercase(self) -> None:
        assert _normalize_topic("CUT") == "cut"

    def test_strips_brackets(self) -> None:
        assert _normalize_topic("[cut]") == "cut"

    def test_alias_resolution(self) -> None:
        assert _normalize_topic("start") == "quickstart"
        assert _normalize_topic("cmd") == "commands"
        assert _normalize_topic("lint") == "validate"
        assert _normalize_topic("repl") == "quick"
        assert _normalize_topic("demo") == "examples"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown manual topic"):
            _normalize_topic("nonexistent_topic")


# ── _parse_class_fields ─────────────────────────────────────────────


class TestParseClassFields:
    def test_simple_dataclass(self) -> None:
        source = """
class FakeConfig:
    alpha: float = 1.0
    mode: str = "default"
    layers: list[int] | None = None
"""
        module = ast.parse(source)
        class_node = module.body[0]
        assert isinstance(class_node, ast.ClassDef)
        fields = _parse_class_fields(class_node)
        assert "alpha" in fields
        assert "mode" in fields
        assert "layers" in fields
        assert fields["alpha"].type_name == "float"
        assert fields["alpha"].required is False
        assert fields["alpha"].default_repr == "1.0"
        assert fields["mode"].default_repr == '"default"'

    def test_required_field(self) -> None:
        source = """
class FakeConfig:
    name: str
"""
        module = ast.parse(source)
        class_node = module.body[0]
        assert isinstance(class_node, ast.ClassDef)
        fields = _parse_class_fields(class_node)
        assert fields["name"].required is True
        assert fields["name"].default_repr is None

    def test_bool_default(self) -> None:
        source = """
class FakeConfig:
    enabled: bool = True
    disabled: bool = False
"""
        module = ast.parse(source)
        class_node = module.body[0]
        assert isinstance(class_node, ast.ClassDef)
        fields = _parse_class_fields(class_node)
        assert fields["enabled"].default_repr == "true"
        assert fields["disabled"].default_repr == "false"


# ── _format_default ─────────────────────────────────────────────────


class TestFormatDefault:
    def test_none(self) -> None:
        assert _format_default(None) == "null"

    def test_bool_true(self) -> None:
        assert _format_default(True) == "true"

    def test_bool_false(self) -> None:
        assert _format_default(False) == "false"

    def test_string(self) -> None:
        assert _format_default("hello") == '"hello"'

    def test_int(self) -> None:
        assert _format_default(42) == "42"

    def test_float(self) -> None:
        assert _format_default(0.5) == "0.5"

    def test_list(self) -> None:
        assert _format_default([1, 2, 3]) == "[1, 2, 3]"

    def test_empty_list(self) -> None:
        assert _format_default([]) == "[]"

    def test_nested_list(self) -> None:
        assert _format_default(["a", "b"]) == '["a", "b"]'

    def test_dict(self) -> None:
        result = _format_default({"key": "val"})
        assert '"key"' in result
        assert '"val"' in result


# ── _build_section ──────────────────────────────────────────────────


class TestBuildSection:
    def test_builds_from_spec_without_config_class(self) -> None:
        spec = SectionSpec(
            name="test_section",
            description="A test section.",
            fields=(
                FieldSpec(key="alpha", description="The alpha value"),
            ),
        )
        section = _build_section(spec)
        assert section.name == "test_section"
        assert section.description == "A test section."
        assert len(section.fields) == 1
        assert section.fields[0].key == "alpha"
        assert section.fields[0].description == "The alpha value"
        # Without config_class, type defaults to "object"
        assert section.fields[0].type_name == "object"

    def test_fields_preserve_order(self) -> None:
        spec = SectionSpec(
            name="test",
            description="desc",
            fields=(
                FieldSpec(key="first", description="1st"),
                FieldSpec(key="second", description="2nd"),
                FieldSpec(key="third", description="3rd"),
            ),
        )
        section = _build_section(spec)
        assert [f.key for f in section.fields][:3] == [
            "first", "second", "third",
        ]


# ── _build_sections ─────────────────────────────────────────────────


class TestBuildSections:
    def test_returns_non_empty(self) -> None:
        sections = _build_sections()
        assert len(sections) > 0

    def test_all_sections_have_names(self) -> None:
        sections = _build_sections()
        for section in sections:
            assert section.name

    def test_known_sections_present(self) -> None:
        sections = _build_sections()
        names = {s.name for s in sections}
        assert "model" in names
        assert "cut" in names
        assert "measure" in names


# ── _select_sections ────────────────────────────────────────────────


class TestSelectSections:
    def test_none_returns_all(self) -> None:
        sections = _build_sections()
        selected = _select_sections(sections, None)
        assert selected == sections

    def test_all_returns_all(self) -> None:
        sections = _build_sections()
        selected = _select_sections(sections, "all")
        assert selected == sections

    def test_meta_topic_returns_empty(self) -> None:
        sections = _build_sections()
        for topic in ("quickstart", "commands", "validate", "playbook",
                       "quick", "examples", "print", "modes", "formats"):
            selected = _select_sections(sections, topic)
            assert selected == ()

    def test_section_topic_returns_single(self) -> None:
        sections = _build_sections()
        selected = _select_sections(sections, "cut")
        assert len(selected) == 1
        assert selected[0].name == "cut"

    def test_unknown_section_returns_empty(self) -> None:
        sections = _build_sections()
        selected = _select_sections(sections, "nonexistent_section")
        assert selected == ()


# ── render_manual for every topic ────────────────────────────────────


@pytest.mark.parametrize("topic", manual_topics())
def test_render_manual_produces_output(topic: str) -> None:
    text = render_manual(topic)
    assert len(text) > 0
    assert "VAUBAN(1)" in text


# ── DataClass round-trip tests ──────────────────────────────────────


class TestManualDataclasses:
    def test_field_spec_frozen(self) -> None:
        fs = FieldSpec(key="k", description="d")
        with pytest.raises(AttributeError):
            fs.key = "other"  # type: ignore[misc]

    def test_section_spec_defaults(self) -> None:
        ss = SectionSpec(name="n", description="d")
        assert ss.required is False
        assert ss.early_return is False
        assert ss.table is True
        assert ss.config_class is None
        assert ss.fields == ()

    def test_auto_field_slots(self) -> None:
        af = AutoField(type_name="str", default_repr='"hi"', required=False)
        assert af.type_name == "str"
        assert af.required is False

    def test_manual_field_defaults(self) -> None:
        mf = ManualField(
            key="k", type_name="int", required=True,
            default_repr=None, description="desc", constraints=None,
        )
        assert mf.notes == ()

    def test_manual_section_defaults(self) -> None:
        ms = ManualSection(
            name="n", description="d", required=False,
            early_return=False, table=True, fields=(),
            notes=(),
        )
        assert ms.notes == ()


# ── No auto-discovered fields ───────────────────────────────────────


def test_no_auto_discovered_fields() -> None:
    """Every config dataclass field must have an explicit FieldSpec.

    If this test fails, a field was added to a config dataclass in
    types.py without a corresponding FieldSpec in manual.py.  Add a
    FieldSpec to the relevant SectionSpec in ``_SECTION_SPECS`` to fix.
    """
    sections = _build_sections()
    undocumented: list[str] = []
    for section in sections:
        for field in section.fields:
            if field.description.startswith("Auto-discovered"):
                undocumented.append(
                    f"[{section.name}].{field.key}"
                )
    assert undocumented == [], (
        "Fields missing explicit FieldSpec in manual.py "
        "(add a FieldSpec to _SECTION_SPECS):\n  "
        + "\n  ".join(undocumented)
    )

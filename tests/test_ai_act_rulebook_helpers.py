# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Focused coverage tests for AI Act rulebook and document helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

import vauban.ai_act as ai_act_module

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _clear_rulebook_caches() -> Iterator[None]:
    """Reset rulebook helper caches around each test."""
    ai_act_module._rulebook_v1.cache_clear()
    ai_act_module._rulebook_resource_bytes.cache_clear()
    yield
    ai_act_module._rulebook_v1.cache_clear()
    ai_act_module._rulebook_resource_bytes.cache_clear()


class TestRulebookHelpers:
    """Tests for packaged rulebook validation helpers."""

    def test_rulebook_v1_requires_object_payload(self) -> None:
        with (
            patch.object(ai_act_module, "_rulebook_resource_bytes", return_value=b"[]"),
            patch.object(ai_act_module.json, "loads", return_value=["bad"]),
            pytest.raises(TypeError, match="must decode to a JSON object"),
        ):
            ai_act_module._rulebook_v1()

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            (
                {"source_snapshot_date": "2026-04-04"},
                "missing rulebook_version",
            ),
            (
                {"rulebook_version": "v1"},
                "missing source_snapshot_date",
            ),
        ],
    )
    def test_rulebook_metadata_requires_string_fields(
        self,
        payload: dict[str, object],
        message: str,
    ) -> None:
        with (
            patch.object(ai_act_module, "_rulebook_v1", return_value=payload),
            patch.object(ai_act_module, "_rulebook_resource_bytes", return_value=b"{}"),
            pytest.raises(TypeError, match=message),
        ):
            ai_act_module._rulebook_metadata()

    def test_rulebook_controls_rejects_non_list(self) -> None:
        with (
            patch.object(ai_act_module, "_rulebook_v1", return_value={"controls": {}}),
            pytest.raises(TypeError, match="controls must be a list"),
        ):
            ai_act_module._rulebook_controls()

    def test_rulebook_controls_rejects_non_object_entries(self) -> None:
        with (
            patch.object(
                ai_act_module,
                "_rulebook_v1",
                return_value={"controls": ["bad"]},
            ),
            pytest.raises(TypeError, match="control entries must be objects"),
        ):
            ai_act_module._rulebook_controls()

    @pytest.mark.parametrize(
        ("control", "message"),
        [
            (
                {"source_ids": "s1"},
                "source_ids must be a list",
            ),
            (
                {"source_ids": ["s1", 2]},
                "source_ids must be strings",
            ),
        ],
    )
    def test_control_source_citations_rejects_invalid_source_ids(
        self,
        control: dict[str, object],
        message: str,
    ) -> None:
        with pytest.raises(TypeError, match=message):
            ai_act_module._control_source_citations(control)

    def test_control_source_citations_rejects_unknown_source_ids(self) -> None:
        control = {"source_ids": ["missing"]}
        with (
            patch.object(ai_act_module, "_sources_by_id", return_value={}),
            pytest.raises(TypeError, match="unknown source_id 'missing'"),
        ):
            ai_act_module._control_source_citations(control)

    def test_control_source_citations_resolves_known_source_ids(self) -> None:
        control = {"source_ids": ["s1"]}
        sources = {"s1": {"source_id": "s1", "title": "AI Act"}}

        with patch.object(ai_act_module, "_sources_by_id", return_value=sources):
            assert ai_act_module._control_source_citations(control) == [
                {"source_id": "s1", "title": "AI Act"},
            ]

    def test_annex_iii_catalog_rejects_non_list(self) -> None:
        with (
            patch.object(
                ai_act_module,
                "_rulebook_v1",
                return_value={"annex_iii_catalog": {}},
            ),
            pytest.raises(TypeError, match="annex_iii_catalog must be a list"),
        ):
            ai_act_module._annex_iii_catalog()

    def test_annex_iii_catalog_rejects_non_object_entries(self) -> None:
        with (
            patch.object(
                ai_act_module,
                "_rulebook_v1",
                return_value={"annex_iii_catalog": ["bad"]},
            ),
            pytest.raises(TypeError, match="catalog entries must be objects"),
        ):
            ai_act_module._annex_iii_catalog()

    def test_annex_iii_catalog_by_id_requires_use_case_id(self) -> None:
        with (
            patch.object(ai_act_module, "_annex_iii_catalog", return_value=[{}]),
            pytest.raises(TypeError, match="missing use_case_id"),
        ):
            ai_act_module._annex_iii_catalog_by_id()

    @pytest.mark.parametrize(
        ("rulebook", "message"),
        [
            (
                {"markers": []},
                "markers must be an object",
            ),
            (
                {"markers": {}},
                "markers\\['minimum'\\] must be a list",
            ),
            (
                {"markers": {"minimum": ["bad"]}},
                "marker entry for 'minimum' must be an object",
            ),
            (
                {"markers": {"minimum": [{"patterns": ["x"]}]}},
                "missing string label",
            ),
            (
                {"markers": {"minimum": [{"label": "x"}]}},
                "missing patterns list",
            ),
            (
                {"markers": {"minimum": [{"label": "x", "patterns": ["ok", 1]}]}},
                "non-string patterns",
            ),
        ],
    )
    def test_rulebook_markers_validate_entry_shapes(
        self,
        rulebook: dict[str, object],
        message: str,
    ) -> None:
        with (
            patch.object(ai_act_module, "_rulebook_v1", return_value=rulebook),
            pytest.raises(TypeError, match=message),
        ):
            ai_act_module._rulebook_markers("minimum")

    def test_rulebook_markers_returns_typed_marker_rules(self) -> None:
        rulebook = {
            "markers": {
                "minimum": [
                    {"label": "owner", "patterns": ["owner:", "lead:"]},
                ],
            },
        }

        with patch.object(ai_act_module, "_rulebook_v1", return_value=rulebook):
            assert ai_act_module._rulebook_markers("minimum") == (
                ("owner", ("owner:", "lead:")),
            )

    @pytest.mark.parametrize(
        ("rulebook", "message"),
        [
            (
                {"structured_fields": []},
                "structured_fields must be an object",
            ),
            (
                {"structured_fields": {}},
                "structured_fields\\['schema'\\] must be a list",
            ),
            (
                {"structured_fields": {"schema": ["bad"]}},
                "structured field entry for 'schema' must be an object",
            ),
            (
                {"structured_fields": {"schema": [{"aliases": ["owner"]}]}},
                "missing label",
            ),
            (
                {"structured_fields": {"schema": [{"label": "Owner"}]}},
                "missing aliases list",
            ),
            (
                {
                    "structured_fields": {
                        "schema": [{"label": "Owner", "aliases": ["owner", 1]}],
                    },
                },
                "non-string aliases",
            ),
        ],
    )
    def test_rulebook_structured_fields_validate_entry_shapes(
        self,
        rulebook: dict[str, object],
        message: str,
    ) -> None:
        with (
            patch.object(ai_act_module, "_rulebook_v1", return_value=rulebook),
            pytest.raises(TypeError, match=message),
        ):
            ai_act_module._rulebook_structured_fields("schema")

    def test_rulebook_structured_fields_returns_typed_rules(self) -> None:
        rulebook = {
            "structured_fields": {
                "schema": [
                    {"label": "Owner", "aliases": ["owner", "owner name"]},
                ],
            },
        }

        with patch.object(ai_act_module, "_rulebook_v1", return_value=rulebook):
            assert ai_act_module._rulebook_structured_fields("schema") == (
                ("Owner", ("owner", "owner name")),
            )

    @pytest.mark.parametrize(
        ("rulebook", "message"),
        [
            (
                {"technical_metric_rules": {}},
                "technical_metric_rules must be a list",
            ),
            (
                {"technical_metric_rules": ["bad"]},
                "rule entries must be objects",
            ),
        ],
    )
    def test_rulebook_technical_metric_rules_validate_shape(
        self,
        rulebook: dict[str, object],
        message: str,
    ) -> None:
        with (
            patch.object(ai_act_module, "_rulebook_v1", return_value=rulebook),
            pytest.raises(TypeError, match=message),
        ):
            ai_act_module._rulebook_technical_metric_rules()

    def test_control_definition_builds_expected_payload(self) -> None:
        result = ai_act_module._control_definition(
            "A4-LIT-001",
            "AI literacy",
            "regulation",
            "Article 4",
            "Train operators.",
            "Always",
            ["literacy_record"],
            "manual_review",
            blocking=True,
        )

        assert result == {
            "control_id": "A4-LIT-001",
            "title": "AI literacy",
            "source_type": "regulation",
            "source_ref": "Article 4",
            "description": "Train operators.",
            "applies_when": "Always",
            "evidence_required": ["literacy_record"],
            "check_method": "manual_review",
            "blocking": True,
        }


class TestDocumentHelpers:
    """Tests for document parsing helpers used by AI Act readiness."""

    def test_path_hash_and_size_helpers_handle_success_and_missing(
        self,
        tmp_path: Path,
    ) -> None:
        existing = tmp_path / "evidence.txt"
        existing.write_text("alpha", encoding="utf-8")
        missing = tmp_path / "missing.txt"

        assert ai_act_module._path_sha256(existing) == ai_act_module._sha256_bytes(
            b"alpha",
        )
        assert ai_act_module._path_size_bytes(existing) == 5
        assert ai_act_module._path_sha256(missing) is None
        assert ai_act_module._path_size_bytes(missing) is None

    def test_collect_json_field_names_recurses_nested_dicts_and_lists(self) -> None:
        detected_fields: set[str] = set()

        ai_act_module._collect_json_field_names(
            {
                "Owner Name": {"Risk Topics": "misuse"},
                "nested": [{"Human Oversight": True}],
                1: {"ignored": True},
            },
            detected_fields,
        )

        assert detected_fields == {
            "owner_name",
            "risk_topics",
            "nested",
            "human_oversight",
            "ignored",
        }

    def test_extract_structured_field_names_returns_empty_for_missing_or_invalid_json(
        self,
        tmp_path: Path,
    ) -> None:
        missing = tmp_path / "missing.json"
        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("{not valid json", encoding="utf-8")

        assert ai_act_module._extract_structured_field_names(missing) == set()
        assert ai_act_module._extract_structured_field_names(invalid_json) == set()

    def test_extract_structured_field_names_reads_json_and_markdown(
        self,
        tmp_path: Path,
    ) -> None:
        json_path = tmp_path / "fields.json"
        json_path.write_text(
            '{"Owner":"Alice","items":[{"Risk Topics":"misuse"}]}',
            encoding="utf-8",
        )
        markdown_path = tmp_path / "fields.md"
        markdown_path.write_text(
            "\n# Oversight Plan\nOwner: Alice\n- Risk Topics: misuse\n",
            encoding="utf-8",
        )

        assert ai_act_module._extract_structured_field_names(json_path) == {
            "owner",
            "items",
            "risk_topics",
        }
        assert ai_act_module._extract_structured_field_names(markdown_path) == {
            "oversight_plan",
            "owner",
            "risk_topics",
        }

    def test_extract_structured_field_names_returns_empty_on_text_read_error(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "fields.md"
        path.write_text("Owner: Alice\n", encoding="utf-8")

        with patch.object(Path, "read_text", side_effect=OSError("boom")):
            assert ai_act_module._extract_structured_field_names(path) == set()

    def test_document_has_draft_placeholders_handles_scaffold_and_read_error(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "draft.md"
        path.write_text("Template status: draft\n", encoding="utf-8")

        assert ai_act_module._document_has_draft_placeholders(None) is False
        assert ai_act_module._document_has_draft_placeholders(path) is True

        with patch.object(Path, "read_text", side_effect=OSError("boom")):
            assert ai_act_module._document_has_draft_placeholders(path) is False

    def test_document_marker_assessment_handles_placeholder_schema_and_read_error(
        self,
        tmp_path: Path,
    ) -> None:
        placeholder_path = tmp_path / "placeholder.md"
        placeholder_path.write_text("Replace before use: yes\n", encoding="utf-8")
        markers = (("owner", ("owner:",)),)
        structured_fields = (("Owner", ("owner",)),)

        with patch.object(
            ai_act_module,
            "_rulebook_structured_fields",
            return_value=structured_fields,
        ):
            present, missing = ai_act_module._document_marker_assessment(
                placeholder_path,
                markers,
                schema_name="schema",
            )
            assert present == []
            assert missing == [
                "replace_scaffold_placeholders",
                "owner",
                "field:Owner",
            ]

        readable_path = tmp_path / "readable.md"
        readable_path.write_text("Owner: Alice\n", encoding="utf-8")

        with (
            patch.object(
                ai_act_module,
                "_rulebook_structured_fields",
                return_value=structured_fields,
            ),
            patch.object(
                ai_act_module,
                "_document_has_draft_placeholders",
                return_value=False,
            ),
            patch.object(Path, "read_text", side_effect=OSError("boom")),
        ):
            present, missing = ai_act_module._document_marker_assessment(
                readable_path,
                markers,
                schema_name="schema",
            )
            assert present == []
            assert missing == ["owner", "field:Owner"]

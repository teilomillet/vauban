# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Cross-registry consistency tests.

These tests catch silent drift between the many registries that must
stay in sync when a new pipeline module is added.  Each test names a
canonical source of truth and asserts that downstream registries are
complete with respect to it.
"""

from __future__ import annotations

from vauban._init import KNOWN_MODES
from vauban._pipeline._modes import EARLY_MODE_RUNNERS
from vauban.config._mode_registry import (
    EARLY_MODE_DESCRIPTION_BY_MODE,
    EARLY_MODE_LABEL_BY_SECTION,
    EARLY_MODE_SPECS,
    early_mode_label,
)
from vauban.config._registry import SECTION_PARSE_SPECS
from vauban.config._schema import (
    _DATACLASS_SECTION_SPECS,
    KNOWN_SECTION_KEYS,
    KNOWN_TOP_LEVEL_KEYS,
)
from vauban.manual import (
    _PIPELINE_MODES,
    _SECTION_SPECS,
    EARLY_RETURN_PRECEDENCE,
)

# ---------------------------------------------------------------------------
# Canonical sets derived from the two primary registries
# ---------------------------------------------------------------------------

# All TOML section names that have a parser.
_PARSE_SECTIONS: frozenset[str] = frozenset(
    spec.section for spec in SECTION_PARSE_SPECS
)

# All early-return mode names (bare, e.g. "depth").
_EARLY_MODES: frozenset[str] = frozenset(
    spec.mode for spec in EARLY_MODE_SPECS
)

# All early-return section names (bracketed, e.g. "[depth]").
_EARLY_SECTIONS: frozenset[str] = frozenset(
    spec.section for spec in EARLY_MODE_SPECS
)


# ---------------------------------------------------------------------------
# 1. EARLY_MODE_RUNNERS must cover every EARLY_MODE_SPEC
# ---------------------------------------------------------------------------


class TestModeRunnersCoverSpecs:
    """EARLY_MODE_RUNNERS keys must match EARLY_MODE_SPECS modes."""

    def test_runners_cover_all_specs(self) -> None:
        runner_keys = frozenset(EARLY_MODE_RUNNERS.keys())
        assert runner_keys == _EARLY_MODES, (
            f"Modes in EARLY_MODE_SPECS but missing from EARLY_MODE_RUNNERS: "
            f"{_EARLY_MODES - runner_keys}"
        )

    def test_no_orphan_runners(self) -> None:
        runner_keys = frozenset(EARLY_MODE_RUNNERS.keys())
        orphans = runner_keys - _EARLY_MODES
        assert not orphans, (
            f"Modes in EARLY_MODE_RUNNERS with no EARLY_MODE_SPEC: {orphans}"
        )


# ---------------------------------------------------------------------------
# 2. Schema covers all parsed sections
# ---------------------------------------------------------------------------

# Sections that are parsed but don't come from a dataclass spec
# (they have manual schema definitions in generate_config_schema).
_MANUAL_SCHEMA_SECTIONS: frozenset[str] = frozenset({
    "behavior_report", "model", "data", "output",
})


class TestSchemaCoversParsers:
    """_DATACLASS_SECTION_SPECS must cover every parser section
    that isn't handled by a manual schema."""

    def test_schema_covers_all_parsed_sections(self) -> None:
        schema_sections = frozenset(
            spec.name for spec in _DATACLASS_SECTION_SPECS
        )
        # meta is in schema but not in parse_specs — that's fine.
        # We only check that parsed sections appear in schema or manual.
        uncovered = _PARSE_SECTIONS - schema_sections - _MANUAL_SCHEMA_SECTIONS
        assert not uncovered, (
            f"Sections in SECTION_PARSE_SPECS but missing from "
            f"_DATACLASS_SECTION_SPECS: {uncovered}"
        )

    def test_known_section_keys_cover_all_parsed_sections(self) -> None:
        known = frozenset(KNOWN_SECTION_KEYS.keys())
        uncovered = _PARSE_SECTIONS - known - _MANUAL_SCHEMA_SECTIONS
        assert not uncovered, (
            f"Sections in SECTION_PARSE_SPECS but missing from "
            f"KNOWN_SECTION_KEYS: {uncovered}"
        )

    def test_known_top_level_keys_cover_all_parsed_sections(self) -> None:
        uncovered = _PARSE_SECTIONS - KNOWN_TOP_LEVEL_KEYS
        assert not uncovered, (
            f"Sections in SECTION_PARSE_SPECS but missing from "
            f"KNOWN_TOP_LEVEL_KEYS: {uncovered}"
        )


# ---------------------------------------------------------------------------
# 3. Suggestions cover all parsed sections
# ---------------------------------------------------------------------------


class TestSuggestionsCoverParsers:
    """KNOWN_SECTION_KEYS must cover every parsed section."""

    def test_suggestions_cover_all_parsed_sections(self) -> None:
        suggestion_sections = frozenset(KNOWN_SECTION_KEYS.keys())
        uncovered = _PARSE_SECTIONS - suggestion_sections - _MANUAL_SCHEMA_SECTIONS
        assert not uncovered, (
            f"Sections in SECTION_PARSE_SPECS but missing from "
            f"KNOWN_SECTION_KEYS: {uncovered}"
        )


# ---------------------------------------------------------------------------
# 4. Manual covers all parsed sections and modes
# ---------------------------------------------------------------------------


class TestManualCoversAll:
    """manual.py registries must cover all sections and modes."""

    def test_section_specs_cover_all_parsed_sections(self) -> None:
        manual_sections = frozenset(spec.name for spec in _SECTION_SPECS)
        uncovered = _PARSE_SECTIONS - manual_sections
        assert not uncovered, (
            f"Sections in SECTION_PARSE_SPECS but missing from "
            f"manual._SECTION_SPECS: {uncovered}"
        )

    def test_pipeline_modes_cover_all_early_modes(self) -> None:
        manual_modes = frozenset(m.mode for m in _PIPELINE_MODES)
        # manual has an extra "default" mode
        uncovered = _EARLY_MODES - manual_modes
        assert not uncovered, (
            f"Modes in EARLY_MODE_SPECS but missing from "
            f"manual._PIPELINE_MODES: {uncovered}"
        )

    def test_early_return_precedence_matches_registry(self) -> None:
        expected = tuple(spec.mode for spec in EARLY_MODE_SPECS)
        assert expected == EARLY_RETURN_PRECEDENCE, (
            "manual.EARLY_RETURN_PRECEDENCE does not match "
            "_mode_registry.EARLY_MODE_SPECS order"
        )


# ---------------------------------------------------------------------------
# 5. Init scaffolds cover all early-return modes
# ---------------------------------------------------------------------------


class TestInitCoversAllModes:
    """_MODE_TEMPLATES should cover every early-return mode."""

    def test_init_covers_all_early_modes(self) -> None:
        uncovered = _EARLY_MODES - KNOWN_MODES
        assert not uncovered, (
            f"Modes in EARLY_MODE_SPECS but missing from "
            f"_init._MODE_TEMPLATES: {uncovered}"
        )

    def test_init_descriptions_cover_all_early_modes(self) -> None:
        uncovered = _EARLY_MODES - frozenset(EARLY_MODE_DESCRIPTION_BY_MODE)
        assert not uncovered, (
            "Modes in EARLY_MODE_SPECS but missing from "
            f"EARLY_MODE_DESCRIPTION_BY_MODE: {uncovered}"
        )


# ---------------------------------------------------------------------------
# 6. Validation render covers all early-return modes
# ---------------------------------------------------------------------------


class TestValidationRenderCoversAllModes:
    """Shared validation labels must cover every early mode."""

    def test_mode_labels_cover_all_early_sections(self) -> None:
        for spec in EARLY_MODE_SPECS:
            assert spec.section in EARLY_MODE_LABEL_BY_SECTION, (
                "EARLY_MODE_LABEL_BY_SECTION is missing "
                f"an entry for {spec.section!r}"
            )
            assert early_mode_label(spec.section) == spec.validation_label, (
                f"early_mode_label returned unexpected label for {spec.section!r}"
            )

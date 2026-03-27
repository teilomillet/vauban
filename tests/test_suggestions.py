# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban._suggestions — typo detection and help hints."""

from pathlib import Path

import pytest

from vauban._suggestions import (
    check_unknown_keys,
    check_unknown_sections,
    check_value_constraints,
)


class TestCheckUnknownSections:
    def test_typo_suggests_correction(self) -> None:
        raw: dict[str, object] = {"modle": {"path": "x"}}
        warnings = check_unknown_sections(raw)
        assert len(warnings) == 1
        assert "modle" in warnings[0]
        assert "model" in warnings[0]
        assert "did you mean" in warnings[0]

    def test_no_match_gives_generic_warning(self) -> None:
        raw: dict[str, object] = {"foobar": {"x": 1}}
        warnings = check_unknown_sections(raw)
        assert len(warnings) == 1
        assert "foobar" in warnings[0]
        assert "not a recognized" in warnings[0]

    def test_known_sections_no_warnings(self) -> None:
        raw: dict[str, object] = {
            "model": {"path": "x"},
            "data": {"harmful": "default", "harmless": "default"},
            "cut": {},
        }
        warnings = check_unknown_sections(raw)
        assert warnings == []

    def test_multiple_typos(self) -> None:
        raw: dict[str, object] = {"modle": {}, "ctu": {}}
        warnings = check_unknown_sections(raw)
        assert len(warnings) == 2

    def test_close_typo_for_surface(self) -> None:
        raw: dict[str, object] = {"surfce": {}}
        warnings = check_unknown_sections(raw)
        assert len(warnings) == 1
        assert "surface" in warnings[0]


class TestCheckUnknownKeys:
    def test_key_typo_suggests_correction(self) -> None:
        raw: dict[str, object] = {"cut": {"alph": 1.0}}
        warnings = check_unknown_keys(raw)
        assert len(warnings) == 1
        assert "alph" in warnings[0]
        assert "alpha" in warnings[0]
        assert "did you mean" in warnings[0]

    def test_unknown_key_generic_warning(self) -> None:
        raw: dict[str, object] = {"cut": {"zzzzz": 42}}
        warnings = check_unknown_keys(raw)
        assert len(warnings) == 1
        assert "zzzzz" in warnings[0]
        assert "not a recognized key" in warnings[0]

    def test_known_keys_no_warnings(self) -> None:
        raw: dict[str, object] = {
            "model": {"path": "x"},
            "cut": {"alpha": 1.0, "layers": [1, 2]},
            "measure": {"mode": "diff", "diff_model": "base", "measure_only": True},
        }
        warnings = check_unknown_keys(raw)
        assert warnings == []

    def test_hf_subtable_not_flagged(self) -> None:
        """HF dataset sub-tables in [data] should be skipped."""
        raw: dict[str, object] = {
            "data": {
                "harmful": {"hf": "repo/id", "split": "train"},
                "harmless": "default",
            },
        }
        warnings = check_unknown_keys(raw)
        assert warnings == []

    def test_skips_sections_not_in_known_keys(self) -> None:
        """Sections without a _KNOWN_KEYS entry are ignored."""
        raw: dict[str, object] = {"verbose": True}
        warnings = check_unknown_keys(raw)
        assert warnings == []

    def test_multiple_keys_in_one_section(self) -> None:
        raw: dict[str, object] = {"eval": {"propmts": "x", "max_toekns": 100}}
        warnings = check_unknown_keys(raw)
        assert len(warnings) == 2


class TestCheckValueConstraints:
    def test_bad_enum_suggests_correction(self) -> None:
        raw: dict[str, object] = {"measure": {"mode": "directon"}}
        warnings = check_value_constraints(raw)
        assert len(warnings) == 1
        assert "directon" in warnings[0]
        assert "did you mean" in warnings[0]
        assert "direction" in warnings[0]

    def test_bad_enum_no_close_match(self) -> None:
        raw: dict[str, object] = {"measure": {"mode": "zzzzz"}}
        warnings = check_value_constraints(raw)
        assert len(warnings) == 1
        assert "zzzzz" in warnings[0]
        assert "expected one of" in warnings[0]

    def test_valid_enum_no_warnings(self) -> None:
        raw: dict[str, object] = {
            "measure": {"mode": "direction"},
            "cut": {"layer_strategy": "top_k"},
            "softprompt": {"mode": "gcg", "loss_mode": "targeted"},
        }
        warnings = check_value_constraints(raw)
        assert warnings == []

    def test_numeric_below_range_warns(self) -> None:
        raw: dict[str, object] = {"measure": {"clip_quantile": -0.1}}
        warnings = check_value_constraints(raw)
        assert len(warnings) == 1
        assert ">= 0.0" in warnings[0]

    def test_numeric_above_range_warns(self) -> None:
        raw: dict[str, object] = {"cut": {"sparsity": 1.0}}
        warnings = check_value_constraints(raw)
        assert len(warnings) == 1
        assert "< 1.0" in warnings[0]

    def test_numeric_in_range_no_warnings(self) -> None:
        raw: dict[str, object] = {
            "measure": {"clip_quantile": 0.05},
            "softprompt": {"n_tokens": 16, "n_steps": 200},
        }
        warnings = check_value_constraints(raw)
        assert warnings == []

    def test_optional_none_skipped(self) -> None:
        """None-valued optional keys should not produce warnings."""
        raw: dict[str, object] = {"cut": {"layer_type_filter": None}}
        warnings = check_value_constraints(raw)
        assert warnings == []

    def test_missing_section_skipped(self) -> None:
        raw: dict[str, object] = {"model": {"path": "x"}}
        warnings = check_value_constraints(raw)
        assert warnings == []

    def test_unbounded_upper_range(self) -> None:
        """Unbounded upper range (None) should not warn for large values."""
        raw: dict[str, object] = {"softprompt": {"n_tokens": 999999}}
        warnings = check_value_constraints(raw)
        assert warnings == []


class TestValidateIntegration:
    def test_typo_warning_in_validate(self, tmp_path: Path) -> None:
        """Unknown sections should appear in validate() output."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            '[data]\nharmful = "default"\nharmless = "default"\n'
            "[modle]\nfoo = 1\n"
        )
        import vauban

        warnings = vauban.validate(toml_file)
        typo_warnings = [w for w in warnings if "modle" in w]
        assert len(typo_warnings) == 1
        assert "model" in typo_warnings[0]

    def test_key_typo_in_validate(self, tmp_path: Path) -> None:
        """Unknown keys within sections should appear in validate() output."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            '[data]\nharmful = "default"\nharmless = "default"\n'
            '[cut]\nalph = 1.0\n'
        )
        import vauban

        warnings = vauban.validate(toml_file)
        key_warnings = [w for w in warnings if "alph" in w]
        assert len(key_warnings) == 1
        assert "alpha" in key_warnings[0]

    def test_value_constraint_runs_in_validate(self, tmp_path: Path) -> None:
        """Value constraints run before load_config; bad values raise in loader."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(
            '[model]\npath = "test"\n'
            '[data]\nharmful = "default"\nharmless = "default"\n'
            '[measure]\nmode = "directon"\n'
        )
        import vauban

        # load_config raises ValueError for invalid enum before warnings return
        with pytest.raises(ValueError, match="directon"):
            vauban.validate(toml_file)

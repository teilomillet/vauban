"""Tests for vauban._suggestions — typo detection and help hints."""

from pathlib import Path

from vauban._suggestions import check_unknown_keys, check_unknown_sections


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

# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra tests for ``vauban.config._parse_helpers``."""

import pytest

from vauban.config._parse_helpers import SectionReader, require_toml_table


class TestRequireTomlTableExtra:
    def test_rejects_non_string_keys(self) -> None:
        with pytest.raises(TypeError, match="keys must be strings"):
            require_toml_table("[demo]", {1: "bad"})


class TestSectionReaderExtra:
    def test_int_list_rejects_non_int_elements(self) -> None:
        reader = SectionReader("[demo]", {"layers": [0, "bad"]})

        with pytest.raises(TypeError, match="elements must be integers"):
            reader.int_list("layers")

    def test_optional_number_list_rejects_non_numeric_elements(self) -> None:
        reader = SectionReader("[demo]", {"weights": [0.1, "bad"]})

        with pytest.raises(TypeError, match="elements must be numbers"):
            reader.optional_number_list("weights")

    def test_str_float_table_rejects_non_table_and_non_numeric_values(self) -> None:
        reader = SectionReader("[demo]", {"composition": "bad"})
        with pytest.raises(TypeError, match="must be a table"):
            reader.str_float_table("composition")

        reader = SectionReader("[demo]", {"composition": {"a": "bad"}})
        with pytest.raises(TypeError, match=r"\.a must be a number"):
            reader.str_float_table("composition")

    def test_number_pairs_rejects_non_numeric_values(self) -> None:
        reader = SectionReader("[demo]", {"tiers": [[0.1, "bad"]]})

        with pytest.raises(TypeError, match="values must be numbers"):
            reader.number_pairs("tiers")

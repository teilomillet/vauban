# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for flywheel payload library."""

from pathlib import Path

from vauban.flywheel._payloads import (
    BUILTIN_PAYLOADS,
    extend_library,
    load_payload_library,
    save_payload_library,
)
from vauban.types import Payload


class TestPayloadLibrary:
    def test_builtins_nonempty(self) -> None:
        assert len(BUILTIN_PAYLOADS) >= 5

    def test_load_none_returns_builtins(self) -> None:
        payloads = load_payload_library(None)
        assert len(payloads) == len(BUILTIN_PAYLOADS)

    def test_load_missing_file_returns_builtins(self) -> None:
        payloads = load_payload_library(Path("/nonexistent/path"))
        assert len(payloads) == len(BUILTIN_PAYLOADS)

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        original = [
            Payload(text="test1", source="lib", cycle_discovered=0),
            Payload(
                text="test2", source="gcg",
                cycle_discovered=1, domain="email",
            ),
        ]
        path = tmp_path / "payloads.jsonl"
        save_payload_library(original, path)
        loaded = load_payload_library(path)

        assert len(loaded) == 2
        assert loaded[0].text == "test1"
        assert loaded[0].source == "lib"
        assert loaded[1].text == "test2"
        assert loaded[1].domain == "email"

    def test_extend_deduplicates(self) -> None:
        existing = [
            Payload(text="a", source="lib", cycle_discovered=0),
            Payload(text="b", source="lib", cycle_discovered=0),
        ]
        extended = extend_library(
            existing, ["b", "c", "d"], "new", 1, None,
        )
        assert len(extended) == 4
        texts = [p.text for p in extended]
        assert texts == ["a", "b", "c", "d"]

    def test_extend_preserves_original(self) -> None:
        existing = [
            Payload(text="a", source="lib", cycle_discovered=0),
        ]
        extended = extend_library(
            existing, ["b"], "new", 1, "email",
        )
        assert len(existing) == 1  # original not mutated
        assert len(extended) == 2
        assert extended[1].source == "new"
        assert extended[1].cycle_discovered == 1
        assert extended[1].domain == "email"

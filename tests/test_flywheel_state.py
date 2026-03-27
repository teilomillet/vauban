# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for flywheel state persistence."""

from pathlib import Path

from vauban.flywheel._state import load_state, save_state
from vauban.types import DefendedTrace, FlywheelDefenseParams, Payload


class TestStatePersistence:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        defense = FlywheelDefenseParams(
            cast_alpha=2.5,
            cast_threshold=0.1,
            sic_threshold=0.3,
            sic_iterations=4,
            sic_mode="direction",
            cast_layers=[3, 7],
        )
        payloads = [
            Payload(text="p1", source="lib", cycle_discovered=0),
            Payload(
                text="p2", source="gcg",
                cycle_discovered=1, domain="email",
            ),
        ]
        evasions = [
            DefendedTrace(
                world_index=0, payload_index=0,
                payload_text="p1", reward=0.9,
                target_called=True, turns_used=3,
                tool_calls_made=2, defense_blocked=False,
            ),
        ]

        save_state(path, defense, payloads, 5, evasions)
        result = load_state(path)

        assert result is not None
        loaded_defense, loaded_payloads, cycle, loaded_evasions = result
        assert loaded_defense.cast_alpha == 2.5
        assert loaded_defense.sic_mode == "direction"
        assert loaded_defense.cast_layers == [3, 7]
        assert cycle == 5
        assert len(loaded_payloads) == 2
        assert loaded_payloads[1].domain == "email"
        assert len(loaded_evasions) == 1
        assert loaded_evasions[0].reward == 0.9

    def test_load_missing_file_returns_none(self, tmp_path: Path) -> None:
        result = load_state(tmp_path / "nonexistent.json")
        assert result is None

    def test_empty_evasions(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        defense = FlywheelDefenseParams(
            cast_alpha=1.0,
            cast_threshold=0.0,
            sic_threshold=0.5,
            sic_iterations=3,
            sic_mode="direction",
        )
        save_state(path, defense, [], 0, [])
        result = load_state(path)

        assert result is not None
        _, loaded_payloads, cycle, loaded_evasions = result
        assert loaded_payloads == []
        assert loaded_evasions == []
        assert cycle == 0

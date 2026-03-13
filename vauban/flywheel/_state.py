"""Flywheel state persistence for resumability."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from vauban.types import (
    DefendedTrace,
    FlywheelDefenseParams,
    Payload,
)

if TYPE_CHECKING:
    from pathlib import Path


def save_state(
    path: Path,
    defense: FlywheelDefenseParams,
    payloads: list[Payload],
    cycle: int,
    previous_evasions: list[DefendedTrace],
) -> None:
    """Save flywheel state for resumption.

    Args:
        path: Path to write the state JSON file.
        defense: Current defense parameters.
        payloads: Current payload library.
        cycle: Last completed cycle number.
        previous_evasions: Evasions from the last cycle.
    """
    state: dict[str, object] = {
        "cycle": cycle,
        "defense": {
            "cast_alpha": defense.cast_alpha,
            "cast_threshold": defense.cast_threshold,
            "sic_threshold": defense.sic_threshold,
            "sic_iterations": defense.sic_iterations,
            "sic_mode": defense.sic_mode,
            "cast_layers": defense.cast_layers,
        },
        "payloads": [
            {
                "text": p.text,
                "source": p.source,
                "cycle_discovered": p.cycle_discovered,
                "domain": p.domain,
            }
            for p in payloads
        ],
        "previous_evasions": [
            {
                "world_index": t.world_index,
                "payload_index": t.payload_index,
                "payload_text": t.payload_text,
                "reward": t.reward,
                "target_called": t.target_called,
                "turns_used": t.turns_used,
                "tool_calls_made": t.tool_calls_made,
                "defense_blocked": t.defense_blocked,
                "cast_refusal_rate": t.cast_refusal_rate,
                "sic_blocked": t.sic_blocked,
                "cast_interventions": t.cast_interventions,
            }
            for t in previous_evasions
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def load_state(
    path: Path,
) -> (
    tuple[
        FlywheelDefenseParams,
        list[Payload],
        int,
        list[DefendedTrace],
    ]
    | None
):
    """Load flywheel state from a JSON file.

    Returns None if the file does not exist.

    Returns:
        Tuple of (defense_params, payloads, cycle, previous_evasions)
        or None if state file not found.
    """
    if not path.exists():
        return None

    raw = json.loads(path.read_text())

    defense_raw = raw["defense"]
    defense = FlywheelDefenseParams(
        cast_alpha=defense_raw["cast_alpha"],
        cast_threshold=defense_raw["cast_threshold"],
        sic_threshold=defense_raw["sic_threshold"],
        sic_iterations=defense_raw["sic_iterations"],
        sic_mode=defense_raw["sic_mode"],
        cast_layers=defense_raw.get("cast_layers"),
    )

    payloads = [
        Payload(
            text=p["text"],
            source=p["source"],
            cycle_discovered=p["cycle_discovered"],
            domain=p.get("domain"),
        )
        for p in raw["payloads"]
    ]

    cycle: int = raw["cycle"]

    evasions = [
        DefendedTrace(
            world_index=t["world_index"],
            payload_index=t["payload_index"],
            payload_text=t["payload_text"],
            reward=t["reward"],
            target_called=t["target_called"],
            turns_used=t["turns_used"],
            tool_calls_made=t["tool_calls_made"],
            defense_blocked=t.get("defense_blocked", False),
            cast_refusal_rate=t.get("cast_refusal_rate", 0.0),
            sic_blocked=t.get("sic_blocked", False),
            cast_interventions=t.get("cast_interventions", 0),
        )
        for t in raw.get("previous_evasions", [])
    ]

    return defense, payloads, cycle, evasions

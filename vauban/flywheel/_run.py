"""Flywheel orchestrator: the main cycle loop."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from vauban.flywheel._attack import warmstart_gcg_payloads
from vauban.flywheel._convergence import check_convergence
from vauban.flywheel._defend import defend_traces, triage
from vauban.flywheel._defended_loop import run_defended_agent_loop
from vauban.flywheel._execute import execute_attack_matrix
from vauban.flywheel._harden import harden_defense
from vauban.flywheel._payloads import extend_library, load_payload_library
from vauban.flywheel._state import save_state
from vauban.flywheel._utility import measure_utility
from vauban.flywheel._worldgen import generate_worlds
from vauban.types import (
    DefendedTrace,
    FlywheelCycleMetrics,
    FlywheelDefenseParams,
    FlywheelResult,
    FlywheelTrace,
)

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import (
        CausalLM,
        EnvironmentConfig,
        FlywheelConfig,
        Payload,
        Tokenizer,
    )


def run_flywheel(
    model: CausalLM,
    tokenizer: Tokenizer,
    config: FlywheelConfig,
    direction: Array | None,
    layer_index: int,
    output_dir: Path,
    verbose: bool = True,
    t0: float = 0.0,
) -> FlywheelResult:
    """Run the flywheel attack-defense co-evolution loop.

    Each cycle: generate worlds, attack, defend, triage, harden, repeat.
    Outputs are written incrementally to JSONL files.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        config: Flywheel configuration.
        direction: Refusal direction (optional).
        layer_index: Layer index for direction application.
        output_dir: Directory for output files.
        verbose: Whether to print progress.
        t0: Start time for elapsed reporting.

    Returns:
        FlywheelResult with all cycle metrics and final defense state.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize defense parameters
    defense = FlywheelDefenseParams(
        cast_alpha=config.cast_alpha,
        cast_threshold=config.cast_threshold,
        sic_threshold=config.sic_threshold,
        sic_iterations=config.sic_iterations,
        sic_mode=config.sic_mode,
        cast_layers=config.cast_layers,
    )

    # Load or initialize payload library
    lib_path = (
        Path(config.payload_library_path)
        if config.payload_library_path
        else None
    )
    payloads = load_payload_library(lib_path)

    all_metrics: list[FlywheelCycleMetrics] = []
    defense_history: list[FlywheelDefenseParams] = [defense]
    previous_evasions: list[DefendedTrace] = []
    previous_evasion_cases: list[tuple[EnvironmentConfig, Payload, DefendedTrace]] = []
    total_worlds = 0
    total_evasions = 0
    convergence_cycle: int | None = None

    for cycle in range(config.n_cycles):
        if verbose:
            elapsed = time.monotonic() - t0
            print(
                f"  [{elapsed:7.1f}s] Flywheel cycle"
                f" {cycle + 1}/{config.n_cycles}"
                f" (alpha={defense.cast_alpha:.2f},"
                f" sic_thresh={defense.sic_threshold:.3f})"
            )

        # 1. Generate worlds
        world_metas = generate_worlds(
            skeletons=config.skeletons,
            n_worlds=config.worlds_per_cycle,
            difficulty_range=config.difficulty_range,
            positions=config.positions,
            seed=(
                config.seed + cycle
                if config.seed is not None
                else None
            ),
            model=model if config.model_expand else None,
            tokenizer=(
                tokenizer if config.model_expand else None
            ),
            model_expand=config.model_expand,
            expand_temperature=config.expand_temperature,
            expand_max_tokens=config.expand_max_tokens,
        )
        worlds = [wm[0] for wm in world_metas]
        total_worlds += len(worlds)

        # 2. Optional GCG warm-start
        n_new_payloads = 0
        if config.warmstart_gcg:
            new_texts = warmstart_gcg_payloads(
                model, tokenizer, worlds,
                payloads, config, direction,
            )
            if new_texts:
                payloads = extend_library(
                    payloads, new_texts, "gcg", cycle, None,
                )
                n_new_payloads = len(new_texts)

        # Select payloads for this cycle — take the most recent so
        # GCG-discovered payloads (appended at the tail) are included.
        cycle_payloads = payloads[-config.payloads_per_world:]

        # 3. Execute attack matrix
        traces = execute_attack_matrix(
            model, tokenizer, worlds, cycle_payloads,
            config.max_turns, config.max_gen_tokens,
        )

        # 4. Defense evaluation
        defended = defend_traces(
            model, tokenizer, traces, worlds, cycle_payloads,
            direction, layer_index, defense,
        )
        blocked, evaded, _borderline = triage(defended)

        # 5. Measure utility (random sample for unbiased estimate)
        utility_seed = (
            config.seed * 1000 + cycle
            if config.seed is not None
            else None
        )
        utility_score = measure_utility(
            model, tokenizer, worlds, direction,
            layer_index, defense,
            n_samples=min(20, len(worlds)),
            seed=utility_seed,
        )

        # 6. Validate previous evasions against current defense
        n_previous_blocked = 0
        if config.validate_previous and previous_evasion_cases:
            n_previous_blocked = _count_now_blocked(
                model, tokenizer, previous_evasion_cases,
                direction, layer_index, defense,
            )

        # 7. Compute attack metrics (needed by hardening)
        n_attacks = len(traces)
        n_successful = sum(
            1 for t in traces if t.reward >= 0.5
        )

        # 8. Harden defense
        cycle_defense = defense
        if config.harden and evaded:
            defense = harden_defense(
                defense, evaded,
                config.adaptation_rate,
                utility_score,
                config.utility_floor,
                n_successful=n_successful,
            )
            defense_history.append(defense)
        attack_success_rate = (
            n_successful / n_attacks if n_attacks > 0 else 0.0
        )
        defense_block_rate = (
            len(blocked) / n_successful
            if n_successful > 0
            else 1.0
        )
        evasion_rate = (
            len(evaded) / n_successful
            if n_successful > 0
            else 0.0
        )
        total_evasions += len(evaded)

        cycle_metrics = FlywheelCycleMetrics(
            cycle=cycle,
            n_worlds=len(worlds),
            n_attacks=n_attacks,
            attack_success_rate=attack_success_rate,
            defense_block_rate=defense_block_rate,
            evasion_rate=evasion_rate,
            utility_score=utility_score,
            cast_alpha=cycle_defense.cast_alpha,
            sic_threshold=cycle_defense.sic_threshold,
            n_new_payloads=n_new_payloads,
            n_previous_blocked=n_previous_blocked,
        )
        all_metrics.append(cycle_metrics)

        # Track evasions for next cycle validation
        previous_evasions = evaded
        previous_evasion_cases = [
            (
                worlds[trace.world_index],
                cycle_payloads[trace.payload_index],
                trace,
            )
            for trace in evaded
        ]

        # 9. Write incremental outputs
        _append_jsonl(
            output_dir / "flywheel_traces.jsonl", traces, cycle,
        )
        _append_jsonl(
            output_dir / "flywheel_failures.jsonl", evaded, cycle,
        )
        _write_report(
            output_dir / "flywheel_report.json",
            all_metrics, defense_history,
        )
        save_state(
            output_dir / "flywheel_state.json",
            defense, payloads, cycle, previous_evasions,
        )

        if verbose:
            elapsed = time.monotonic() - t0
            print(
                f"  [{elapsed:7.1f}s]  "
                f" ASR={attack_success_rate:.1%}"
                f" blocked={defense_block_rate:.1%}"
                f" evaded={evasion_rate:.1%}"
                f" utility={utility_score:.1%}"
            )

        # 10. Check convergence
        if check_convergence(
            all_metrics,
            config.convergence_window,
            config.convergence_threshold,
        ):
            convergence_cycle = cycle
            if verbose:
                elapsed = time.monotonic() - t0
                print(
                    f"  [{elapsed:7.1f}s]  "
                    f" Converged at cycle {cycle + 1}"
                )
            break

    return FlywheelResult(
        cycles=all_metrics,
        defense_history=defense_history,
        final_defense=defense,
        converged=convergence_cycle is not None,
        convergence_cycle=convergence_cycle,
        total_worlds=total_worlds,
        total_evasions=total_evasions,
        total_payloads=len(payloads),
    )


def _count_now_blocked(
    model: CausalLM,
    tokenizer: Tokenizer,
    previous_evasion_cases: list[tuple[EnvironmentConfig, Payload, DefendedTrace]],
    direction: Array | None,
    layer_index: int,
    defense: FlywheelDefenseParams,
) -> int:
    """Count how many previous evasions are now blocked.

    Exceptions during re-evaluation are NOT counted as blocked —
    a crash is ambiguous (not the same as defense success).
    """
    count = 0
    for world, payload, _trace in previous_evasion_cases:
        try:
            defended_result = run_defended_agent_loop(
                model,
                tokenizer,
                world,
                payload.text,
                direction,
                layer_index,
                defense,
            )
        except Exception:  # re-evaluation crash — skip, don't count
            continue
        if defended_result.sic_blocked or defended_result.env_result.reward < 0.5:
            count += 1
    return count


def _append_jsonl(
    path: Path,
    items: list[FlywheelTrace] | list[DefendedTrace],
    cycle: int,
) -> None:
    """Append trace items to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for item in items:
            obj = asdict(item)
            obj["_cycle"] = cycle
            f.write(json.dumps(obj) + "\n")


def _write_report(
    path: Path,
    metrics: list[FlywheelCycleMetrics],
    defense_history: list[FlywheelDefenseParams],
) -> None:
    """Write the cumulative report JSON."""
    report: dict[str, object] = {
        "n_cycles": len(metrics),
        "cycles": [asdict(m) for m in metrics],
        "defense_history": [asdict(d) for d in defense_history],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))

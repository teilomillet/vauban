"""Shared mutable state for the main pipeline runner."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from vauban._array import Array
    from vauban.types import (
        CausalLM,
        DBDIResult,
        DiffResult,
        DirectionResult,
        PipelineConfig,
        SubspaceResult,
        SurfacePrompt,
        SurfaceResult,
        Tokenizer,
    )


@dataclass(slots=True)
class RunState:
    """Mutable state passed between pipeline phases."""

    config_path: str | Path
    config: PipelineConfig
    model: CausalLM
    tokenizer: Tokenizer
    t0: float
    verbose: bool
    harmful: list[str] | None = None
    harmless: list[str] | None = None
    refusal_phrases: list[str] | None = None
    direction_result: DirectionResult | None = None
    subspace_result: SubspaceResult | None = None
    dbdi_result: DBDIResult | None = None
    diff_result: DiffResult | None = None
    cosine_scores: list[float] = field(default_factory=list)
    measure_reports: list[str] = field(default_factory=list)
    transfer_reports: list[str] = field(default_factory=list)
    target_layers: list[int] = field(default_factory=list)
    flat_weights: dict[str, Array] = field(default_factory=dict)
    modified_weights: dict[str, Array] = field(default_factory=dict)
    modified_model: CausalLM | None = None
    surface_direction: Array | None = None
    surface_layer: int = 0
    surface_prompts: list[SurfacePrompt] | None = None
    surface_before: SurfaceResult | None = None
    eval_refusal_rate: float | None = None
    report_files: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def elapsed(self) -> float:
        """Return elapsed runtime in seconds."""
        return time.monotonic() - self.t0

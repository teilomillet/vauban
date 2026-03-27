# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Early-return mode dispatch registry."""

from __future__ import annotations

from collections.abc import Callable

from vauban._pipeline._context import EarlyModeContext
from vauban._pipeline._mode_ai_act import _run_ai_act_mode
from vauban._pipeline._mode_api_eval import _run_api_eval_mode
from vauban._pipeline._mode_awareness import _run_awareness_mode
from vauban._pipeline._mode_cast import _run_cast_mode
from vauban._pipeline._mode_circuit import _run_circuit_mode
from vauban._pipeline._mode_compose_optimize import _run_compose_optimize_mode
from vauban._pipeline._mode_defend import _run_defend_mode
from vauban._pipeline._mode_depth import _run_depth_mode
from vauban._pipeline._mode_features import _run_features_mode
from vauban._pipeline._mode_flywheel import _run_flywheel_mode
from vauban._pipeline._mode_fusion import _run_fusion_mode
from vauban._pipeline._mode_jailbreak import _run_jailbreak_mode
from vauban._pipeline._mode_linear_probe import _run_linear_probe_mode
from vauban._pipeline._mode_lora_analysis import _run_lora_analysis_mode
from vauban._pipeline._mode_lora_export import _run_lora_export_mode
from vauban._pipeline._mode_optimize import _run_optimize_mode
from vauban._pipeline._mode_probe import _run_probe_mode
from vauban._pipeline._mode_remote import _run_remote_mode
from vauban._pipeline._mode_repbend import _run_repbend_mode
from vauban._pipeline._mode_sic import _run_sic_mode
from vauban._pipeline._mode_softprompt import _run_softprompt_mode
from vauban._pipeline._mode_sss import _run_sss_mode
from vauban._pipeline._mode_steer import _run_steer_mode
from vauban._pipeline._mode_svf import _run_svf_mode
from vauban.config._mode_registry import (
    EarlyModePhase,
    active_early_mode_for_phase,
)

type EarlyModeRunner = Callable[[EarlyModeContext], None]

EARLY_MODE_RUNNERS: dict[str, EarlyModeRunner] = {
    "remote": _run_remote_mode,
    "api_eval": _run_api_eval_mode,
    "ai_act": _run_ai_act_mode,
    "depth": _run_depth_mode,
    "svf": _run_svf_mode,
    "features": _run_features_mode,
    "probe": _run_probe_mode,
    "steer": _run_steer_mode,
    "sss": _run_sss_mode,
    "awareness": _run_awareness_mode,
    "cast": _run_cast_mode,
    "sic": _run_sic_mode,
    "optimize": _run_optimize_mode,
    "compose_optimize": _run_compose_optimize_mode,
    "softprompt": _run_softprompt_mode,
    "jailbreak": _run_jailbreak_mode,
    "defend": _run_defend_mode,
    "circuit": _run_circuit_mode,
    "linear_probe": _run_linear_probe_mode,
    "fusion": _run_fusion_mode,
    "repbend": _run_repbend_mode,
    "lora_export": _run_lora_export_mode,
    "lora_analysis": _run_lora_analysis_mode,
    "flywheel": _run_flywheel_mode,
}


def dispatch_early_mode(
    phase: EarlyModePhase,
    context: EarlyModeContext,
) -> bool:
    """Run the active early-return mode for *phase* if one is enabled."""
    spec = active_early_mode_for_phase(context.config, phase)
    if spec is None:
        return False
    if spec.requires_direction and context.direction_result is None:
        return False
    runner = EARLY_MODE_RUNNERS[spec.mode]
    runner(context)
    return True

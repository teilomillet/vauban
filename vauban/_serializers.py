# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""JSON serialization helpers for pipeline result types."""

from dataclasses import asdict

from vauban.taxonomy import TaxonomyCoverage
from vauban.types import (
    AwarenessResult,
    CastResult,
    CircuitResult,
    DefenseStackResult,
    DepthDirectionResult,
    DepthResult,
    DetectResult,
    DiffResult,
    DirectionTransferResult,
    FeaturesResult,
    FlywheelResult,
    GuardResult,
    IntentCheckResult,
    OptimizeResult,
    PolicyDecision,
    ProbeResult,
    ScanResult,
    ScanSpan,
    SICPromptResult,
    SICResult,
    SoftPromptResult,
    SSSResult,
    SteerResult,
    SurfaceComparison,
    SurfaceGroupDelta,
    TransferEvalResult,
    TrialResult,
)


def _diff_result_to_dict(result: DiffResult) -> dict[str, object]:
    """Serialize a DiffResult to a JSON-compatible dict (skips mx.array)."""
    return {
        "singular_values": result.singular_values,
        "explained_variance": result.explained_variance,
        "best_layer": result.best_layer,
        "d_model": result.d_model,
        "source_model": result.source_model,
        "target_model": result.target_model,
        "per_layer_singular_values": result.per_layer_singular_values,
    }


def _cast_to_dict(result: CastResult) -> dict[str, object]:
    """Serialize a CastResult to a JSON-compatible dict."""
    return {
        "prompt": result.prompt,
        "text": result.text,
        "projections_before": result.projections_before,
        "projections_after": result.projections_after,
        "interventions": result.interventions,
        "considered": result.considered,
        "displacement_interventions": result.displacement_interventions,
        "max_displacement": result.max_displacement,
    }


def _guard_to_dict(result: GuardResult) -> dict[str, object]:
    """Serialize a GuardResult to a JSON-compatible dict."""
    return {
        "prompt": result.prompt,
        "text": result.text,
        "events": [
            {
                "token_index": e.token_index,
                "token_id": e.token_id,
                "token_str": e.token_str,
                "projection": e.projection,
                "zone": e.zone,
                "action": e.action,
                "alpha_applied": e.alpha_applied,
                "rewind_count": e.rewind_count,
                "checkpoint_offset": e.checkpoint_offset,
            }
            for e in result.events
        ],
        "total_rewinds": result.total_rewinds,
        "circuit_broken": result.circuit_broken,
        "tokens_generated": result.tokens_generated,
        "tokens_rewound": result.tokens_rewound,
        "final_zone_counts": result.final_zone_counts,
    }


def _depth_to_dict(result: DepthResult) -> dict[str, object]:
    """Serialize a DepthResult to a JSON-compatible dict."""
    return {
        "prompt": result.prompt,
        "deep_thinking_ratio": result.deep_thinking_ratio,
        "deep_thinking_count": result.deep_thinking_count,
        "mean_settling_depth": result.mean_settling_depth,
        "layer_count": result.layer_count,
        "settling_threshold": result.settling_threshold,
        "deep_fraction": result.deep_fraction,
        "tokens": [
            {
                "token_id": t.token_id,
                "token_str": t.token_str,
                "settling_depth": t.settling_depth,
                "is_deep_thinking": t.is_deep_thinking,
                "jsd_profile": t.jsd_profile,
            }
            for t in result.tokens
        ],
    }


def _depth_direction_to_dict(
    result: DepthDirectionResult,
) -> dict[str, object]:
    """Serialize a DepthDirectionResult to a JSON-compatible dict."""
    return {
        "layer_index": result.layer_index,
        "cosine_scores": result.cosine_scores,
        "d_model": result.d_model,
        "refusal_cosine": result.refusal_cosine,
        "deep_prompts": result.deep_prompts,
        "shallow_prompts": result.shallow_prompts,
        "median_dtr": result.median_dtr,
    }


def _taxonomy_coverage_to_dict(
    coverage: TaxonomyCoverage,
) -> dict[str, object]:
    """Serialize a TaxonomyCoverage to a JSON-compatible dict."""
    return {
        "present": sorted(coverage.present),
        "missing": sorted(coverage.missing),
        "aliased": dict(sorted(coverage.aliased.items())),
        "coverage_ratio": coverage.coverage_ratio,
    }


def _surface_comparison_to_dict(
    comparison: SurfaceComparison,
) -> dict[str, object]:
    """Serialize a SurfaceComparison to a JSON-compatible dict."""
    def _group_delta_to_dict(d: SurfaceGroupDelta) -> dict[str, object]:
        return {
            "name": d.name,
            "count": d.count,
            "refusal_rate_before": d.refusal_rate_before,
            "refusal_rate_after": d.refusal_rate_after,
            "refusal_rate_delta": d.refusal_rate_delta,
            "mean_projection_before": d.mean_projection_before,
            "mean_projection_after": d.mean_projection_after,
            "mean_projection_delta": d.mean_projection_delta,
        }

    result: dict[str, object] = {
        "summary": {
            "refusal_rate_before": comparison.refusal_rate_before,
            "refusal_rate_after": comparison.refusal_rate_after,
            "refusal_rate_delta": comparison.refusal_rate_delta,
            "threshold_before": comparison.threshold_before,
            "threshold_after": comparison.threshold_after,
            "threshold_delta": comparison.threshold_delta,
            "coverage_score_before": comparison.coverage_score_before,
            "coverage_score_after": comparison.coverage_score_after,
            "coverage_score_delta": comparison.coverage_score_delta,
            "worst_cell_refusal_rate_before": (
                comparison.worst_cell_refusal_rate_before
            ),
            "worst_cell_refusal_rate_after": (
                comparison.worst_cell_refusal_rate_after
            ),
            "worst_cell_refusal_rate_delta": (
                comparison.worst_cell_refusal_rate_delta
            ),
            "total_scanned": comparison.before.total_scanned,
        },
        "category_deltas": [
            _group_delta_to_dict(d) for d in comparison.category_deltas
        ],
        "label_deltas": [
            _group_delta_to_dict(d) for d in comparison.label_deltas
        ],
        "style_deltas": [
            _group_delta_to_dict(d) for d in comparison.style_deltas
        ],
        "language_deltas": [
            _group_delta_to_dict(d) for d in comparison.language_deltas
        ],
        "turn_depth_deltas": [
            _group_delta_to_dict(d) for d in comparison.turn_depth_deltas
        ],
        "framing_deltas": [
            _group_delta_to_dict(d) for d in comparison.framing_deltas
        ],
        "cell_deltas": [
            _group_delta_to_dict(d) for d in comparison.cell_deltas
        ],
    }

    # Include taxonomy coverage from the before-cut surface if available
    tc = comparison.before.taxonomy_coverage
    if tc is not None:
        result["taxonomy_coverage"] = _taxonomy_coverage_to_dict(tc)

    return result


def _detect_to_dict(result: DetectResult) -> dict[str, object]:
    """Serialize a DetectResult to a JSON-compatible dict."""
    return {
        "hardened": result.hardened,
        "confidence": result.confidence,
        "effective_rank": result.effective_rank,
        "cosine_concentration": result.cosine_concentration,
        "silhouette_peak": result.silhouette_peak,
        "hdd_red_distance": result.hdd_red_distance,
        "residual_refusal_rate": result.residual_refusal_rate,
        "mean_refusal_position": result.mean_refusal_position,
        "evidence": result.evidence,
    }


def _trial_to_dict(t: TrialResult) -> dict[str, object]:
    """Serialize a TrialResult to a JSON-compatible dict."""
    return {
        "trial_number": t.trial_number,
        "alpha": t.alpha,
        "sparsity": t.sparsity,
        "norm_preserve": t.norm_preserve,
        "layer_strategy": t.layer_strategy,
        "layer_top_k": t.layer_top_k,
        "target_layers": t.target_layers,
        "refusal_rate": t.refusal_rate,
        "perplexity_delta": t.perplexity_delta,
        "kl_divergence": t.kl_divergence,
    }


def _optimize_to_dict(result: OptimizeResult) -> dict[str, object]:
    """Serialize an OptimizeResult to a JSON-compatible dict."""
    return {
        "n_trials": result.n_trials,
        "baseline_refusal_rate": result.baseline_refusal_rate,
        "baseline_perplexity": result.baseline_perplexity,
        "best_refusal": (
            _trial_to_dict(result.best_refusal)
            if result.best_refusal is not None
            else None
        ),
        "best_balanced": (
            _trial_to_dict(result.best_balanced)
            if result.best_balanced is not None
            else None
        ),
        "pareto_trials": [_trial_to_dict(t) for t in result.pareto_trials],
        "all_trials": [_trial_to_dict(t) for t in result.all_trials],
    }


def _sic_to_dict(result: SICResult) -> dict[str, object]:
    """Serialize a SICResult to a JSON-compatible dict."""
    return {
        "prompts_clean": result.prompts_clean,
        "prompts_blocked": result.prompts_blocked,
        "iterations_used": result.iterations_used,
        "initial_scores": result.initial_scores,
        "final_scores": result.final_scores,
        "total_blocked": result.total_blocked,
        "total_sanitized": result.total_sanitized,
        "total_clean": result.total_clean,
        "calibrated_threshold": result.calibrated_threshold,
    }


def _transfer_eval_to_dict(
    result: TransferEvalResult,
) -> dict[str, object]:
    """Serialize a TransferEvalResult to a JSON-compatible dict."""
    return {
        "model_id": result.model_id,
        "success_rate": result.success_rate,
        "eval_responses": result.eval_responses,
    }


def _softprompt_to_dict(result: SoftPromptResult) -> dict[str, object]:
    """Serialize a SoftPromptResult to a JSON-compatible dict."""
    return {
        "mode": result.mode,
        "success_rate": result.success_rate,
        "final_loss": result.final_loss,
        "loss_history": result.loss_history,
        "n_steps": result.n_steps,
        "n_tokens": result.n_tokens,
        "token_ids": result.token_ids,
        "token_text": result.token_text,
        "eval_responses": result.eval_responses,
        "accessibility_score": result.accessibility_score,
        "per_prompt_losses": result.per_prompt_losses,
        "early_stopped": result.early_stopped,
        "transfer_results": [
            _transfer_eval_to_dict(t) for t in result.transfer_results
        ],
        "defense_eval": (
            result.defense_eval.to_dict()
            if result.defense_eval is not None
            else None
        ),
        "gan_history": [r.to_dict() for r in result.gan_history],
    }


def _direction_transfer_to_dict(
    result: DirectionTransferResult,
) -> dict[str, object]:
    """Serialize a DirectionTransferResult to a JSON-compatible dict."""
    return {
        "model_id": result.model_id,
        "cosine_separation": result.cosine_separation,
        "best_native_separation": result.best_native_separation,
        "transfer_efficiency": result.transfer_efficiency,
        "per_layer_cosines": result.per_layer_cosines,
    }


def _probe_to_dict(result: ProbeResult) -> dict[str, object]:
    """Serialize a ProbeResult to a JSON-compatible dict."""
    return {
        "prompt": result.prompt,
        "layer_count": result.layer_count,
        "projections": result.projections,
    }


def _steer_to_dict(result: SteerResult) -> dict[str, object]:
    """Serialize a SteerResult to a JSON-compatible dict."""
    return {
        "text": result.text,
        "projections_before": result.projections_before,
        "projections_after": result.projections_after,
    }


def _sss_to_dict(result: SSSResult) -> dict[str, object]:
    """Serialize an SSSResult to a JSON-compatible dict."""
    return {
        "text": result.text,
        "prompt": result.prompt,
        "seed_layers": result.seed_layers,
        "seed_strength": result.seed_strength,
        "per_token_gains": result.per_token_gains,
        "projections_before": result.projections_before,
        "projections_after": result.projections_after,
    }


def _awareness_result_to_dict(result: AwarenessResult) -> dict[str, object]:
    """Serialize an AwarenessResult to a JSON-compatible dict."""
    return {
        "prompt": result.prompt,
        "steered": result.steered,
        "confidence": result.confidence,
        "anomalous_layers": result.anomalous_layers,
        "layers": [
            {
                "layer_index": lr.layer_index,
                "baseline_gain": lr.baseline_gain,
                "test_gain": lr.test_gain,
                "gain_ratio": lr.gain_ratio,
                "baseline_rank": lr.baseline_rank,
                "test_rank": lr.test_rank,
                "rank_ratio": lr.rank_ratio,
                "baseline_correlation": lr.baseline_correlation,
                "test_correlation": lr.test_correlation,
                "correlation_delta": lr.correlation_delta,
                "anomalous": lr.anomalous,
            }
            for lr in result.layers
        ],
        "evidence": result.evidence,
    }


def _scan_span_to_dict(span: ScanSpan) -> dict[str, object]:
    """Serialize a ScanSpan to a JSON-compatible dict."""
    return {
        "start": span.start,
        "end": span.end,
        "text": span.text,
        "mean_projection": span.mean_projection,
    }


def _scan_result_to_dict(result: ScanResult) -> dict[str, object]:
    """Serialize a ScanResult to a JSON-compatible dict."""
    return {
        "injection_probability": result.injection_probability,
        "overall_projection": result.overall_projection,
        "spans": [_scan_span_to_dict(s) for s in result.spans],
        "per_token_projections": result.per_token_projections,
        "flagged": result.flagged,
    }


def _sic_prompt_result_to_dict(
    result: SICPromptResult,
) -> dict[str, object]:
    """Serialize a SICPromptResult to a JSON-compatible dict."""
    return {
        "clean_prompt": result.clean_prompt,
        "blocked": result.blocked,
        "iterations": result.iterations,
        "initial_score": result.initial_score,
        "final_score": result.final_score,
    }


def _policy_decision_to_dict(
    result: PolicyDecision,
) -> dict[str, object]:
    """Serialize a PolicyDecision to a JSON-compatible dict."""
    return {
        "action": result.action,
        "matched_rules": result.matched_rules,
        "reasons": result.reasons,
    }


def _intent_check_to_dict(
    result: IntentCheckResult,
) -> dict[str, object]:
    """Serialize an IntentCheckResult to a JSON-compatible dict."""
    return {
        "aligned": result.aligned,
        "score": result.score,
        "mode": result.mode,
    }


def _defend_to_dict(result: DefenseStackResult) -> dict[str, object]:
    """Serialize a DefenseStackResult to a JSON-compatible dict."""
    return {
        "blocked": result.blocked,
        "layer_that_blocked": result.layer_that_blocked,
        "scan_result": (
            _scan_result_to_dict(result.scan_result)
            if result.scan_result is not None
            else None
        ),
        "sic_result": (
            _sic_prompt_result_to_dict(result.sic_result)
            if result.sic_result is not None
            else None
        ),
        "policy_decision": (
            _policy_decision_to_dict(result.policy_decision)
            if result.policy_decision is not None
            else None
        ),
        "intent_check": (
            _intent_check_to_dict(result.intent_check)
            if result.intent_check is not None
            else None
        ),
        "reasons": result.reasons,
    }


def _circuit_to_dict(result: CircuitResult) -> dict[str, object]:
    """Serialize a CircuitResult to a JSON-compatible dict."""
    return result.to_dict()


def _features_to_dict(result: FeaturesResult) -> dict[str, object]:
    """Serialize a FeaturesResult to a JSON-compatible dict."""
    return result.to_dict()


def _flywheel_to_dict(result: FlywheelResult) -> dict[str, object]:
    """Serialize a FlywheelResult to a JSON-compatible dict."""
    return {
        "n_cycles": len(result.cycles),
        "converged": result.converged,
        "convergence_cycle": result.convergence_cycle,
        "total_worlds": result.total_worlds,
        "total_evasions": result.total_evasions,
        "total_payloads": result.total_payloads,
        "cycles": [asdict(m) for m in result.cycles],
        "defense_history": [asdict(d) for d in result.defense_history],
        "final_defense": asdict(result.final_defense),
        "objective": (
            asdict(result.objective)
            if result.objective is not None
            else None
        ),
        "objective_assessment": (
            asdict(result.objective_assessment)
            if result.objective_assessment is not None
            else None
        ),
    }

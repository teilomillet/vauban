"""JSON serialization helpers for pipeline result types."""

from vauban.types import (
    DetectResult,
    OptimizeResult,
    ProbeResult,
    SICResult,
    SoftPromptResult,
    SteerResult,
    SurfaceComparison,
    SurfaceGroupDelta,
    TransferEvalResult,
    TrialResult,
)


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

    return {
        "summary": {
            "refusal_rate_before": comparison.refusal_rate_before,
            "refusal_rate_after": comparison.refusal_rate_after,
            "refusal_rate_delta": comparison.refusal_rate_delta,
            "threshold_before": comparison.threshold_before,
            "threshold_after": comparison.threshold_after,
            "threshold_delta": comparison.threshold_delta,
            "total_scanned": comparison.before.total_scanned,
        },
        "category_deltas": [
            _group_delta_to_dict(d) for d in comparison.category_deltas
        ],
        "label_deltas": [
            _group_delta_to_dict(d) for d in comparison.label_deltas
        ],
    }


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

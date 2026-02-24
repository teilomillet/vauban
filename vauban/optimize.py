"""Multi-objective optimization of cut hyperparameters using Optuna.

Searches over alpha, sparsity, norm_preserve, layer strategy, and layer_top_k.
Returns a Pareto front of (refusal_rate, perplexity_delta, kl_divergence).

Optuna is an **optional dependency** — lazily imported so the rest of
vauban works without it installed.
"""

import logging
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from vauban.cut import cut, sparsify_direction
from vauban.evaluate import (
    DEFAULT_REFUSAL_PHRASES,
    _extract_logits,
    _perplexity,
    _refusal_rate,
)
from vauban.measure import select_target_layers
from vauban.types import (
    CausalLM,
    DirectionResult,
    OptimizeConfig,
    OptimizeResult,
    Tokenizer,
    TrialResult,
)


def optimize(
    model: CausalLM,
    tokenizer: Tokenizer,
    direction_result: DirectionResult,
    eval_prompts: list[str],
    config: OptimizeConfig,
) -> OptimizeResult:
    """Run multi-objective optimization over cut parameters.

    Takes a pre-measured direction and eval prompts. Creates an Optuna
    study with TPE sampler searching over alpha, sparsity, norm_preserve,
    layer strategy, and layer_top_k. Each trial applies ``cut()`` to a
    copy of the weights, measures refusal_rate, perplexity_delta, and
    KL divergence, then restores the originals.

    Args:
        model: The causal language model (unmodified).
        tokenizer: Tokenizer with chat template support.
        direction_result: Pre-measured refusal direction from ``measure()``.
        eval_prompts: Prompts for evaluation.
        config: Optimization configuration.

    Returns:
        OptimizeResult with all trials, Pareto front, and convenience picks.

    Raises:
        ImportError: If optuna is not installed.
    """
    try:
        import optuna
    except ImportError:
        msg = (
            "optuna is required for optimization. "
            "Install it with: pip install vauban[optimize]"
        )
        raise ImportError(msg) from None

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)

    num_layers = len(model.model.layers)
    cosine_scores = direction_result.cosine_scores

    # Precompute baselines (once)
    model_module: nn.Module = model  # type: ignore[assignment]
    original_weights: dict[str, mx.array] = {
        k: v
        for k, v in tree_flatten(model_module.parameters())
        if isinstance(v, mx.array)
    }

    baseline_refusal = _refusal_rate(
        model, tokenizer, eval_prompts,  # type: ignore[arg-type]
        DEFAULT_REFUSAL_PHRASES, config.max_tokens,
    )
    baseline_perplexity = _perplexity(
        model, tokenizer, eval_prompts,  # type: ignore[arg-type]
    )
    original_logits = _precompute_logits(
        model, tokenizer, eval_prompts,  # type: ignore[arg-type]
    )

    # Resolve layer_top_k_max
    layer_top_k_max = config.layer_top_k_max or num_layers

    all_trials: list[TrialResult] = []

    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        alpha = trial.suggest_float("alpha", config.alpha_min, config.alpha_max)
        sparsity = trial.suggest_float(
            "sparsity", config.sparsity_min, config.sparsity_max,
        )

        if config.search_norm_preserve:
            norm_preserve = trial.suggest_categorical(
                "norm_preserve", [True, False],
            )
        else:
            norm_preserve = False

        strategy = trial.suggest_categorical(
            "layer_strategy", config.search_strategies,
        )

        layer_top_k: int | None = None
        if strategy == "top_k":
            layer_top_k = trial.suggest_int(
                "layer_top_k", config.layer_top_k_min, layer_top_k_max,
            )

        # Determine target layers
        if strategy == "all":
            target_layers = list(range(num_layers))
        elif cosine_scores:
            top_k_val = layer_top_k if layer_top_k is not None else 10
            target_layers = select_target_layers(
                cosine_scores, strategy, top_k_val,
            )
        else:
            target_layers = list(range(num_layers))

        # Optionally sparsify direction
        direction = direction_result.direction
        if sparsity > 0.0:
            direction = sparsify_direction(direction, sparsity)

        # Apply cut to original weights
        modified_weights = cut(
            original_weights,
            direction,
            target_layers,
            alpha,
            bool(norm_preserve),
        )

        # Load modified weights
        model_module.load_weights(list(modified_weights.items()))

        # Measure objectives
        refusal = _refusal_rate(
            model, tokenizer, eval_prompts,  # type: ignore[arg-type]
            DEFAULT_REFUSAL_PHRASES, config.max_tokens,
        )
        perplexity = _perplexity(
            model, tokenizer, eval_prompts,  # type: ignore[arg-type]
        )
        kl = _kl_from_precomputed(
            model, tokenizer, eval_prompts, original_logits,  # type: ignore[arg-type]
        )

        # Restore original weights
        restore_items = [
            (k, original_weights[k]) for k in modified_weights
        ]
        model_module.load_weights(restore_items)

        perplexity_delta = perplexity - baseline_perplexity

        result = TrialResult(
            trial_number=trial.number,
            alpha=alpha,
            sparsity=sparsity,
            norm_preserve=bool(norm_preserve),
            layer_strategy=str(strategy),
            layer_top_k=layer_top_k,
            target_layers=target_layers,
            refusal_rate=refusal,
            perplexity_delta=perplexity_delta,
            kl_divergence=kl,
        )
        all_trials.append(result)

        return refusal, perplexity_delta, kl

    # Create study
    sampler = optuna.samplers.TPESampler(seed=config.seed)
    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout,
    )

    # Extract Pareto front
    pareto_numbers = {t.number for t in study.best_trials}
    pareto_trials = [t for t in all_trials if t.trial_number in pareto_numbers]

    # Convenience picks
    best_refusal = min(all_trials, key=lambda t: t.refusal_rate) if all_trials else None
    best_balanced = _pick_balanced(
        all_trials, baseline_refusal, baseline_perplexity,
    )

    return OptimizeResult(
        all_trials=all_trials,
        pareto_trials=pareto_trials,
        baseline_refusal_rate=baseline_refusal,
        baseline_perplexity=baseline_perplexity,
        n_trials=len(all_trials),
        best_refusal=best_refusal,
        best_balanced=best_balanced,
    )


def _precompute_logits(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompts: list[str],
) -> list[mx.array]:
    """Run each prompt through the model once and store logits.

    Used as a KL baseline so we don't re-run the original model
    per trial.
    """
    results: list[mx.array] = []
    for prompt in prompts:
        token_ids = mx.array(tokenizer.encode(prompt))[None, :]
        logits = _extract_logits(model(token_ids))
        mx.eval(logits)
        results.append(logits)
    return results


def _kl_from_precomputed(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompts: list[str],
    original_logits: list[mx.array],
) -> float:
    """Compute KL divergence using precomputed original logits.

    Avoids running the original model again per trial. Computes
    KL(P_original || Q_modified) token-averaged.
    """
    if not prompts:
        return 0.0

    total_kl = 0.0
    total_tokens = 0

    for prompt, orig_logits in zip(prompts, original_logits, strict=True):
        token_ids = mx.array(tokenizer.encode(prompt))[None, :]
        mod_logits = _extract_logits(model(token_ids))

        p = mx.softmax(orig_logits, axis=-1)
        q = mx.softmax(mod_logits, axis=-1)

        kl = p * (mx.log(p + 1e-10) - mx.log(q + 1e-10))
        kl_per_token = mx.sum(kl, axis=-1)
        mean_kl = mx.mean(kl_per_token)
        mx.eval(mean_kl)
        total_kl += float(mean_kl.item())
        total_tokens += 1

    return total_kl / total_tokens if total_tokens > 0 else 0.0


def _pick_balanced(
    trials: list[TrialResult],
    baseline_refusal: float,
    baseline_perplexity: float,
) -> TrialResult | None:
    """Pick the trial with the best normalized sum of all 3 objectives.

    Min-max normalizes refusal_rate, perplexity_delta, and kl_divergence
    across all trials, then picks the trial with the lowest sum.
    Returns None if trials is empty.
    """
    if not trials:
        return None

    if len(trials) == 1:
        return trials[0]

    refusals = [t.refusal_rate for t in trials]
    ppl_deltas = [t.perplexity_delta for t in trials]
    kls = [t.kl_divergence for t in trials]

    def _normalize(values: list[float]) -> list[float]:
        lo = min(values)
        hi = max(values)
        span = hi - lo
        if span < 1e-10:
            return [0.0] * len(values)
        return [(v - lo) / span for v in values]

    norm_r = _normalize(refusals)
    norm_p = _normalize(ppl_deltas)
    norm_k = _normalize(kls)

    best_idx = 0
    best_score = math.inf
    for i in range(len(trials)):
        score = norm_r[i] + norm_p[i] + norm_k[i]
        if score < best_score:
            best_score = score
            best_idx = i

    return trials[best_idx]

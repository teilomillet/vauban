"""AmpleGCG collection phase: run GCG internally and harvest successful suffixes.

Runs multiple GCG restarts and collects any suffix whose loss falls below
the configured threshold. These collected suffixes form the training data
for the generator MLP.

Reference: arxiv.org/abs/2404.07921
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban import _ops as ops
from vauban._forward import force_eval
from vauban.softprompt._constraints import _build_vocab_mask
from vauban.softprompt._encoding import _pre_encode_prompts
from vauban.softprompt._gcg_candidates import (
    _allowed_indices_from_mask,
    _initialize_restart_tokens,
    _sample_greedy_candidates,
    _score_token_candidates,
    _select_step_prompts,
    _top_candidate_indices,
)
from vauban.softprompt._gcg_objective import (
    _build_gcg_shared_state,
    _compute_average_objective_loss,
    _evaluate_candidate_loss,
)
from vauban.softprompt._runtime import _encode_targets
from vauban.softprompt._search import _split_into_batches

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM, SoftPromptConfig, Tokenizer


def collect_gcg_suffixes(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SoftPromptConfig,
    direction: Array | None,
    ref_model: CausalLM | None = None,
) -> list[tuple[list[int], float]]:
    """Run GCG restarts and collect suffixes below the loss threshold.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with encode/decode support.
        prompts: Attack prompts to optimize against.
        config: SoftPromptConfig with amplecgc_* fields.
        direction: Optional refusal direction.
        ref_model: Optional reference model for KL collision loss.

    Returns:
        List of (token_ids, loss) tuples for suffixes below threshold.
    """
    transformer = model.model
    vocab_size = transformer.embed_tokens.weight.shape[0]
    embed_matrix = transformer.embed_tokens.weight

    target_ids = _encode_targets(
        tokenizer, config.target_prefixes, config.target_repeat_count,
    )
    force_eval(target_ids)

    effective_prompts = prompts if prompts else ["Hello"]
    all_prompt_ids = _pre_encode_prompts(
        tokenizer, effective_prompts, config.system_prompt,
    )

    objective_state = _build_gcg_shared_state(
        model, tokenizer, config, target_ids, direction,
        ref_model=ref_model,
    )

    vocab_mask = _build_vocab_mask(
        tokenizer, vocab_size, config.token_constraint,
    )
    allowed_indices = _allowed_indices_from_mask(vocab_mask, vocab_size)

    collected: list[tuple[list[int], float]] = []
    threshold = config.amplecgc_collect_threshold
    n_restarts = config.amplecgc_collect_restarts
    n_steps = config.amplecgc_collect_steps

    for restart_idx in range(n_restarts):
        current_ids = _initialize_restart_tokens(
            config, allowed_indices, restart_idx,
        )
        best_ids = list(current_ids)
        best_loss = float("inf")
        steps_without_improvement = 0

        for step in range(n_steps):
            token_array = ops.array(current_ids)[None, :]
            soft_embeds = transformer.embed_tokens(token_array)
            force_eval(soft_embeds)

            selected_ids = _select_step_prompts(
                config, objective_state, all_prompt_ids, soft_embeds, step,
            )
            batches = _split_into_batches(
                selected_ids, config.grad_accum_steps,
            )
            total_loss = 0.0
            accum_grad = ops.zeros_like(soft_embeds)

            for batch in batches:
                def loss_fn(
                    embeds: Array,
                    _sel: list[Array] = batch,
                    _tok_arr: Array = token_array,
                ) -> Array:
                    suffix_ids = (
                        _tok_arr
                        if objective_state.perplexity_weight > 0.0
                        else None
                    )
                    return _compute_average_objective_loss(
                        objective_state, embeds, _sel,
                        suffix_token_ids=suffix_ids,
                    )

                batch_loss, batch_grad = ops.value_and_grad(loss_fn)(
                    soft_embeds,
                )
                force_eval(batch_loss, batch_grad)
                total_loss += float(batch_loss.item())
                accum_grad = accum_grad + batch_grad

            current_loss = total_loss / len(batches)
            grad = accum_grad / len(batches)

            if current_loss < best_loss:
                best_loss = current_loss
                best_ids = list(current_ids)
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            # Collect if below threshold
            if current_loss < threshold:
                collected.append((list(current_ids), current_loss))

            # Early stopping per restart
            if (
                config.patience > 0
                and steps_without_improvement >= config.patience
            ):
                break

            # GCG token replacement step
            scores = _score_token_candidates(grad, embed_matrix, vocab_mask)
            force_eval(scores)
            top_indices, effective_k = _top_candidate_indices(
                scores, config.top_k, vocab_size,
            )

            candidates = _sample_greedy_candidates(
                current_ids, top_indices, config.batch_size,
                config.n_tokens, effective_k,
            )
            candidate_losses = [
                _evaluate_candidate_loss(
                    objective_state, candidate, selected_ids,
                )
                for candidate in candidates
            ]
            best_idx = candidate_losses.index(min(candidate_losses))
            if candidate_losses[best_idx] < current_loss:
                current_ids = candidates[best_idx]

        # Always collect the best from each restart
        if best_loss < threshold:
            collected.append((best_ids, best_loss))

    return collected

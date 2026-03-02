"""GCG orchestration over shared objective, candidate, and rerank helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban import _ops as ops
from vauban._forward import force_eval
from vauban.softprompt._constraints import _build_vocab_mask
from vauban.softprompt._encoding import _pre_encode_prompts
from vauban.softprompt._gcg_candidates import (
    _allowed_indices_from_mask,
    _initialize_beam,
    _initialize_restart_tokens,
    _sample_beam_candidates,
    _sample_greedy_candidates,
    _score_token_candidates,
    _select_step_prompts,
    _top_candidate_indices,
    _update_beam,
)
from vauban.softprompt._gcg_objective import (
    _build_gcg_shared_state,
    _compute_average_objective_loss,
    _evaluate_candidate_loss,
)
from vauban.softprompt._gcg_rerank import (
    GCGRerankContext,
    _apply_rollout_reranking,
    _apply_transfer_reranking,
)
from vauban.softprompt._generation import _evaluate_attack
from vauban.softprompt._runtime import (
    _compute_accessibility_score,
    _encode_targets,
    _prepare_transfer_data,
)
from vauban.softprompt._search import _compute_per_prompt_losses, _split_into_batches
from vauban.types import (
    CausalLM,
    EnvironmentConfig,
    SoftPromptConfig,
    SoftPromptResult,
    Tokenizer,
)

if TYPE_CHECKING:
    from vauban._array import Array


def _gcg_attack(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SoftPromptConfig,
    direction: Array | None,
    ref_model: CausalLM | None = None,
    all_prompt_ids_override: list[Array] | None = None,
    transfer_models: list[tuple[str, CausalLM, Tokenizer]] | None = None,
    infix_map: dict[int, int] | None = None,
    environment_config: EnvironmentConfig | None = None,
) -> SoftPromptResult:
    """Run GCG discrete token search for the configured soft prompt objective."""
    transformer = model.model
    vocab_size = transformer.embed_tokens.weight.shape[0]
    embed_matrix = transformer.embed_tokens.weight

    target_ids = _encode_targets(
        tokenizer,
        config.target_prefixes,
        config.target_repeat_count,
    )
    force_eval(target_ids)
    if all_prompt_ids_override is not None:
        all_prompt_ids = all_prompt_ids_override
    else:
        effective_prompts = prompts if prompts else ["Hello"]
        all_prompt_ids = _pre_encode_prompts(
            tokenizer,
            effective_prompts,
            config.system_prompt,
        )

    objective_state = _build_gcg_shared_state(
        model,
        tokenizer,
        config,
        target_ids,
        direction,
        ref_model=ref_model,
        infix_map=infix_map,
    )
    transfer_data = _prepare_transfer_data(transfer_models, config, prompts)
    rerank_context = GCGRerankContext(
        model=model,
        tokenizer=tokenizer,
        transfer_data=transfer_data,
        transfer_loss_weight=config.transfer_loss_weight,
        transfer_rerank_count=config.transfer_rerank_count,
        environment_config=environment_config,
    )

    vocab_mask = _build_vocab_mask(
        tokenizer,
        vocab_size,
        config.token_constraint,
    )
    allowed_indices = _allowed_indices_from_mask(vocab_mask, vocab_size)

    overall_best_ids: list[int] = []
    overall_best_loss = float("inf")
    all_loss_history: list[float] = []
    total_steps = 0
    early_stopped = False

    for restart_idx in range(config.n_restarts):
        current_ids = _initialize_restart_tokens(
            config,
            allowed_indices,
            restart_idx,
        )
        restart_best_ids = list(current_ids)
        restart_best_loss = float("inf")
        steps_without_improvement = 0
        beam, beam_losses = _initialize_beam(
            current_ids,
            config.beam_width,
            config.n_tokens,
            allowed_indices,
        )

        for step in range(config.n_steps):
            total_steps += 1

            token_array = ops.array(current_ids)[None, :]
            soft_embeds = transformer.embed_tokens(token_array)
            force_eval(soft_embeds)

            selected_ids = _select_step_prompts(
                config,
                objective_state,
                all_prompt_ids,
                soft_embeds,
                step,
            )
            batches = _split_into_batches(selected_ids, config.grad_accum_steps)
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
                        objective_state,
                        embeds,
                        _sel,
                        suffix_token_ids=suffix_ids,
                    )

                batch_loss, batch_grad = ops.value_and_grad(loss_fn)(soft_embeds)
                force_eval(batch_loss, batch_grad)
                total_loss += float(batch_loss.item())
                accum_grad = accum_grad + batch_grad

            current_loss = total_loss / len(batches)
            grad = accum_grad / len(batches)
            all_loss_history.append(current_loss)

            if current_loss < restart_best_loss:
                restart_best_loss = current_loss
                restart_best_ids = list(current_ids)
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            if (
                config.patience > 0
                and steps_without_improvement >= config.patience
            ):
                early_stopped = True
                break

            scores = _score_token_candidates(grad, embed_matrix, vocab_mask)
            force_eval(scores)
            top_indices, effective_k = _top_candidate_indices(
                scores,
                config.top_k,
                vocab_size,
            )

            if config.beam_width <= 1:
                candidates = _sample_greedy_candidates(
                    current_ids,
                    top_indices,
                    config.batch_size,
                    config.n_tokens,
                    effective_k,
                )
                candidate_losses = [
                    _evaluate_candidate_loss(objective_state, candidate, selected_ids)
                    for candidate in candidates
                ]
                _apply_transfer_reranking(
                    rerank_context,
                    candidates,
                    candidate_losses,
                )
                _apply_rollout_reranking(
                    rerank_context,
                    candidates,
                    candidate_losses,
                    step,
                )
                best_idx = candidate_losses.index(min(candidate_losses))
                if candidate_losses[best_idx] < current_loss:
                    current_ids = candidates[best_idx]
                continue

            candidates = _sample_beam_candidates(
                beam,
                top_indices,
                config.batch_size,
                config.beam_width,
                config.n_tokens,
                effective_k,
            )
            candidate_losses = [
                _evaluate_candidate_loss(objective_state, candidate, selected_ids)
                for candidate in candidates
            ]
            _apply_transfer_reranking(
                rerank_context,
                candidates,
                candidate_losses,
            )
            _apply_rollout_reranking(
                rerank_context,
                candidates,
                candidate_losses,
                step,
            )
            beam, beam_losses = _update_beam(
                beam,
                beam_losses,
                candidates,
                candidate_losses,
                config.beam_width,
            )
            current_ids = beam[0]
            if beam_losses[0] < restart_best_loss:
                restart_best_loss = beam_losses[0]
                restart_best_ids = list(beam[0])
                steps_without_improvement = 0

        if restart_best_loss < overall_best_loss:
            overall_best_loss = restart_best_loss
            overall_best_ids = restart_best_ids

    final_token_array = ops.array(overall_best_ids)[None, :]
    final_embeds = transformer.embed_tokens(final_token_array)
    force_eval(final_embeds)

    per_prompt_losses = _compute_per_prompt_losses(
        model,
        final_embeds,
        all_prompt_ids,
        target_ids,
        config.n_tokens,
        direction,
        config.direction_weight,
        config.direction_mode,
        objective_state.direction_layers,
        objective_state.eos_token_id,
        config.eos_loss_mode,
        config.eos_loss_weight,
        ref_model,
        config.kl_ref_weight,
        loss_mode=config.loss_mode,
        refusal_ids=objective_state.refusal_ids,
        defense_aware_weight=objective_state.defense.weight,
        sic_layer=objective_state.defense.sic_layer,
        sic_threshold=objective_state.defense.sic_threshold,
        cast_layers=objective_state.defense.cast_layers,
        cast_threshold=objective_state.defense.cast_threshold,
        perplexity_weight=objective_state.perplexity_weight,
        token_position=objective_state.token_position,
        suffix_token_ids=final_token_array,
    )
    final_loss = overall_best_loss
    accessibility_score = _compute_accessibility_score(final_loss)
    success_rate, responses = _evaluate_attack(
        model,
        tokenizer,
        prompts,
        final_embeds,
        config,
    )
    token_text = tokenizer.decode(overall_best_ids)
    return SoftPromptResult(
        mode="gcg",
        success_rate=success_rate,
        final_loss=final_loss,
        loss_history=all_loss_history,
        n_steps=total_steps,
        n_tokens=config.n_tokens,
        embeddings=None,
        token_ids=overall_best_ids,
        token_text=token_text,
        eval_responses=responses,
        accessibility_score=accessibility_score,
        per_prompt_losses=per_prompt_losses,
        early_stopped=early_stopped,
    )

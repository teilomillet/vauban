"""GCG (Greedy Coordinate Gradient) discrete token search."""

import random

from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import force_eval
from vauban.softprompt._generation import _evaluate_attack
from vauban.softprompt._loss import (
    _compute_defensive_loss,
    _compute_loss,
    _compute_untargeted_loss,
)
from vauban.softprompt._utils import (
    _build_vocab_mask,
    _compute_accessibility_score,
    _compute_per_prompt_losses,
    _encode_refusal_tokens,
    _encode_targets,
    _pre_encode_prompts,
    _prepare_transfer_data,
    _sample_prompt_ids,
    _score_transfer_loss,
    _select_prompt_ids,
    _select_worst_k_prompt_ids,
    _split_into_batches,
)
from vauban.types import CausalLM, SoftPromptConfig, SoftPromptResult, Tokenizer


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
) -> SoftPromptResult:
    """GCG (Greedy Coordinate Gradient) discrete token search.

    Finds adversarial token sequences by using gradient information
    to guide a discrete search over the vocabulary. Supports
    multi-prompt optimization, multiple restarts, and early stopping.
    """
    transformer = model.model
    vocab_size = transformer.embed_tokens.weight.shape[0]
    embed_matrix = transformer.embed_tokens.weight

    target_ids = _encode_targets(
        tokenizer, config.target_prefixes, config.target_repeat_count,
    )
    force_eval(target_ids)

    # Pre-encode all prompts (or use override with history baked in)
    if all_prompt_ids_override is not None:
        all_prompt_ids = all_prompt_ids_override
    else:
        effective_prompts = prompts if prompts else ["Hello"]
        all_prompt_ids = _pre_encode_prompts(
            tokenizer, effective_prompts, config.system_prompt,
        )

    # Pre-encode transfer model data for multi-model re-ranking
    transfer_data = _prepare_transfer_data(
        transfer_models, config, prompts,
    )

    # Pre-compute direction config
    direction_layers_set: set[int] | None = (
        set(config.direction_layers) if config.direction_layers is not None
        else None
    )
    refusal_ids: Array | None = None
    if config.loss_mode in ("untargeted", "defensive"):
        refusal_ids = _encode_refusal_tokens(tokenizer)
        force_eval(refusal_ids)

    # Pre-compute vocab mask and EOS token ID
    vocab_mask = _build_vocab_mask(
        tokenizer, vocab_size, config.token_constraint,
    )
    eos_token_id: int | None = getattr(tokenizer, "eos_token_id", None)

    # Pre-compute defense-aware loss config
    da_weight = config.defense_aware_weight
    da_sic_layer = config.defense_eval_layer
    da_sic_threshold = config.defense_eval_threshold
    da_cast_layers = config.defense_eval_cast_layers
    da_cast_threshold = config.defense_eval_threshold

    # Build list of allowed token indices for constrained random init
    if vocab_mask is not None:
        force_eval(vocab_mask)
        allowed_indices: list[int] = [
            i for i in range(vocab_size) if bool(vocab_mask[i].item())
        ]
    else:
        allowed_indices = list(range(vocab_size))

    overall_best_ids: list[int] = []
    overall_best_loss = float("inf")
    all_loss_history: list[float] = []
    total_steps = 0
    early_stopped = False
    beam_width = config.beam_width

    # Pre-compute perplexity and position config
    ppl_weight = config.perplexity_weight
    tok_pos = config.token_position

    # --- Evaluate a single candidate's average loss ---
    def _eval_cand(
        cand_ids: list[int],
        sel: list[Array],
    ) -> float:
        cand_array = ops.array(cand_ids)[None, :]
        cand_embeds = transformer.embed_tokens(cand_array)
        cand_total = ops.array(0.0)
        for pid in sel:
            _infix_split = (
                infix_map.get(id(pid)) if infix_map is not None else None
            )
            if (
                config.loss_mode == "defensive"
                and refusal_ids is not None
            ):
                cand_total = cand_total + _compute_defensive_loss(
                    model, cand_embeds, pid,
                    config.n_tokens, refusal_ids,
                    direction, config.direction_weight,
                    config.direction_mode, direction_layers_set,
                    eos_token_id, config.eos_loss_mode,
                    config.eos_loss_weight,
                    ref_model, config.kl_ref_weight,
                    da_weight, da_sic_layer, da_sic_threshold,
                    da_cast_layers, da_cast_threshold,
                    perplexity_weight=ppl_weight,
                    suffix_token_ids=cand_array,
                    token_position=tok_pos,
                    infix_split=_infix_split,
                )
            elif (
                config.loss_mode == "untargeted"
                and refusal_ids is not None
            ):
                cand_total = cand_total + _compute_untargeted_loss(
                    model, cand_embeds, pid,
                    config.n_tokens, refusal_ids,
                    direction, config.direction_weight,
                    config.direction_mode, direction_layers_set,
                    eos_token_id, config.eos_loss_mode,
                    config.eos_loss_weight,
                    ref_model, config.kl_ref_weight,
                    da_weight, da_sic_layer, da_sic_threshold,
                    da_cast_layers, da_cast_threshold,
                    perplexity_weight=ppl_weight,
                    suffix_token_ids=cand_array,
                    token_position=tok_pos,
                    infix_split=_infix_split,
                )
            else:
                cand_total = cand_total + _compute_loss(
                    model, cand_embeds, pid, target_ids,
                    config.n_tokens, direction,
                    config.direction_weight,
                    config.direction_mode, direction_layers_set,
                    eos_token_id, config.eos_loss_mode,
                    config.eos_loss_weight,
                    ref_model, config.kl_ref_weight,
                    da_weight, da_sic_layer, da_sic_threshold,
                    da_cast_layers, da_cast_threshold,
                    perplexity_weight=ppl_weight,
                    suffix_token_ids=cand_array,
                    token_position=tok_pos,
                    infix_split=_infix_split,
                )
        cand_avg = cand_total / len(sel)
        force_eval(cand_avg)
        return float(cand_avg.item())

    # --- Transfer re-ranking helpers ---
    # NOTE: Transfer scoring always evaluates against *all* pre-encoded
    # prompts (t_prompts), not the per-step selected_ids subset.  This is
    # intentional — transferability should be measured over the full
    # prompt set, not the primary model's worst-k or cycle selection.
    def _apply_transfer_reranking(
        candidates: list[list[int]],
        candidate_losses: list[float],
    ) -> None:
        """Re-rank top candidates using transfer model scores (in-place)."""
        if not transfer_data:
            return
        n_rerank = min(config.transfer_rerank_count, len(candidates))
        ranked = sorted(
            range(len(candidate_losses)),
            key=lambda i: candidate_losses[i],
        )[:n_rerank]
        for idx in ranked:
            t_loss = _score_transfer_loss(
                tokenizer.decode(candidates[idx]), transfer_data,
            )
            candidate_losses[idx] += (
                config.transfer_loss_weight * t_loss
            )

    for _restart in range(config.n_restarts):
        # Initialize: warm-start from init_tokens or random
        if config.init_tokens is not None and _restart == 0:
            # Pad or truncate to n_tokens
            init = list(config.init_tokens)
            if len(init) < config.n_tokens:
                init.extend(
                    random.choice(allowed_indices)
                    for _ in range(config.n_tokens - len(init))
                )
            current_ids = init[: config.n_tokens]
        else:
            current_ids = [
                random.choice(allowed_indices)
                for _ in range(config.n_tokens)
            ]
        restart_best_ids = list(current_ids)
        restart_best_loss = float("inf")
        steps_without_improvement = 0

        # Initialize beam for beam search (beam_width > 1)
        beam: list[list[int]] = [list(current_ids)]
        beam_losses_tracked: list[float] = [float("inf")]
        if beam_width > 1:
            for _ in range(beam_width - 1):
                member = [
                    random.choice(allowed_indices)
                    for _ in range(config.n_tokens)
                ]
                beam.append(member)
                beam_losses_tracked.append(float("inf"))

        for step in range(config.n_steps):
            total_steps += 1

            # Get embeddings for current tokens
            token_array = ops.array(current_ids)[None, :]
            soft_embeds = transformer.embed_tokens(token_array)
            force_eval(soft_embeds)

            # Select prompts for this step
            if config.prompt_strategy == "worst_k":
                selected_ids = _select_worst_k_prompt_ids(
                    model, soft_embeds, all_prompt_ids, target_ids,
                    config.n_tokens, config.worst_k,
                    direction, config.direction_weight,
                    config.direction_mode, direction_layers_set,
                    eos_token_id, config.eos_loss_mode,
                    config.eos_loss_weight,
                    ref_model, config.kl_ref_weight,
                    loss_mode=config.loss_mode, refusal_ids=refusal_ids,
                    defense_aware_weight=da_weight,
                    sic_layer=da_sic_layer, sic_threshold=da_sic_threshold,
                    cast_layers=da_cast_layers, cast_threshold=da_cast_threshold,
                )
            elif config.prompt_strategy == "sample":
                selected_ids = _sample_prompt_ids(
                    all_prompt_ids, config.worst_k,
                )
            else:
                selected_ids = _select_prompt_ids(
                    all_prompt_ids, step, config.prompt_strategy,
                )

            # Gradient accumulation over mini-batches
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
                    _suf_ids = _tok_arr if ppl_weight > 0.0 else None
                    total = ops.array(0.0)
                    for pid in _sel:
                        _isplit = (
                            infix_map.get(id(pid))
                            if infix_map is not None
                            else None
                        )
                        if (
                            config.loss_mode == "defensive"
                            and refusal_ids is not None
                        ):
                            total = total + _compute_defensive_loss(
                                model, embeds, pid,
                                config.n_tokens, refusal_ids,
                                direction, config.direction_weight,
                                config.direction_mode, direction_layers_set,
                                eos_token_id, config.eos_loss_mode,
                                config.eos_loss_weight,
                                ref_model, config.kl_ref_weight,
                                da_weight, da_sic_layer, da_sic_threshold,
                                da_cast_layers, da_cast_threshold,
                                perplexity_weight=ppl_weight,
                                suffix_token_ids=_suf_ids,
                                token_position=tok_pos,
                                infix_split=_isplit,
                            )
                        elif (
                            config.loss_mode == "untargeted"
                            and refusal_ids is not None
                        ):
                            total = total + _compute_untargeted_loss(
                                model, embeds, pid,
                                config.n_tokens, refusal_ids,
                                direction, config.direction_weight,
                                config.direction_mode, direction_layers_set,
                                eos_token_id, config.eos_loss_mode,
                                config.eos_loss_weight,
                                ref_model, config.kl_ref_weight,
                                da_weight, da_sic_layer, da_sic_threshold,
                                da_cast_layers, da_cast_threshold,
                                perplexity_weight=ppl_weight,
                                suffix_token_ids=_suf_ids,
                                token_position=tok_pos,
                                infix_split=_isplit,
                            )
                        else:
                            total = total + _compute_loss(
                                model, embeds, pid, target_ids,
                                config.n_tokens, direction,
                                config.direction_weight,
                                config.direction_mode, direction_layers_set,
                                eos_token_id, config.eos_loss_mode,
                                config.eos_loss_weight,
                                ref_model, config.kl_ref_weight,
                                da_weight, da_sic_layer, da_sic_threshold,
                                da_cast_layers, da_cast_threshold,
                                perplexity_weight=ppl_weight,
                                suffix_token_ids=_suf_ids,
                                token_position=tok_pos,
                                infix_split=_isplit,
                            )
                    return total / len(_sel)

                batch_loss, batch_grad = ops.value_and_grad(loss_fn)(
                    soft_embeds,
                )
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

            # Early stopping within restart
            if (
                config.patience > 0
                and steps_without_improvement >= config.patience
            ):
                early_stopped = True
                break

            # Score token candidates: -grad @ embed_matrix.T
            # grad shape: (1, n_tokens, d_model)
            scores = -ops.matmul(grad[0], embed_matrix.T)  # (n_tokens, vocab_size)
            # Apply vocab mask to exclude disallowed tokens
            if vocab_mask is not None:
                scores = ops.where(vocab_mask, scores, ops.array(-1e10))
            force_eval(scores)

            # Top-k per position
            effective_k = min(config.top_k, vocab_size)
            sorted_indices = ops.argsort(-scores, axis=-1)  # descending order
            top_indices = sorted_indices[:, :effective_k]  # (n_tokens, top_k)
            force_eval(top_indices)

            if beam_width <= 1:
                # Fast path: exact greedy behavior (no regression)
                candidates: list[list[int]] = []
                for _ in range(config.batch_size):
                    pos = random.randint(0, config.n_tokens - 1)
                    tok_idx = random.randint(0, effective_k - 1)
                    new_token = int(top_indices[pos, tok_idx].item())
                    candidate = list(current_ids)
                    candidate[pos] = new_token
                    candidates.append(candidate)

                candidate_losses = [
                    _eval_cand(c, selected_ids)
                    for c in candidates
                ]

                _apply_transfer_reranking(candidates, candidate_losses)

                best_candidate_idx = candidate_losses.index(
                    min(candidate_losses),
                )
                if candidate_losses[best_candidate_idx] < current_loss:
                    current_ids = candidates[best_candidate_idx]
            else:
                # Beam search: distribute candidates across beam members
                candidates_per_member = max(
                    1, config.batch_size // beam_width,
                )
                candidates = []
                for member in beam:
                    for _ in range(candidates_per_member):
                        pos = random.randint(0, config.n_tokens - 1)
                        tok_idx = random.randint(0, effective_k - 1)
                        candidate = list(member)
                        candidate[pos] = int(
                            top_indices[pos, tok_idx].item(),
                        )
                        candidates.append(candidate)

                candidate_losses = [
                    _eval_cand(c, selected_ids)
                    for c in candidates
                ]

                _apply_transfer_reranking(candidates, candidate_losses)

                # Pool candidates + current beam, deduplicate,
                # keep top beam_width
                pool: list[tuple[list[int], float]] = list(
                    zip(candidates, candidate_losses, strict=True),
                )
                for i, member in enumerate(beam):
                    pool.append((member, beam_losses_tracked[i]))
                seen: set[tuple[int, ...]] = set()
                unique: list[tuple[list[int], float]] = []
                for ids, loss in pool:
                    key = tuple(ids)
                    if key not in seen:
                        seen.add(key)
                        unique.append((ids, loss))
                unique.sort(key=lambda x: x[1])
                beam = [ids for ids, _ in unique[:beam_width]]
                beam_losses_tracked = [
                    loss for _, loss in unique[:beam_width]
                ]
                current_ids = beam[0]

                # Update restart best from beam selection
                if beam_losses_tracked[0] < restart_best_loss:
                    restart_best_loss = beam_losses_tracked[0]
                    restart_best_ids = list(beam[0])
                    steps_without_improvement = 0

        # Update overall best across restarts
        if restart_best_loss < overall_best_loss:
            overall_best_loss = restart_best_loss
            overall_best_ids = restart_best_ids

    # Evaluate with best tokens
    current_ids = overall_best_ids
    final_token_array = ops.array(current_ids)[None, :]
    final_embeds = transformer.embed_tokens(final_token_array)
    force_eval(final_embeds)

    # Compute per-prompt losses and accessibility score
    per_prompt_losses = _compute_per_prompt_losses(
        model, final_embeds, all_prompt_ids, target_ids,
        config.n_tokens, direction, config.direction_weight,
        config.direction_mode, direction_layers_set,
        eos_token_id, config.eos_loss_mode, config.eos_loss_weight,
        ref_model, config.kl_ref_weight,
        loss_mode=config.loss_mode, refusal_ids=refusal_ids,
        defense_aware_weight=da_weight,
        sic_layer=da_sic_layer, sic_threshold=da_sic_threshold,
        cast_layers=da_cast_layers, cast_threshold=da_cast_threshold,
    )
    final_loss = overall_best_loss
    accessibility_score = _compute_accessibility_score(final_loss)

    success_rate, responses = _evaluate_attack(
        model, tokenizer, prompts, final_embeds, config,
    )

    token_text = tokenizer.decode(current_ids)

    return SoftPromptResult(
        mode="gcg",
        success_rate=success_rate,
        final_loss=final_loss,
        loss_history=all_loss_history,
        n_steps=total_steps,
        n_tokens=config.n_tokens,
        embeddings=None,
        token_ids=current_ids,
        token_text=token_text,
        eval_responses=responses,
        accessibility_score=accessibility_score,
        per_prompt_losses=per_prompt_losses,
        early_stopped=early_stopped,
    )

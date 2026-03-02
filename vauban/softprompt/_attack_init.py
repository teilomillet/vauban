"""Shared initialization for attack algorithms.

Consolidates the common setup (transformer extraction, target encoding,
prompt pre-encoding, objective state construction) shared across GCG,
EGD, COLD, continuous, and AmpleGCG attack modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vauban._forward import force_eval, get_transformer
from vauban.softprompt._encoding import _pre_encode_prompts
from vauban.softprompt._gcg_objective import GCGSharedState, _build_gcg_shared_state
from vauban.softprompt._runtime import _encode_targets, _prepare_transfer_data

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import (
        CausalLM,
        SoftPromptConfig,
        Tokenizer,
        TransformerModel,
    )


@dataclass(frozen=True, slots=True)
class AttackInitState:
    """Shared initialization state for attack algorithms.

    Holds the resolved transformer, embedding matrix, encoded targets,
    pre-encoded prompts, objective state, and transfer data so that
    each attack module can skip the identical ~30-line setup block.
    """

    transformer: TransformerModel
    vocab_size: int
    d_model: int
    embed_matrix: Array
    target_ids: Array
    all_prompt_ids: list[Array]
    objective_state: GCGSharedState
    transfer_data: list[tuple[CausalLM, Tokenizer, list[Array], Array]]


def prepare_attack(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SoftPromptConfig,
    direction: Array | None,
    *,
    ref_model: CausalLM | None = None,
    all_prompt_ids_override: list[Array] | None = None,
    transfer_models: list[tuple[str, CausalLM, Tokenizer]] | None = None,
    infix_map: dict[int, int] | None = None,
    perplexity_weight_override: float | None = None,
    svf_boundary: object | None = None,
) -> AttackInitState:
    """Prepare shared initialization state for an attack algorithm.

    Extracts the inner transformer, encodes target prefixes, pre-encodes
    prompts, builds the shared objective state, and prepares transfer
    model data.  Each attack module calls this once at entry.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with encode/decode support.
        prompts: Attack prompts to optimize against.
        config: Soft prompt configuration.
        direction: Optional refusal direction for direction-guided mode.
        ref_model: Optional reference model for KL collision loss.
        all_prompt_ids_override: Pre-encoded prompt IDs (injection/infix).
        transfer_models: Optional transfer models for post-hoc scoring.
        infix_map: Per-prompt infix split positions.
        perplexity_weight_override: Override for perplexity weight in
            objective state (continuous mode sets this to 0.0).
        svf_boundary: Optional SVF boundary for context-dependent directions.
    """
    transformer = get_transformer(model)
    vocab_size = transformer.embed_tokens.weight.shape[0]
    d_model = transformer.embed_tokens.weight.shape[1]
    embed_matrix = transformer.embed_tokens.weight

    target_ids = _encode_targets(
        tokenizer, config.target_prefixes, config.target_repeat_count,
    )
    force_eval(target_ids)

    if all_prompt_ids_override is not None:
        all_prompt_ids = all_prompt_ids_override
    else:
        effective_prompts = prompts if prompts else ["Hello"]
        all_prompt_ids = _pre_encode_prompts(
            tokenizer, effective_prompts, config.system_prompt,
        )

    objective_state = _build_gcg_shared_state(
        model,
        tokenizer,
        config,
        target_ids,
        direction,
        ref_model=ref_model,
        infix_map=infix_map,
        perplexity_weight_override=perplexity_weight_override,
        svf_boundary=svf_boundary,
    )

    transfer_data = _prepare_transfer_data(
        transfer_models, config, prompts,
    )

    return AttackInitState(
        transformer=transformer,
        vocab_size=vocab_size,
        d_model=d_model,
        embed_matrix=embed_matrix,
        target_ids=target_ids,
        all_prompt_ids=all_prompt_ids,
        objective_state=objective_state,
        transfer_data=transfer_data,
    )

"""Soft prompt attack: continuous, GCG, EGD, COLD, and AmpleGCG modes."""

from vauban._forward import make_cache as _make_cache
from vauban.softprompt._amplecgc import _amplecgc_attack
from vauban.softprompt._cold import _cold_attack
from vauban.softprompt._constraints import _build_vocab_mask
from vauban.softprompt._continuous import _continuous_attack
from vauban.softprompt._defense_eval import (
    evaluate_against_defenses,
    evaluate_against_defenses_multiturn,
)
from vauban.softprompt._dispatcher import softprompt_attack
from vauban.softprompt._egd import _egd_attack
from vauban.softprompt._encoding import (
    _compute_infix_split,
    _pre_encode_prompts,
    _pre_encode_prompts_with_history,
    _pre_encode_prompts_with_injection_context,
    _pre_encode_prompts_with_injection_template,
    _resolve_infix_overrides,
    _resolve_injection_ids,
)
from vauban.softprompt._gan import gan_loop
from vauban.softprompt._gcg import _gcg_attack
from vauban.softprompt._generation import (
    _decode_step,
    _evaluate_attack,
    _evaluate_attack_with_history,
    _prefill_with_cache,
)
from vauban.softprompt._largo import largo_loop
from vauban.softprompt._loss import (
    _add_perplexity_term,
    _compute_defense_aware_penalty,
    _compute_defensive_loss,
    _compute_eos_loss,
    _compute_externality_loss,
    _compute_kl_collision_loss,
    _compute_loss,
    _compute_perplexity_loss,
    _compute_untargeted_loss,
)
from vauban.softprompt._paraphrase import paraphrase_prompts
from vauban.softprompt._runtime import (
    _compute_accessibility_score,
    _compute_embed_regularization,
    _compute_learning_rate,
    _encode_refusal_tokens,
    _encode_targets,
    _forward_with_prefix,
    _prepare_transfer_data,
    _project_to_tokens,
    _score_transfer_loss,
)
from vauban.softprompt._search import (
    _compute_per_prompt_losses,
    _sample_prompt_ids,
    _select_prompt_ids,
    _select_worst_k_prompt_ids,
    _split_into_batches,
)

__all__ = [
    "_add_perplexity_term",
    "_amplecgc_attack",
    "_build_vocab_mask",
    "_cold_attack",
    "_compute_accessibility_score",
    "_compute_defense_aware_penalty",
    "_compute_defensive_loss",
    "_compute_embed_regularization",
    "_compute_eos_loss",
    "_compute_externality_loss",
    "_compute_infix_split",
    "_compute_kl_collision_loss",
    "_compute_learning_rate",
    "_compute_loss",
    "_compute_per_prompt_losses",
    "_compute_perplexity_loss",
    "_compute_untargeted_loss",
    "_continuous_attack",
    "_decode_step",
    "_egd_attack",
    "_encode_refusal_tokens",
    "_encode_targets",
    "_evaluate_attack",
    "_evaluate_attack_with_history",
    "_forward_with_prefix",
    "_gcg_attack",
    "_make_cache",
    "_pre_encode_prompts",
    "_pre_encode_prompts_with_history",
    "_pre_encode_prompts_with_injection_context",
    "_pre_encode_prompts_with_injection_template",
    "_prefill_with_cache",
    "_prepare_transfer_data",
    "_project_to_tokens",
    "_resolve_infix_overrides",
    "_resolve_injection_ids",
    "_sample_prompt_ids",
    "_score_transfer_loss",
    "_select_prompt_ids",
    "_select_worst_k_prompt_ids",
    "_split_into_batches",
    "evaluate_against_defenses",
    "evaluate_against_defenses_multiturn",
    "gan_loop",
    "largo_loop",
    "paraphrase_prompts",
    "softprompt_attack",
]

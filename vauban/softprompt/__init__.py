"""Soft prompt attack: continuous embedding optimization, GCG, and EGD."""

from vauban.probe import _make_cache
from vauban.softprompt._continuous import _continuous_attack
from vauban.softprompt._dispatcher import softprompt_attack
from vauban.softprompt._egd import _egd_attack
from vauban.softprompt._gcg import _gcg_attack
from vauban.softprompt._generation import (
    _decode_step,
    _evaluate_attack,
    _prefill_with_cache,
)
from vauban.softprompt._loss import (
    _compute_defensive_loss,
    _compute_eos_loss,
    _compute_kl_collision_loss,
    _compute_loss,
    _compute_untargeted_loss,
)
from vauban.softprompt._utils import (
    _build_vocab_mask,
    _compute_accessibility_score,
    _compute_embed_regularization,
    _compute_learning_rate,
    _encode_refusal_tokens,
    _encode_targets,
    _forward_with_prefix,
    _pre_encode_prompts,
    _project_to_tokens,
    _select_prompt_ids,
    _select_worst_k_prompt_ids,
    _split_into_batches,
)

__all__ = [
    "_build_vocab_mask",
    "_compute_accessibility_score",
    "_compute_defensive_loss",
    "_compute_embed_regularization",
    "_compute_eos_loss",
    "_compute_kl_collision_loss",
    "_compute_learning_rate",
    "_compute_loss",
    "_compute_untargeted_loss",
    "_continuous_attack",
    "_decode_step",
    "_egd_attack",
    "_encode_refusal_tokens",
    "_encode_targets",
    "_evaluate_attack",
    "_forward_with_prefix",
    "_gcg_attack",
    "_make_cache",
    "_pre_encode_prompts",
    "_prefill_with_cache",
    "_project_to_tokens",
    "_select_prompt_ids",
    "_select_worst_k_prompt_ids",
    "_split_into_batches",
    "softprompt_attack",
]

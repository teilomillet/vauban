# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Compatibility re-exports for softprompt helpers."""

from vauban.softprompt._constraints import (
    _build_vocab_mask,
    _is_emoji_char,
    _is_invisible_char,
    _matches_constraint,
)
from vauban.softprompt._encoding import (
    _INJECTION_CONTEXT_PRESETS,
    _INJECTION_CONTEXT_WRAPPERS,
    _build_messages,
    _compute_infix_split,
    _encode_messages,
    _pre_encode_prompts,
    _pre_encode_prompts_with_history,
    _pre_encode_prompts_with_injection_context,
    _pre_encode_prompts_with_injection_template,
    _resolve_infix_overrides,
    _resolve_infix_overrides_with_history,
    _resolve_injection_ids,
)
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
    "_INJECTION_CONTEXT_PRESETS",
    "_INJECTION_CONTEXT_WRAPPERS",
    "_build_messages",
    "_build_vocab_mask",
    "_compute_accessibility_score",
    "_compute_embed_regularization",
    "_compute_infix_split",
    "_compute_learning_rate",
    "_compute_per_prompt_losses",
    "_encode_messages",
    "_encode_refusal_tokens",
    "_encode_targets",
    "_forward_with_prefix",
    "_is_emoji_char",
    "_is_invisible_char",
    "_matches_constraint",
    "_pre_encode_prompts",
    "_pre_encode_prompts_with_history",
    "_pre_encode_prompts_with_injection_context",
    "_pre_encode_prompts_with_injection_template",
    "_prepare_transfer_data",
    "_project_to_tokens",
    "_resolve_infix_overrides",
    "_resolve_infix_overrides_with_history",
    "_resolve_injection_ids",
    "_sample_prompt_ids",
    "_score_transfer_loss",
    "_select_prompt_ids",
    "_select_worst_k_prompt_ids",
    "_split_into_batches",
]

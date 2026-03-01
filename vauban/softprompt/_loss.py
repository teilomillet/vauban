"""Compatibility facade for soft prompt loss helpers and objectives."""

from vauban.softprompt._loss_common import (
    ForwardTrace,
    LayerPenaltyAccumulator,
    LossAuxConfig,
    LossPlacementConfig,
    _add_perplexity_term,
    _apply_shared_aux_terms,
    _assemble_prefix_only_sequence,
    _assemble_targeted_sequence,
    _compute_defense_aware_penalty,
    _compute_eos_loss,
    _compute_kl_collision_loss,
    _compute_perplexity_loss,
    _run_transformer_with_penalties,
    _soft_token_offset,
)
from vauban.softprompt._loss_externality import _compute_externality_loss
from vauban.softprompt._loss_refusal import (
    _compute_defensive_loss,
    _compute_untargeted_loss,
)
from vauban.softprompt._loss_targeted import _compute_loss

__all__ = [
    "ForwardTrace",
    "LayerPenaltyAccumulator",
    "LossAuxConfig",
    "LossPlacementConfig",
    "_add_perplexity_term",
    "_apply_shared_aux_terms",
    "_assemble_prefix_only_sequence",
    "_assemble_targeted_sequence",
    "_compute_defense_aware_penalty",
    "_compute_defensive_loss",
    "_compute_eos_loss",
    "_compute_externality_loss",
    "_compute_kl_collision_loss",
    "_compute_loss",
    "_compute_perplexity_loss",
    "_compute_untargeted_loss",
    "_run_transformer_with_penalties",
    "_soft_token_offset",
]

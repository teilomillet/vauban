"""Loss-related field parsing for the [softprompt] config section."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from vauban.config._parse_helpers import SectionReader

if TYPE_CHECKING:
    from vauban.config._types import TomlDict


@dataclass(frozen=True, slots=True)
class _SoftPromptLossSection:
    """Parsed loss and optimization knobs for softprompt."""

    direction_weight: float
    embed_reg_weight: float
    loss_mode: str
    egd_temperature: float
    token_constraint: str | list[str] | None
    eos_loss_mode: str
    eos_loss_weight: float
    kl_ref_weight: float
    ref_model: str | None
    worst_k: int
    grad_accum_steps: int
    target_repeat_count: int
    defense_aware_weight: float
    transfer_loss_weight: float
    transfer_rerank_count: int
    perplexity_weight: float
    externality_target: str | None


def _parse_softprompt_loss(
    sec: TomlDict,
    base_dir: Path | None,
) -> _SoftPromptLossSection:
    """Parse the loss-oriented [softprompt] fields."""
    reader = SectionReader("[softprompt]", sec)

    direction_weight = reader.number("direction_weight", default=0.0)
    if direction_weight < 0.0:
        msg = (
            "[softprompt].direction_weight must be >= 0.0,"
            f" got {direction_weight}"
        )
        raise ValueError(msg)

    embed_reg_weight = reader.number("embed_reg_weight", default=0.0)
    if embed_reg_weight < 0.0:
        msg = (
            "[softprompt].embed_reg_weight must be >= 0.0,"
            f" got {embed_reg_weight}"
        )
        raise ValueError(msg)

    loss_mode = reader.literal(
        "loss_mode",
        ("targeted", "untargeted", "defensive", "externality"),
        default="targeted",
    )

    egd_temperature = reader.number("egd_temperature", default=1.0)
    if egd_temperature <= 0:
        msg = (
            f"[softprompt].egd_temperature must be > 0, got {egd_temperature}"
        )
        raise ValueError(msg)

    token_constraint_raw = reader.data.get("token_constraint")
    token_constraint: str | list[str] | None = None
    valid_constraints = (
        "ascii",
        "alpha",
        "alphanumeric",
        "non_latin",
        "chinese",
        "non_alphabetic",
        "invisible",
        "zalgo",
        "emoji",
    )
    if token_constraint_raw is not None:
        if isinstance(token_constraint_raw, str):
            if token_constraint_raw not in valid_constraints:
                msg = (
                    "[softprompt].token_constraint must be one of"
                    f" {valid_constraints!r}, got {token_constraint_raw!r}"
                )
                raise ValueError(msg)
            token_constraint = token_constraint_raw
        elif isinstance(token_constraint_raw, list):
            constraint_list: list[str] = []
            for item in token_constraint_raw:
                if not isinstance(item, str):
                    msg = (
                        "[softprompt].token_constraint list elements must be"
                        f" strings, got {type(item).__name__}"
                    )
                    raise TypeError(msg)
                if item not in valid_constraints:
                    msg = (
                        "[softprompt].token_constraint element"
                        f" {item!r} is not one of {valid_constraints!r}"
                    )
                    raise ValueError(msg)
                constraint_list.append(item)
            token_constraint = constraint_list
        else:
            msg = (
                "[softprompt].token_constraint must be a string or list of"
                f" strings, got {type(token_constraint_raw).__name__}"
            )
            raise TypeError(msg)

    eos_loss_mode = reader.literal(
        "eos_loss_mode",
        ("none", "force", "suppress"),
        default="none",
    )

    eos_loss_weight = reader.number("eos_loss_weight", default=0.0)
    if eos_loss_weight < 0.0:
        msg = (
            "[softprompt].eos_loss_weight must be >= 0.0,"
            f" got {eos_loss_weight}"
        )
        raise ValueError(msg)

    kl_ref_weight = reader.number("kl_ref_weight", default=0.0)
    if kl_ref_weight < 0.0:
        msg = (
            "[softprompt].kl_ref_weight must be >= 0.0,"
            f" got {kl_ref_weight}"
        )
        raise ValueError(msg)

    ref_model = reader.optional_string("ref_model")

    worst_k = reader.integer("worst_k", default=5)
    if worst_k < 1:
        msg = f"[softprompt].worst_k must be >= 1, got {worst_k}"
        raise ValueError(msg)

    grad_accum_steps = reader.integer("grad_accum_steps", default=1)
    if grad_accum_steps < 1:
        msg = (
            "[softprompt].grad_accum_steps must be >= 1,"
            f" got {grad_accum_steps}"
        )
        raise ValueError(msg)

    target_repeat_count = reader.integer("target_repeat_count", default=0)
    if target_repeat_count < 0:
        msg = (
            "[softprompt].target_repeat_count must be >= 0,"
            f" got {target_repeat_count}"
        )
        raise ValueError(msg)

    defense_aware_weight = reader.number("defense_aware_weight", default=0.0)
    if defense_aware_weight < 0.0:
        msg = (
            "[softprompt].defense_aware_weight must be >= 0.0,"
            f" got {defense_aware_weight}"
        )
        raise ValueError(msg)

    transfer_loss_weight = reader.number("transfer_loss_weight", default=0.0)
    if transfer_loss_weight < 0.0:
        msg = (
            "[softprompt].transfer_loss_weight must be >= 0.0,"
            f" got {transfer_loss_weight}"
        )
        raise ValueError(msg)

    transfer_rerank_count = reader.integer("transfer_rerank_count", default=8)
    if transfer_rerank_count < 1:
        msg = (
            "[softprompt].transfer_rerank_count must be >= 1,"
            f" got {transfer_rerank_count}"
        )
        raise ValueError(msg)

    perplexity_weight = reader.number("perplexity_weight", default=0.0)
    if perplexity_weight < 0.0:
        msg = (
            "[softprompt].perplexity_weight must be >= 0.0,"
            f" got {perplexity_weight}"
        )
        raise ValueError(msg)

    externality_target_raw = reader.optional_string("externality_target")
    externality_target: str | None = None
    if externality_target_raw is not None:
        resolve_dir = base_dir if base_dir is not None else Path.cwd()
        externality_target = str((resolve_dir / externality_target_raw).resolve())

    return _SoftPromptLossSection(
        direction_weight=direction_weight,
        embed_reg_weight=embed_reg_weight,
        loss_mode=loss_mode,
        egd_temperature=egd_temperature,
        token_constraint=token_constraint,
        eos_loss_mode=eos_loss_mode,
        eos_loss_weight=eos_loss_weight,
        kl_ref_weight=kl_ref_weight,
        ref_model=ref_model,
        worst_k=worst_k,
        grad_accum_steps=grad_accum_steps,
        target_repeat_count=target_repeat_count,
        defense_aware_weight=defense_aware_weight,
        transfer_loss_weight=transfer_loss_weight,
        transfer_rerank_count=transfer_rerank_count,
        perplexity_weight=perplexity_weight,
        externality_target=externality_target,
    )

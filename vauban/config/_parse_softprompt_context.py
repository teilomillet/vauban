# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Context-related field parsing for the [softprompt] config section."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vauban.config._parse_helpers import SectionReader

if TYPE_CHECKING:
    from vauban.config._types import TomlDict


@dataclass(frozen=True, slots=True)
class _SoftPromptContextSection:
    """Parsed context and prompting fields for softprompt."""

    transfer_models: list[str]
    system_prompt: str | None
    injection_context: str | None
    injection_context_template: str | None
    token_position: str
    paraphrase_strategies: list[str]


def _parse_softprompt_context(sec: TomlDict) -> _SoftPromptContextSection:
    """Parse the prompt-context [softprompt] fields."""
    reader = SectionReader("[softprompt]", sec)

    transfer_models = reader.string_list(
        "transfer_models",
        default=[],
        coerce=True,
    )
    system_prompt = reader.optional_string("system_prompt")
    injection_context = reader.optional_literal(
        "injection_context",
        ("web_page", "tool_output", "code_file"),
    )
    injection_context_template = reader.optional_string("injection_context_template")
    token_position = reader.literal(
        "token_position",
        ("prefix", "suffix", "infix"),
        default="prefix",
    )

    valid_paraphrase = (
        "narrative",
        "deceptive_delight",
        "technical",
        "historical",
        "code_block",
        "educational",
    )
    paraphrase_strategies = reader.string_list(
        "paraphrase_strategies",
        default=[],
    )
    for item in paraphrase_strategies:
        if item not in valid_paraphrase:
            msg = (
                "[softprompt].paraphrase_strategies element"
                f" {item!r} is not one of {valid_paraphrase!r}"
            )
            raise ValueError(msg)

    return _SoftPromptContextSection(
        transfer_models=transfer_models,
        system_prompt=system_prompt,
        injection_context=injection_context,
        injection_context_template=injection_context_template,
        token_position=token_position,
        paraphrase_strategies=paraphrase_strategies,
    )

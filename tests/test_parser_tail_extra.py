# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Small edge-case tests for remaining parser branches."""

from pathlib import Path

import pytest

from vauban.config._parse_guard import _parse_guard
from vauban.config._parse_meta import parse_meta
from vauban.config._parse_softprompt_defense import _parse_softprompt_defense
from vauban.config._parse_softprompt_gan import _parse_softprompt_gan


def test_parse_guard_requires_tiers_list() -> None:
    with pytest.raises(TypeError, match=r"\[guard\]\.tiers must be a list"):
        _parse_guard({
            "guard": {
                "prompts": ["test"],
                "tiers": "bad",
            },
        })


def test_parse_meta_docs_requires_list() -> None:
    with pytest.raises(TypeError, match=r"\[meta\]\.docs must be an array"):
        parse_meta(
            {"meta": {"docs": "bad"}},
            Path("/tmp/run.toml"),
        )


def test_parse_softprompt_defense_iterations_must_be_positive() -> None:
    with pytest.raises(ValueError, match="defense_eval_sic_max_iterations"):
        _parse_softprompt_defense({"defense_eval_sic_max_iterations": 0})


def test_parse_softprompt_gan_token_escalation_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="gan_token_escalation"):
        _parse_softprompt_gan({"gan_token_escalation": -1})

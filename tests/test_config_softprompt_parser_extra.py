# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra edge-case coverage for softprompt config parsers."""

import pytest

from vauban.config._parse_softprompt import _parse_softprompt
from vauban.config._parse_softprompt_core import _parse_softprompt_core


class TestParseSoftpromptExtra:
    """Target uncovered validation branches in the top-level parser."""

    def test_requires_table(self) -> None:
        with pytest.raises(TypeError, match=r"\[softprompt\] must be a table"):
            _parse_softprompt({"softprompt": 3})

    def test_externality_loss_requires_target(self) -> None:
        with pytest.raises(ValueError, match="externality_target"):
            _parse_softprompt({
                "softprompt": {
                    "mode": "gcg",
                    "loss_mode": "externality",
                },
            })

    def test_largo_and_gan_rounds_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            _parse_softprompt({
                "softprompt": {
                    "largo_reflection_rounds": 1,
                    "gan_rounds": 1,
                },
            })

    def test_largo_requires_continuous_mode(self) -> None:
        with pytest.raises(ValueError, match="requires mode"):
            _parse_softprompt({
                "softprompt": {
                    "mode": "gcg",
                    "largo_reflection_rounds": 1,
                },
            })

    def test_alpha_tiers_must_be_sorted(self) -> None:
        with pytest.raises(ValueError, match="must be sorted"):
            _parse_softprompt({
                "softprompt": {
                    "mode": "gcg",
                    "defense_eval_alpha_tiers": [
                        [0.6, 2.0],
                        [0.3, 1.0],
                    ],
                },
            })


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("n_steps", 0, "n_steps"),
        ("learning_rate", 0.0, "learning_rate"),
        ("batch_size", 0, "batch_size"),
        ("top_k", 0, "top_k"),
    ],
)
def test_parse_softprompt_core_validates_positive_values(
    key: str,
    value: int | float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _parse_softprompt_core({key: value})

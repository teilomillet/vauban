# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Public softprompt entrypoints.

The package root exposes only high-level workflows. Private helpers live in
submodules such as ``vauban.softprompt._utils`` and ``vauban.softprompt._loss``.
"""

from vauban.softprompt._defense_eval import (
    evaluate_against_defenses,
    evaluate_against_defenses_multiturn,
)
from vauban.softprompt._dispatcher import softprompt_attack
from vauban.softprompt._gan import gan_loop
from vauban.softprompt._largo import largo_loop
from vauban.softprompt._paraphrase import paraphrase_prompts

__all__ = [
    "evaluate_against_defenses",
    "evaluate_against_defenses_multiturn",
    "gan_loop",
    "largo_loop",
    "paraphrase_prompts",
    "softprompt_attack",
]

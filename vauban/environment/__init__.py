# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Agent environment harness for indirect prompt injection simulation.

Simulates tool-use workflows locally, defining success as tool-call
behavior (not text compliance), and provides reward signals to the
existing GCG/EGD optimizer.
"""

from vauban.environment._loop import run_agent_loop
from vauban.environment._reward import compute_reward
from vauban.environment._rollout import score_candidates_via_rollout

__all__ = [
    "compute_reward",
    "run_agent_loop",
    "score_candidates_via_rollout",
]

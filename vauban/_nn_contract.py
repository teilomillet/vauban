# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Neural network utilities contract.

Every symbol listed in NN_CONTRACT must be exported by each NN backend module.
"""

NN_CONTRACT: list[str] = [
    "create_additive_causal_mask",
    "cross_entropy",
]

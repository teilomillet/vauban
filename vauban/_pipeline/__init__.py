# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Pipeline orchestration subpackage.

This package contains the runtime pipeline logic that was previously
inlined in ``vauban/__init__.py``.  The public entry point is :func:`run`.
"""

from vauban._pipeline._run import run

__all__ = ["run"]

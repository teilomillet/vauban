# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SPDX header automation script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def _load_script_module() -> ModuleType:
    """Load the header automation script as a module for testing."""
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "license_headers.py"
    )
    spec = importlib.util.spec_from_file_location("license_headers", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_insert_header_before_module_docstring() -> None:
    """The script should prepend the SPDX block before a module docstring."""
    module = _load_script_module()
    source = '"""Example module."""\n\nVALUE = 1\n'
    rewritten = module.insert_header(source, Path("example.py"))
    assert rewritten.startswith(
        "# SPDX-FileCopyrightText: 2026 Teilo Millet\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
        '"""Example module."""\n',
    )


def test_insert_header_preserves_shebang() -> None:
    """A shebang must remain the first line after header insertion."""
    module = _load_script_module()
    source = "#!/usr/bin/env python3\nprint('hi')\n"
    rewritten = module.insert_header(source, Path("script.py"))
    assert rewritten.startswith(
        "#!/usr/bin/env python3\n"
        "# SPDX-FileCopyrightText: 2026 Teilo Millet\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n",
    )


def test_has_required_header_detects_existing_block() -> None:
    """An existing SPDX block should not be inserted twice."""
    module = _load_script_module()
    source = (
        "# SPDX-FileCopyrightText: 2026 Teilo Millet\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
        '"""Example module."""\n'
    )
    assert module.has_required_header(source, Path("example.py")) is True
    assert module.insert_header(source, Path("example.py")) == source


def test_insert_header_for_markdown_uses_html_comments() -> None:
    """Markdown files should receive HTML comment SPDX headers."""
    module = _load_script_module()
    source = "# Example\n\nSome text.\n"
    rewritten = module.insert_header(source, Path("README.md"))
    assert rewritten.startswith(
        "<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->\n"
        "<!-- SPDX-License-Identifier: Apache-2.0 -->\n\n"
        "# Example\n",
    )


def test_insert_header_for_toml_uses_hash_comments() -> None:
    """TOML files should receive hash-comment SPDX headers."""
    module = _load_script_module()
    source = '[project]\nname = "vauban"\n'
    rewritten = module.insert_header(source, Path("pyproject.toml"))
    assert rewritten.startswith(
        "# SPDX-FileCopyrightText: 2026 Teilo Millet\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
        "[project]\n",
    )

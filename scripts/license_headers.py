# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Apply or check SPDX headers for repo-owned source files."""

from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_COPYRIGHT_OWNER = "Teilo Millet"
DEFAULT_COPYRIGHT_YEAR = "2026"
LICENSE_IDENTIFIER = "Apache-2.0"
HASH_COMMENT_SUFFIXES: frozenset[str] = frozenset({
    ".py",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
})
MARKDOWN_SUFFIXES: frozenset[str] = frozenset({".md"})
TARGET_PATTERNS: tuple[str, ...] = (
    "vauban/**/*.py",
    "tests/**/*.py",
    "scripts/**/*.py",
    ".github/workflows/*.yml",
    "docs/**/*.md",
    "docs/**/*.txt",
    "examples/**/*.toml",
    "*.md",
    "pyproject.toml",
    "mkdocs.yml",
    ".readthedocs.yaml",
)


@dataclass(frozen=True, slots=True)
class HeaderResult:
    """Summary of a header application or validation run."""

    updated: list[Path]
    missing: list[Path]


@dataclass(frozen=True, slots=True)
class HeaderConfig:
    """Copyright settings used to build SPDX file headers."""

    copyright_owner: str
    copyright_year: str


def repo_root() -> Path:
    """Return the repository root for this script."""
    return Path(__file__).resolve().parents[1]


def _dict_value(value: object) -> dict[str, object] | None:
    """Narrow a parsed TOML value to a string-keyed dict when possible."""
    if isinstance(value, dict):
        return {
            str(key): item
            for key, item in value.items()
        }
    return None


def load_header_config(root: Path) -> HeaderConfig:
    """Load header settings from ``pyproject.toml`` when configured."""
    config = HeaderConfig(
        copyright_owner=DEFAULT_COPYRIGHT_OWNER,
        copyright_year=DEFAULT_COPYRIGHT_YEAR,
    )
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.exists():
        return config
    with pyproject_path.open("rb") as handle:
        parsed = tomllib.load(handle)
    tool = _dict_value(parsed.get("tool"))
    if tool is None:
        return config
    vauban = _dict_value(tool.get("vauban"))
    if vauban is None:
        return config
    headers = _dict_value(vauban.get("license_headers"))
    if headers is None:
        return config
    owner = headers.get("owner")
    year = headers.get("year")
    copyright_owner = (
        owner
        if isinstance(owner, str) and owner
        else config.copyright_owner
    )
    copyright_year = (
        year
        if isinstance(year, str) and year
        else config.copyright_year
    )
    return HeaderConfig(
        copyright_owner=copyright_owner,
        copyright_year=copyright_year,
    )


def target_files(root: Path) -> list[Path]:
    """Return repo-owned source files that must carry SPDX headers."""
    files: list[Path] = []
    for pattern in TARGET_PATTERNS:
        files.extend(sorted(root.glob(pattern)))
    return [
        path
        for path in sorted(set(files))
        if "__pycache__" not in path.parts
    ]


def spdx_lines(config: HeaderConfig) -> tuple[str, str]:
    """Return the raw SPDX lines for the active header configuration."""
    return (
        (
            "SPDX-FileCopyrightText: "
            f"{config.copyright_year} {config.copyright_owner}"
        ),
        f"SPDX-License-Identifier: {LICENSE_IDENTIFIER}",
    )


def header_lines_for_path(path: Path, config: HeaderConfig) -> tuple[str, str]:
    """Return the correctly formatted SPDX lines for a path."""
    line_a, line_b = spdx_lines(config)
    if path.suffix in HASH_COMMENT_SUFFIXES:
        return (f"# {line_a}", f"# {line_b}")
    if path.suffix in MARKDOWN_SUFFIXES:
        return (f"<!-- {line_a} -->", f"<!-- {line_b} -->")
    msg = f"Unsupported header target type: {path}"
    raise ValueError(msg)


def header_block_for_path(path: Path, config: HeaderConfig) -> str:
    """Return the correctly formatted SPDX block for a path."""
    return "\n".join((*header_lines_for_path(path, config), "", ""))


def has_required_header(
    text: str,
    path: Path,
    config: HeaderConfig | None = None,
) -> bool:
    """Return whether text already starts with the required SPDX header."""
    active_config = config if config is not None else load_header_config(repo_root())
    first_lines = text.splitlines()[:6]
    return all(
        line in first_lines for line in header_lines_for_path(path, active_config)
    )


def insert_header(
    text: str,
    path: Path,
    config: HeaderConfig | None = None,
) -> str:
    """Insert the SPDX header after a shebang or encoding line when present."""
    active_config = config if config is not None else load_header_config(repo_root())
    if has_required_header(text, path, active_config):
        return text

    header_block = header_block_for_path(path, active_config)
    if path.suffix in MARKDOWN_SUFFIXES:
        suffix = text[1:] if text.startswith("\n") else text
        return f"{header_block}{suffix}"

    lines = text.splitlines(keepends=True)
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    if insert_at < len(lines) and "coding" in lines[insert_at]:
        insert_at += 1

    prefix = "".join(lines[:insert_at])
    suffix = "".join(lines[insert_at:])
    if suffix.startswith("\n"):
        suffix = suffix[1:]
    return f"{prefix}{header_block}{suffix}"


def apply_headers(*, write: bool) -> HeaderResult:
    """Apply or validate SPDX headers across the managed source set."""
    root = repo_root()
    config = load_header_config(root)
    updated: list[Path] = []
    missing: list[Path] = []
    for path in target_files(root):
        original = path.read_text(encoding="utf-8")
        rewritten = insert_header(original, path, config)
        if rewritten == original:
            continue
        rel_path = path.relative_to(root)
        if write:
            path.write_text(rewritten, encoding="utf-8")
            updated.append(rel_path)
        else:
            missing.append(rel_path)
    return HeaderResult(updated=updated, missing=missing)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Apply or check SPDX headers for repo-owned source files.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write",
        action="store_true",
        help="Insert SPDX headers where they are missing.",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Fail if any managed file is missing an SPDX header.",
    )
    return parser


def main() -> int:
    """Run the header tool CLI."""
    parser = build_parser()
    args = parser.parse_args()
    result = apply_headers(write=args.write)
    if args.write:
        print(f"Updated {len(result.updated)} files.")
        return 0
    if result.missing:
        print("Missing SPDX headers:")
        for path in result.missing:
            print(path.as_posix())
        return 1
    print("All managed source files carry SPDX headers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

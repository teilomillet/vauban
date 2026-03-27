# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Apply or check SPDX headers for repo-owned source files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

COPYRIGHT_OWNER = "Teilo Millet"
COPYRIGHT_YEAR = "2026"
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


def repo_root() -> Path:
    """Return the repository root for this script."""
    return Path(__file__).resolve().parents[1]


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


def spdx_lines() -> tuple[str, str]:
    """Return the raw SPDX lines for the maintainer-controlled header policy."""
    return (
        (
            "SPDX-FileCopyrightText: "
            f"{COPYRIGHT_YEAR} {COPYRIGHT_OWNER}"
        ),
        f"SPDX-License-Identifier: {LICENSE_IDENTIFIER}",
    )


def header_lines_for_path(path: Path) -> tuple[str, str]:
    """Return the correctly formatted SPDX lines for a path."""
    line_a, line_b = spdx_lines()
    if path.suffix in HASH_COMMENT_SUFFIXES:
        return (f"# {line_a}", f"# {line_b}")
    if path.suffix in MARKDOWN_SUFFIXES:
        return (f"<!-- {line_a} -->", f"<!-- {line_b} -->")
    msg = f"Unsupported header target type: {path}"
    raise ValueError(msg)


def header_block_for_path(path: Path) -> str:
    """Return the correctly formatted SPDX block for a path."""
    return "\n".join((*header_lines_for_path(path), "", ""))


def has_required_header(text: str, path: Path) -> bool:
    """Return whether text already starts with the required SPDX header."""
    first_lines = text.splitlines()[:6]
    return all(line in first_lines for line in header_lines_for_path(path))


def insert_header(text: str, path: Path) -> str:
    """Insert the SPDX header after a shebang or encoding line when present."""
    if has_required_header(text, path):
        return text

    header_block = header_block_for_path(path)
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
    updated: list[Path] = []
    missing: list[Path] = []
    for path in target_files(root):
        original = path.read_text(encoding="utf-8")
        rewritten = insert_header(original, path)
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

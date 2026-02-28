"""Experiment tech tree viewer.

Discovers ``[meta]`` sections in TOML configs and renders the
experiment lineage as a text tree or Mermaid flowchart.

Usage::

    python -m vauban.tree [directory]
    python -m vauban.tree experiments/ --format mermaid
    python -m vauban.tree experiments/ --status promising
    python -m vauban.tree experiments/ --tag gcg
"""

from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path

from vauban.config._parse_meta import _VALID_STATUSES, parse_meta

# Status badge mapping for the text tree
_STATUS_BADGES: dict[str, str] = {
    "wip": "[WIP]",
    "promising": "[OK+]",
    "dead_end": "[X]",
    "baseline": "[BASE]",
    "superseded": "[OLD]",
    "archived": "[ARC]",
}


@dataclass
class ExperimentNode:
    """A single experiment node in the tech tree."""

    id: str
    title: str
    status: str
    parents: list[str]
    tags: list[str]
    path: Path
    date: str = ""

    @property
    def badge(self) -> str:
        """Status badge string for display."""
        return _STATUS_BADGES.get(self.status, f"[{self.status.upper()}]")

    @property
    def display_label(self) -> str:
        """Node label: badge + title or id."""
        label = self.title if self.title else self.id
        return f"{self.badge} {label}"


def discover_experiments(directory: str | Path) -> list[ExperimentNode]:
    """Recursively find TOML files and extract experiment nodes.

    TOMLs without a ``[meta]`` section get a default node derived
    from the filename.  Files that fail to parse are silently skipped.
    """
    directory = Path(directory)
    nodes: list[ExperimentNode] = []

    for toml_path in sorted(directory.rglob("*.toml")):
        try:
            with toml_path.open("rb") as f:
                raw = tomllib.load(f)
        except Exception:
            continue

        meta = parse_meta(raw, config_path=toml_path)
        if meta is not None:
            nodes.append(ExperimentNode(
                id=meta.id,
                title=meta.title,
                status=meta.status,
                parents=list(meta.parents),
                tags=list(meta.tags),
                path=toml_path,
                date=meta.date,
            ))
        else:
            # Include TOMLs without [meta] as stub nodes
            nodes.append(ExperimentNode(
                id=toml_path.stem,
                title="",
                status="wip",
                parents=[],
                tags=[],
                path=toml_path,
            ))

    return nodes


def _validate_graph(
    nodes: list[ExperimentNode],
) -> list[str]:
    """Validate the experiment graph and return warnings."""
    warnings: list[str] = []
    ids = {n.id for n in nodes}

    # Duplicate IDs
    seen: set[str] = set()
    for node in nodes:
        if node.id in seen:
            warnings.append(f"Duplicate id: {node.id!r}")
        seen.add(node.id)

    # Dangling parent references
    for node in nodes:
        for parent_id in node.parents:
            if parent_id not in ids:
                warnings.append(
                    f"Dangling parent: {node.id!r} references"
                    f" unknown parent {parent_id!r}",
                )

    # Cycle detection (DFS)
    children_map: dict[str, list[str]] = {n.id: [] for n in nodes}
    for node in nodes:
        for parent_id in node.parents:
            if parent_id in children_map:
                children_map[parent_id].append(node.id)

    visited: set[str] = set()
    in_stack: set[str] = set()

    def _dfs(nid: str) -> bool:
        visited.add(nid)
        in_stack.add(nid)
        for child in children_map.get(nid, []):
            if child in in_stack:
                return True
            if child not in visited and _dfs(child):
                return True
        in_stack.discard(nid)
        return False

    for node in nodes:
        if node.id not in visited and _dfs(node.id):
            warnings.append("Cycle detected in experiment graph")
            break

    return warnings


def build_tree_text(nodes: list[ExperimentNode]) -> str:
    """Build an ASCII text tree from experiment nodes.

    Groups nodes into roots (no parents), children, and orphans.
    """
    if not nodes:
        return "(no experiments found)"

    node_map: dict[str, ExperimentNode] = {}
    for n in nodes:
        node_map[n.id] = n

    children_map: dict[str, list[str]] = {n.id: [] for n in nodes}
    for node in nodes:
        for parent_id in node.parents:
            if parent_id in children_map:
                children_map[parent_id].append(node.id)

    roots = [n for n in nodes if not n.parents]
    # Nodes whose parents are all missing are also roots
    all_ids = {n.id for n in nodes}
    for node in nodes:
        if (node.parents
                and all(p not in all_ids for p in node.parents)
                and node not in roots):
            roots.append(node)

    visited: set[str] = set()
    lines: list[str] = []

    def _render(nid: str, prefix: str, is_last: bool) -> None:
        if nid in visited or nid not in node_map:
            return
        visited.add(nid)
        node = node_map[nid]
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{node.display_label}")
        child_prefix = prefix + ("    " if is_last else "│   ")
        kids = children_map.get(nid, [])
        for i, kid in enumerate(kids):
            _render(kid, child_prefix, i == len(kids) - 1)

    for i, root in enumerate(roots):
        if root.id in visited:
            continue
        visited.add(root.id)
        lines.append(root.display_label)
        kids = children_map.get(root.id, [])
        for j, kid in enumerate(kids):
            _render(kid, "", j == len(kids) - 1)
        if i < len(roots) - 1:
            lines.append("")

    # Orphans (nodes not reached from any root)
    orphans = [n for n in nodes if n.id not in visited]
    if orphans:
        lines.append("")
        lines.append("Orphans:")
        for o in orphans:
            lines.append(f"  {o.display_label}")

    # Warnings
    warnings = _validate_graph(nodes)
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"  ⚠ {w}")

    # Tag summary
    all_tags: dict[str, int] = {}
    for n in nodes:
        for t in n.tags:
            all_tags[t] = all_tags.get(t, 0) + 1
    if all_tags:
        lines.append("")
        tag_parts = [f"{t}({c})" for t, c in sorted(all_tags.items())]
        lines.append(f"Tags: {', '.join(tag_parts)}")

    return "\n".join(lines)


def build_mermaid(nodes: list[ExperimentNode]) -> str:
    """Build a Mermaid flowchart from experiment nodes."""
    if not nodes:
        return "flowchart TD\n    empty[No experiments found]"

    _status_styles: dict[str, str] = {
        "wip": "fill:#ffd,stroke:#aa0",
        "promising": "fill:#dfd,stroke:#0a0",
        "dead_end": "fill:#fdd,stroke:#a00",
        "baseline": "fill:#ddf,stroke:#00a",
        "superseded": "fill:#eee,stroke:#888",
        "archived": "fill:#eee,stroke:#888,stroke-dasharray:5",
    }

    lines: list[str] = ["flowchart TD"]

    # Node definitions
    for node in nodes:
        label = node.title if node.title else node.id
        safe_label = label.replace('"', "'")
        lines.append(f'    {node.id}["{safe_label}"]')

    # Edges (parent → child)
    for node in nodes:
        for parent_id in node.parents:
            lines.append(f"    {parent_id} --> {node.id}")

    # Style classes
    lines.append("")
    for status, style in _status_styles.items():
        class_nodes = [n.id for n in nodes if n.status == status]
        if class_nodes:
            lines.append(f"    style {','.join(class_nodes)} {style}")

    return "\n".join(lines)


def _filter_nodes(
    nodes: list[ExperimentNode],
    *,
    status: str | None = None,
    tag: str | None = None,
) -> list[ExperimentNode]:
    """Filter nodes by status and/or tag."""
    result = nodes
    if status is not None:
        result = [n for n in result if n.status == status]
    if tag is not None:
        result = [n for n in result if tag in n.tags]
    return result


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the tree viewer."""
    parser = argparse.ArgumentParser(
        description="Visualize experiment tech tree from TOML configs.",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to scan for TOML files (default: current dir)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "mermaid"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--status",
        choices=sorted(_VALID_STATUSES),
        default=None,
        help="Filter by experiment status",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Filter by tag",
    )

    args = parser.parse_args(argv)
    nodes = discover_experiments(args.directory)
    nodes = _filter_nodes(nodes, status=args.status, tag=args.tag)

    if args.format == "mermaid":
        print(build_mermaid(nodes))
    else:
        print(build_tree_text(nodes))


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban.tree: experiment tech tree discovery and rendering."""

import runpy
import sys
import warnings
from pathlib import Path

import pytest

from vauban.__main__ import main as cli_main
from vauban.tree import (
    ExperimentNode,
    _filter_nodes,
    _sanitize_mermaid_id,
    _validate_graph,
    build_mermaid,
    build_tree_text,
    discover_experiments,
)
from vauban.tree import (
    main as tree_main,
)


def _make_node(
    node_id: str,
    *,
    title: str = "",
    status: str = "wip",
    parents: list[str] | None = None,
    tags: list[str] | None = None,
) -> ExperimentNode:
    return ExperimentNode(
        id=node_id,
        title=title,
        status=status,
        parents=parents or [],
        tags=tags or [],
        path=Path(f"/tmp/{node_id}.toml"),
    )


class TestDiscoverExperiments:
    """Tests for discover_experiments()."""

    def test_discover_experiments(self, tmp_path: Path) -> None:
        # Create TOML files with and without [meta]
        (tmp_path / "exp_a.toml").write_text(
            '[meta]\ntitle = "Experiment A"\nstatus = "promising"\n'
            '[model]\npath = "test"\n'
        )
        (tmp_path / "exp_b.toml").write_text(
            '[model]\npath = "test"\n'
        )
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "exp_c.toml").write_text(
            '[meta]\nstatus = "dead_end"\nparents = ["exp_a"]\n'
            '[model]\npath = "test"\n'
        )

        nodes, warnings = discover_experiments(tmp_path)
        assert len(nodes) == 3
        assert warnings == []
        ids = {n.id for n in nodes}
        assert ids == {"exp_a", "exp_b", "exp_c"}

        # exp_a has meta
        node_a = next(n for n in nodes if n.id == "exp_a")
        assert node_a.title == "Experiment A"
        assert node_a.status == "promising"

        # exp_b has no meta — defaults
        node_b = next(n for n in nodes if n.id == "exp_b")
        assert node_b.status == "wip"
        assert node_b.title == ""

        # exp_c has parent
        node_c = next(n for n in nodes if n.id == "exp_c")
        assert node_c.parents == ["exp_a"]

    def test_discover_skips_bad_toml(self, tmp_path: Path) -> None:
        (tmp_path / "bad.toml").write_text("this is not valid toml {{{{")
        nodes, warnings = discover_experiments(tmp_path)
        assert len(nodes) == 0
        assert warnings == []

    def test_discover_empty_directory(self, tmp_path: Path) -> None:
        nodes, warnings = discover_experiments(tmp_path)
        assert nodes == []
        assert warnings == []

    def test_no_meta_graceful(self, tmp_path: Path) -> None:
        """TOMLs without [meta] still appear with defaults from filename."""
        (tmp_path / "plain_experiment.toml").write_text(
            '[model]\npath = "x"\n'
        )
        nodes, warnings = discover_experiments(tmp_path)
        assert len(nodes) == 1
        assert warnings == []
        assert nodes[0].id == "plain_experiment"
        assert nodes[0].status == "wip"

    def test_discover_invalid_meta_warns(self, tmp_path: Path) -> None:
        """Invalid [meta] content produces a warning, not a crash."""
        (tmp_path / "bad_meta.toml").write_text(
            '[meta]\nstatus = "bogus"\n'
        )
        nodes, warnings = discover_experiments(tmp_path)
        # Still included as stub node
        assert len(nodes) == 1
        assert nodes[0].id == "bad_meta"
        assert nodes[0].status == "wip"
        # Warning about the invalid status
        assert len(warnings) == 1
        assert "invalid [meta]" in warnings[0]
        assert "bogus" in warnings[0]


class TestBuildTreeText:
    """Tests for build_tree_text()."""

    def test_empty(self) -> None:
        assert "no experiments found" in build_tree_text([])

    def test_build_tree_linear(self) -> None:
        """A → B → C renders as a linear chain."""
        nodes = [
            _make_node("a", title="Root"),
            _make_node("b", title="Middle", parents=["a"]),
            _make_node("c", title="Leaf", parents=["b"]),
        ]
        text = build_tree_text(nodes)
        lines = text.split("\n")
        # Root should be first
        assert "[WIP] Root" in lines[0]
        # B and C should be indented under it
        assert any("Middle" in line for line in lines)
        assert any("Leaf" in line for line in lines)

    def test_build_tree_multi_parent(self) -> None:
        """Diamond graph: A → B, A → C, B → D, C → D."""
        nodes = [
            _make_node("a", title="Root"),
            _make_node("b", parents=["a"]),
            _make_node("c", parents=["a"]),
            _make_node("d", parents=["b", "c"]),
        ]
        text = build_tree_text(nodes)
        # All nodes should appear
        assert "Root" in text
        assert "b" in text
        assert "c" in text
        assert "d" in text

    def test_status_badges(self) -> None:
        nodes = [
            _make_node("a", status="promising"),
            _make_node("b", status="dead_end"),
            _make_node("c", status="baseline"),
        ]
        text = build_tree_text(nodes)
        assert "[OK+]" in text
        assert "[X]" in text
        assert "[BASE]" in text

    def test_tag_summary(self) -> None:
        nodes = [
            _make_node("a", tags=["gcg", "transfer"]),
            _make_node("b", tags=["gcg"]),
        ]
        text = build_tree_text(nodes)
        assert "Tags:" in text
        assert "gcg(2)" in text
        assert "transfer(1)" in text

    def test_missing_parent_nodes_are_promoted_to_roots(self) -> None:
        nodes = [
            _make_node("root", title="Root"),
            _make_node("child", title="Child", parents=["missing"]),
        ]

        text = build_tree_text(nodes)
        assert "[WIP] Root" in text
        assert "[WIP] Child" in text
        assert "Orphans:" not in text

    def test_duplicate_root_ids_are_rendered_once(self) -> None:
        nodes = [
            _make_node("dup", title="First"),
            _make_node("dup", title="Second"),
        ]

        text = build_tree_text(nodes)
        assert "[WIP] First" in text
        assert "Second" not in text
        assert "Warnings:" in text
        assert "Duplicate id" in text

    def test_cycle_graph_emits_orphans_and_warning(self) -> None:
        nodes = [
            _make_node("a", parents=["b"]),
            _make_node("b", parents=["a"]),
        ]

        text = build_tree_text(nodes)
        assert "Orphans:" in text
        assert "[WIP] a" in text
        assert "[WIP] b" in text
        assert "Cycle detected in experiment graph" in text


class TestBuildMermaid:
    """Tests for build_mermaid()."""

    def test_empty(self) -> None:
        output = build_mermaid([])
        assert "flowchart TD" in output
        assert "No experiments found" in output

    def test_build_mermaid_output(self) -> None:
        nodes = [
            _make_node("a", title="Root Exp", status="promising"),
            _make_node("b", title="Child Exp", parents=["a"]),
        ]
        output = build_mermaid(nodes)
        assert "flowchart TD" in output
        assert 'a["Root Exp"]' in output
        assert 'b["Child Exp"]' in output
        assert "a --> b" in output

    def test_mermaid_sanitizes_ids(self) -> None:
        """Hyphens and dots in IDs are replaced with underscores."""
        nodes = [
            _make_node("gan-loop-v3", title="GAN v3"),
            _make_node("exp.v2", title="Exp v2", parents=["gan-loop-v3"]),
        ]
        output = build_mermaid(nodes)
        assert "gan_loop_v3" in output
        assert "exp_v2" in output
        assert "gan_loop_v3 --> exp_v2" in output
        # Raw IDs with hyphens/dots should NOT appear as Mermaid node IDs
        assert "gan-loop-v3[" not in output
        assert "exp.v2[" not in output

    def test_mermaid_style_classes(self) -> None:
        nodes = [
            _make_node("a", status="promising"),
            _make_node("b", status="dead_end"),
        ]
        output = build_mermaid(nodes)
        assert "style a fill:#dfd" in output
        assert "style b fill:#fdd" in output


class TestValidateGraph:
    """Tests for graph validation warnings."""

    def test_dangling_parent_warning(self) -> None:
        nodes = [
            _make_node("a", parents=["nonexistent"]),
        ]
        warnings = _validate_graph(nodes)
        assert any("Dangling parent" in w for w in warnings)
        assert any("nonexistent" in w for w in warnings)

    def test_duplicate_id_warning(self) -> None:
        nodes = [
            _make_node("a", title="First"),
            _make_node("a", title="Second"),
        ]
        warnings = _validate_graph(nodes)
        assert any("Duplicate id" in w for w in warnings)

    def test_cycle_warning(self) -> None:
        nodes = [
            _make_node("a", parents=["b"]),
            _make_node("b", parents=["a"]),
        ]
        warnings = _validate_graph(nodes)
        assert any("Cycle" in w for w in warnings)

    def test_no_warnings_clean_graph(self) -> None:
        nodes = [
            _make_node("a"),
            _make_node("b", parents=["a"]),
        ]
        warnings = _validate_graph(nodes)
        assert warnings == []


class TestSanitizeMermaidId:
    """Tests for Mermaid ID sanitization."""

    def test_hyphens_replaced(self) -> None:
        assert _sanitize_mermaid_id("gan-loop-v3") == "gan_loop_v3"

    def test_dots_replaced(self) -> None:
        assert _sanitize_mermaid_id("exp.v2") == "exp_v2"

    def test_clean_id_unchanged(self) -> None:
        assert _sanitize_mermaid_id("simple_id") == "simple_id"

    def test_spaces_replaced(self) -> None:
        assert _sanitize_mermaid_id("my experiment") == "my_experiment"


class TestFilterNodes:
    """Tests for the private node filtering helper."""

    def test_filters_by_status_and_tag(self) -> None:
        nodes = [
            _make_node("a", status="promising", tags=["gcg"]),
            _make_node("b", status="dead_end", tags=["transfer"]),
            _make_node("c", status="promising", tags=["transfer"]),
        ]

        assert [node.id for node in _filter_nodes(nodes, status="promising")] == [
            "a",
            "c",
        ]
        assert [node.id for node in _filter_nodes(nodes, tag="transfer")] == [
            "b",
            "c",
        ]
        assert [node.id for node in _filter_nodes(
            nodes,
            status="promising",
            tag="transfer",
        )] == ["c"]


class TestMainEntrypoint:
    """Tests for the tree CLI entrypoint."""

    def test_main_mermaid_prints_warnings_to_stderr(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (tmp_path / "bad_meta.toml").write_text(
            '[meta]\nstatus = "bogus"\n',
        )

        tree_main([str(tmp_path), "--format", "mermaid"])
        captured = capsys.readouterr()
        assert "flowchart TD" in captured.out
        assert "bad_meta.toml" in captured.err

    def test_module_execution_runs_main(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (tmp_path / "meta.toml").write_text(
            '[meta]\nstatus = "baseline"\n',
        )

        monkeypatch.setattr(
            sys,
            "argv",
            ["python", str(tmp_path)],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            runpy.run_module("vauban.tree", run_name="__main__")
        captured = capsys.readouterr()
        assert "[BASE]" in captured.out


class TestDiscoveryWarningsInTree:
    """Test that discovery warnings appear in tree text output."""

    def test_discovery_warnings_shown(self) -> None:
        nodes = [_make_node("a")]
        text = build_tree_text(
            nodes,
            discovery_warnings=["bad_meta.toml: invalid [meta]: bad status"],
        )
        assert "Warnings:" in text
        assert "bad_meta.toml" in text


def test_tree_subcommand_renders_text_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "baseline.toml").write_text(
        '[meta]\n'
        'title = "Baseline"\n'
        'status = "baseline"\n'
        '[model]\n'
        'path = "test"\n',
    )

    monkeypatch.setattr(sys, "argv", ["vauban", "tree", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        cli_main()
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "[BASE] Baseline" in captured.out

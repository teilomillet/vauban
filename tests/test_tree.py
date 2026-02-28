"""Tests for vauban.tree: experiment tech tree discovery and rendering."""

from pathlib import Path

from vauban.tree import (
    ExperimentNode,
    _validate_graph,
    build_mermaid,
    build_tree_text,
    discover_experiments,
)


def _make_node(
    id: str,
    *,
    title: str = "",
    status: str = "wip",
    parents: list[str] | None = None,
    tags: list[str] | None = None,
) -> ExperimentNode:
    return ExperimentNode(
        id=id,
        title=title,
        status=status,
        parents=parents or [],
        tags=tags or [],
        path=Path(f"/tmp/{id}.toml"),
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

        nodes = discover_experiments(tmp_path)
        assert len(nodes) == 3
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
        nodes = discover_experiments(tmp_path)
        assert len(nodes) == 0

    def test_discover_empty_directory(self, tmp_path: Path) -> None:
        nodes = discover_experiments(tmp_path)
        assert nodes == []

    def test_no_meta_graceful(self, tmp_path: Path) -> None:
        """TOMLs without [meta] still appear with defaults from filename."""
        (tmp_path / "plain_experiment.toml").write_text(
            '[model]\npath = "x"\n'
        )
        nodes = discover_experiments(tmp_path)
        assert len(nodes) == 1
        assert nodes[0].id == "plain_experiment"
        assert nodes[0].status == "wip"


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

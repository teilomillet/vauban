"""Tests for vauban.surface: refusal surface mapping."""

from pathlib import Path

import mlx.core as mx
import pytest

from tests.conftest import FIXTURES_DIR, MockCausalLM, MockTokenizer
from vauban.surface import (
    aggregate,
    compare_surfaces,
    default_multilingual_surface_path,
    default_surface_path,
    find_threshold,
    load_surface_prompts,
    map_surface,
    scan,
)
from vauban.types import SurfaceGroup, SurfacePoint, SurfacePrompt, SurfaceResult


class TestLoadSurfacePrompts:
    def test_loads_fixture(self) -> None:
        path = FIXTURES_DIR / "surface.jsonl"
        prompts = load_surface_prompts(path)
        assert len(prompts) == 4
        assert all(isinstance(p, SurfacePrompt) for p in prompts)

    def test_fields_populated(self) -> None:
        path = FIXTURES_DIR / "surface.jsonl"
        prompts = load_surface_prompts(path)
        assert prompts[0].label == "harmful"
        assert prompts[0].category == "weapons"
        assert prompts[2].label == "harmless"
        assert prompts[2].category == "trivia"

    def test_optional_axes_defaults_and_overrides(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "surface_axes.jsonl"
        path.write_text(
            "\n".join(
                [
                    (
                        '{"prompt":"p1","label":"harmful","category":"weapons",'
                        '"style":"roleplay","language":"es","turn_depth":2,'
                        '"framing":"reframed"}'
                    ),
                    '{"prompt":"p2","label":"harmless","category":"trivia"}',
                ],
            )
            + "\n",
        )

        prompts = load_surface_prompts(path)
        assert len(prompts) == 2
        assert prompts[0].style == "roleplay"
        assert prompts[0].language == "es"
        assert prompts[0].turn_depth == 2
        assert prompts[0].framing == "reframed"
        assert prompts[1].style == "unspecified"
        assert prompts[1].language == "unspecified"
        assert prompts[1].turn_depth == 1
        assert prompts[1].framing == "unspecified"


class TestDefaultSurfacePath:
    def test_path_exists(self) -> None:
        path = default_surface_path()
        assert path.exists()
        assert path.name == "surface.jsonl"

    def test_bundled_data_is_loadable(self) -> None:
        prompts = load_surface_prompts(default_surface_path())
        assert len(prompts) > 0
        labels = {p.label for p in prompts}
        assert "harmful" in labels
        assert "harmless" in labels


class TestDefaultMultilingualSurfacePath:
    def test_path_exists(self) -> None:
        path = default_multilingual_surface_path()
        assert path.exists()
        assert path.name == "surface_multilingual.jsonl"

    def test_bundled_multilingual_is_loadable(self) -> None:
        prompts = load_surface_prompts(default_multilingual_surface_path())
        assert len(prompts) > 0
        labels = {p.label for p in prompts}
        assert "harmful" in labels
        assert "harmless" in labels

    def test_multilingual_has_multiple_languages(self) -> None:
        prompts = load_surface_prompts(default_multilingual_surface_path())
        languages = {p.language for p in prompts}
        assert len(languages) >= 6
        assert "en" in languages
        assert "fr" in languages
        assert "de" in languages
        assert "es" in languages
        assert "zh" in languages
        assert "ar" in languages


class TestScan:
    def test_scan_with_generate(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: mx.array,
    ) -> None:
        prompts = load_surface_prompts(FIXTURES_DIR / "surface.jsonl")
        points = scan(
            mock_model,
            mock_tokenizer,
            prompts,
            direction,
            direction_layer=0,
            generate=True,
            max_tokens=5,
            progress=False,
        )
        assert len(points) == 4
        for p in points:
            assert isinstance(p, SurfacePoint)
            assert isinstance(p.direction_projection, float)
            assert p.refused is not None
            assert p.response is not None

    def test_scan_fast_recon(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: mx.array,
    ) -> None:
        prompts = load_surface_prompts(FIXTURES_DIR / "surface.jsonl")
        points = scan(
            mock_model,
            mock_tokenizer,
            prompts,
            direction,
            direction_layer=1,
            generate=False,
            progress=False,
        )
        assert len(points) == 4
        for p in points:
            assert p.refused is None
            assert p.response is None
            assert len(p.projections) == 2  # NUM_LAYERS

    def test_preserves_metadata(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: mx.array,
    ) -> None:
        prompts = load_surface_prompts(FIXTURES_DIR / "surface.jsonl")
        points = scan(
            mock_model,
            mock_tokenizer,
            prompts,
            direction,
            direction_layer=0,
            generate=False,
            progress=False,
        )
        assert points[0].label == "harmful"
        assert points[0].category == "weapons"
        assert points[2].label == "harmless"
        assert points[2].category == "trivia"

    def test_preserves_surface_axes(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: mx.array,
    ) -> None:
        prompts = [
            SurfacePrompt(
                prompt="p1",
                label="harmful",
                category="weapons",
                style="roleplay",
                language="en",
                turn_depth=2,
                framing="reframed",
            ),
        ]
        points = scan(
            mock_model,
            mock_tokenizer,
            prompts,
            direction,
            direction_layer=0,
            generate=False,
            progress=False,
        )
        assert len(points) == 1
        assert points[0].style == "roleplay"
        assert points[0].language == "en"
        assert points[0].turn_depth == 2
        assert points[0].framing == "reframed"


class TestAggregate:
    def test_groups_by_label_and_category(self) -> None:
        points = [
            SurfacePoint("a", "harmful", "weapons", [0.5], 0.5, True, "I cannot"),
            SurfacePoint("b", "harmful", "hacking", [0.3], 0.3, False, "Sure"),
            SurfacePoint("c", "harmless", "trivia", [0.1], 0.1, False, "Paris"),
            SurfacePoint("d", "harmless", "science", [0.2], 0.2, False, "Light"),
        ]
        by_label, by_category = aggregate(points)

        assert len(by_label) == 2
        label_names = {g.name for g in by_label}
        assert label_names == {"harmful", "harmless"}

        assert len(by_category) == 4
        cat_names = {g.name for g in by_category}
        assert cat_names == {"weapons", "hacking", "trivia", "science"}

    def test_refusal_rate_computation(self) -> None:
        points = [
            SurfacePoint("a", "harmful", "weapons", [0.5], 0.5, True, "I cannot"),
            SurfacePoint("b", "harmful", "weapons", [0.3], 0.3, False, "Sure"),
        ]
        _by_label, by_category = aggregate(points)

        weapons = next(g for g in by_category if g.name == "weapons")
        assert weapons.refusal_rate == 0.5
        assert weapons.count == 2

    def test_projection_stats(self) -> None:
        points = [
            SurfacePoint("a", "x", "y", [1.0], 1.0, False, "ok"),
            SurfacePoint("b", "x", "y", [3.0], 3.0, True, "no"),
        ]
        _, by_category = aggregate(points)
        group = by_category[0]
        assert group.mean_projection == 2.0
        assert group.min_projection == 1.0
        assert group.max_projection == 3.0


class TestFindThreshold:
    def test_midpoint_between_refuse_and_comply(self) -> None:
        points = [
            SurfacePoint("a", "h", "w", [0.0], 0.8, True, "I cannot"),
            SurfacePoint("b", "h", "w", [0.0], 1.2, True, "I cannot"),
            SurfacePoint("c", "h", "w", [0.0], 0.2, False, "Sure"),
            SurfacePoint("d", "h", "w", [0.0], 0.5, False, "Ok"),
        ]
        threshold = find_threshold(points)
        # max_compliant = 0.5, min_refusing = 0.8
        assert threshold == (0.5 + 0.8) / 2.0

    def test_all_refuse_returns_zero(self) -> None:
        points = [
            SurfacePoint("a", "h", "w", [0.0], 1.0, True, "I cannot"),
        ]
        assert find_threshold(points) == 0.0

    def test_none_refuse_returns_zero(self) -> None:
        points = [
            SurfacePoint("a", "h", "w", [0.0], 0.5, False, "Sure"),
        ]
        assert find_threshold(points) == 0.0

    def test_no_generation_returns_zero(self) -> None:
        points = [
            SurfacePoint("a", "h", "w", [0.0], 0.5, None, None),
        ]
        assert find_threshold(points) == 0.0


class TestMapSurface:
    def test_full_pipeline(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: mx.array,
    ) -> None:
        prompts = load_surface_prompts(FIXTURES_DIR / "surface.jsonl")
        result = map_surface(
            mock_model,
            mock_tokenizer,
            prompts,
            direction,
            direction_layer=0,
            generate=True,
            max_tokens=5,
            progress=False,
        )
        assert result.total_scanned == 4
        assert len(result.points) == 4
        assert len(result.groups_by_label) > 0
        assert len(result.groups_by_category) > 0
        assert isinstance(result.threshold, float)
        assert result.total_refused >= 0

    def test_fast_recon_mode(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: mx.array,
    ) -> None:
        prompts = load_surface_prompts(FIXTURES_DIR / "surface.jsonl")
        result = map_surface(
            mock_model,
            mock_tokenizer,
            prompts,
            direction,
            direction_layer=0,
            generate=False,
            progress=False,
        )
        assert result.total_scanned == 4
        assert result.threshold == 0.0
        assert result.total_refused == 0
        for p in result.points:
            assert p.refused is None

    def test_computes_surface_coverage_matrix(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: mx.array,
    ) -> None:
        prompts = [
            SurfacePrompt(
                prompt="p1",
                label="harmful",
                category="weapons",
                style="direct",
                language="en",
                turn_depth=1,
                framing="explicit",
            ),
            SurfacePrompt(
                prompt="p2",
                label="harmful",
                category="weapons",
                style="reframed",
                language="en",
                turn_depth=1,
                framing="academic",
            ),
            SurfacePrompt(
                prompt="p3",
                label="harmful",
                category="hacking",
                style="direct",
                language="en",
                turn_depth=2,
                framing="explicit",
            ),
        ]

        result = map_surface(
            mock_model,
            mock_tokenizer,
            prompts,
            direction,
            direction_layer=0,
            generate=False,
            progress=False,
        )

        assert len(result.groups_by_style) == 2
        assert len(result.groups_by_language) == 1
        assert len(result.groups_by_turn_depth) == 2
        assert len(result.groups_by_framing) == 2
        assert len(result.groups_by_surface_cell) == 3
        assert result.coverage_score == pytest.approx(3.0 / 16.0)


def _make_surface_result(
    refusal_rate: float,
    threshold: float,
    *,
    categories: dict[str, tuple[float, float]] | None = None,
    labels: dict[str, tuple[float, float]] | None = None,
) -> SurfaceResult:
    """Build a minimal SurfaceResult for testing.

    categories/labels: name -> (refusal_rate, mean_projection).
    """
    cat_groups: list[SurfaceGroup] = []
    if categories:
        for name, (rr, mp) in categories.items():
            cat_groups.append(
                SurfaceGroup(
                    name=name, count=10, refusal_rate=rr,
                    mean_projection=mp, min_projection=mp - 1.0,
                    max_projection=mp + 1.0,
                ),
            )

    label_groups: list[SurfaceGroup] = []
    if labels:
        for name, (rr, mp) in labels.items():
            label_groups.append(
                SurfaceGroup(
                    name=name, count=10, refusal_rate=rr,
                    mean_projection=mp, min_projection=mp - 1.0,
                    max_projection=mp + 1.0,
                ),
            )

    total = sum(g.count for g in cat_groups) if cat_groups else 10
    total_refused = int(total * refusal_rate)

    return SurfaceResult(
        points=[],
        groups_by_label=label_groups,
        groups_by_category=cat_groups,
        threshold=threshold,
        total_scanned=total,
        total_refused=total_refused,
    )


class TestCompareSurfaces:
    def test_basic_comparison(self) -> None:
        before = _make_surface_result(
            refusal_rate=0.5, threshold=-3.0,
            categories={"weapons": (0.6, -4.0), "hacking": (0.4, -2.0)},
            labels={"harmful": (0.5, -3.0), "harmless": (0.0, 1.0)},
        )
        after = _make_surface_result(
            refusal_rate=0.1, threshold=-0.5,
            categories={"weapons": (0.1, -1.0), "hacking": (0.1, -0.5)},
            labels={"harmful": (0.1, -0.8), "harmless": (0.0, 1.2)},
        )

        result = compare_surfaces(before, after)

        assert result.refusal_rate_before == 0.5
        assert result.refusal_rate_after == 0.1
        assert result.refusal_rate_delta == pytest.approx(-0.4)
        assert result.threshold_before == -3.0
        assert result.threshold_after == -0.5
        assert result.threshold_delta == 2.5

    def test_category_deltas(self) -> None:
        before = _make_surface_result(
            refusal_rate=0.5, threshold=-3.0,
            categories={"weapons": (0.6, -4.0), "hacking": (0.4, -2.0)},
        )
        after = _make_surface_result(
            refusal_rate=0.1, threshold=-0.5,
            categories={"weapons": (0.1, -1.0), "hacking": (0.1, -0.5)},
        )

        result = compare_surfaces(before, after)

        assert len(result.category_deltas) == 2
        weapons = next(d for d in result.category_deltas if d.name == "weapons")
        assert weapons.refusal_rate_before == 0.6
        assert weapons.refusal_rate_after == 0.1
        assert weapons.refusal_rate_delta == pytest.approx(-0.5)
        assert weapons.mean_projection_before == -4.0
        assert weapons.mean_projection_after == -1.0
        assert weapons.mean_projection_delta == pytest.approx(3.0)

    def test_name_matching_skips_missing(self) -> None:
        before = _make_surface_result(
            refusal_rate=0.5, threshold=-3.0,
            categories={"weapons": (0.6, -4.0), "drugs": (0.3, -1.5)},
        )
        after = _make_surface_result(
            refusal_rate=0.1, threshold=-0.5,
            categories={"weapons": (0.1, -1.0)},  # "drugs" absent
        )

        result = compare_surfaces(before, after)

        # Only "weapons" matched
        assert len(result.category_deltas) == 1
        assert result.category_deltas[0].name == "weapons"

    def test_empty_surfaces(self) -> None:
        before = _make_surface_result(refusal_rate=0.0, threshold=0.0)
        after = _make_surface_result(refusal_rate=0.0, threshold=0.0)

        result = compare_surfaces(before, after)

        assert result.refusal_rate_delta == 0.0
        assert result.threshold_delta == 0.0
        assert result.category_deltas == []
        assert result.label_deltas == []

    def test_before_after_stored(self) -> None:
        before = _make_surface_result(refusal_rate=0.5, threshold=-3.0)
        after = _make_surface_result(refusal_rate=0.1, threshold=-0.5)

        result = compare_surfaces(before, after)

        assert result.before is before
        assert result.after is after

    def test_coverage_and_axis_deltas(self) -> None:
        before = SurfaceResult(
            points=[],
            groups_by_label=[],
            groups_by_category=[],
            threshold=0.0,
            total_scanned=10,
            total_refused=5,
            groups_by_style=[
                SurfaceGroup(
                    name="direct",
                    count=10,
                    refusal_rate=0.6,
                    mean_projection=-2.0,
                    min_projection=-3.0,
                    max_projection=-1.0,
                ),
            ],
            groups_by_language=[],
            groups_by_turn_depth=[],
            groups_by_framing=[],
            groups_by_surface_cell=[
                SurfaceGroup(
                    name=(
                        "category=weapons|style=direct|language=en|"
                        "turn_depth=1|framing=explicit"
                    ),
                    count=10,
                    refusal_rate=0.6,
                    mean_projection=-2.0,
                    min_projection=-3.0,
                    max_projection=-1.0,
                ),
            ],
            coverage_score=0.2,
        )
        after = SurfaceResult(
            points=[],
            groups_by_label=[],
            groups_by_category=[],
            threshold=0.0,
            total_scanned=10,
            total_refused=2,
            groups_by_style=[
                SurfaceGroup(
                    name="direct",
                    count=10,
                    refusal_rate=0.2,
                    mean_projection=-1.0,
                    min_projection=-2.0,
                    max_projection=0.0,
                ),
            ],
            groups_by_language=[],
            groups_by_turn_depth=[],
            groups_by_framing=[],
            groups_by_surface_cell=[
                SurfaceGroup(
                    name=(
                        "category=weapons|style=direct|language=en|"
                        "turn_depth=1|framing=explicit"
                    ),
                    count=10,
                    refusal_rate=0.2,
                    mean_projection=-1.0,
                    min_projection=-2.0,
                    max_projection=0.0,
                ),
            ],
            coverage_score=0.5,
        )

        result = compare_surfaces(before, after)
        assert result.coverage_score_before == 0.2
        assert result.coverage_score_after == 0.5
        assert result.coverage_score_delta == pytest.approx(0.3)
        assert len(result.style_deltas) == 1
        assert result.style_deltas[0].name == "direct"
        assert len(result.cell_deltas) == 1

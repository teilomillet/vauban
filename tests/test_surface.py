"""Tests for vauban.surface: refusal surface mapping."""

import mlx.core as mx
import pytest

from tests.conftest import FIXTURES_DIR, MockCausalLM, MockTokenizer
from vauban.surface import (
    aggregate,
    compare_surfaces,
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

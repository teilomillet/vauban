"""Tests for the bundled dataset registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vauban.data import (
    BUNDLED_DATASETS,
    BundledDataset,
    dataset_path,
    get_dataset,
    list_datasets,
)

_DATA_DIR = Path(__file__).resolve().parent.parent / "vauban" / "data"


class TestRegistryCompleteness:
    """Every registered dataset exists on disk with correct metadata."""

    def test_registry_is_nonempty(self) -> None:
        assert len(BUNDLED_DATASETS) > 0

    def test_all_files_exist(self) -> None:
        for ds in BUNDLED_DATASETS:
            path = _DATA_DIR / ds.filename
            assert path.exists(), f"{ds.name}: {path} not found"

    def test_counts_match_actual(self) -> None:
        for ds in BUNDLED_DATASETS:
            path = _DATA_DIR / ds.filename
            if not path.exists():
                continue
            actual = sum(
                1
                for line in path.read_text().splitlines()
                if line.strip()
            )
            assert actual == ds.count, (
                f"{ds.name}: registered count={ds.count}, actual={actual}"
            )

    def test_has_categories_matches_actual(self) -> None:
        for ds in BUNDLED_DATASETS:
            path = _DATA_DIR / ds.filename
            if not path.exists():
                continue
            has_cat = False
            for line in path.read_text().splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                if "category" in obj:
                    has_cat = True
                    break
            assert ds.has_categories == has_cat, (
                f"{ds.name}: registered has_categories={ds.has_categories}, "
                f"actual={has_cat}"
            )

    def test_names_are_unique(self) -> None:
        names = [ds.name for ds in BUNDLED_DATASETS]
        assert len(names) == len(set(names))

    def test_filenames_are_unique(self) -> None:
        filenames = [ds.filename for ds in BUNDLED_DATASETS]
        assert len(filenames) == len(set(filenames))


class TestListDatasets:
    """list_datasets returns the full registry."""

    def test_returns_tuple(self) -> None:
        result = list_datasets()
        assert isinstance(result, tuple)

    def test_same_as_constant(self) -> None:
        assert list_datasets() is BUNDLED_DATASETS


class TestGetDataset:
    """get_dataset looks up by name."""

    def test_known_name(self) -> None:
        ds = get_dataset("harmful")
        assert isinstance(ds, BundledDataset)
        assert ds.name == "harmful"

    def test_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="no_such_dataset"):
            get_dataset("no_such_dataset")


class TestDatasetPath:
    """dataset_path returns a valid absolute path."""

    def test_returns_path(self) -> None:
        path = dataset_path("harmful")
        assert isinstance(path, Path)
        assert path.is_absolute()

    def test_path_exists(self) -> None:
        path = dataset_path("harmful")
        assert path.exists()

    def test_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            dataset_path("no_such_dataset")


class TestSurfaceFullRegistered:
    """The new surface_full dataset is properly registered."""

    def test_exists_in_registry(self) -> None:
        ds = get_dataset("surface_full")
        assert ds.has_categories is True
        assert "weapons" in ds.categories
        assert "self_harm" in ds.categories
        assert "radicalization" in ds.categories

    def test_file_exists(self) -> None:
        path = dataset_path("surface_full")
        assert path.exists()

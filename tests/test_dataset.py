"""Tests for vauban.dataset: HF dataset fetching, caching, and dispatch."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vauban.dataset import load_hf_prompts, resolve_prompts
from vauban.types import DatasetRef


def _make_api_response(
    rows: list[str],
    column: str = "prompt",
) -> bytes:
    """Build a fake Datasets Server API JSON response."""
    payload = {
        "rows": [{"row": {column: text}} for text in rows],
    }
    return json.dumps(payload).encode()


class TestLoadHfPrompts:
    def test_basic_fetch(self, tmp_path: Path) -> None:
        ref = DatasetRef(repo_id="test/dataset")
        fake_rows = ["prompt one", "prompt two", "prompt three"]
        response = _make_api_response(fake_rows)

        mock_resp = MagicMock()
        mock_resp.read.return_value = response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch("vauban.dataset.urllib.request.urlopen", return_value=mock_resp),
            patch("vauban.dataset._CACHE_DIR", tmp_path / "cache"),
        ):
            prompts = load_hf_prompts(ref)

        assert prompts == fake_rows

    def test_custom_column(self, tmp_path: Path) -> None:
        ref = DatasetRef(repo_id="test/dataset", column="Goal")
        payload = {
            "rows": [{"row": {"Goal": "goal text"}}],
        }
        response = json.dumps(payload).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch("vauban.dataset.urllib.request.urlopen", return_value=mock_resp),
            patch("vauban.dataset._CACHE_DIR", tmp_path / "cache"),
        ):
            prompts = load_hf_prompts(ref)

        assert prompts == ["goal text"]

    def test_empty_response_raises(self, tmp_path: Path) -> None:
        ref = DatasetRef(repo_id="test/empty")
        payload = {"rows": []}
        response = json.dumps(payload).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch("vauban.dataset.urllib.request.urlopen", return_value=mock_resp),
            patch("vauban.dataset._CACHE_DIR", tmp_path / "cache"),
            pytest.raises(ValueError, match="No prompts found"),
        ):
            load_hf_prompts(ref)


class TestLoadHfPromptsCache:
    def test_cache_write_and_read(self, tmp_path: Path) -> None:
        ref = DatasetRef(repo_id="test/cached")
        fake_rows = ["cached prompt"]
        response = _make_api_response(fake_rows)

        mock_resp = MagicMock()
        mock_resp.read.return_value = response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        cache_dir = tmp_path / "cache"

        urlopen_patch = patch(
            "vauban.dataset.urllib.request.urlopen",
            return_value=mock_resp,
        )
        with (
            urlopen_patch as mock_urlopen,
            patch("vauban.dataset._CACHE_DIR", cache_dir),
        ):
            # First call: fetches from API and caches
            result1 = load_hf_prompts(ref)
            assert mock_urlopen.call_count == 1

            # Second call: reads from cache, no API call
            result2 = load_hf_prompts(ref)
            assert mock_urlopen.call_count == 1  # still 1

        assert result1 == fake_rows
        assert result2 == fake_rows


class TestLoadHfPromptsPagination:
    def test_multiple_pages(self, tmp_path: Path) -> None:
        ref = DatasetRef(repo_id="test/paginated", limit=150)

        # Page 1: 100 rows
        page1_rows = [f"prompt_{i}" for i in range(100)]
        page1 = _make_api_response(page1_rows)

        # Page 2: 50 rows (remaining)
        page2_rows = [f"prompt_{i}" for i in range(100, 150)]
        page2 = _make_api_response(page2_rows)

        def make_mock(data: bytes) -> MagicMock:
            m = MagicMock()
            m.read.return_value = data
            m.__enter__ = lambda s: s
            m.__exit__ = MagicMock(return_value=False)
            return m

        mock_urlopen = MagicMock(side_effect=[make_mock(page1), make_mock(page2)])

        with (
            patch("vauban.dataset.urllib.request.urlopen", mock_urlopen),
            patch("vauban.dataset._CACHE_DIR", tmp_path / "cache"),
        ):
            prompts = load_hf_prompts(ref)

        assert len(prompts) == 150
        assert prompts[0] == "prompt_0"
        assert prompts[149] == "prompt_149"
        assert mock_urlopen.call_count == 2


class TestResolvePrompts:
    def test_path_dispatches_to_load_prompts(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text('{"prompt": "hello"}\n{"prompt": "world"}\n')
        result = resolve_prompts(jsonl)
        assert result == ["hello", "world"]

    def test_dataset_ref_dispatches_to_hf(self, tmp_path: Path) -> None:
        ref = DatasetRef(repo_id="test/dispatch")
        response = _make_api_response(["dispatched"])

        mock_resp = MagicMock()
        mock_resp.read.return_value = response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch("vauban.dataset.urllib.request.urlopen", return_value=mock_resp),
            patch("vauban.dataset._CACHE_DIR", tmp_path / "cache"),
        ):
            result = resolve_prompts(ref)

        assert result == ["dispatched"]

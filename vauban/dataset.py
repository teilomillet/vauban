# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace dataset fetching via the Datasets Server REST API."""

import json
import urllib.parse
import urllib.request
from pathlib import Path

from vauban._network import validate_http_url
from vauban.measure import load_prompts
from vauban.types import DatasetRef

_API_BASE = "https://datasets-server.huggingface.co/rows"
_PAGE_SIZE = 100
_CACHE_DIR = Path.home() / ".cache" / "vauban" / "datasets"


def load_hf_prompts(ref: DatasetRef) -> list[str]:
    """Fetch prompts from a HuggingFace dataset.

    Uses the Datasets Server REST API (no ``datasets`` library needed).
    Results are cached locally as JSONL in ``~/.cache/vauban/datasets/``.
    """
    cache_path = _cache_path_for(ref)
    if cache_path.exists():
        return _read_cache(cache_path)

    prompts = _fetch_all(ref)
    _write_cache(cache_path, prompts)
    return prompts


def resolve_prompts(source: Path | DatasetRef) -> list[str]:
    """Load prompts from a local JSONL file or a HuggingFace dataset reference."""
    if isinstance(source, DatasetRef):
        return load_hf_prompts(source)
    return load_prompts(source)


def _cache_path_for(ref: DatasetRef) -> Path:
    """Build the local cache path for a dataset reference."""
    safe_repo = ref.repo_id.replace("/", "__")
    config_part = ref.config or "default"
    limit_part = f"_limit{ref.limit}" if ref.limit is not None else ""
    filename = f"{config_part}__{ref.split}__{ref.column}{limit_part}.jsonl"
    return _CACHE_DIR / safe_repo / filename


def _fetch_all(ref: DatasetRef) -> list[str]:
    """Fetch all rows from the HF Datasets Server API, paginating as needed."""
    prompts: list[str] = []
    offset = 0
    target = ref.limit  # None means fetch everything

    while True:
        length = _PAGE_SIZE
        if target is not None:
            remaining = target - len(prompts)
            if remaining <= 0:
                break
            length = min(_PAGE_SIZE, remaining)

        rows = _fetch_page(ref, offset, length)
        if not rows:
            break

        for row in rows:
            row_data = row.get("row", {})
            if not isinstance(row_data, dict):
                continue
            value = row_data.get(ref.column)  # type: ignore[arg-type]
            if isinstance(value, str):
                prompts.append(value)

        if len(rows) < length:
            break
        offset += len(rows)

    if not prompts:
        msg = (
            f"No prompts found in column {ref.column!r} of "
            f"{ref.repo_id} (split={ref.split!r})"
        )
        raise ValueError(msg)

    return prompts


def _fetch_page(
    ref: DatasetRef,
    offset: int,
    length: int,
) -> list[dict[str, object]]:
    """Fetch a single page of rows from the Datasets Server API."""
    params = (
        f"dataset={_url_quote(ref.repo_id)}"
        f"&split={_url_quote(ref.split)}"
        f"&offset={offset}"
        f"&length={length}"
    )
    if ref.config is not None:
        params += f"&config={_url_quote(ref.config)}"

    url = f"{_API_BASE}?{params}"
    validate_http_url(url, context="datasets server URL")
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "vauban/0.1")

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())

    rows_data = data.get("rows")
    if not isinstance(rows_data, list):
        msg = f"Unexpected API response structure from {url}"
        raise ValueError(msg)

    return rows_data


def _read_cache(path: Path) -> list[str]:
    """Read cached prompts from a JSONL file."""
    prompts: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])
    return prompts


def _write_cache(path: Path, prompts: list[str]) -> None:
    """Write prompts to a JSONL cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for prompt in prompts:
            f.write(json.dumps({"prompt": prompt}) + "\n")


def _url_quote(s: str) -> str:
    """Percent-encode a string for use in URL query parameters."""
    return urllib.parse.quote(s, safe="")

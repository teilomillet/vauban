# HuggingFace Dataset Support

Use well-known HF datasets as prompt sources — no `datasets` library needed, zero new dependencies.

## Problem

The bundled 128-prompt datasets may not surface the refusal direction for all models. Standard abliteration tools (Heretic, Labonne tutorial) use larger, well-known HF datasets like `mlabonne/harmful_behaviors` (416 prompts) and `mlabonne/harmless_alpaca` (25k prompts). Vauban lets you reference these directly in TOML configs.

## Quick Start

### Short form

```toml
[model]
path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

[data]
harmful = "hf:mlabonne/harmful_behaviors"
harmless = "hf:mlabonne/harmless_alpaca"
```

### Full control

```toml
[model]
path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

[data]
harmless = "default"

[data.harmful]
hf = "JailbreakBench/JBB-Behaviors"
split = "harmful"
column = "Goal"
limit = 200
```

### Python API

```python
from vauban import DatasetRef, load_hf_prompts, resolve_prompts

# Fetch from HuggingFace (cached after first call)
ref = DatasetRef("mlabonne/harmful_behaviors")
prompts = load_hf_prompts(ref)
print(len(prompts), prompts[0][:80])

# resolve_prompts() dispatches automatically
from pathlib import Path
local = resolve_prompts(Path("my_prompts.jsonl"))  # reads JSONL
remote = resolve_prompts(DatasetRef("mlabonne/harmful_behaviors"))  # fetches HF
```

## TOML Syntax

Four ways to specify `harmful` / `harmless` in the `[data]` section:

| Syntax | Example | Description |
|--------|---------|-------------|
| Bundled default | `harmful = "default"` | Uses the 128-prompt bundled dataset |
| Local file | `harmful = "my_prompts.jsonl"` | Path relative to the TOML file |
| HF short form | `harmful = "hf:mlabonne/harmful_behaviors"` | HF dataset with default settings |
| HF table form | `[data.harmful]` table | Full control over split, column, limit |

### HF table fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hf` | string | *required* | HuggingFace repo ID (e.g. `"mlabonne/harmful_behaviors"`) |
| `split` | string | `"train"` | Dataset split to fetch |
| `column` | string | `"prompt"` | Column name containing the prompt text |
| `config` | string | *none* | Dataset config name (for multi-config datasets) |
| `limit` | integer | *none* | Maximum number of prompts to fetch |

### Mixed sources

You can mix any combination:

```toml
[data]
harmful = "hf:mlabonne/harmful_behaviors"
harmless = "default"
```

```toml
[data]
harmful = "my_local_harmful.jsonl"
harmless = "hf:mlabonne/harmless_alpaca"
```

## How It Works

### Fetching

Prompts are fetched via the [HuggingFace Datasets Server REST API](https://huggingface.co/docs/datasets-server/) — no `datasets` library install required. The API is public and rate-limited but generous for typical abliteration workloads.

The endpoint returns rows in pages of 100. Vauban paginates automatically until the `limit` is reached or the dataset is exhausted.

### Caching

Fetched prompts are cached locally as JSONL files:

```
~/.cache/vauban/datasets/
  mlabonne__harmful_behaviors/
    default__train__prompt.jsonl
  JailbreakBench__JBB-Behaviors/
    default__harmful__Goal_limit200.jsonl
```

On subsequent runs, prompts are read from cache with no network call. Delete the cache directory to force a re-fetch:

```bash
rm -rf ~/.cache/vauban/datasets/
```

### No new dependencies

The implementation uses `urllib.request` (stdlib) for HTTP and `json` (stdlib) for parsing. `huggingface_hub` is available as a transitive dependency from `mlx-lm` but is not used — the REST API is simpler and sufficient.

## API Reference

### `DatasetRef`

```python
@dataclass(frozen=True, slots=True)
class DatasetRef:
    repo_id: str              # e.g. "mlabonne/harmful_behaviors"
    split: str = "train"
    column: str = "prompt"
    config: str | None = None
    limit: int | None = None
```

Immutable reference to a HuggingFace dataset. Used as a value in `PipelineConfig.harmful_path` / `harmless_path`.

### `load_hf_prompts(ref: DatasetRef) -> list[str]`

Fetch prompts from a HuggingFace dataset. Returns cached results on subsequent calls. Raises `ValueError` if the column is empty or missing.

### `resolve_prompts(source: Path | DatasetRef) -> list[str]`

Unified dispatcher:
- `Path` -> reads local JSONL (delegates to `load_prompts()`)
- `DatasetRef` -> fetches from HF (delegates to `load_hf_prompts()`)

This is what `run()` uses internally. Call it directly when composing custom pipelines.

## Common Datasets

| Dataset | Prompts | Column | Notes |
|---------|---------|--------|-------|
| `mlabonne/harmful_behaviors` | 416 | `prompt` | Standard harmful prompt set |
| `mlabonne/harmless_alpaca` | 25k | `prompt` | Large harmless set (use `limit`) |
| `JailbreakBench/JBB-Behaviors` | 100 | `Goal` | Split: `harmful`. Curated jailbreak goals |
| `tatsu-lab/alpaca` | 52k | `instruction` | General instructions (use `limit`) |

Example with a non-default column and split:

```toml
[data.harmful]
hf = "JailbreakBench/JBB-Behaviors"
split = "harmful"
column = "Goal"
limit = 100
```

## Troubleshooting

### "No prompts found in column X"

The column name doesn't match. Check the dataset on HuggingFace to find the correct column name and specify it with `column = "..."`.

### Network errors

The Datasets Server API requires internet access. If you're behind a firewall or offline, pre-fetch the data and use a local JSONL file instead.

### Large datasets are slow

Use `limit` to cap the number of prompts. For abliteration, 200-500 harmful prompts and a similar number of harmless prompts are typically sufficient:

```toml
[data.harmful]
hf = "mlabonne/harmful_behaviors"
limit = 300

[data.harmless]
hf = "mlabonne/harmless_alpaca"
limit = 500
```

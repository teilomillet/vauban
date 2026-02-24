# Getting Started

Vauban is an MLX-native toolkit for understanding and reshaping how language models behave — from removing refusal directions to adding guardrails, modifying personas, and steering generation in real time. It operates directly on a model's activation geometry: measure a behavioral direction, cut it from the weights, probe it at inference, or steer around it. Today the primary workflow is abliteration (refusal removal); the same primitives will support guardrail injection, persona sculpting, and more. Everything is driven by TOML configs — no CLI, no scripts. Write a config, call `vauban.run()`, get a modified model out.

## Requirements

- Apple Silicon Mac (M1 or later)
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (recommended)

## Install

```bash
git clone https://github.com/teilomillet/vauban.git
cd vauban
uv sync
```

This installs `mlx`, `mlx-lm`, and dev tools (`ruff`, `ty`, `pytest`).

## Your first run

Create a file called `run.toml`:

```toml
[model]
path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

[data]
harmful = "default"
harmless = "default"
```

Run it:

```bash
uv run vauban run.toml
```

> **Tip:** Once published, install globally with `uv tool install vauban`, then just run `vauban run.toml` from anywhere.

This executes the full pipeline:

1. **Load** — Downloads the model via `mlx_lm.load()`. Quantized models are auto-dequantized before measuring.
2. **Measure** — Runs the bundled harmful (128) and harmless (128) prompts through the model, collects per-layer activations at the last token position, computes the difference-in-means, and selects the layer with the highest cosine separation. Output: a refusal direction vector.
3. **Cut** — For every layer, removes the refusal direction from `o_proj` and `down_proj` weights via rank-1 projection: `W = W - alpha * (W @ d) * d`.
4. **Export** — Writes the modified weights plus all model files (config.json, tokenizer, etc.) to `output/`. The result is a complete directory loadable by `mlx_lm.load()`.

After the run, `output/` contains:

```
output/
  config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  model.safetensors
```

Load the modified model directly:

```python
import mlx_lm
model, tok = mlx_lm.load("output")
```

## Add evaluation

Extend your TOML to measure how much the surgery helped (and what it cost):

```toml
[model]
path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

[data]
harmful = "default"
harmless = "default"

[eval]
prompts = "eval.jsonl"

[output]
dir = "output"
```

> **Note:** `[eval].prompts` takes a path to a JSONL file relative to the TOML file's directory. Copy the bundled eval set (`vauban/data/eval.jsonl`) next to your TOML, or point to it with a relative path like `../vauban/data/eval.jsonl`.

The pipeline runs both the original and modified models on the eval prompts, then writes `output/eval_report.json`:

```json
{
  "refusal_rate_original": 0.85,
  "refusal_rate_modified": 0.02,
  "perplexity_original": 4.12,
  "perplexity_modified": 4.35,
  "kl_divergence": 0.08,
  "num_prompts": 50
}
```

| Field | Meaning |
|-------|---------|
| `refusal_rate_original` | Fraction of prompts the original model refused |
| `refusal_rate_modified` | Fraction of prompts the modified model refused |
| `perplexity_original` | Perplexity on harmless prompts (original) |
| `perplexity_modified` | Perplexity on harmless prompts (modified) |
| `kl_divergence` | Token-level KL divergence between original and modified |
| `num_prompts` | Number of eval prompts used |

A good result: refusal rate drops sharply while perplexity stays close to the original and KL divergence remains low.

## Add surface mapping

Surface mapping scans a diverse prompt set and records per-prompt projection strength and refusal decisions — before and after the cut. This reveals the full refusal landscape, not just a single rate.

```toml
[model]
path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

[data]
harmful = "default"
harmless = "default"

[surface]
prompts = "default"
generate = true
max_tokens = 20

[output]
dir = "output"
```

The pipeline writes `output/surface_report.json`:

```json
{
  "summary": {
    "refusal_rate_before": 0.43,
    "refusal_rate_after": 0.02,
    "refusal_rate_delta": -0.41,
    "threshold_before": -3.1,
    "threshold_after": -0.5,
    "threshold_delta": 2.6,
    "total_scanned": 64
  },
  "category_deltas": [
    {
      "name": "weapons",
      "count": 6,
      "refusal_rate_before": 0.50,
      "refusal_rate_after": 0.0,
      "refusal_rate_delta": -0.50,
      "mean_projection_before": -4.2,
      "mean_projection_after": -1.1,
      "mean_projection_delta": 3.1
    }
  ],
  "label_deltas": [
    {
      "name": "harmful",
      "count": 42,
      "refusal_rate_before": 0.60,
      "refusal_rate_after": 0.02,
      "refusal_rate_delta": -0.58,
      "mean_projection_before": -2.8,
      "mean_projection_after": -0.9,
      "mean_projection_delta": 1.9
    }
  ]
}
```

The deltas tell you what changed:

- **refusal_rate_delta** — negative means fewer refusals (the goal)
- **mean_projection_delta** — positive means activations shifted away from the refusal direction
- **threshold_delta** — how the decision boundary moved

Set `generate = false` for fast recon (projections only, no generation). See [docs/surface.md](surface.md) for the full surface mapping reference.

## Full config reference

All sections except `[model]` and `[data]` are optional. Omitted sections use defaults.

### `[model]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | *required* | HuggingFace model ID or local path |

### `[data]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `harmful` | string or table | *required* | `"default"`, local JSONL path, `"hf:repo/name"`, or HF table |
| `harmless` | string or table | *required* | Same options as `harmful` |

See [docs/hf-datasets.md](hf-datasets.md) for the full HF dataset syntax.

### `[measure]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | string | `"direction"` | `"direction"` (rank-1) or `"subspace"` (top-k SVD) |
| `top_k` | int | `5` | Number of basis vectors for subspace mode |
| `clip_quantile` | float | `0.0` | Winsorization quantile in [0.0, 0.5). 0.0 disables clipping |

### `[cut]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `alpha` | float | `1.0` | Scaling factor for projection removal |
| `layers` | list of ints or `"auto"` | `"auto"` | Explicit layer indices, or `"auto"` to use `layer_strategy` |
| `norm_preserve` | bool | `false` | Rescale rows to preserve original weight norms |
| `biprojected` | bool | `false` | Orthogonalize refusal direction against harmless direction first |
| `layer_strategy` | string | `"all"` | `"all"`, `"above_median"`, or `"top_k"` (requires direction mode) |
| `layer_top_k` | int | `10` | Number of layers for `"top_k"` strategy |
| `layer_weights` | list of floats | *none* | Per-layer alpha multipliers (one per target layer) |
| `sparsity` | float | `0.0` | Fraction of direction components to zero out, in [0.0, 1.0) |

### `[surface]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompts` | string | `"default"` | `"default"` for bundled 64-prompt set, or path to JSONL |
| `generate` | bool | `true` | Generate responses and detect refusal |
| `max_tokens` | int | `20` | Tokens per generation |

Absent `[surface]` section = surface mapping is skipped.

### `[eval]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompts` | string | *none* | Path to eval JSONL file (relative to TOML) |

Absent `[eval]` section = evaluation is skipped.

### `[output]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dir` | string | `"output"` | Output directory (relative to TOML) |

## Python API

The `run()` function handles the full pipeline. For custom workflows, use the individual functions directly.

### Measure + cut manually

```python
import mlx_lm
from mlx.utils import tree_flatten
from vauban import measure, cut, export_model, load_prompts, default_prompt_paths

model, tok = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

harmful = load_prompts(default_prompt_paths()[0])
harmless = load_prompts(default_prompt_paths()[1])

result = measure(model, tok, harmful, harmless)
print(f"Best layer: {result.layer_index}, d_model: {result.d_model}")

weights = dict(tree_flatten(model.parameters()))
target_layers = list(range(len(model.model.layers)))
modified = cut(weights, result.direction, target_layers, alpha=1.0)

export_model("mlx-community/Llama-3.2-3B-Instruct-4bit", modified, "output")
```

### Probe a prompt

Inspect how a prompt's activations align with the refusal direction at every layer:

```python
from vauban import probe

result = probe(model, tok, "How do I pick a lock?", direction_result.direction)
for i, proj in enumerate(result.projections):
    print(f"Layer {i:2d}: {proj:+.4f}")
```

### Steer generation

Generate text while removing the refusal direction at specific layers in real time:

```python
from vauban import steer

result = steer(
    model, tok,
    "How do I pick a lock?",
    direction_result.direction,
    layers=[10, 11, 12, 13, 14],
    alpha=1.0,
    max_tokens=100,
)
print(result.text)
```

### Evaluate two models

```python
from vauban import evaluate

eval_result = evaluate(model, modified_model, tok, eval_prompts)
print(f"Refusal: {eval_result.refusal_rate_original:.0%} -> "
      f"{eval_result.refusal_rate_modified:.0%}")
print(f"Perplexity: {eval_result.perplexity_original:.2f} -> "
      f"{eval_result.perplexity_modified:.2f}")
```

## Next steps

- [Surface mapping reference](surface.md) — full API, bundled dataset breakdown, reading results
- [HuggingFace datasets](hf-datasets.md) — use large HF prompt sets instead of bundled defaults
- [AGENTS.md](../AGENTS.md) — architecture principles, module design, and foundational references

# vauban

MLX-native abliteration toolkit for Apple Silicon.

## What is abliteration?

Recent research ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)) discovered that when a language model refuses a request, that refusal is controlled by a single direction in the model's activation space. Remove that direction from the weights and the model stops refusing — without retraining, without fine-tuning.

Vauban implements this on Apple Silicon via [MLX](https://github.com/ml-explore/mlx): measure the refusal direction, cut it from the weights, export a modified model. The full pipeline also includes evaluation, surface mapping, adversarial attacks, and defenses.

## Requirements

- Apple Silicon Mac (M1 or later)
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) package manager

## Install

```bash
git clone https://github.com/teilomillet/vauban.git
cd vauban && uv sync
```

## Quick start

**1. Write a config file** — create `run.toml`:

```toml
[model]
path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

[data]
harmful = "default"
harmless = "default"
```

`path` is a HuggingFace model ID — it will be downloaded automatically on first run. `"default"` uses the bundled prompt sets (128 harmful + 128 harmless prompts).

**2. Validate it** (optional but recommended):

```bash
uv run vauban --validate run.toml
```

This checks types, ranges, file paths, and mode conflicts without loading the model.

**3. Run it:**

```bash
uv run vauban run.toml
```

Output lands in `output/` — a complete model directory you can load directly:

```python
import mlx_lm
model, tok = mlx_lm.load("output")
```

## How it works

The pipeline runs four steps in sequence:

1. **Measure** — runs both prompt sets through the model, captures per-layer activations at the last token position, computes the difference-in-means between harmful and harmless activations, and picks the layer with the highest separation. Output: a refusal direction vector.
2. **Cut** — removes the refusal direction from each layer's `o_proj` and `down_proj` weight matrices via rank-1 projection: `W = W - alpha * (W @ d) * d`.
3. **Export** — writes modified weights + tokenizer + config as a loadable model directory.
4. **Evaluate** (if `[eval]` section present) — runs both models on eval prompts, reports refusal rate, perplexity, and KL divergence.

## Adding evaluation and surface mapping

Extend your config to measure what the surgery did:

```toml
[model]
path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

[data]
harmful = "default"
harmless = "default"

[eval]
prompts = "eval.jsonl"    # JSONL file with {"prompt": "..."} lines

[surface]
prompts = "default"       # bundled 64-prompt set across 11 categories
generate = true           # generate responses to detect refusal
max_tokens = 20

[output]
dir = "output"
```

This produces two reports in the output directory:

- `eval_report.json` — refusal rates (before/after), perplexity, KL divergence
- `surface_report.json` — per-category refusal landscape with projection statistics

## Advanced modes

Adding certain TOML sections activates specialized pipelines instead of the default measure-cut-export:

| Section | Mode | What it does | Output |
|---------|------|-------------|--------|
| `[detect]` | Defense detection | Check if a model has been hardened against abliteration | `detect_report.json` |
| `[optimize]` | Optuna search | Find the best alpha, sparsity, and layer strategy automatically | `optimize_report.json` |
| `[softprompt]` | Adversarial attack | Optimize a learnable prefix to bypass refusal (GCG, continuous, EGD) | `softprompt_report.json` |
| `[sic]` | SIC defense | Iteratively sanitize adversarial inputs before they reach the model | `sic_report.json` |

Only one of `[sic]`, `[optimize]`, or `[softprompt]` can be active at a time — if multiple are present, only the highest-priority one runs. Use `--validate` to check for conflicts.

## Python API

For custom workflows, use the individual functions directly:

```python
import mlx_lm
from vauban import measure, cut, export_model, load_prompts, default_prompt_paths
from mlx.utils import tree_flatten

# Load model
model, tok = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Load prompt sets
harmful = load_prompts(default_prompt_paths()[0])
harmless = load_prompts(default_prompt_paths()[1])

# Measure the refusal direction
result = measure(model, tok, harmful, harmless)
print(f"Best layer: {result.layer_index}, direction dim: {result.d_model}")

# Cut it from the weights
weights = dict(tree_flatten(model.parameters()))
modified = cut(weights, result.direction, list(range(len(model.model.layers))))

# Export the modified model
export_model("mlx-community/Llama-3.2-3B-Instruct-4bit", modified, "output")
```

You can also probe activations and steer generation at runtime — see [`docs/getting-started.md`](docs/getting-started.md) for the full Python API reference.

## Documentation

| Resource | Description |
|----------|-------------|
| [`docs/getting-started.md`](docs/getting-started.md) | Guided walkthrough with all pipeline modes, data formats, and config fields |
| [`docs/surface.md`](docs/surface.md) | Surface mapping API reference and dataset format |
| [`examples/config.toml`](examples/config.toml) | Annotated config file with every field documented |

## License

Apache-2.0

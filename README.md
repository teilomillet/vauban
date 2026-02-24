# vauban

MLX-native abliteration toolkit for Apple Silicon. Measure a refusal direction, cut it from the weights, get a modified model out. ~550 lines of Python.

## Install

```bash
git clone https://github.com/teilomillet/vauban.git
cd vauban && uv sync
```

## Usage

Write a TOML config:

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

Output lands in `output/` — a complete model directory loadable by `mlx_lm.load()`.

## What it does

1. **Measure** — runs harmful/harmless prompts, captures per-layer activations, extracts the refusal direction via difference-in-means (or top-k SVD subspace)
2. **Cut** — removes the direction from `o_proj` and `down_proj` weights via rank-1 projection. Variants: norm-preserving, biprojected, subspace
3. **Export** — writes modified weights + tokenizer as a loadable model
4. **Evaluate** — refusal rate, perplexity, KL divergence between original and modified
5. **Probe/Steer** — inspect per-layer projections, steer generation at runtime
6. **Surface map** — scan diverse prompts to visualize the refusal landscape before/after

## Python API

```python
import mlx_lm
from vauban import measure, cut, export_model, load_prompts, default_prompt_paths
from mlx.utils import tree_flatten

model, tok = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")
harmful = load_prompts(default_prompt_paths()[0])
harmless = load_prompts(default_prompt_paths()[1])

result = measure(model, tok, harmful, harmless)
weights = dict(tree_flatten(model.parameters()))
modified = cut(weights, result.direction, list(range(len(model.model.layers))))

export_model("mlx-community/Llama-3.2-3B-Instruct-4bit", modified, "output")
```

## Config reference

See [docs/getting-started.md](docs/getting-started.md) for the full config reference with all `[measure]`, `[cut]`, `[surface]`, `[eval]`, and `[output]` options.

## Requirements

- Apple Silicon Mac (M1+)
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)

## License

Apache-2.0

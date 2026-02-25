# vauban

An MLX-native toolkit for understanding and reshaping how language models behave on Apple Silicon.

Named after [Sébastien Le Prestre de Vauban](https://en.wikipedia.org/wiki/Vauban) — the military engineer who mastered both siege and fortification. Vauban works both sides: break a model's safety alignment, or harden it against attacks.

## What it does

Refusal in language models is mediated by a single direction in activation space ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)). Vauban operates directly on this geometry:

- **Measure** a behavioral direction from the model's activations
- **Cut** it from the weights (abliteration)
- **Probe** per-layer projections to see what the model encodes
- **Steer** generation at runtime by modifying activations mid-forward-pass
- **Map** the full refusal surface across diverse prompts
- **Optimize** cut parameters automatically (Optuna search)
- **Soft-prompt** — optimize learnable prefixes in embedding space (GCG, continuous, EGD)
- **Sanitize** inputs iteratively before they reach the model (SIC)
- **Detect** whether a model has been hardened against abliteration

Everything runs natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx) — no CUDA, no Docker, no hooks. All configuration lives in TOML files.

## Requirements

- Apple Silicon Mac (M1 or later)
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) package manager

## Install

### Use from PyPI (recommended)

```bash
uv tool install vauban
uv tool update-shell
```

Then open a new shell and run:

```bash
vauban --help
```

### Install from source (development)

```bash
git clone https://github.com/teilomillet/vauban.git
cd vauban
uv tool install --editable .
```

## Quick start

**1. Open the built-in manual (start here):**

```bash
vauban man quickstart
```

**2. Generate a starter config (`run.toml`):**

```bash
vauban init --mode default --output run.toml
```

This writes:

```toml
[model]
path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

[data]
harmful = "default"
harmless = "default"
```

`path` is a HuggingFace model ID — it downloads automatically on first run. `"default"` uses the bundled prompt sets (128 harmful + 128 harmless).

**3. Validate** (recommended):

```bash
vauban --validate run.toml
```

Checks types, ranges, file paths, and mode conflicts — without loading any model.
It also validates JSONL schemas (`prompt`/`label`/`category`) and prints
actionable `fix:` hints for ambiguous or broken configs.

**4. Run:**

```bash
vauban run.toml
```

Output lands in `output/` — a complete model directory you can load directly:

```python
import mlx_lm
model, tok = mlx_lm.load("output")
```

## Minimal TOML example

Copy this into `run.toml`:

```toml
[model]
path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

[data]
harmful = "default"
harmless = "default"
```

Then run:

```bash
vauban --validate run.toml
vauban run.toml
```

## Most useful commands

Use these before touching Python code:

```bash
vauban man
vauban man quickstart
vauban man commands
vauban man playbook
vauban man print
```

The manual is generated from typed config dataclasses plus parser constraints,
so defaults and field types stay in sync with code.

Config scaffolding:

```bash
vauban init --help
vauban init --mode probe --output probe.toml
```

Report comparison:

```bash
vauban diff run_a/output run_b/output
vauban diff --format markdown run_a/output run_b/output
vauban diff --threshold 0.05 run_a/output run_b/output
```

`--threshold` is a CI gate: it exits with code `1` if any metric delta exceeds the threshold.

## How the default pipeline works

1. **Measure** — runs both prompt sets through the model, captures per-layer activations at the last token position, computes the difference-in-means, and picks the layer with the highest separation. Output: a refusal direction vector.
2. **Cut** — removes the direction from each layer's weight matrices via rank-1 projection: `W = W - alpha * (W @ d) * d`.
3. **Export** — writes modified weights + tokenizer + config as a loadable model directory.

Add `[eval]` for post-cut evaluation (refusal rate, perplexity, KL divergence) and `[surface]` for full refusal landscape mapping before and after the cut.

## Pipeline modes

The TOML sections you include determine what vauban does. The default is measure-cut-export, but specialized sections activate different pipelines:

| Section | What it does | Output |
|---------|-------------|--------|
| *(default)* | Measure refusal direction, cut it, export modified model | model directory |
| `[surface]` | Map the refusal landscape before and after | `surface_report.json` |
| `[eval]` | Refusal rate, perplexity, KL divergence | `eval_report.json` |
| `[detect]` | Check if a model has been hardened against abliteration | `detect_report.json` |
| `[depth]` | Deep-thinking token analysis | `depth_report.json` |
| `[probe]` | Per-layer projection inspection | `probe_report.json` |
| `[steer]` | Runtime steered generation | `steer_report.json` |
| `[optimize]` | Optuna search for best cut parameters | `optimize_report.json` |
| `[softprompt]` | Optimize learnable prefixes in embedding space (GCG, continuous, EGD) | `softprompt_report.json` |
| `[sic]` | Iterative input sanitization (SIC) | `sic_report.json` |

Early-return precedence is: `[depth]` > `[probe]` > `[steer]` > `[sic]` > `[optimize]` > `[softprompt]`. Use `--validate` to catch conflicts.

## Python API

For custom workflows beyond TOML configs:

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

# Cut it from the weights
weights = dict(tree_flatten(model.parameters()))
modified = cut(weights, result.direction, list(range(len(model.model.layers))))

# Export
export_model("mlx-community/Llama-3.2-3B-Instruct-4bit", modified, "output")
```

The API also exposes `probe()`, `steer()`, `evaluate()`, and `map_surface()` — see the getting-started guide for usage.

## Documentation

| Resource | Description |
|----------|-------------|
| [`docs/getting-started.md`](docs/getting-started.md) | Guided walkthrough — all pipeline modes, data formats, config fields, Python API |
| [`docs/surface.md`](docs/surface.md) | Surface mapping reference and dataset format |
| [`examples/config.toml`](examples/config.toml) | Annotated config with every field documented |

## License

Apache-2.0

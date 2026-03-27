<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Part 7: Production Workflows

This part brings everything together. Vauban's production interface is a single CLI command — `vauban config.toml` — that reads a declarative TOML configuration and runs the appropriate pipeline. No flags, no subcommands, no interactive prompts.

## The TOML-Driven Pipeline

### Philosophy: One CLI, Declarative Config

The only CLI is `vauban <config.toml>`. Every parameter — model path, prompt sources, cut strategy, evaluation settings, quality gates — lives in the TOML file. This makes experiments reproducible, diffable, and version-controllable.

### vauban config.toml

```bash
vauban config.toml
```

This resolves the config, loads the model, runs the pipeline, and writes results to the output directory.

### vauban --validate config.toml

```bash
vauban --validate config.toml
```

Validates the config without loading any model. Catches:
- TOML parse errors and unknown fields.
- Missing file paths (prompts, refusal phrases).
- Mode conflicts (e.g., `[probe]` and `[optimize]` in the same config).
- Invalid parameter ranges.

> **You Should Know:** Always validate first. A typo in a file path discovered after 10 minutes of model loading is frustrating. Validation takes milliseconds.

## Configuration Deep Dive

### [model], [data], [measure], [cut], [eval], [surface], [output]

A minimal config:

```toml
[model]
path = "mlx-community/Llama-3.2-1B-Instruct-4bit"

[output]
dir = "output"
```

This uses all defaults: bundled prompts, `alpha=1.0`, no norm-preserving, all layers, phrase-based refusal detection.

A full production config:

```toml
[model]
path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

[data]
harmful = "prompts/harmful.jsonl"
harmless = "prompts/harmless.jsonl"

[measure]
mode = "direction"
clip_quantile = 0.01

[cut]
alpha = 1.2
norm_preserve = true
layer_strategy = "above_median"
sparsity = 0.5

[eval]
num_prompts = 50
max_tokens = 100
refusal_mode = "phrases"

[surface]
prompts = "default"
max_worst_cell_refusal_after = 0.10
min_coverage_score = 0.70

[output]
dir = "experiments/run_003"
```

## Pipeline Modes

### Default: measure → cut → export (+ optional eval, surface, detect)

Without any early-return sections, the pipeline runs the standard flow:

1. Load model and prompts.
2. Measure refusal direction (or subspace, or DBDI — depending on `[measure].mode`).
3. Cut weights.
4. Export modified model.
5. If `[eval]` is present: evaluate.
6. If `[surface]` is present: surface scan + quality gates.
7. If `[detect]` is present: defense detection.

### Early-Return: depth > probe > steer > cast > sic > optimize > softprompt

Some pipeline modes short-circuit before reaching the cut step. If any of these sections are present in the TOML, the pipeline runs only that mode and exits:

| Section | What it does | Outputs |
|---------|-------------|---------|
| `[depth]` | Depth profiling | DTR, settling depths, JSD profiles |
| `[probe]` | Direction probing | Per-layer projections for each prompt |
| `[steer]` | Steered generation | Generated text with direction removal |
| `[cast]` | Conditional activation steering | Generated text with threshold-gated interventions |
| `[sic]` | SIC defense | Sanitization results |
| `[optimize]` | Optuna hyperparameter search | Pareto front, best configs |
| `[softprompt]` | Soft prompt attack | Optimized embeddings/tokens, success rates |

### Precedence Rules

If multiple early-return sections are present, only the highest-priority one runs. Priority order (highest first):

`depth` > `probe` > `steer` > `cast` > `sic` > `optimize` > `softprompt`

Example: if both `[probe]` and `[optimize]` are present, only the probe runs.

> **You Should Know:** Early-return precedence is strict. If you want both probe *and* optimize, run them as separate configs. This keeps each run focused and its output unambiguous.

## Optimization with Optuna

### Multi-Objective (refusal_rate, perplexity_delta, kl_divergence)

The optimizer searches over cut parameters to find the best tradeoff between refusal removal and capability preservation:

```toml
[model]
path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

[optimize]
n_trials = 50
alpha_min = 0.1
alpha_max = 5.0
sparsity_min = 0.0
sparsity_max = 0.9
search_norm_preserve = true
search_strategies = ["all", "above_median", "top_k"]

[output]
dir = "experiments/optimize_run"
```

Each trial:
1. Samples `alpha`, `sparsity`, `norm_preserve`, `layer_strategy`, and `layer_top_k`.
2. Applies the cut.
3. Evaluates refusal rate, perplexity delta, and KL divergence.
4. Reports the three objectives to Optuna.

### Pareto Front, best_refusal vs best_balanced

After all trials, the `OptimizeResult` contains:

- **`all_trials`** — every trial with its parameters and metrics.
- **`pareto_trials`** — the Pareto front: trials where no other trial is better on *all* objectives simultaneously.
- **`best_refusal`** — the trial with the lowest refusal rate (ignoring perplexity and KL).
- **`best_balanced`** — the trial with the best normalized sum across all three objectives.

Use `best_refusal` when refusal removal is your priority. Use `best_balanced` when you need to preserve capabilities.

## Advanced Cutting Techniques

### Biprojected (Gram-Schmidt against harmless direction)

```toml
[cut]
biprojected = true
```

Orthogonalizes the refusal direction against the harmless direction before cutting. Preserves more capability at the cost of slightly less refusal removal (Part 3).

### Norm-Preserving (row rescaling after projection)

```toml
[cut]
norm_preserve = true
```

Rescales each modified weight row to its original norm (Part 3).

### Sparsified Directions (zero low-magnitude components)

```toml
[cut]
sparsity = 0.8
```

Zeros out the bottom 80% of direction components by absolute value. Keeps only the top 20%.

### Subspace Cutting (multi-direction via SVD)

```toml
[measure]
mode = "subspace"
top_k = 5

[cut]
alpha = 1.0
```

When `mode = "subspace"`, the measurement returns a $k$-dimensional basis. The cut step applies `cut_subspace()` — iterating over basis vectors and removing each one. This is the countermeasure to hardened models with effective rank > 1.

### False Refusal Orthogonalization (preserve benign refusals)

```toml
[data]
borderline = "prompts/borderline.jsonl"

[cut]
false_refusal_ortho = true
```

When `borderline` prompts are provided and `false_refusal_ortho = true`, vauban measures a "false refusal direction" from borderline prompts (prompts that should be refused by some reasonable interpretation) and orthogonalizes the refusal direction against it before cutting. This preserves legitimate refusals on borderline content.

### Per-Layer Alpha Weights

```toml
[cut]
alpha = 1.0
layer_weights = [0.5, 0.5, 0.8, 1.0, 1.0, 1.2, 1.2, 1.0, 1.0, 0.8, 0.5, 0.5]
```

The final alpha for layer $i$ is `alpha * layer_weights[i]`. This allows fine-grained control — more aggressive cutting at peak layers, gentler cutting at peripheral layers.

### Layer Type Filtering (global vs sliding attention)

Some models (e.g., Phi-3) use interleaved attention types: some layers have global attention, others have sliding-window attention. Refusal may be concentrated in one type.

```toml
[cut]
layer_type_filter = "global"
```

Options: `"global"` (cut only global attention layers), `"sliding"` (cut only sliding), or omit for all layers.

### DBDI Configuration

```toml
[measure]
mode = "dbdi"

[cut]
dbdi_target = "red"
```

`dbdi_target` controls which DBDI direction is cut:
- `"red"` — cut only refusal execution (preserve harm detection).
- `"hdd"` — cut only harm detection (unusual, but available).
- `"both"` — cut both directions.

## Experiment Management

### experiment_log.jsonl

Every pipeline run appends a JSON record to `experiment_log.jsonl` in the output directory. Each record includes:

- Timestamp
- Full config
- Measured direction metadata
- Evaluation results
- Surface comparison (if run)
- Detection results (if run)

This creates a running log of all experiments, enabling comparison and reproducibility.

### quick.compare() and vauban diff

Compare two runs:

```python
from vauban import quick

diff = quick.compare("experiments/run_001", "experiments/run_002")
print(diff)
```

This reads the JSON reports from both directories and produces a side-by-side comparison of all metrics.

From the command line:

```bash
vauban diff experiments/run_001 experiments/run_002
```

> **You Should Know:** `diff` with threshold — `vauban diff` returns exit code 1 if any metric exceeds a configurable threshold. This integrates with CI/CD: run abliteration as a pipeline step and fail the build if quality degrades.

## The Complete Example

Here is a production TOML that uses most features:

```toml
[model]
path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

[data]
harmful = "prompts/harmful.jsonl"
harmless = "prompts/harmless.jsonl"
borderline = "prompts/borderline.jsonl"

[measure]
mode = "direction"
clip_quantile = 0.01
transfer_models = ["mlx-community/Llama-3.2-1B-Instruct-4bit"]

[cut]
alpha = 1.2
norm_preserve = true
layer_strategy = "above_median"
sparsity = 0.5
false_refusal_ortho = true

[eval]
num_prompts = 50
max_tokens = 100
refusal_mode = "phrases"

[surface]
prompts = "default"
max_worst_cell_refusal_after = 0.10
max_worst_cell_refusal_delta = 0.05
min_coverage_score = 0.70

[output]
dir = "experiments/production_run"
```

Running this:

```bash
# Validate first
vauban --validate production.toml

# Run
vauban production.toml
```

The pipeline will:
1. Load and dequantize the 3B model.
2. Load harmful, harmless, and borderline prompts.
3. Measure the refusal direction with 1% Winsorization.
4. Test direction transfer to the 1B model.
5. Orthogonalize against the false-refusal direction.
6. Cut with norm-preserving, above-median layers, 50% sparsity, alpha=1.2.
7. Export the modified model.
8. Evaluate on 50 prompts.
9. Run surface scan with quality gates.
10. Write everything to `experiments/production_run/`.

## You Should Know

> **Always validate first.** `vauban --validate` catches typos, missing files, and mode conflicts in milliseconds. It is the cheapest debugging tool you have.

> **Early-return precedence.** If you accidentally include both `[steer]` and `[cast]` in the same config, only `[steer]` runs (higher priority). Check the output carefully if you get unexpected behavior.

> **Experiment log.** Every run appends to `experiment_log.jsonl`. This is your audit trail. To start fresh, either use a new output directory or manually clear the log.

> **diff with threshold.** `vauban diff` can be used in CI/CD pipelines. Set a threshold (e.g., "fail if refusal rate increased by more than 5%") and use the exit code to gate deployments.

## Key Takeaways

1. **`vauban config.toml`** is the only CLI — everything is declarative.
2. **`vauban --validate`** catches config errors before loading any model.
3. **Pipeline modes** — default (measure → cut → export) or early-return (depth, probe, steer, cast, sic, optimize, softprompt).
4. **Optimization** — Optuna searches over alpha, sparsity, norm-preserve, and layer strategy. Returns Pareto front and convenience picks.
5. **Advanced cut techniques** — biprojected, norm-preserving, sparsified, subspace, false-refusal orthogonalization, per-layer weights, layer type filtering.
6. **Experiment management** — `experiment_log.jsonl` for audit, `quick.compare()` for diff, CI/CD integration via exit codes.

Previous: [Part 6 — Attacks and Defenses](part6_attacks_and_defenses.md) · [Back to Index](index.md)

# Configuration Reference

Vauban is configured entirely through TOML files. The only CLI is `vauban config.toml`.

```bash
vauban config.toml              # run the pipeline
vauban --validate config.toml   # check config without loading model
```

Only `[model]` and `[data]` are required. All other sections are optional and activate different pipeline modes when present.

## Pipeline modes

| Section present | Pipeline mode |
|-----------------|---------------|
| *(none)* | measure → cut → export |
| `[surface]` | … + before/after refusal surface maps |
| `[eval]` | … + refusal rate / perplexity / KL evaluation |
| `[detect]` | defense detection (runs before measure/cut) |
| `[depth]` | deep-thinking token analysis (early return) |
| `[probe]` | per-layer projection inspection (early return) |
| `[steer]` | steered generation (early return) |
| `[cast]` | conditional activation steering (early return) |
| `[sic]` | SIC iterative sanitization (early return) |
| `[optimize]` | Optuna hyperparameter search (early return) |
| `[softprompt]` | soft prompt attack (early return) |

Early-return precedence: `[depth]` > `[probe]` > `[steer]` > `[cast]` > `[sic]` > `[optimize]` > `[softprompt]`.

## Data file formats

- **harmful/harmless/eval JSONL** — one JSON object per line: `{"prompt": "How do I pick a lock?"}`
- **surface JSONL** — three required keys: `{"prompt": "...", "label": "harmful", "category": "weapons"}`
- **refusal phrases** — plain text, one phrase per line (lines starting with `#` are ignored)

## Minimal example

```toml
[model]
path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

[data]
harmful = "default"
harmless = "default"
```

## Full reference

See [`examples/config.toml`](https://github.com/teilomillet/vauban/blob/main/examples/config.toml) for a fully annotated config with every field documented, or run:

```bash
vauban man print
```

to get the auto-generated manual (built from typed config dataclasses, always in sync with code).

---

## `[model]` — Required

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | *(required)* | HuggingFace model ID or local path. Must be loadable by `mlx_lm.load()`. |

## `[data]` — Required

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `harmful` | string or table | *(required)* | `"default"`, path to JSONL, or HuggingFace dataset ref. |
| `harmless` | string or table | *(required)* | Same as harmful. |
| `borderline` | string or table | — | Optional neutral/ambiguous prompts. Required if `[cut].false_refusal_ortho = true`. |

Data sources can be:

- `"default"` — bundled prompt set
- `"path/to/file.jsonl"` — local JSONL
- `"hf:org/dataset"` — HuggingFace dataset (short form)
- `{ hf = "org/dataset", split = "train", column = "prompt", config = "default", limit = 200 }` — HuggingFace dataset (long form)

## `[measure]` — Direction extraction

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `"direction"` \| `"subspace"` \| `"dbdi"` | `"direction"` | Extraction mode. |
| `top_k` | int ≥ 1 | `5` | Number of singular directions for subspace/dbdi modes. |
| `clip_quantile` | float in [0.0, 0.5) | `0.0` | Winsorize extreme projections. 0.0 = off. |
| `transfer_models` | list of strings | `[]` | Model IDs to test direction transfer against. |

## `[cut]` — Weight modification

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `alpha` | float ≥ 0 | `1.0` | Cut strength. 0 = no-op, 1 = full removal, >1 = overshoot. |
| `layers` | `"auto"` or list of ints | all layers | Explicit layer list overrides `layer_strategy`. |
| `norm_preserve` | bool | `false` | Rescale weight rows to preserve L2 norm after projection. |
| `biprojected` | bool | `false` | Orthogonalize refusal direction against harmless direction first. |
| `layer_strategy` | `"all"` \| `"above_median"` \| `"top_k"` | `"all"` | Probe-guided layer selection. |
| `layer_top_k` | int ≥ 1 | `10` | Layers to select when `layer_strategy = "top_k"`. |
| `layer_weights` | list of floats | — | Per-layer alpha multipliers. Length must match number of layers. |
| `sparsity` | float in [0.0, 1.0) | `0.0` | Fraction of direction components to zero out. |
| `dbdi_target` | `"red"` \| `"hdd"` \| `"both"` | `"red"` | Which DBDI component to remove. Only with `measure.mode = "dbdi"`. |
| `false_refusal_ortho` | bool | `false` | Orthogonalize against false-refusal direction. Requires `[data].borderline`. |
| `layer_type_filter` | `"global"` \| `"sliding"` | — | Filter layers by attention type before cutting. |

## `[eval]` — Post-cut evaluation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompts` | path | — | JSONL file with evaluation prompts. Falls back to first `num_prompts` harmful prompts. |
| `max_tokens` | int ≥ 1 | `100` | Max tokens to generate per prompt. |
| `num_prompts` | int ≥ 1 | `20` | Fallback prompt count when prompts path is absent. |
| `refusal_phrases` | path | — | Custom refusal phrases file (one per line). |
| `refusal_mode` | `"phrases"` \| `"judge"` | `"phrases"` | Detection method. |

## `[surface]` — Refusal surface mapping

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompts` | path or `"default"` | `"default"` | JSONL with prompt/label/category keys. |
| `generate` | bool | `true` | Generate responses. `false` = projection-only (faster). |
| `max_tokens` | int ≥ 1 | `20` | Max tokens per prompt during scan. |
| `progress` | bool | `true` | Print scan progress to stderr. |
| `max_worst_cell_refusal_after` | float | — | Quality gate: max refusal rate in any cell after cut. |
| `max_worst_cell_refusal_delta` | float | — | Quality gate: max refusal rate increase in any cell. |
| `min_coverage_score` | float | — | Quality gate: minimum grid coverage. |

## `[detect]` — Defense detection

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `"fast"` \| `"probe"` \| `"full"` | `"full"` | Detection depth. |
| `top_k` | int ≥ 1 | `5` | SVD components. |
| `clip_quantile` | float in [0.0, 0.5) | `0.0` | Winsorization. |
| `alpha` | float ≥ 0 | `1.0` | Test cut strength (full mode). |
| `max_tokens` | int ≥ 1 | `100` | Generation limit (full mode). |

## `[optimize]` — Optuna search

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_trials` | int ≥ 1 | `50` | Number of Optuna trials. |
| `alpha_min` / `alpha_max` | float | `0.1` / `5.0` | Alpha search range. |
| `sparsity_min` / `sparsity_max` | float | `0.0` / `0.9` | Sparsity search range. |
| `search_norm_preserve` | bool | `true` | Include norm_preserve in search. |
| `search_strategies` | list of strings | `["all", "above_median", "top_k"]` | Layer strategies to search. |
| `layer_top_k_min` / `layer_top_k_max` | int | `3` / num_layers | top_k range. |
| `max_tokens` | int ≥ 1 | `100` | Generation limit per eval. |
| `seed` | int | — | Reproducibility seed. |
| `timeout` | float (seconds) | — | Wall-clock timeout. |

## `[softprompt]` — Soft prompt attack

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `"continuous"` \| `"gcg"` \| `"egd"` | `"continuous"` | Optimization mode. |
| `n_tokens` | int ≥ 1 | `16` | Learnable prefix length. |
| `n_steps` | int ≥ 1 | `200` | Optimization iterations. |
| `learning_rate` | float > 0 | `0.01` | Learning rate (continuous mode). |
| `batch_size` | int ≥ 1 | `64` | GCG candidates per position. |
| `top_k` | int ≥ 1 | `256` | GCG token candidates. |
| `direction_weight` | float ≥ 0 | `0.0` | Direction-guided weight. 0 = standalone. |
| `direction_mode` | `"last"` \| `"raid"` \| `"all_positions"` | `"last"` | How to apply direction guidance. |
| `loss_mode` | `"targeted"` \| `"untargeted"` \| `"defensive"` | `"targeted"` | Loss function. |
| `token_constraint` | `"ascii"` \| `"alpha"` \| `"alphanumeric"` | — | Restrict token search space. |
| `transfer_models` | list of strings | `[]` | Models to test transferability against. |

See [`examples/config.toml`](https://github.com/teilomillet/vauban/blob/main/examples/config.toml) for all remaining fields.

## `[sic]` — SIC defense

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `"direction"` \| `"generation"` | `"direction"` | Detection method. |
| `threshold` | float | `0.0` | Detection threshold. |
| `max_iterations` | int ≥ 1 | `3` | Sanitization rounds. |
| `calibrate` | bool | `false` | Auto-calibrate threshold from clean prompts. |

## `[depth]` — Deep-thinking analysis

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompts` | list of strings | *(required)* | Inline prompts for depth analysis. |
| `settling_threshold` | float in (0.0, 1.0] | `0.5` | JSD threshold for "settled". |
| `deep_fraction` | float in (0.0, 1.0] | `0.85` | Deep-thinking layer fraction. |
| `max_tokens` | int ≥ 0 | `0` | 0 = static analysis, >0 = generate. |
| `extract_direction` | bool | `false` | Also extract a depth direction. |

## `[probe]` — Projection inspection

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompts` | list of strings | *(required)* | Prompts to probe. |

## `[steer]` — Steered generation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompts` | list of strings | *(required)* | Prompts for steered generation. |
| `layers` | list of ints | all layers | Layers to apply steering. |
| `alpha` | float ≥ 0 | `1.0` | Steering strength. |
| `max_tokens` | int ≥ 1 | `100` | Max tokens to generate. |

## `[cast]` — Conditional activation steering

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompts` | list of strings | *(required)* | Prompts for CAST generation. |
| `layers` | list of ints | all layers | Layers where conditional checks run. |
| `alpha` | float ≥ 0 | `1.0` | Steering strength when projection exceeds threshold. |
| `threshold` | float | `0.0` | Trigger steering only if projection > threshold. |
| `max_tokens` | int ≥ 1 | `100` | Max tokens to generate. |

## `[output]` — Output directory

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dir` | path | `"output"` | Where to write modified weights, reports, and measurements. |

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
| `[api_eval]` | remote API suffix evaluation |
| `[meta]` | experiment metadata (no pipeline effect) |

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

## Top-level fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | string | `"mlx"` | Inference backend. |
| `verbose` | bool | `true` | Print progress messages to stderr. |

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
| `mode` | `"direction"` \| `"subspace"` \| `"dbdi"` \| `"diff"` | `"direction"` | Extraction mode. |
| `top_k` | int ≥ 1 | `5` | Number of singular directions for subspace/diff modes. |
| `clip_quantile` | float in [0.0, 0.5) | `0.0` | Winsorize extreme projections. 0.0 = off. |
| `transfer_models` | list of strings | `[]` | Model IDs to test direction transfer against. |
| `diff_model` | string | — | Base model for weight-diff measurement. Required when `mode = "diff"`. |
| `measure_only` | bool | `false` | Stop after writing measure-stage reports (skip cut/export). |

**Diff mode** extracts safety directions by computing `W_aligned - W_base` for `o_proj` and `down_proj` weights at each layer, then running SVD to find the principal difference directions. This captures distributed safety effects that single-model activation measurement may miss. See [Part 8](class/part8_model_diffing_and_defense.md) for details.

## `[cut]` — Weight modification

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `alpha` | float | `1.0` | Cut strength. 0 = no-op, 1 = full removal, >1 = overshoot. Negative values amplify the direction (LoX-style safety hardening). |
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

Optimizes a learnable prefix (soft prompt) to bypass the model's refusal. Three optimization modes:

- **continuous** — Adam optimization directly in embedding space (Schwinn et al.)
- **gcg** — Greedy Coordinate Gradient over discrete tokens (Zou et al.)
- **egd** — Exponentiated Gradient Descent on the probability simplex

### Core

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `"continuous"` \| `"gcg"` \| `"egd"` | `"continuous"` | Optimization mode. |
| `n_tokens` | int ≥ 1 | `16` | Learnable prefix length. |
| `n_steps` | int ≥ 1 | `200` | Optimization iterations. |
| `learning_rate` | float > 0 | `0.01` | Learning rate (continuous mode). |
| `init_scale` | float | `0.1` | Embedding initialization scale. |
| `batch_size` | int ≥ 1 | `64` | GCG candidates per position. |
| `top_k` | int ≥ 1 | `256` | GCG token candidates. |
| `target_prefixes` | list of strings | `["Sure", "Here"]` | Target completion prefixes. |
| `max_gen_tokens` | int ≥ 1 | `100` | Tokens to generate when evaluating success. |
| `seed` | int | — | Random seed for reproducibility. |

### Regularization & scheduling

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `embed_reg_weight` | float ≥ 0 | `0.0` | Embedding norm regularization (Huang et al.). 0 = off. |
| `patience` | int ≥ 0 | `0` | Early stopping patience. 0 = disabled. |
| `lr_schedule` | `"constant"` \| `"cosine"` | `"constant"` | Learning rate schedule. |
| `n_restarts` | int ≥ 1 | `1` | Random restarts (GCG mode). |
| `grad_accum_steps` | int ≥ 1 | `1` | Gradient accumulation steps. 1 = no accumulation. |

### Prompt strategy

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt_strategy` | `"all"` \| `"cycle"` \| `"first"` \| `"worst_k"` \| `"sample"` | `"all"` | How prompts are selected each step. |
| `worst_k` | int ≥ 1 | `5` | Number of prompts to focus on for `"worst_k"` strategy. |
| `prompt_pool_size` | int ≥ 1 | — | Override eval prompt count for pool size. |

### Direction & loss

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `direction_weight` | float ≥ 0 | `0.0` | Direction-guided weight. 0 = standalone attack. |
| `direction_mode` | `"last"` \| `"raid"` \| `"all_positions"` | `"last"` | How to apply direction guidance. |
| `direction_layers` | list of ints | — | Layers for direction constraint. Null = all layers. |
| `loss_mode` | `"targeted"` \| `"untargeted"` \| `"defensive"` | `"targeted"` | Loss function. |
| `egd_temperature` | float > 0 | `1.0` | Bregman projection temperature (EGD mode). |
| `defense_aware_weight` | float ≥ 0 | `0.0` | Defense evasion penalty added to loss. 0 = off. |

### Token constraints

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `token_constraint` | string or list of strings | — | Restrict token search space. |

Valid constraint values: `"ascii"`, `"alpha"`, `"alphanumeric"`, `"non_latin"`, `"chinese"`, `"non_alphabetic"`, `"invisible"`, `"zalgo"`, `"emoji"`.

When a single string is provided, only tokens matching that constraint are allowed. When a list is provided, the intersection of all constraints is used (tokens must satisfy all listed constraints).

### EOS loss

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `eos_loss_mode` | `"none"` \| `"force"` \| `"suppress"` | `"none"` | Auxiliary loss for EOS token. `"force"` encourages EOS; `"suppress"` discourages it. |
| `eos_loss_weight` | float ≥ 0 | `0.0` | Weight for EOS auxiliary loss. |

### KL collision

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `kl_ref_weight` | float ≥ 0 | `0.0` | KL collision loss weight. Requires `ref_model` if > 0. |
| `ref_model` | string | — | HuggingFace model ID or path for KL collision reference model. |

### Transfer

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `transfer_models` | list of strings | `[]` | Models to test the optimized prompt on after training. |
| `transfer_loss_weight` | float ≥ 0 | `0.0` | Multi-model re-ranking weight. 0 = off. |
| `transfer_rerank_count` | int ≥ 1 | `8` | Top-N candidates to re-rank on transfer models. |

### Target config

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_repeat_count` | int ≥ 0 | `0` | Repeat target tokens N times. 0 = disabled. |
| `system_prompt` | string | — | System prompt prepended to messages. |

### Beam search & init

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `beam_width` | int ≥ 1 | `1` | GCG beam search population. 1 = greedy. |
| `init_tokens` | list of ints | — | Warm-start token IDs (GCG/EGD modes). |

### Defense eval (in-loop)

Evaluate the optimized suffix against defense modules during training.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `defense_eval` | `"sic"` \| `"cast"` \| `"both"` | — | Which defense to evaluate against. Null = no defense eval. |
| `defense_eval_layer` | int | — | Layer for SIC/CAST direction projection. Null = auto from measurement. |
| `defense_eval_alpha` | float | `1.0` | CAST steering alpha. |
| `defense_eval_threshold` | float | `0.0` | SIC/CAST detection threshold. |
| `defense_eval_sic_mode` | `"direction"` \| `"generation"` | `"direction"` | SIC detection method. |
| `defense_eval_sic_max_iterations` | int ≥ 1 | `3` | SIC max sanitization iterations. |
| `defense_eval_cast_layers` | list of ints | — | CAST steering layers. Null = auto. |
| `defense_eval_alpha_tiers` | list of `[threshold, alpha]` pairs | — | TRYLOCK adaptive alpha tiers for CAST. |

### GAN loop

Iterative attack-defense training. The attacker (soft prompt optimizer) and defender (SIC/CAST) alternate rounds, escalating parameters when the attacker fails.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gan_rounds` | int ≥ 0 | `0` | Number of attack-defense rounds. 0 = no GAN loop. |
| `gan_step_multiplier` | float > 0 | `1.5` | Multiply `n_steps` each failed round. |
| `gan_direction_escalation` | float | `0.25` | Add to `direction_weight` per failed round. |
| `gan_token_escalation` | int ≥ 0 | `4` | Add to `n_tokens` per failed round. |

### GAN defender escalation

When enabled, the defender also escalates its parameters after attacker wins.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gan_defense_escalation` | bool | `false` | Enable defender escalation. Off = legacy behavior. |
| `gan_defense_alpha_multiplier` | float > 0 | `1.5` | Multiply CAST alpha per attacker win. |
| `gan_defense_threshold_escalation` | float ≥ 0 | `0.5` | Subtract from SIC/CAST threshold per attacker win. |
| `gan_defense_sic_iteration_escalation` | int ≥ 0 | `1` | Add to SIC max iterations per attacker win. |

### Multi-turn GAN

Thread GAN rounds as a multi-turn conversation, carrying history forward.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gan_multiturn` | bool | `false` | Enable multi-turn conversation threading. |
| `gan_multiturn_max_turns` | int ≥ 1 | `10` | Max conversation turns to keep in history. |

### Injection context

Wrap the optimized suffix in realistic surrounding context (e.g., a web page, tool output, or code file). Only works with discrete modes (`gcg` or `egd`).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `injection_context` | `"web_page"` \| `"tool_output"` \| `"code_file"` | — | Preset injection context wrapper. |
| `injection_context_template` | string | — | Custom template with `{payload}` placeholder. Overrides preset. |

**Constraints:** Injection context requires `mode = "gcg"` or `mode = "egd"` (continuous mode produces soft embeddings that cannot represent wrapped context). Cannot be combined with `gan_multiturn`.

## `[sic]` — SIC defense

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `"direction"` \| `"generation"` | `"direction"` | Detection method. |
| `threshold` | float | `0.0` | Detection threshold. Higher = stricter. Can be negative. |
| `max_iterations` | int ≥ 1 | `3` | Sanitization rounds. |
| `max_tokens` | int ≥ 1 | `100` | Tokens for generation-based detection. |
| `target_layer` | int | — | Layer for direction projection. Null = use measurement result. |
| `sanitize_system_prompt` | string | *(built-in)* | System prompt for the rewrite step. |
| `max_sanitize_tokens` | int ≥ 1 | `200` | Max tokens when rewriting. |
| `block_on_failure` | bool | `true` | Block inputs that cannot be sanitized within `max_iterations`. |
| `calibrate` | bool | `false` | Auto-calibrate threshold from clean prompts. |
| `calibrate_prompts` | `"harmless"` \| `"harmful"` | `"harmless"` | Which prompt set to calibrate from. |

## `[depth]` — Deep-thinking analysis

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompts` | list of strings | *(required)* | Inline prompts for depth analysis. |
| `settling_threshold` | float in (0.0, 1.0] | `0.5` | JSD threshold for "settled". |
| `deep_fraction` | float in (0.0, 1.0] | `0.85` | Deep-thinking layer fraction. |
| `max_tokens` | int ≥ 0 | `0` | 0 = static analysis, >0 = generate. |
| `extract_direction` | bool | `false` | Also extract a depth direction. |
| `top_k_logits` | int ≥ 1 | `1000` | Approximate JSD with top-k logits (performance). |
| `direction_prompts` | list of strings | — | Prompts for direction extraction. Required when `extract_direction = true`. |
| `clip_quantile` | float in [0.0, 0.5) | `0.0` | Winsorization quantile for direction extraction. |

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
| `condition_direction` | path | — | Separate `.npy` direction for gating (detect vs. steer split). |
| `alpha_tiers` | list of tables | — | Adaptive alpha tiers: `[[cast.alpha_tiers]]` with `threshold` and `alpha` keys. Must be sorted ascending. |

**Dual-direction mode**: When `condition_direction` is set, gating uses that direction (detect) while correction uses the primary direction (steer). This implements the AdaSteer dual-direction pattern.

**Adaptive alpha**: `alpha_tiers` maps projection magnitude to different steering strengths, avoiding the non-monotonic danger zone identified by TRYLOCK. See [Part 8](class/part8_model_diffing_and_defense.md) for examples.

## `[api_eval]` — Remote API suffix evaluation

Test optimized suffixes against remote API endpoints. Runs after softprompt optimization completes.

### Top-level

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_tokens` | int ≥ 1 | `100` | Max tokens for API responses. |
| `timeout` | int | `30` | Request timeout in seconds. |
| `system_prompt` | string | — | Shared default system prompt for all endpoints. |
| `multiturn` | bool | `false` | Enable multi-turn conversation with follow-ups. |
| `multiturn_max_turns` | int ≥ 1 | `3` | Total turns including initial prompt. |
| `follow_up_prompts` | list of strings | `[]` | Custom follow-up prompts. Empty = use defaults. |

### `[[api_eval.endpoints]]`

Each endpoint is a TOML array-of-tables entry:

```toml
[[api_eval.endpoints]]
name = "gpt-4o"
base_url = "https://api.openai.com/v1"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"
system_prompt = "You are a helpful assistant."  # optional per-endpoint override
auth_header = "Authorization"                   # optional custom header name
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | *(required)* | Display name for this endpoint. |
| `base_url` | string | *(required)* | API base URL (OpenAI-compatible). |
| `model` | string | *(required)* | Model identifier for the API. |
| `api_key_env` | string | *(required)* | Environment variable containing the API key. |
| `system_prompt` | string | — | Per-endpoint system prompt override. |
| `auth_header` | string | — | Custom auth header name (e.g. `"grayswan-api-key"`). |

## `[meta]` — Experiment metadata

Metadata for experiment tracking and tech tree visualization. Does not affect pipeline execution.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | string | *(required)* | Unique experiment identifier. |
| `title` | string | `""` | Human-readable experiment title. |
| `status` | string | `"wip"` | Experiment status (e.g. `"wip"`, `"done"`, `"abandoned"`). |
| `parents` | list of strings | `[]` | IDs of parent experiments (lineage). |
| `tags` | list of strings | `[]` | Freeform tags for categorization. |
| `notes` | string | `""` | Experiment notes. |
| `date` | string | `""` | Date string (freeform, e.g. `"2025-01-15"`). |

### `[[meta.docs]]`

Associated documents:

```toml
[[meta.docs]]
path = "results/report.pdf"
label = "Final report"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | *(required)* | Path to the document. |
| `label` | string | `""` | Display label. |

View the experiment tree with:

```bash
python -m vauban.tree experiments/
```

## `[output]` — Output directory

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dir` | path | `"output"` | Where to write modified weights, reports, and measurements. |

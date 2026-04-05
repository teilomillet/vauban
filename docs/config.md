<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

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
| `[svf]` | steering vector field boundary training (early return) |
| `[probe]` | per-layer projection inspection (early return) |
| `[steer]` | steered generation (early return) |
| `[cast]` | conditional activation steering (early return) |
| `[sic]` | iterative input sanitization (early return) |
| `[optimize]` | Optuna hyperparameter search (early return) |
| `[compose_optimize]` | Bayesian composition weight optimization (early return) |
| `[softprompt]` | soft prompt attack (early return) |
| `[defend]` | composed defense stack evaluation (early return) |
| `[environment]` | agent tool harness (used with `[softprompt]`) |
| `[api_eval]` | remote API suffix evaluation |
| `[meta]` | experiment metadata (no pipeline effect) |

Early-return precedence: `[depth]` > `[svf]` > `[probe]` > `[steer]` > `[cast]` > `[sic]` > `[optimize]` > `[compose_optimize]` > `[softprompt]` > `[defend]`.

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
| `refusal_mode` | `"phrases"` \| `"judge"` | `"phrases"` | Detection method. `"phrases"` = substring matching against refusal phrase list. `"judge"` = model-based classification. |

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
| `loss_mode` | `"targeted"` \| `"untargeted"` \| `"defensive"` \| `"externality"` | `"targeted"` | Loss function. `"externality"` requires `externality_target`. |
| `egd_temperature` | float > 0 | `1.0` | Bregman projection temperature (EGD mode). |
| `defense_aware_weight` | float ≥ 0 | `0.0` | Defense evasion penalty added to loss. 0 = off. |
| `externality_target` | path | — | Path to `.npy` direction file for externality loss. Required when `loss_mode = "externality"`. Penalizes safety margin erosion (Xiong et al. 2026). |

### Token constraints

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `token_constraint` | string or list of strings | — | Restrict token search space. |

**Positive constraints** (include only matching tokens): `"ascii"`, `"alpha"`, `"alphanumeric"`, `"non_latin"`, `"chinese"`, `"non_alphabetic"`, `"invisible"`, `"zalgo"`, `"emoji"`.

**Negative constraints** (exclude matching tokens): `"exclude_glitch"`.

When a single string is provided, only tokens matching that constraint are allowed (or excluded, for negative constraints). When a list is provided, positive constraints are intersected first, then negative constraints remove tokens from the result.

#### `exclude_glitch`

Detects and excludes under-trained ("glitch") tokens from the adversarial search space. These are tokens with anomalously low or high embedding norms (beyond 3 standard deviations from the mean) that cause model collapse when encountered during generation.

**Why it matters:** Under-trained tokens produce random multilingual word salad instead of coherent text. If GCG/EGD selects one, the optimization step is wasted — the model cannot parse the input at all, so the gradient signal is noise. Excluding these tokens improves optimization efficiency.

**Research basis:** "Fishing for Magikarp" (Rumbelow & Watkins, EMNLP 2024, arxiv 2405.05417) showed that ~0.3-0.6% of tokens in modern LLMs are under-trained and cause anomalous behavior. Our cross-model validation on Qwen2.5-0.5B/1.5B confirmed this rate with calibrated multi-template behavioral entropy testing.

```toml
[softprompt]
mode = "gcg"
token_constraint = "exclude_glitch"

# Can combine with positive constraints:
# token_constraint = ["ascii", "exclude_glitch"]
```

Detection runs automatically from the embedding matrix at mask-build time (~1 second). No additional configuration needed.

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
| `defense_eval_sic_mode` | `"direction"` \| `"generation"` \| `"svf"` | `"direction"` | SIC detection method. `"svf"` uses trained boundary MLP. |
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

### Perplexity regularization

Cross-entropy penalty pushing optimized suffixes toward fluent text. Encourages token sequences that look like natural language instead of adversarial noise.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `perplexity_weight` | float >= 0 | `0.0` | Weight for perplexity (CE) auxiliary loss. 0 = off. Higher values produce more fluent but potentially less effective suffixes. |

### Token position

Controls where the learnable tokens are inserted relative to the prompt.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `token_position` | `"prefix"` \| `"suffix"` \| `"infix"` | `"prefix"` | Placement of optimized tokens. `"prefix"` = before the prompt, `"suffix"` = after the prompt, `"infix"` = split and inserted within the prompt. |

**Constraints:** `token_position = "infix"` requires `mode = "gcg"` or `mode = "egd"` (continuous mode cannot resolve infix split positions).

### Prompt paraphrasing

Augment the prompt pool with paraphrased variants. Each strategy applies a different rewriting style to diversify the attack surface.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `paraphrase_strategies` | list of strings | `[]` | Paraphrase strategies to apply. Empty = no paraphrasing. |

Valid strategies: `"narrative"`, `"deceptive_delight"`, `"technical"`, `"historical"`, `"code_block"`, `"educational"`.

### Environment rollout integration

When `[environment]` is present alongside `[softprompt]`, the optimizer periodically runs the top candidates through the agent environment to compute reward-based re-ranking. See the `[environment]` section for full configuration.

## `[sic]` — Iterative input sanitization

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

## `[defend]` — Defense stack composition

Composes multiple defense layers (scan, SIC, policy, intent) into a single evaluation pipeline. When `[defend]` is present, it acts as an early-return mode that runs each enabled defense layer in sequence against evaluation prompts and writes a combined report.

The `[defend]` section itself has only one field. The individual defense layers are configured via their own top-level sections (`[scan]`, `[sic]`, `[policy]`, `[intent]`), which are pulled in automatically when `[defend]` is present.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fail_fast` | bool | `true` | Stop at the first defense layer that blocks. `false` = run all layers and report all results. |

**Layer composition:** The defense stack runs layers in order: scan (Layer 0, injection detection) -> SIC (Layer 1, iterative sanitization) -> policy (Layer 3, tool-call filtering) -> intent (Layer 4, intent alignment). Each layer is optional; only layers with a corresponding top-level section are active.

**Example:**

```toml
[defend]
fail_fast = false

[scan]
threshold = 0.5
calibrate = true

[sic]
mode = "direction"
max_iterations = 3
```

### `[scan]` — Injection content scanning

Configure the Layer 0 injection scanner. Only active when `[defend]` is present.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_layer` | int | — | Layer for direction projection. Null = use measurement result. |
| `span_threshold` | float | `0.5` | Minimum mean projection for a token span to be flagged. |
| `threshold` | float | `0.0` | Overall detection threshold. |
| `calibrate` | bool | `false` | Auto-calibrate threshold from clean prompts. |

### `[policy]` — Tool-call policy engine

Configure the Layer 3 tool-call policy engine. Only active when `[defend]` is present.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_action` | `"allow"` \| `"block"` | `"allow"` | Default action for unmatched tool calls. |
| `rules` | list of tables | `[]` | Policy rules (see below). |
| `data_flow_rules` | list of tables | `[]` | Data flow restriction rules. |
| `rate_limits` | list of tables | `[]` | Rate limit rules. |

Each `[[policy.rules]]` entry:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | *(required)* | Rule name for logging. |
| `action` | `"allow"` \| `"block"` \| `"confirm"` | *(required)* | Action when matched. |
| `tool_pattern` | string | *(required)* | fnmatch pattern for tool names. |
| `argument_key` | string | — | Optional argument key to check. |
| `argument_pattern` | string | — | Regex pattern for argument value. |

### `[intent]` — Intent alignment checking

Configure the Layer 4 intent alignment checker. Only active when `[defend]` is present.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `"embedding"` \| `"judge"` | `"embedding"` | Alignment detection method. |
| `target_layer` | int | — | Layer for embedding similarity. Null = auto. |
| `similarity_threshold` | float | `0.7` | Cosine similarity threshold for `"embedding"` mode. |
| `judge_prompt` | string | *(built-in)* | System prompt for `"judge"` mode. |
| `max_tokens` | int >= 1 | `10` | Max tokens for judge generation. |

## `[environment]` — Agent tool harness

Defines a simulated agent environment for indirect prompt injection testing. The environment provides tools, a benign user task, and a target tool call that the injection payload should elicit. Used alongside `[softprompt]` to evaluate whether optimized suffixes can hijack agent behavior.

### Top-level

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `scenario` | string | — | Optional built-in benchmark scenario name. Seeds the environment with a named baseline such as `share_doc`; explicit TOML fields override the seeded defaults. |
| `system_prompt` | string | *(required)* | System prompt for the agent. |
| `injection_surface` | string | *(required)* | Name of the tool whose output contains the injection payload. Must match a defined tool name. |
| `injection_position` | `"prefix"` \| `"suffix"` \| `"infix"` | `"suffix"` | Where the injected payload is inserted into the tool result. |
| `benign_expected_tools` | list of strings | `[]` | Tools expected during benign execution. Used by flywheel utility checks and benchmark scenarios. |
| `max_turns` | int >= 1 | `6` | Maximum agent loop turns. |
| `max_gen_tokens` | int >= 1 | `200` | Max tokens per agent generation step. |
| `temperature` | float >= 0 | `0.0` | Sampling temperature. 0.0 = greedy (argmax). |
| `rollout_top_n` | int >= 1 | `8` | Number of top candidates to evaluate via environment rollout. |
| `rollout_every_n` | int >= 1 | `1` | Run environment rollouts every N optimization steps. 1 = every step. |

Built-in scenarios are available through `vauban init --help`. A minimal scenario-backed harness can be as short as:

```toml
[environment]
scenario = "share_doc"
max_turns = 5
```

### `[environment.target]`

The tool call the injection payload should trigger.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `function` | string | *(required)* | Target tool name. Must match a defined tool. |
| `required_args` | list of strings | `[]` | Argument keys that must be present. |
| `arg_contains` | table | `{}` | Key-value pairs the arguments must contain (substring match). |

### `[environment.task]`

The benign user task that initiates the agent loop.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `content` | string | *(required)* | The user's task prompt. |

### `[[environment.tools]]`

Each tool available to the agent is defined as an array-of-tables entry:

```toml
[[environment.tools]]
name = "web_search"
description = "Search the web for information."
parameters = { query = "string" }
result = "Search results as text."
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | *(required)* | Tool name (referenced by `injection_surface` and `target.function`). |
| `description` | string | `""` | Human-readable description shown to the agent. |
| `parameters` | table | `{}` | Parameter names mapped to type descriptions. |
| `result` | string | — | Description of the tool's return value. |

### `[environment.policy]`

Optional inline tool-call policy for the environment harness (separate from the `[policy]` defense layer).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `blocked_functions` | list of strings | `[]` | Tool names to block outright. |
| `require_confirmation` | list of strings | `[]` | Tool names requiring confirmation. |
| `arg_blocklist` | table of lists | `{}` | Per-argument blocked value patterns. |

**Cross-field validation:** `injection_surface` and `target.function` must both reference tools defined in `[[environment.tools]]`.

## `[svf]` — Steering vector field training

Trains a boundary MLP that learns a differentiable decision surface in activation space. The gradient of the boundary function at each activation gives the steering direction, replacing static linear vectors with context-dependent steering.

Reference: Li, Li & Huang (2026) -- arxiv.org/abs/2602.01654

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompts_target` | path | *(required)* | JSONL file with target-behavior prompts (e.g., harmful prompts). |
| `prompts_opposite` | path | *(required)* | JSONL file with opposite-behavior prompts (e.g., harmless prompts). |
| `projection_dim` | int >= 1 | `16` | Dimensionality of the projected activation space fed to the boundary MLP. |
| `hidden_dim` | int >= 1 | `64` | Hidden layer width in the boundary MLP. |
| `n_epochs` | int >= 1 | `10` | Training epochs. |
| `learning_rate` | float > 0 | `0.001` | Adam learning rate. |
| `layers` | list of ints | all layers | Layers to train boundary MLPs on. Null = all layers. |

The trained boundary model can be referenced by `[steer]`, `[cast]`, and `[sic]` via their `direction_source = "svf"` and `svf_boundary_path` fields.

## `[compose_optimize]` — Composition weight optimization

Bayesian optimization over linear composition weights for Steer2Adapt composed steering. Given a subspace bank (`.safetensors` file with named basis vectors), searches for the weight combination that optimizes refusal rate and perplexity trade-off.

Reference: Han et al. (2026) -- arxiv.org/abs/2602.07276

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `bank_path` | path | *(required)* | Path to a `.safetensors` subspace bank file. Each key is a named subspace with shape `(k, d_model)`. |
| `n_trials` | int >= 1 | `50` | Number of Optuna trials. |
| `max_tokens` | int >= 1 | `100` | Max tokens per evaluation generation. |
| `timeout` | float (seconds) | — | Wall-clock timeout. Null = no timeout. |
| `seed` | int | — | Reproducibility seed. Null = non-deterministic. |

The bank file is produced by the `[measure]` stage when `measure.bank` entries are configured. Each entry in the bank contains the SVD basis vectors for a named behavioral subspace. The optimizer searches for weights `w_i` such that the composed direction `sum(w_i * basis_i[0])` (L2-normalized) achieves the best refusal/quality balance.

## `[output]` — Output directory

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dir` | path | `"output"` | Where to write modified weights, reports, and measurements. |

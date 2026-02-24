"""Shared types for the vauban pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Protocols — structural typing decoupled from mlx-lm internals
# ---------------------------------------------------------------------------


@runtime_checkable
class TransformerModel(Protocol):
    """Inner transformer model (e.g. model.model in mlx-lm)."""

    embed_tokens: nn.Embedding
    layers: list[nn.Module]
    norm: nn.Module


@runtime_checkable
class CausalLM(Protocol):
    """Top-level causal language model (e.g. what mlx_lm.load returns)."""

    model: TransformerModel


@runtime_checkable
class Tokenizer(Protocol):
    """Tokenizer with chat template support."""

    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
    ) -> str | list[int]: ...


class LayerCache(Protocol):
    """Per-layer KV cache (matches mlx-lm KVCache interface)."""

    offset: int

    def update_and_fetch(
        self, keys: mx.array, values: mx.array,
    ) -> tuple[mx.array, mx.array]: ...


# ---------------------------------------------------------------------------
# Dataclasses — immutable value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DirectionResult:
    """Output of the measure step: a refusal direction vector."""

    direction: mx.array
    layer_index: int
    cosine_scores: list[float]
    d_model: int
    model_path: str


@dataclass(frozen=True, slots=True)
class SubspaceResult:
    """Output of the subspace measure step: top-k orthonormal directions."""

    basis: mx.array  # shape (k, d_model)
    singular_values: list[float]
    explained_variance: list[float]
    layer_index: int
    d_model: int
    model_path: str
    per_layer_bases: list[mx.array]  # basis for every layer

    def best_direction(self) -> DirectionResult:
        """Extract the rank-1 direction (first basis vector) for compatibility."""
        return DirectionResult(
            direction=self.basis[0],
            layer_index=self.layer_index,
            cosine_scores=[],
            d_model=self.d_model,
            model_path=self.model_path,
        )


@dataclass(frozen=True, slots=True)
class DBDIResult:
    """Output of DBDI decomposition: harm detection + refusal execution directions."""

    hdd: mx.array  # harm detection direction (d_model,)
    red: mx.array  # refusal execution direction (d_model,)
    hdd_layer_index: int
    red_layer_index: int
    hdd_cosine_scores: list[float]
    red_cosine_scores: list[float]
    d_model: int
    model_path: str


@dataclass(frozen=True, slots=True)
class CutConfig:
    """Configuration for the cut step."""

    alpha: float = 1.0
    layers: list[int] | None = None
    norm_preserve: bool = False
    biprojected: bool = False
    layer_strategy: str = "all"  # "all", "above_median", "top_k"
    layer_top_k: int = 10
    layer_weights: list[float] | None = None  # per-layer alpha multipliers
    sparsity: float = 0.0  # fraction of direction components to zero out
    dbdi_target: str = "red"  # "red", "hdd", or "both" (for DBDI mode)
    false_refusal_ortho: bool = False  # orthogonalize against false refusal


@dataclass(frozen=True, slots=True)
class DatasetRef:
    """Reference to a HuggingFace dataset for prompt loading."""

    repo_id: str
    split: str = "train"
    column: str = "prompt"
    config: str | None = None
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class MeasureConfig:
    """Configuration for the measure step."""

    mode: str = "direction"  # "direction", "subspace", or "dbdi"
    top_k: int = 5
    clip_quantile: float = 0.0  # winsorization quantile (0.0 = disabled)


@dataclass(frozen=True, slots=True)
class SurfaceConfig:
    """Configuration for pre/post surface mapping."""

    prompts_path: Path | str  # resolved Path or "default" sentinel
    generate: bool = True
    max_tokens: int = 20


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Output of the evaluate step."""

    refusal_rate_original: float
    refusal_rate_modified: float
    perplexity_original: float
    perplexity_modified: float
    kl_divergence: float
    num_prompts: int


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Output of the probe step: per-layer projection magnitudes."""

    projections: list[float]
    layer_count: int
    prompt: str


@dataclass(frozen=True, slots=True)
class SteerResult:
    """Output of the steer step: generated text with projection info."""

    text: str
    projections_before: list[float]
    projections_after: list[float]


@dataclass(frozen=True, slots=True)
class SurfacePrompt:
    """A prompt with metadata for refusal surface mapping."""

    prompt: str
    label: str
    category: str


@dataclass(frozen=True, slots=True)
class SurfacePoint:
    """Result for a single prompt in a surface scan."""

    prompt: str
    label: str
    category: str
    projections: list[float]
    direction_projection: float
    refused: bool | None
    response: str | None


@dataclass(frozen=True, slots=True)
class SurfaceGroup:
    """Aggregated stats for a group of surface points."""

    name: str
    count: int
    refusal_rate: float
    mean_projection: float
    min_projection: float
    max_projection: float


@dataclass(frozen=True, slots=True)
class SurfaceResult:
    """Complete refusal surface map."""

    points: list[SurfacePoint]
    groups_by_label: list[SurfaceGroup]
    groups_by_category: list[SurfaceGroup]
    threshold: float
    total_scanned: int
    total_refused: int


@dataclass(frozen=True, slots=True)
class SurfaceGroupDelta:
    """Change in a surface group between before and after cut."""

    name: str
    count: int
    refusal_rate_before: float
    refusal_rate_after: float
    refusal_rate_delta: float
    mean_projection_before: float
    mean_projection_after: float
    mean_projection_delta: float


@dataclass(frozen=True, slots=True)
class SurfaceComparison:
    """Comparison of refusal surface before and after cut."""

    before: SurfaceResult
    after: SurfaceResult
    refusal_rate_before: float
    refusal_rate_after: float
    refusal_rate_delta: float
    threshold_before: float
    threshold_after: float
    threshold_delta: float
    category_deltas: list[SurfaceGroupDelta]
    label_deltas: list[SurfaceGroupDelta]


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Full pipeline configuration loaded from TOML."""

    model_path: str
    harmful_path: Path | DatasetRef
    harmless_path: Path | DatasetRef
    cut: CutConfig = field(default_factory=CutConfig)
    measure: MeasureConfig = field(default_factory=MeasureConfig)
    surface: SurfaceConfig | None = None
    eval_prompts_path: Path | None = None
    output_dir: Path = field(default_factory=lambda: Path("output"))
    borderline_path: Path | DatasetRef | None = None

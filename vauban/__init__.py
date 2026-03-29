# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Vauban - MLX-native abliteration toolkit for Apple Silicon.

All public symbols are lazily loaded on first access (PEP 562).
``from vauban import X`` and ``vauban.X`` both work — the first access
imports the owning module, subsequent accesses hit a cached global.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban._version import __version__

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Lazy-import registry: symbol → module  (or (module, original_name) for
# renamed imports like ``scan as injection_scan``).
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, str | tuple[str, str]] = {
    # -- types (vauban.types) --
    "AgentTurn": "vauban.types",
    "AlphaTier": "vauban.types",
    "ApiEvalEndpoint": "vauban.types",
    "AIActArtifacts": "vauban.ai_act",
    "AIActConfig": "vauban.types",
    "AwarenessConfig": "vauban.types",
    "AwarenessLayerResult": "vauban.types",
    "AwarenessResult": "vauban.types",
    "CastConfig": "vauban.types",
    "CastResult": "vauban.types",
    "CausalLM": "vauban.types",
    "CircuitConfig": "vauban.types",
    "CircuitResult": "vauban.types",
    "GuardConfig": "vauban.types",
    "GuardEvent": "vauban.types",
    "GuardResult": "vauban.types",
    "GuardSession": "vauban.guard",
    "GuardTierSpec": "vauban.types",
    "GuardVerdict": "vauban.types",
    "ComponentEffect": "vauban.types",
    "ComposeOptimizeConfig": "vauban.types",
    "ComposeOptimizeResult": "vauban.types",
    "CompositionTrialResult": "vauban.types",
    "CutConfig": "vauban.types",
    "DBDIResult": "vauban.types",
    "DataFlowRule": "vauban.types",
    "DatasetRef": "vauban.types",
    "DefendedTrace": "vauban.types",
    "DefenseProxyResult": "vauban.types",
    "DefenseStackConfig": "vauban.types",
    "DefenseStackResult": "vauban.types",
    "DepthConfig": "vauban.types",
    "DepthDirectionResult": "vauban.types",
    "DepthResult": "vauban.types",
    "DetectConfig": "vauban.types",
    "DetectResult": "vauban.types",
    "DiffResult": "vauban.types",
    "DirectionResult": "vauban.types",
    "DirectionTransferResult": "vauban.types",
    "EnvironmentConfig": "vauban.types",
    "EnvironmentResult": "vauban.types",
    "EvalConfig": "vauban.types",
    "EvalResult": "vauban.types",
    "FeaturesConfig": "vauban.types",
    "FeaturesResult": "vauban.types",
    "FlywheelConfig": "vauban.types",
    "FlywheelCycleMetrics": "vauban.types",
    "FlywheelDefenseParams": "vauban.types",
    "FlywheelResult": "vauban.types",
    "FlywheelTrace": "vauban.types",
    "FusionConfig": "vauban.types",
    "FusionGeneration": "vauban.types",
    "FusionResult": "vauban.types",
    "IntentCheckResult": "vauban.types",
    "IntentConfig": "vauban.types",
    "JailbreakConfig": "vauban.types",
    "JailbreakResult": "vauban.types",
    "JailbreakStrategyResult": "vauban.types",
    "JailbreakTemplate": "vauban.types",
    "IntentState": "vauban.types",
    "LinearProbeConfig": "vauban.types",
    "LinearProbeLayerResult": "vauban.types",
    "LinearProbeResult": "vauban.types",
    "LoraAnalysisConfig": "vauban.types",
    "LoraAnalysisResult": "vauban.types",
    "LoraExportConfig": "vauban.types",
    "LoraExportResult": "vauban.types",
    "LoraLayerAnalysis": "vauban.types",
    "LoraLoadConfig": "vauban.types",
    "LoraMatrices": "vauban.types",
    "MeasureConfig": "vauban.types",
    "MetaConfig": "vauban.types",
    "OptimizeConfig": "vauban.types",
    "OptimizeResult": "vauban.types",
    "Payload": "vauban.types",
    "PerturbConfig": "vauban.types",
    "PipelineConfig": "vauban.types",
    "PolicyConfig": "vauban.types",
    "PolicyDecision": "vauban.types",
    "PolicyRule": "vauban.types",
    "ProbeConfig": "vauban.types",
    "ProbeResult": "vauban.types",
    "RateLimitRule": "vauban.types",
    "RemoteActivationResult": "vauban.types",
    "RemoteBackend": "vauban.types",
    "RemoteChatResult": "vauban.types",
    "RemoteConfig": "vauban.types",
    "RepBendConfig": "vauban.types",
    "ResponseScoreResult": "vauban.types",
    "ResponseScoreWeights": "vauban.types",
    "RepBendResult": "vauban.types",
    "SAELayerResult": "vauban.types",
    "SICConfig": "vauban.types",
    "SICPromptResult": "vauban.types",
    "SICResult": "vauban.types",
    "SSSConfig": "vauban.types",
    "SSSResult": "vauban.types",
    "SVFConfig": "vauban.types",
    "SVFResult": "vauban.types",
    "ScanConfig": "vauban.types",
    "ScanResult": "vauban.types",
    "ScanSpan": "vauban.types",
    "SoftPromptConfig": "vauban.types",
    "SoftPromptResult": "vauban.types",
    "SteerConfig": "vauban.types",
    "SteerResult": "vauban.types",
    "SubspaceResult": "vauban.types",
    "SurfaceComparison": "vauban.types",
    "SurfaceConfig": "vauban.types",
    "SurfaceGroup": "vauban.types",
    "SurfaceGroupDelta": "vauban.types",
    "SurfacePoint": "vauban.types",
    "SurfacePrompt": "vauban.types",
    "SurfaceResult": "vauban.types",
    "TokenDepth": "vauban.types",
    "Tokenizer": "vauban.types",
    "ToolCall": "vauban.types",
    "ToolSchema": "vauban.types",
    "TransferEvalResult": "vauban.types",
    "TrialResult": "vauban.types",
    "WorldMeta": "vauban.types",
    # -- functions --
    "BUNDLED_DATASETS": "vauban.data",
    "DirectionGeometryResult": "vauban.geometry",
    "DirectionPair": "vauban.geometry",
    "HarmScore": "vauban.taxonomy",
    "TAXONOMY": "vauban.taxonomy",
    "TaxonomyCoverage": "vauban.taxonomy",
    "aggregate": "vauban.surface",
    "all_categories": "vauban.taxonomy",
    "classify_text": "vauban.taxonomy",
    "analyze_adapter": "vauban.lora",
    "analyze_directions": "vauban.geometry",
    "awareness_calibrate": "vauban.awareness",
    "awareness_detect": "vauban.awareness",
    "build_lora_weights": "vauban.lora",
    "calibrate_scan_threshold": "vauban.scan",
    "calibrate_threshold": "vauban.sic",
    "capture_intent": "vauban.intent",
    "cast_generate": "vauban.cast",
    "cast_generate_svf": "vauban.cast",
    "check_alignment": "vauban.intent",
    "compare_surfaces": "vauban.surface",
    "compute_reward": "vauban.environment",
    "compute_sensitivity_profile": "vauban.sensitivity",
    "coverage_report": "vauban.taxonomy",
    "domain_for_category": "vauban.taxonomy",
    "cut": "vauban.cut",
    "cut_biprojected": "vauban.cut",
    "cut_false_refusal_ortho": "vauban.cut",
    "cut_subspace": "vauban.cut",
    "dataset_path": "vauban.data",
    "default_eval_path": "vauban.measure",
    "default_full_surface_path": "vauban.surface",
    "default_multilingual_surface_path": "vauban.surface",
    "default_prompt_paths": "vauban.measure",
    "default_surface_path": "vauban.surface",
    "defend_content": "vauban.defend",
    "defend_tool_call": "vauban.defend",
    "depth_direction": "vauban.depth",
    "depth_generate": "vauban.depth",
    "depth_profile": "vauban.depth",
    "dequantize_model": "vauban.dequantize",
    "detect": "vauban.detect",
    "detect_layer_types": "vauban.measure",
    "direction_to_lora": "vauban.lora",
    "effective_rank": "vauban.subspace",
    "evaluate": "vauban.evaluate",
    "evaluate_data_flow": "vauban.policy",
    "evaluate_suffix_via_api": "vauban.api_eval",
    "evaluate_tool_call": "vauban.policy",
    "evaluate_with_defense_proxy": "vauban.api_eval_proxy",
    "explained_variance_ratio": "vauban.subspace",
    "export_model": "vauban.export",
    "find_instruction_boundary": "vauban.measure",
    "find_threshold": "vauban.surface",
    "fuse_and_generate": "vauban.fusion",
    "fuse_batch": "vauban.fusion",
    "generate_config_schema": "vauban.config",
    "generate_deployer_readiness_artifacts": "vauban.ai_act",
    "generate_deployer_readiness_bundle": "vauban.ai_act",
    "get_dataset": "vauban.data",
    "grassmann_distance": "vauban.subspace",
    "injection_scan": ("vauban.scan", "scan"),
    "load_jailbreak_templates": ("vauban.jailbreak", "load_templates"),
    "is_quantized": "vauban.dequantize",
    "list_datasets": "vauban.data",
    "load_and_apply_adapter": "vauban.lora",
    "load_and_merge_adapters": "vauban.lora",
    "load_config": "vauban.config",
    "load_hf_prompts": "vauban.dataset",
    "load_prompts": "vauban.measure",
    "load_surface_prompts": "vauban.surface",
    "load_svf_boundary": "vauban.svf",
    "map_surface": "vauban.surface",
    "measure": "vauban.measure",
    "measure_dbdi": "vauban.measure",
    "measure_diff": "vauban.measure",
    "measure_subspace": "vauban.measure",
    "measure_subspace_bank": "vauban.measure",
    "merge_adapters": "vauban.lora",
    "multi_probe": "vauban.probe",
    "perturb": "vauban.perturb",
    "perturb_batch": "vauban.perturb",
    "optimize": "vauban.optimize",
    "optimize_composition": "vauban.optimize",
    "orthonormalize": "vauban.subspace",
    "principal_angles": "vauban.subspace",
    "probe": "vauban.probe",
    "project_subspace": "vauban.subspace",
    "remove_subspace": "vauban.subspace",
    "repbend": "vauban.repbend",
    "resolve_category": "vauban.taxonomy",
    "score_response": "vauban.scoring",
    "score_responses": "vauban.scoring",
    "score_batch": "vauban.taxonomy",
    "score_text": "vauban.taxonomy",
    "resolve_prompts": "vauban.dataset",
    "run": "vauban._pipeline",
    "run_agent_loop": "vauban.environment",
    "run_flywheel": "vauban.flywheel",
    "run_remote_probe": "vauban.remote",
    "save_adapter_mlx": "vauban.lora",
    "save_adapter_peft": "vauban.lora",
    "save_svf_boundary": "vauban.svf",
    "save_weights": "vauban.cut",
    "scan": "vauban.surface",
    "score_candidates_via_rollout": "vauban.environment",
    "select_target_layers": "vauban.measure",
    "sic_sanitize": ("vauban.sic", "sic"),
    "sic_single": "vauban.sic",
    "silhouette_scores": "vauban.measure",
    "softprompt_attack": "vauban.softprompt",
    "sparsify_direction": "vauban.cut",
    "sss_generate": "vauban.sss",
    "steer": "vauban.probe",
    "steer_svf": "vauban.probe",
    "subspace_overlap": "vauban.subspace",
    "subspace_to_lora": "vauban.lora",
    "svf_gradient": "vauban.svf",
    "target_weight_keys": "vauban.cut",
    "trace_circuit": "vauban.circuit",
    "train_probe": "vauban.linear_probe",
    "train_sae": "vauban.features",
    "train_sae_multi_layer": "vauban.features",
    "train_svf_boundary": "vauban.svf",
    "write_config_schema": "vauban.config",
}


def __getattr__(name: str) -> object:
    spec = _LAZY_IMPORTS.get(name)
    if spec is None:
        msg = f"module 'vauban' has no attribute {name!r}"
        raise AttributeError(msg)

    import importlib

    if isinstance(spec, tuple):
        module_path, original_name = spec
    else:
        module_path, original_name = spec, name

    mod = importlib.import_module(module_path)
    attr = getattr(mod, original_name)
    globals()[name] = attr  # cache for zero-overhead next access
    return attr


def __dir__() -> list[str]:
    return __all__


__all__ = [
    *sorted(_LAZY_IMPORTS),
    "__version__",
    "validate",
]


def validate(config_path: str | Path) -> list[str]:
    """Validate a TOML config without loading any model."""
    from vauban.config._validation import validate_config

    return validate_config(config_path)

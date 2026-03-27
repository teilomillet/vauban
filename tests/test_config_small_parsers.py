# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for small config parsers: detect, eval, measure, optimize, sic."""

from pathlib import Path

import pytest

from vauban.config._parse_detect import _parse_detect
from vauban.config._parse_eval import _parse_eval
from vauban.config._parse_measure import _parse_measure
from vauban.config._parse_optimize import _parse_optimize
from vauban.config._parse_sic import _parse_sic

# ── [detect] ─────────────────────────────────────────────────────────


class TestParseDetect:
    def test_absent_returns_none(self) -> None:
        assert _parse_detect({}) is None

    def test_section_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_detect({"detect": "bad"})

    def test_minimal_valid(self) -> None:
        cfg = _parse_detect({"detect": {}})
        assert cfg is not None
        assert cfg.mode == "full"
        assert cfg.top_k == 5
        assert cfg.clip_quantile == 0.0
        assert cfg.alpha == 1.0
        assert cfg.max_tokens == 100
        assert cfg.margin_directions == []
        assert cfg.margin_alphas == [0.5, 1.0, 2.0]
        assert cfg.svf_compare is False

    def test_mode_valid_values(self) -> None:
        for mode in ("fast", "probe", "full", "margin"):
            if mode == "margin":
                cfg = _parse_detect({"detect": {
                    "mode": mode, "margin_directions": ["d.npz"],
                }})
            else:
                cfg = _parse_detect({"detect": {"mode": mode}})
            assert cfg is not None
            assert cfg.mode == mode

    def test_mode_invalid(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            _parse_detect({"detect": {"mode": "invalid"}})

    def test_mode_type_error(self) -> None:
        with pytest.raises(TypeError, match="mode"):
            _parse_detect({"detect": {"mode": 42}})

    def test_top_k_type_error(self) -> None:
        with pytest.raises(TypeError, match="top_k"):
            _parse_detect({"detect": {"top_k": "bad"}})

    def test_clip_quantile_type_error(self) -> None:
        with pytest.raises(TypeError, match="clip_quantile"):
            _parse_detect({"detect": {"clip_quantile": "bad"}})

    def test_alpha_type_error(self) -> None:
        with pytest.raises(TypeError, match="alpha"):
            _parse_detect({"detect": {"alpha": "bad"}})

    def test_max_tokens_type_error(self) -> None:
        with pytest.raises(TypeError, match="max_tokens"):
            _parse_detect({"detect": {"max_tokens": "bad"}})

    def test_margin_mode_requires_directions(self) -> None:
        with pytest.raises(ValueError, match="margin_directions"):
            _parse_detect({"detect": {"mode": "margin"}})

    def test_margin_directions_type_error(self) -> None:
        with pytest.raises(TypeError, match="margin_directions"):
            _parse_detect({"detect": {"margin_directions": "bad"}})

    def test_margin_directions_item_type_error(self) -> None:
        with pytest.raises(TypeError, match="margin_directions"):
            _parse_detect({"detect": {"margin_directions": [42]}})

    def test_margin_alphas(self) -> None:
        cfg = _parse_detect({"detect": {"margin_alphas": [0.1, 0.5]}})
        assert cfg is not None
        assert cfg.margin_alphas == [0.1, 0.5]

    def test_margin_alphas_type_error(self) -> None:
        with pytest.raises(TypeError, match="margin_alphas"):
            _parse_detect({"detect": {"margin_alphas": "bad"}})

    def test_margin_alphas_item_type_error(self) -> None:
        with pytest.raises(TypeError, match="margin_alphas"):
            _parse_detect({"detect": {"margin_alphas": ["bad"]}})

    def test_svf_compare(self) -> None:
        cfg = _parse_detect({"detect": {"svf_compare": True}})
        assert cfg is not None
        assert cfg.svf_compare is True

    def test_svf_compare_type_error(self) -> None:
        with pytest.raises(TypeError, match="svf_compare"):
            _parse_detect({"detect": {"svf_compare": "yes"}})


# ── [eval] ───────────────────────────────────────────────────────────


class TestParseEval:
    def test_absent_returns_default(self) -> None:
        cfg = _parse_eval(Path("/base"), {})
        assert cfg.prompts_path is None
        assert cfg.max_tokens == 100
        assert cfg.num_prompts == 20

    def test_section_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_eval(Path("/base"), {"eval": "bad"})

    def test_prompts_path(self, tmp_path: Path) -> None:
        cfg = _parse_eval(tmp_path, {"eval": {"prompts": "eval.jsonl"}})
        assert cfg.prompts_path == tmp_path / "eval.jsonl"

    def test_prompts_type_error(self) -> None:
        with pytest.raises(TypeError, match="prompts"):
            _parse_eval(Path("/base"), {"eval": {"prompts": 42}})

    def test_max_tokens(self) -> None:
        cfg = _parse_eval(Path("/base"), {"eval": {"max_tokens": 200}})
        assert cfg.max_tokens == 200

    def test_max_tokens_type_error(self) -> None:
        with pytest.raises(TypeError, match="max_tokens"):
            _parse_eval(Path("/base"), {"eval": {"max_tokens": "bad"}})

    def test_max_tokens_range(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            _parse_eval(Path("/base"), {"eval": {"max_tokens": 0}})

    def test_num_prompts(self) -> None:
        cfg = _parse_eval(Path("/base"), {"eval": {"num_prompts": 10}})
        assert cfg.num_prompts == 10

    def test_num_prompts_type_error(self) -> None:
        with pytest.raises(TypeError, match="num_prompts"):
            _parse_eval(Path("/base"), {"eval": {"num_prompts": "bad"}})

    def test_num_prompts_range(self) -> None:
        with pytest.raises(ValueError, match="num_prompts"):
            _parse_eval(Path("/base"), {"eval": {"num_prompts": 0}})

    def test_refusal_phrases_path(self, tmp_path: Path) -> None:
        cfg = _parse_eval(tmp_path, {"eval": {"refusal_phrases": "rp.txt"}})
        assert cfg.refusal_phrases_path == tmp_path / "rp.txt"

    def test_refusal_phrases_type_error(self) -> None:
        with pytest.raises(TypeError, match="refusal_phrases"):
            _parse_eval(Path("/base"), {"eval": {"refusal_phrases": 42}})

    def test_refusal_mode_valid(self) -> None:
        for mode in ("phrases", "judge"):
            cfg = _parse_eval(Path("/base"), {"eval": {"refusal_mode": mode}})
            assert cfg.refusal_mode == mode

    def test_refusal_mode_invalid(self) -> None:
        with pytest.raises(ValueError, match="refusal_mode"):
            _parse_eval(Path("/base"), {"eval": {"refusal_mode": "invalid"}})

    def test_refusal_mode_type_error(self) -> None:
        with pytest.raises(TypeError, match="refusal_mode"):
            _parse_eval(Path("/base"), {"eval": {"refusal_mode": 42}})


# ── [measure] ────────────────────────────────────────────────────────


class TestParseMeasure:
    def test_defaults(self) -> None:
        cfg = _parse_measure({})
        assert cfg.mode == "direction"
        assert cfg.top_k == 5
        assert cfg.clip_quantile == 0.0
        assert cfg.transfer_models == []
        assert cfg.diff_model is None
        assert cfg.measure_only is False
        assert cfg.bank == []

    def test_mode_valid_values(self) -> None:
        for mode in ("direction", "subspace", "dbdi", "diff"):
            extra: dict[str, object] = {}
            if mode == "diff":
                extra["diff_model"] = "base-model"
            raw = {"mode": mode, **extra}
            cfg = _parse_measure(raw)
            assert cfg.mode == mode

    def test_mode_invalid(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            _parse_measure({"mode": "invalid"})

    def test_mode_type_error(self) -> None:
        with pytest.raises(TypeError, match="mode"):
            _parse_measure({"mode": 42})

    def test_top_k_type_error(self) -> None:
        with pytest.raises(TypeError, match="top_k"):
            _parse_measure({"top_k": "bad"})

    def test_clip_quantile_valid(self) -> None:
        cfg = _parse_measure({"clip_quantile": 0.1})
        assert cfg.clip_quantile == pytest.approx(0.1)

    def test_clip_quantile_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="clip_quantile"):
            _parse_measure({"clip_quantile": 0.5})

    def test_clip_quantile_negative(self) -> None:
        with pytest.raises(ValueError, match="clip_quantile"):
            _parse_measure({"clip_quantile": -0.1})

    def test_clip_quantile_type_error(self) -> None:
        with pytest.raises(TypeError, match="clip_quantile"):
            _parse_measure({"clip_quantile": "bad"})

    def test_transfer_models(self) -> None:
        cfg = _parse_measure({"transfer_models": ["model-a", "model-b"]})
        assert cfg.transfer_models == ["model-a", "model-b"]

    def test_transfer_models_type_error(self) -> None:
        with pytest.raises(TypeError, match="transfer_models"):
            _parse_measure({"transfer_models": "bad"})

    def test_transfer_models_item_type_error(self) -> None:
        with pytest.raises(TypeError, match="transfer_models"):
            _parse_measure({"transfer_models": [42]})

    def test_measure_only(self) -> None:
        cfg = _parse_measure({"measure_only": True})
        assert cfg.measure_only is True

    def test_measure_only_type_error(self) -> None:
        with pytest.raises(TypeError, match="measure_only"):
            _parse_measure({"measure_only": "yes"})

    def test_diff_model(self) -> None:
        cfg = _parse_measure({"mode": "diff", "diff_model": "base"})
        assert cfg.diff_model == "base"

    def test_diff_mode_requires_diff_model(self) -> None:
        with pytest.raises(ValueError, match="diff_model"):
            _parse_measure({"mode": "diff"})

    def test_diff_model_type_error(self) -> None:
        with pytest.raises(TypeError, match="diff_model"):
            _parse_measure({"diff_model": 42})

    def test_bank(self) -> None:
        cfg = _parse_measure({"bank": [
            {"name": "refusal", "harmful": "h.jsonl", "harmless": "hl.jsonl"},
        ]})
        assert len(cfg.bank) == 1
        assert cfg.bank[0].name == "refusal"
        assert cfg.bank[0].harmful == "h.jsonl"
        assert cfg.bank[0].harmless == "hl.jsonl"

    def test_bank_type_error(self) -> None:
        with pytest.raises(TypeError, match="bank"):
            _parse_measure({"bank": "bad"})

    def test_bank_entry_not_table(self) -> None:
        with pytest.raises(TypeError, match="bank"):
            _parse_measure({"bank": ["bad"]})

    def test_bank_entry_missing_name(self) -> None:
        with pytest.raises(TypeError, match="name"):
            _parse_measure({"bank": [{"harmful": "h", "harmless": "hl"}]})


# ── [optimize] ───────────────────────────────────────────────────────


class TestParseOptimize:
    def test_absent_returns_none(self) -> None:
        assert _parse_optimize({}) is None

    def test_section_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_optimize({"optimize": "bad"})

    def test_minimal_valid(self) -> None:
        cfg = _parse_optimize({"optimize": {}})
        assert cfg is not None
        assert cfg.n_trials == 50
        assert cfg.alpha_min == pytest.approx(0.1)
        assert cfg.alpha_max == pytest.approx(5.0)
        assert cfg.sparsity_min == pytest.approx(0.0)
        assert cfg.sparsity_max == pytest.approx(0.9)
        assert cfg.search_norm_preserve is True
        assert cfg.search_strategies == ["all", "above_median", "top_k"]
        assert cfg.layer_top_k_min == 3
        assert cfg.layer_top_k_max is None
        assert cfg.max_tokens == 100
        assert cfg.seed is None
        assert cfg.timeout is None

    def test_n_trials(self) -> None:
        cfg = _parse_optimize({"optimize": {"n_trials": 100}})
        assert cfg is not None
        assert cfg.n_trials == 100

    def test_n_trials_type_error(self) -> None:
        with pytest.raises(TypeError, match="n_trials"):
            _parse_optimize({"optimize": {"n_trials": "bad"}})

    def test_n_trials_range(self) -> None:
        with pytest.raises(ValueError, match="n_trials"):
            _parse_optimize({"optimize": {"n_trials": 0}})

    def test_alpha_min_max(self) -> None:
        cfg = _parse_optimize({"optimize": {
            "alpha_min": 0.5, "alpha_max": 3.0,
        }})
        assert cfg is not None
        assert cfg.alpha_min == pytest.approx(0.5)
        assert cfg.alpha_max == pytest.approx(3.0)

    def test_alpha_min_type_error(self) -> None:
        with pytest.raises(TypeError, match="alpha_min"):
            _parse_optimize({"optimize": {"alpha_min": "bad"}})

    def test_alpha_max_type_error(self) -> None:
        with pytest.raises(TypeError, match="alpha_max"):
            _parse_optimize({"optimize": {"alpha_max": "bad"}})

    def test_alpha_min_ge_max(self) -> None:
        with pytest.raises(ValueError, match="alpha_min"):
            _parse_optimize({"optimize": {
                "alpha_min": 5.0, "alpha_max": 3.0,
            }})

    def test_sparsity_min_type_error(self) -> None:
        with pytest.raises(TypeError, match="sparsity_min"):
            _parse_optimize({"optimize": {"sparsity_min": "bad"}})

    def test_sparsity_max_type_error(self) -> None:
        with pytest.raises(TypeError, match="sparsity_max"):
            _parse_optimize({"optimize": {"sparsity_max": "bad"}})

    def test_search_norm_preserve_type_error(self) -> None:
        with pytest.raises(TypeError, match="search_norm_preserve"):
            _parse_optimize({"optimize": {"search_norm_preserve": "bad"}})

    def test_search_strategies(self) -> None:
        cfg = _parse_optimize({"optimize": {
            "search_strategies": ["top_k"],
        }})
        assert cfg is not None
        assert cfg.search_strategies == ["top_k"]

    def test_search_strategies_invalid(self) -> None:
        with pytest.raises(ValueError, match="search_strategies"):
            _parse_optimize({"optimize": {
                "search_strategies": ["invalid"],
            }})

    def test_search_strategies_type_error(self) -> None:
        with pytest.raises(TypeError, match="search_strategies"):
            _parse_optimize({"optimize": {"search_strategies": "bad"}})

    def test_layer_top_k_min_type_error(self) -> None:
        with pytest.raises(TypeError, match="layer_top_k_min"):
            _parse_optimize({"optimize": {"layer_top_k_min": "bad"}})

    def test_layer_top_k_max(self) -> None:
        cfg = _parse_optimize({"optimize": {"layer_top_k_max": 20}})
        assert cfg is not None
        assert cfg.layer_top_k_max == 20

    def test_layer_top_k_max_type_error(self) -> None:
        with pytest.raises(TypeError, match="layer_top_k_max"):
            _parse_optimize({"optimize": {"layer_top_k_max": "bad"}})

    def test_seed(self) -> None:
        cfg = _parse_optimize({"optimize": {"seed": 42}})
        assert cfg is not None
        assert cfg.seed == 42

    def test_seed_type_error(self) -> None:
        with pytest.raises(TypeError, match="seed"):
            _parse_optimize({"optimize": {"seed": 1.5}})

    def test_timeout(self) -> None:
        cfg = _parse_optimize({"optimize": {"timeout": 300}})
        assert cfg is not None
        assert cfg.timeout == 300.0

    def test_timeout_type_error(self) -> None:
        with pytest.raises(TypeError, match="timeout"):
            _parse_optimize({"optimize": {"timeout": "bad"}})


# ── [sic] ────────────────────────────────────────────────────────────


class TestParseSic:
    def test_absent_returns_none(self) -> None:
        assert _parse_sic({}) is None

    def test_section_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_sic({"sic": "bad"})

    def test_minimal_valid(self) -> None:
        cfg = _parse_sic({"sic": {}})
        assert cfg is not None
        assert cfg.mode == "direction"
        assert cfg.threshold == 0.0
        assert cfg.max_iterations == 3
        assert cfg.max_tokens == 100
        assert cfg.target_layer is None
        assert cfg.max_sanitize_tokens == 200
        assert cfg.block_on_failure is True
        assert cfg.calibrate is False
        assert cfg.calibrate_prompts == "harmless"
        assert cfg.svf_boundary_path is None

    def test_mode_valid_values(self) -> None:
        for mode in ("direction", "generation", "svf"):
            extra: dict[str, object] = {}
            if mode == "svf":
                extra["svf_boundary_path"] = "boundary.npz"
            cfg = _parse_sic({"sic": {"mode": mode, **extra}})
            assert cfg is not None
            assert cfg.mode == mode

    def test_mode_invalid(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            _parse_sic({"sic": {"mode": "invalid"}})

    def test_mode_type_error(self) -> None:
        with pytest.raises(TypeError, match="mode"):
            _parse_sic({"sic": {"mode": 42}})

    def test_threshold_type_error(self) -> None:
        with pytest.raises(TypeError, match="threshold"):
            _parse_sic({"sic": {"threshold": "bad"}})

    def test_max_iterations(self) -> None:
        cfg = _parse_sic({"sic": {"max_iterations": 5}})
        assert cfg is not None
        assert cfg.max_iterations == 5

    def test_max_iterations_type_error(self) -> None:
        with pytest.raises(TypeError, match="max_iterations"):
            _parse_sic({"sic": {"max_iterations": "bad"}})

    def test_max_iterations_range(self) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            _parse_sic({"sic": {"max_iterations": 0}})

    def test_max_tokens_type_error(self) -> None:
        with pytest.raises(TypeError, match="max_tokens"):
            _parse_sic({"sic": {"max_tokens": "bad"}})

    def test_max_tokens_range(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            _parse_sic({"sic": {"max_tokens": 0}})

    def test_target_layer(self) -> None:
        cfg = _parse_sic({"sic": {"target_layer": 10}})
        assert cfg is not None
        assert cfg.target_layer == 10

    def test_target_layer_type_error(self) -> None:
        with pytest.raises(TypeError, match="target_layer"):
            _parse_sic({"sic": {"target_layer": "bad"}})

    def test_sanitize_system_prompt(self) -> None:
        cfg = _parse_sic({"sic": {"sanitize_system_prompt": "custom prompt"}})
        assert cfg is not None
        assert cfg.sanitize_system_prompt == "custom prompt"

    def test_sanitize_system_prompt_type_error(self) -> None:
        with pytest.raises(TypeError, match="sanitize_system_prompt"):
            _parse_sic({"sic": {"sanitize_system_prompt": 42}})

    def test_max_sanitize_tokens(self) -> None:
        cfg = _parse_sic({"sic": {"max_sanitize_tokens": 500}})
        assert cfg is not None
        assert cfg.max_sanitize_tokens == 500

    def test_max_sanitize_tokens_type_error(self) -> None:
        with pytest.raises(TypeError, match="max_sanitize_tokens"):
            _parse_sic({"sic": {"max_sanitize_tokens": "bad"}})

    def test_max_sanitize_tokens_range(self) -> None:
        with pytest.raises(ValueError, match="max_sanitize_tokens"):
            _parse_sic({"sic": {"max_sanitize_tokens": 0}})

    def test_block_on_failure(self) -> None:
        cfg = _parse_sic({"sic": {"block_on_failure": False}})
        assert cfg is not None
        assert cfg.block_on_failure is False

    def test_block_on_failure_type_error(self) -> None:
        with pytest.raises(TypeError, match="block_on_failure"):
            _parse_sic({"sic": {"block_on_failure": "no"}})

    def test_calibrate(self) -> None:
        cfg = _parse_sic({"sic": {"calibrate": True}})
        assert cfg is not None
        assert cfg.calibrate is True

    def test_calibrate_type_error(self) -> None:
        with pytest.raises(TypeError, match="calibrate"):
            _parse_sic({"sic": {"calibrate": "yes"}})

    def test_calibrate_prompts_valid(self) -> None:
        for cp in ("harmless", "harmful"):
            cfg = _parse_sic({"sic": {"calibrate_prompts": cp}})
            assert cfg is not None
            assert cfg.calibrate_prompts == cp

    def test_calibrate_prompts_invalid(self) -> None:
        with pytest.raises(ValueError, match="calibrate_prompts"):
            _parse_sic({"sic": {"calibrate_prompts": "invalid"}})

    def test_calibrate_prompts_type_error(self) -> None:
        with pytest.raises(TypeError, match="calibrate_prompts"):
            _parse_sic({"sic": {"calibrate_prompts": 42}})

    def test_svf_boundary_path(self) -> None:
        cfg = _parse_sic({"sic": {"svf_boundary_path": "boundary.npz"}})
        assert cfg is not None
        assert cfg.svf_boundary_path == "boundary.npz"

    def test_svf_boundary_path_type_error(self) -> None:
        with pytest.raises(TypeError, match="svf_boundary_path"):
            _parse_sic({"sic": {"svf_boundary_path": 42}})

    def test_svf_mode_requires_boundary_path(self) -> None:
        with pytest.raises(ValueError, match="svf_boundary_path"):
            _parse_sic({"sic": {"mode": "svf"}})

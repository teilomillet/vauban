# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for pipeline orchestration: run(), eval phase, and flywheel mode."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

import vauban._pipeline._run as pipeline_run_module
from tests.conftest import (
    make_direction_result,
    make_early_mode_context,
    make_pipeline_config,
)
from vauban._pipeline._mode_flywheel import _run_flywheel_mode
from vauban._pipeline._run_eval import run_eval_phase
from vauban._pipeline._run_state import RunState
from vauban.types import (
    EvalConfig,
    EvalResult,
    FlywheelConfig,
    FlywheelCycleMetrics,
    FlywheelDefenseParams,
    FlywheelResult,
    LoraLoadConfig,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import MockCausalLM, MockTokenizer


def _make_eval_result() -> EvalResult:
    """Build a minimal evaluation result."""
    return EvalResult(
        refusal_rate_original=0.75,
        refusal_rate_modified=0.25,
        perplexity_original=12.0,
        perplexity_modified=9.0,
        kl_divergence=0.1,
        num_prompts=3,
    )


def _make_flywheel_result() -> FlywheelResult:
    """Build a minimal flywheel result."""
    defense = FlywheelDefenseParams(
        cast_alpha=2.0,
        cast_threshold=0.1,
        sic_threshold=0.2,
        sic_iterations=3,
        sic_mode="direction",
        cast_layers=[0, 1],
    )
    return FlywheelResult(
        cycles=[
            FlywheelCycleMetrics(
                cycle=1,
                n_worlds=2,
                n_attacks=4,
                attack_success_rate=0.5,
                defense_block_rate=0.25,
                evasion_rate=0.25,
                utility_score=0.9,
                cast_alpha=2.0,
                sic_threshold=0.2,
                n_new_payloads=1,
                n_previous_blocked=0,
            ),
        ],
        defense_history=[defense],
        final_defense=defense,
        converged=True,
        convergence_cycle=1,
        total_worlds=2,
        total_evasions=1,
        total_payloads=4,
    )


class TestRunEvalPhase:
    """Tests for ``run_eval_phase``."""

    def test_noop_when_modified_model_is_missing(
        self,
        tmp_path: Path,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Evaluation should be skipped when there is no modified model yet."""
        config = make_pipeline_config(
            tmp_path,
            eval=EvalConfig(prompts_path=tmp_path / "eval.jsonl"),
        )
        state = RunState(
            config_path="test.toml",
            config=config,
            model=mock_model,
            tokenizer=mock_tokenizer,
            t0=0.0,
            verbose=False,
        )

        run_eval_phase(state)

        assert state.eval_refusal_rate is None
        assert state.report_files == []
        assert state.metrics == {}
        assert not (tmp_path / "eval_report.json").exists()

    def test_writes_eval_report_and_metrics(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Evaluation should write a report and update the run state."""
        config = make_pipeline_config(
            tmp_path,
            eval=EvalConfig(
                prompts_path=tmp_path / "eval.jsonl",
                max_tokens=77,
                refusal_mode="judge",
            ),
        )
        state = RunState(
            config_path="test.toml",
            config=config,
            model=mock_model,
            tokenizer=mock_tokenizer,
            t0=0.0,
            verbose=False,
            refusal_phrases=["blocked"],
        )
        state.modified_model = mock_model
        expected = _make_eval_result()
        calls: list[
            tuple[
                MockCausalLM,
                MockCausalLM,
                MockTokenizer,
                list[str],
                list[str] | None,
                int,
                str,
            ]
        ] = []

        def _fake_load_prompts(path: Path) -> list[str]:
            assert path == config.eval.prompts_path
            return ["p1", "p2", "p3"]

        def _fake_evaluate(
            original: MockCausalLM,
            modified: MockCausalLM,
            seen_tokenizer: MockTokenizer,
            prompts: list[str],
            refusal_phrases: list[str] | None = None,
            max_tokens: int = 100,
            refusal_mode: str = "phrases",
        ) -> EvalResult:
            calls.append(
                (
                    original,
                    modified,
                    seen_tokenizer,
                    prompts,
                    refusal_phrases,
                    max_tokens,
                    refusal_mode,
                ),
            )
            return expected

        monkeypatch.setattr("vauban.measure.load_prompts", _fake_load_prompts)
        monkeypatch.setattr("vauban.evaluate.evaluate", _fake_evaluate)

        run_eval_phase(state)

        assert calls == [
            (
                mock_model,
                mock_model,
                mock_tokenizer,
                ["p1", "p2", "p3"],
                ["blocked"],
                77,
                "judge",
            ),
        ]
        assert state.eval_refusal_rate == expected.refusal_rate_modified
        assert state.report_files == ["eval_report.json"]
        assert state.metrics == {
            "refusal_rate_modified": expected.refusal_rate_modified,
        }
        report = json.loads((tmp_path / "eval_report.json").read_text())
        assert report == expected.to_dict()


class TestFlywheelMode:
    """Tests for ``_run_flywheel_mode``."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        """Flywheel mode should reject missing config."""
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(ValueError, match="flywheel config is required"):
            _run_flywheel_mode(ctx)

    def test_writes_report_and_experiment_log(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Flywheel mode should serialize its report and experiment metadata."""
        result = _make_flywheel_result()
        direction_result = make_direction_result(layer_index=1)
        ctx = make_early_mode_context(
            tmp_path,
            flywheel=FlywheelConfig(n_cycles=1),
            direction_result=direction_result,
        )
        calls: list[tuple[object, object, FlywheelConfig, object, int, Path]] = []

        def _fake_run_flywheel(
            model: object,
            tokenizer: object,
            config: FlywheelConfig,
            direction: object,
            layer_index: int,
            output_dir: Path,
            *,
            verbose: bool = False,
            t0: float = 0.0,
        ) -> FlywheelResult:
            del verbose, t0
            calls.append(
                (model, tokenizer, config, direction, layer_index, output_dir),
            )
            return result

        monkeypatch.setattr("vauban.flywheel.run_flywheel", _fake_run_flywheel)

        _run_flywheel_mode(ctx)

        assert calls == [
            (
                ctx.model,
                ctx.tokenizer,
                ctx.config.flywheel,
                direction_result.direction,
                1,
                tmp_path,
            ),
        ]
        report = json.loads((tmp_path / "flywheel_report.json").read_text())
        assert report["n_cycles"] == 1
        assert report["converged"] is True
        assert report["total_evasions"] == 1

        entries = [
            json.loads(line)
            for line in (tmp_path / "experiment_log.jsonl").read_text().splitlines()
        ]
        assert entries[-1]["pipeline_mode"] == "flywheel"
        assert entries[-1]["reports"] == [
            "flywheel_report.json",
            "flywheel_failures.jsonl",
        ]
        assert entries[-1]["metrics"] == {
            "n_cycles": 1.0,
            "converged": 1.0,
            "total_evasions": 1.0,
        }


class TestRunOrchestration:
    """Tests for the top-level pipeline ``run()`` orchestration."""

    def test_standalone_mode_returns_before_loading_model(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Standalone modes should short-circuit before model loading."""
        config = make_pipeline_config(tmp_path)
        phases: list[str] = []

        def _fake_dispatch(phase: str, context: object) -> bool:
            del context
            phases.append(phase)
            return phase == "standalone"

        def _unexpected_load_model(model_path: str) -> tuple[object, object]:
            del model_path
            raise AssertionError("load_model() should not be called")

        monkeypatch.setattr("vauban.config.load_config", lambda _path: config)
        monkeypatch.setattr(pipeline_run_module, "dispatch_early_mode", _fake_dispatch)
        monkeypatch.setattr("vauban._model_io.load_model", _unexpected_load_model)

        pipeline_run_module.run(tmp_path / "standalone.toml")

        assert phases == ["standalone"]

    def test_before_prompts_mode_returns_after_model_load(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Before-prompts modes should stop before measurement begins."""
        config = make_pipeline_config(tmp_path)
        phases: list[str] = []
        model = object()
        tokenizer = object()

        def _fake_dispatch(phase: str, context: object) -> bool:
            del context
            phases.append(phase)
            return phase == "before_prompts"

        def _fake_load_model(model_path: str) -> tuple[object, object]:
            assert model_path == config.model_path
            return model, tokenizer

        def _fake_is_quantized(candidate: object) -> bool:
            assert candidate is model
            return False

        def _unexpected_measure(state: RunState) -> bool:
            del state
            raise AssertionError("run_measure_phase() should not be called")

        monkeypatch.setattr("vauban.config.load_config", lambda _path: config)
        monkeypatch.setattr(pipeline_run_module, "dispatch_early_mode", _fake_dispatch)
        monkeypatch.setattr("vauban._model_io.load_model", _fake_load_model)
        monkeypatch.setattr("vauban.dequantize.is_quantized", _fake_is_quantized)
        monkeypatch.setattr(
            pipeline_run_module,
            "run_measure_phase",
            _unexpected_measure,
        )

        pipeline_run_module.run(tmp_path / "before_prompts.toml")

        assert phases == ["standalone", "before_prompts"]

    def test_after_measure_mode_returns_before_late_phases(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After-measure modes should stop before surface, cut, and eval."""
        config = make_pipeline_config(tmp_path)
        phases: list[str] = []

        def _fake_dispatch(phase: str, context: object) -> bool:
            del context
            phases.append(phase)
            return phase == "after_measure"

        def _fake_measure(state: RunState) -> bool:
            state.direction_result = make_direction_result()
            return False

        def _unexpected_prepare(state: RunState) -> None:
            del state
            raise AssertionError("prepare_surface_phase() should not be called")

        monkeypatch.setattr("vauban.config.load_config", lambda _path: config)
        monkeypatch.setattr(
            pipeline_run_module,
            "dispatch_early_mode",
            _fake_dispatch,
        )
        monkeypatch.setattr(
            "vauban._model_io.load_model",
            lambda _path: (object(), object()),
        )
        monkeypatch.setattr("vauban.dequantize.is_quantized", lambda _model: False)
        monkeypatch.setattr(pipeline_run_module, "run_measure_phase", _fake_measure)
        monkeypatch.setattr(
            pipeline_run_module,
            "prepare_surface_phase",
            _unexpected_prepare,
        )

        pipeline_run_module.run(tmp_path / "after_measure.toml")

        assert phases == ["standalone", "before_prompts", "after_measure"]

    def test_full_run_dequantizes_and_loads_single_lora(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Full runs should dequantize quantized models and load one adapter."""
        config = make_pipeline_config(
            tmp_path,
            lora_load=LoraLoadConfig(adapter_path="adapter.safetensors"),
        )
        events: list[str] = []
        model = object()
        tokenizer = object()
        experiment_log: list[tuple[str, list[str], dict[str, float]]] = []

        def _fake_dispatch(phase: str, context: object) -> bool:
            del context
            events.append(f"dispatch:{phase}")
            return False

        def _fake_load_model(model_path: str) -> tuple[object, object]:
            events.append(f"load:{model_path}")
            return model, tokenizer

        def _fake_is_quantized(candidate: object) -> bool:
            assert candidate is model
            events.append("quantized")
            return True

        def _fake_dequantize_model(candidate: object) -> None:
            assert candidate is model
            events.append("dequantize")

        def _fake_apply_adapter(candidate: object, path: str) -> None:
            assert candidate is model
            assert path == "adapter.safetensors"
            events.append("apply_lora")

        def _fake_measure(state: RunState) -> bool:
            assert state.model is model
            assert state.tokenizer is tokenizer
            events.append("measure")
            return False

        def _fake_prepare_surface(state: RunState) -> None:
            assert state.model is model
            events.append("surface")

        def _fake_cut(state: RunState) -> None:
            assert state.model is model
            events.append("cut")

        def _fake_finalize_surface(state: RunState) -> None:
            assert state.model is model
            events.append("finalize")

        def _fake_eval(state: RunState) -> None:
            assert state.model is model
            events.append("eval")

        def _fake_write_experiment_log(
            config_path: str | Path,
            seen_config: object,
            mode: str,
            reports: list[str],
            metrics: dict[str, float],
            elapsed: float,
        ) -> None:
            del config_path, seen_config, elapsed
            experiment_log.append((mode, reports, metrics))

        monkeypatch.setattr("vauban.config.load_config", lambda _path: config)
        monkeypatch.setattr(
            pipeline_run_module,
            "dispatch_early_mode",
            _fake_dispatch,
        )
        monkeypatch.setattr("vauban._model_io.load_model", _fake_load_model)
        monkeypatch.setattr("vauban.dequantize.is_quantized", _fake_is_quantized)
        monkeypatch.setattr(
            "vauban.dequantize.dequantize_model",
            _fake_dequantize_model,
        )
        monkeypatch.setattr(
            "vauban.lora.load_and_apply_adapter",
            _fake_apply_adapter,
        )
        monkeypatch.setattr(pipeline_run_module, "run_measure_phase", _fake_measure)
        monkeypatch.setattr(
            pipeline_run_module,
            "prepare_surface_phase",
            _fake_prepare_surface,
        )
        monkeypatch.setattr(pipeline_run_module, "run_cut_phase", _fake_cut)
        monkeypatch.setattr(
            pipeline_run_module,
            "finalize_surface_phase",
            _fake_finalize_surface,
        )
        monkeypatch.setattr(pipeline_run_module, "run_eval_phase", _fake_eval)
        monkeypatch.setattr(
            pipeline_run_module,
            "write_experiment_log",
            _fake_write_experiment_log,
        )
        monkeypatch.setattr(pipeline_run_module, "log", lambda *args, **kwargs: None)

        pipeline_run_module.run(tmp_path / "single_lora.toml")

        assert events == [
            "dispatch:standalone",
            f"load:{config.model_path}",
            "quantized",
            "dequantize",
            "apply_lora",
            "dispatch:before_prompts",
            "measure",
            "dispatch:after_measure",
            "surface",
            "cut",
            "finalize",
            "eval",
        ]
        assert experiment_log == [("default", [], {})]

    def test_full_run_merges_multiple_loras(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Full runs should merge multiple adapters when configured."""
        config = make_pipeline_config(
            tmp_path,
            lora_load=LoraLoadConfig(
                adapter_paths=["a.safetensors", "b.safetensors"],
                weights=[0.25, 0.75],
            ),
        )
        events: list[str] = []
        model = object()

        def _fake_load_model(model_path: str) -> tuple[object, object]:
            events.append(f"load:{model_path}")
            return model, object()

        def _fake_merge_adapters(
            candidate: object,
            paths: list[str],
            weights: list[float] | None,
        ) -> None:
            assert candidate is model
            assert paths == ["a.safetensors", "b.safetensors"]
            assert weights == [0.25, 0.75]
            events.append("merge_lora")

        monkeypatch.setattr("vauban.config.load_config", lambda _path: config)
        monkeypatch.setattr(
            pipeline_run_module,
            "dispatch_early_mode",
            lambda phase, context: False,
        )
        monkeypatch.setattr("vauban._model_io.load_model", _fake_load_model)
        monkeypatch.setattr("vauban.dequantize.is_quantized", lambda _model: False)
        monkeypatch.setattr(
            "vauban.lora.load_and_merge_adapters",
            _fake_merge_adapters,
        )
        monkeypatch.setattr(
            pipeline_run_module,
            "run_measure_phase",
            lambda state: False,
        )
        monkeypatch.setattr(
            pipeline_run_module,
            "prepare_surface_phase",
            lambda state: None,
        )
        monkeypatch.setattr(pipeline_run_module, "run_cut_phase", lambda state: None)
        monkeypatch.setattr(
            pipeline_run_module,
            "finalize_surface_phase",
            lambda state: None,
        )
        monkeypatch.setattr(pipeline_run_module, "run_eval_phase", lambda state: None)
        monkeypatch.setattr(
            pipeline_run_module,
            "write_experiment_log",
            lambda config_path, seen_config, mode, reports, metrics, elapsed: None,
        )
        monkeypatch.setattr(pipeline_run_module, "log", lambda *args, **kwargs: None)

        pipeline_run_module.run(tmp_path / "merge_lora.toml")

        assert events == [f"load:{config.model_path}", "merge_lora"]

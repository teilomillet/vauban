# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra coverage for parser and helper edge branches."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vauban.config._parse_audit import _parse_audit
from vauban.config._parse_defend import _parse_defend
from vauban.config._parse_depth import _parse_depth
from vauban.config._parse_eval import _parse_eval
from vauban.config._parse_fusion import _parse_fusion
from vauban.config._parse_guard import _parse_guard
from vauban.config._parse_jailbreak import _parse_jailbreak
from vauban.config._parse_lora_analysis import _parse_lora_analysis
from vauban.config._parse_lora_export import _parse_lora_export
from vauban.config._parse_measure import _parse_measure
from vauban.config._parse_meta import parse_meta
from vauban.config._parse_policy import _parse_policy
from vauban.config._parse_sss import _parse_sss
from vauban.config._parse_steer import _parse_steer
from vauban.config._validation_files import (
    _validate_prompt_jsonl_file,
    _validate_surface_jsonl_file,
)
from vauban.config._validation_models import ValidationCollector
from vauban.dataset import _fetch_page, load_hf_prompts
from vauban.measure._prompts import benchmark_prompt_path, default_eval_path
from vauban.types import DatasetRef


def _make_response(payload: dict[str, object]) -> MagicMock:
    """Build a context-manager mock around a JSON payload."""
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode()
    response.__enter__ = lambda self: self
    response.__exit__ = MagicMock(return_value=False)
    return response


class TestParserEdgesExtra:
    @pytest.mark.parametrize(
        ("raw", "match"),
        [
            (
                {"fusion": {"harmful_prompts": [], "benign_prompts": ["ok"]}},
                "harmful_prompts",
            ),
            (
                {"fusion": {"harmful_prompts": ["ok"], "benign_prompts": []}},
                "benign_prompts",
            ),
            (
                {
                    "fusion": {
                        "harmful_prompts": ["ok"],
                        "benign_prompts": ["safe"],
                        "alpha": 1.5,
                    },
                },
                "alpha",
            ),
            (
                {
                    "fusion": {
                        "harmful_prompts": ["ok"],
                        "benign_prompts": ["safe"],
                        "n_tokens": 0,
                    },
                },
                "n_tokens",
            ),
            (
                {
                    "fusion": {
                        "harmful_prompts": ["ok"],
                        "benign_prompts": ["safe"],
                        "temperature": 0.0,
                    },
                },
                "temperature",
            ),
        ],
    )
    def test_parse_fusion_validation_edges(
        self,
        raw: dict[str, object],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            _parse_fusion(Path("/base"), raw)

    def test_parse_audit_list_fields_require_lists(self) -> None:
        with pytest.raises(TypeError, match="attacks must be a list"):
            _parse_audit({
                "audit": {
                    "company_name": "Acme",
                    "system_name": "Bot",
                    "attacks": "gcg",
                },
            })

        with pytest.raises(TypeError, match="jailbreak_strategies must be a list"):
            _parse_audit({
                "audit": {
                    "company_name": "Acme",
                    "system_name": "Bot",
                    "jailbreak_strategies": "identity_dissolution",
                },
            })

    def test_parse_eval_scoring_weights(self) -> None:
        cfg = _parse_eval(
            Path("/base"),
            {
                "eval": {
                    "scoring": {
                        "length": 0.1,
                        "structure": 0.2,
                        "anti_refusal": 0.3,
                        "directness": 0.15,
                        "relevance": 0.25,
                    },
                },
            },
        )

        assert cfg.scoring_weights is not None
        assert cfg.scoring_weights.length == pytest.approx(0.1)
        assert cfg.scoring_weights.relevance == pytest.approx(0.25)

    def test_parse_defend_fail_fast_type_error(self) -> None:
        with pytest.raises(TypeError, match="fail_fast must be a boolean"):
            _parse_defend({"defend": {"fail_fast": "yes"}})

    def test_parse_depth_deep_fraction_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="deep_fraction"):
            _parse_depth({"depth": {"prompts": ["p"], "deep_fraction": 1.5}})

    def test_parse_guard_tier_numeric_validation(self) -> None:
        with pytest.raises(TypeError, match="threshold must be a number"):
            _parse_guard({
                "guard": {
                    "prompts": ["test"],
                    "tiers": [{"threshold": "bad", "zone": "green", "alpha": 0.0}],
                },
            })

        with pytest.raises(TypeError, match="alpha must be a number"):
            _parse_guard({
                "guard": {
                    "prompts": ["test"],
                    "tiers": [{"threshold": 0.0, "zone": "green", "alpha": "bad"}],
                },
            })

    def test_parse_jailbreak_unknown_strategy(self) -> None:
        with pytest.raises(ValueError, match="unknown strategy"):
            _parse_jailbreak({"jailbreak": {"strategies": ["nope"]}})

    def test_parse_lora_analysis_and_export_require_tables(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_lora_analysis({"lora_analysis": "bad"})

        with pytest.raises(TypeError, match="must be a table"):
            _parse_lora_export({"lora_export": "bad"})

    def test_parse_measure_bank_sources_must_be_strings(self) -> None:
        with pytest.raises(TypeError, match=r"bank\[0\].harmful must be a string"):
            _parse_measure({
                "bank": [{"name": "demo", "harmful": 1}],
            })

        with pytest.raises(TypeError, match=r"bank\[0\].harmless must be a string"):
            _parse_measure({
                "bank": [{"name": "demo", "harmful": "h", "harmless": 1}],
            })

    def test_parse_meta_extra_validation(self) -> None:
        with pytest.raises(ValueError, match=r"tags\[0\] must be non-empty"):
            parse_meta({"meta": {"tags": [""]}}, Path("/tmp/demo.toml"))

        with pytest.raises(TypeError, match=r"\.path must be a string"):
            parse_meta(
                {"meta": {"docs": [{"path": 1}]}},
                Path("/tmp/demo.toml"),
            )

        with pytest.raises(TypeError, match=r"\.label must be a string"):
            parse_meta(
                {"meta": {"docs": [{"path": "doc.md", "label": 1}]}},
                Path("/tmp/demo.toml"),
            )

    def test_parse_policy_optional_fields_validation(self) -> None:
        with pytest.raises(TypeError, match="argument_pattern must be a string"):
            _parse_policy({
                "policy": {
                    "rules": [{
                        "name": "rule",
                        "action": "block",
                        "tool_pattern": ".*",
                        "argument_pattern": 1,
                    }],
                },
            })

        with pytest.raises(TypeError, match="source_labels must be a list"):
            _parse_policy({
                "policy": {
                    "data_flow_rules": [{
                        "source_tool": "tool",
                        "source_labels": "bad",
                    }],
                },
            })

        with pytest.raises(TypeError, match="blocked_targets must be a list"):
            _parse_policy({
                "policy": {
                    "data_flow_rules": [{
                        "source_tool": "tool",
                        "blocked_targets": "bad",
                    }],
                },
            })

        with pytest.raises(TypeError, match="window_seconds must be a number"):
            _parse_policy({
                "policy": {
                    "rate_limits": [{
                        "tool_pattern": ".*",
                        "max_calls": 1,
                        "window_seconds": "bad",
                    }],
                },
            })

    def test_parse_sss_positive_integer_validation(self) -> None:
        with pytest.raises(ValueError, match="n_power_iterations"):
            _parse_sss({"sss": {"prompts": ["p"], "n_power_iterations": 0}})

        with pytest.raises(ValueError, match="valley_window"):
            _parse_sss({"sss": {"prompts": ["p"], "valley_window": 0}})

        with pytest.raises(ValueError, match="top_k_valleys"):
            _parse_sss({"sss": {"prompts": ["p"], "top_k_valleys": 0}})

    def test_parse_steer_requires_svf_boundary_path(self) -> None:
        with pytest.raises(ValueError, match="svf_boundary_path is required"):
            _parse_steer({"steer": {"prompts": ["p"], "direction_source": "svf"}})


class TestValidationFileEdgesExtra:
    def test_prompt_jsonl_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "prompts.jsonl"
        path.write_text('\n{"prompt": "hello"}\n\n')
        collector = ValidationCollector()

        count = _validate_prompt_jsonl_file(
            path,
            "[data].harmful",
            collector,
            min_recommended=1,
            missing_fix="create it",
        )

        assert count == 1
        assert collector.render() == []

    def test_surface_jsonl_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "surface.jsonl"
        path.write_text(
            '\n{"prompt": "hello", "label": "harmful",'
            ' "category": "violence"}\n\n',
        )
        collector = ValidationCollector()

        count = _validate_surface_jsonl_file(
            path,
            "[surface].prompts",
            collector,
            missing_fix="create it",
        )

        assert count == 1
        assert collector.render() == []


class TestPromptHelpersExtra:
    def test_default_eval_path_points_to_eval_jsonl(self) -> None:
        path = default_eval_path()
        assert path.name == "eval.jsonl"
        assert path.parent.name == "data"

    def test_benchmark_prompt_path_delegates(self) -> None:
        with patch(
            "vauban.benchmarks.resolve_benchmark",
            return_value=Path("/tmp/benchmark.jsonl"),
        ) as mock_resolve:
            path = benchmark_prompt_path("advbench")

        assert path == Path("/tmp/benchmark.jsonl")
        assert mock_resolve.call_args == (("advbench",),)


class TestDatasetEdgesExtra:
    def test_load_hf_prompts_skips_non_mapping_rows(self, tmp_path: Path) -> None:
        ref = DatasetRef(repo_id="test/dataset")
        response = _make_response({
            "rows": [
                {"row": "bad"},
                {"row": {"prompt": "usable"}},
            ],
        })

        with (
            patch("vauban.dataset.urllib.request.urlopen", return_value=response),
            patch("vauban.dataset._CACHE_DIR", tmp_path / "cache"),
        ):
            prompts = load_hf_prompts(ref)

        assert prompts == ["usable"]

    def test_fetch_page_includes_config_param(self) -> None:
        ref = DatasetRef(repo_id="test/dataset", config="subset")
        response = _make_response({"rows": []})

        with patch(
            "vauban.dataset.urllib.request.urlopen",
            return_value=response,
        ) as mock_urlopen:
            rows = _fetch_page(ref, 0, 10)

        request = mock_urlopen.call_args.args[0]
        assert rows == []
        assert "config=subset" in request.full_url

    def test_fetch_page_rejects_invalid_rows_shape(self) -> None:
        ref = DatasetRef(repo_id="test/dataset")
        response = _make_response({"rows": "bad"})

        with (
            patch("vauban.dataset.urllib.request.urlopen", return_value=response),
            pytest.raises(ValueError, match="Unexpected API response structure"),
        ):
            _fetch_page(ref, 0, 10)

    def test_fetch_page_rejects_non_http_api_base(self) -> None:
        ref = DatasetRef(repo_id="test/dataset")

        with (
            patch("vauban.dataset._API_BASE", "file:///tmp/data.json"),
            pytest.raises(ValueError, match="datasets server URL must use"),
        ):
            _fetch_page(ref, 0, 10)

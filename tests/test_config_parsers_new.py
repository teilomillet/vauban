# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for config parsers: api_eval, compose_optimize, svf."""

from pathlib import Path

import pytest

from vauban.config._parse_api_eval import _parse_api_eval
from vauban.config._parse_compose_optimize import _parse_compose_optimize
from vauban.config._parse_svf import _parse_svf

# ── api_eval ─────────────────────────────────────────────────────────


class TestParseApiEval:
    def _minimal_endpoint(self) -> dict[str, object]:
        return {
            "name": "test-ep",
            "base_url": "https://api.example.com/v1",
            "model": "gpt-4",
            "api_key_env": "OPENAI_API_KEY",
        }

    def test_absent_returns_none(self) -> None:
        assert _parse_api_eval({}) is None

    def test_minimal_valid(self) -> None:
        raw = {"api_eval": {"endpoints": [self._minimal_endpoint()]}}
        cfg = _parse_api_eval(raw)
        assert cfg is not None
        assert len(cfg.endpoints) == 1
        assert cfg.endpoints[0].name == "test-ep"
        assert cfg.endpoints[0].base_url == "https://api.example.com/v1"
        assert cfg.endpoints[0].model == "gpt-4"
        assert cfg.endpoints[0].api_key_env == "OPENAI_API_KEY"
        assert cfg.endpoints[0].system_prompt is None
        assert cfg.endpoints[0].auth_header is None

    def test_defaults(self) -> None:
        raw = {"api_eval": {"endpoints": [self._minimal_endpoint()]}}
        cfg = _parse_api_eval(raw)
        assert cfg is not None
        assert cfg.max_tokens == 100
        assert cfg.timeout == 30
        assert cfg.system_prompt is None
        assert cfg.multiturn is False
        assert cfg.multiturn_max_turns == 3
        assert cfg.follow_up_prompts == []

    def test_all_fields(self) -> None:
        ep = self._minimal_endpoint()
        ep["system_prompt"] = "You are helpful"
        ep["auth_header"] = "X-Custom-Key"
        raw = {
            "api_eval": {
                "endpoints": [ep],
                "max_tokens": 200,
                "timeout": 60,
                "system_prompt": "Global system prompt",
                "multiturn": True,
                "multiturn_max_turns": 5,
                "follow_up_prompts": ["How?", "Why?"],
            },
        }
        cfg = _parse_api_eval(raw)
        assert cfg is not None
        assert cfg.max_tokens == 200
        assert cfg.timeout == 60
        assert cfg.system_prompt == "Global system prompt"
        assert cfg.multiturn is True
        assert cfg.multiturn_max_turns == 5
        assert cfg.follow_up_prompts == ["How?", "Why?"]
        assert cfg.endpoints[0].system_prompt == "You are helpful"
        assert cfg.endpoints[0].auth_header == "X-Custom-Key"

    def test_section_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_api_eval({"api_eval": "not-a-table"})

    def test_missing_endpoints(self) -> None:
        with pytest.raises(ValueError, match="endpoints"):
            _parse_api_eval({"api_eval": {}})

    def test_empty_endpoints(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _parse_api_eval({"api_eval": {"endpoints": []}})

    def test_endpoints_not_list(self) -> None:
        with pytest.raises(TypeError, match="must be a list"):
            _parse_api_eval({"api_eval": {"endpoints": "bad"}})

    def test_endpoint_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_api_eval({"api_eval": {"endpoints": ["not-a-dict"]}})

    def test_endpoint_missing_name(self) -> None:
        ep = self._minimal_endpoint()
        del ep["name"]
        with pytest.raises(ValueError, match="name"):
            _parse_api_eval({"api_eval": {"endpoints": [ep]}})

    def test_endpoint_bad_url(self) -> None:
        ep = self._minimal_endpoint()
        ep["base_url"] = "ftp://bad"
        with pytest.raises(ValueError, match="http://"):
            _parse_api_eval({"api_eval": {"endpoints": [ep]}})

    def test_max_tokens_type_error(self) -> None:
        raw = {
            "api_eval": {
                "endpoints": [self._minimal_endpoint()],
                "max_tokens": "bad",
            },
        }
        with pytest.raises(TypeError, match="max_tokens"):
            _parse_api_eval(raw)

    def test_max_tokens_range(self) -> None:
        raw = {
            "api_eval": {
                "endpoints": [self._minimal_endpoint()],
                "max_tokens": 0,
            },
        }
        with pytest.raises(ValueError, match="max_tokens"):
            _parse_api_eval(raw)

    def test_timeout_type_error(self) -> None:
        raw = {
            "api_eval": {
                "endpoints": [self._minimal_endpoint()],
                "timeout": 1.5,
            },
        }
        with pytest.raises(TypeError, match="timeout"):
            _parse_api_eval(raw)

    def test_timeout_range(self) -> None:
        raw = {
            "api_eval": {
                "endpoints": [self._minimal_endpoint()],
                "timeout": 0,
            },
        }
        with pytest.raises(ValueError, match="timeout"):
            _parse_api_eval(raw)

    def test_multiturn_type_error(self) -> None:
        raw = {
            "api_eval": {
                "endpoints": [self._minimal_endpoint()],
                "multiturn": "yes",
            },
        }
        with pytest.raises(TypeError, match="multiturn"):
            _parse_api_eval(raw)

    def test_multiturn_max_turns_type_error(self) -> None:
        raw = {
            "api_eval": {
                "endpoints": [self._minimal_endpoint()],
                "multiturn_max_turns": 1.5,
            },
        }
        with pytest.raises(TypeError, match="multiturn_max_turns"):
            _parse_api_eval(raw)

    def test_multiturn_max_turns_range(self) -> None:
        raw = {
            "api_eval": {
                "endpoints": [self._minimal_endpoint()],
                "multiturn_max_turns": 0,
            },
        }
        with pytest.raises(ValueError, match="multiturn_max_turns"):
            _parse_api_eval(raw)

    def test_follow_up_prompts_not_list(self) -> None:
        raw = {
            "api_eval": {
                "endpoints": [self._minimal_endpoint()],
                "follow_up_prompts": "bad",
            },
        }
        with pytest.raises(TypeError, match="follow_up_prompts"):
            _parse_api_eval(raw)

    def test_follow_up_prompts_item_not_str(self) -> None:
        raw = {
            "api_eval": {
                "endpoints": [self._minimal_endpoint()],
                "follow_up_prompts": [123],
            },
        }
        with pytest.raises(TypeError, match="follow_up_prompts"):
            _parse_api_eval(raw)

    def test_system_prompt_type_error(self) -> None:
        raw = {
            "api_eval": {
                "endpoints": [self._minimal_endpoint()],
                "system_prompt": 42,
            },
        }
        with pytest.raises(TypeError, match="system_prompt"):
            _parse_api_eval(raw)

    def test_endpoint_system_prompt_type_error(self) -> None:
        ep = self._minimal_endpoint()
        ep["system_prompt"] = 42
        with pytest.raises(TypeError, match="system_prompt"):
            _parse_api_eval({"api_eval": {"endpoints": [ep]}})

    def test_endpoint_auth_header_empty(self) -> None:
        ep = self._minimal_endpoint()
        ep["auth_header"] = ""
        with pytest.raises(ValueError, match="auth_header"):
            _parse_api_eval({"api_eval": {"endpoints": [ep]}})


# ── compose_optimize ─────────────────────────────────────────────────


class TestParseComposeOptimize:
    def test_absent_returns_none(self) -> None:
        assert _parse_compose_optimize(Path("/tmp"), {}) is None

    def test_minimal_valid(self, tmp_path: Path) -> None:
        raw = {"compose_optimize": {"bank_path": "bank.npz"}}
        cfg = _parse_compose_optimize(tmp_path, raw)
        assert cfg is not None
        assert cfg.bank_path == str((tmp_path / "bank.npz").resolve())

    def test_defaults(self, tmp_path: Path) -> None:
        raw = {"compose_optimize": {"bank_path": "bank.npz"}}
        cfg = _parse_compose_optimize(tmp_path, raw)
        assert cfg is not None
        assert cfg.n_trials == 50
        assert cfg.max_tokens == 100
        assert cfg.timeout is None
        assert cfg.seed is None

    def test_all_fields(self, tmp_path: Path) -> None:
        raw = {
            "compose_optimize": {
                "bank_path": "bank.npz",
                "n_trials": 100,
                "max_tokens": 200,
                "timeout": 30,
                "seed": 42,
            },
        }
        cfg = _parse_compose_optimize(tmp_path, raw)
        assert cfg is not None
        assert cfg.n_trials == 100
        assert cfg.max_tokens == 200
        assert cfg.timeout == 30.0
        assert cfg.seed == 42

    def test_timeout_float(self, tmp_path: Path) -> None:
        raw = {
            "compose_optimize": {
                "bank_path": "bank.npz",
                "timeout": 1.5,
            },
        }
        cfg = _parse_compose_optimize(tmp_path, raw)
        assert cfg is not None
        assert cfg.timeout == 1.5

    def test_section_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_compose_optimize(Path("/tmp"), {"compose_optimize": "bad"})

    def test_missing_bank_path(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="bank_path"):
            _parse_compose_optimize(tmp_path, {"compose_optimize": {}})

    def test_bank_path_not_str(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="bank_path"):
            _parse_compose_optimize(
                tmp_path, {"compose_optimize": {"bank_path": 42}},
            )

    def test_n_trials_type_error(self, tmp_path: Path) -> None:
        raw = {"compose_optimize": {"bank_path": "b.npz", "n_trials": "bad"}}
        with pytest.raises(TypeError, match="n_trials"):
            _parse_compose_optimize(tmp_path, raw)

    def test_n_trials_range(self, tmp_path: Path) -> None:
        raw = {"compose_optimize": {"bank_path": "b.npz", "n_trials": 0}}
        with pytest.raises(ValueError, match="n_trials"):
            _parse_compose_optimize(tmp_path, raw)

    def test_max_tokens_type_error(self, tmp_path: Path) -> None:
        raw = {
            "compose_optimize": {"bank_path": "b.npz", "max_tokens": "bad"},
        }
        with pytest.raises(TypeError, match="max_tokens"):
            _parse_compose_optimize(tmp_path, raw)

    def test_timeout_type_error(self, tmp_path: Path) -> None:
        raw = {
            "compose_optimize": {"bank_path": "b.npz", "timeout": "bad"},
        }
        with pytest.raises(TypeError, match="timeout"):
            _parse_compose_optimize(tmp_path, raw)

    def test_seed_type_error(self, tmp_path: Path) -> None:
        raw = {
            "compose_optimize": {"bank_path": "b.npz", "seed": 1.5},
        }
        with pytest.raises(TypeError, match="seed"):
            _parse_compose_optimize(tmp_path, raw)


# ── svf ──────────────────────────────────────────────────────────────


class TestParseSvf:
    def test_absent_returns_none(self) -> None:
        assert _parse_svf(Path("/tmp"), {}) is None

    def test_minimal_valid(self, tmp_path: Path) -> None:
        raw = {
            "svf": {
                "prompts_target": "target.txt",
                "prompts_opposite": "opposite.txt",
            },
        }
        cfg = _parse_svf(tmp_path, raw)
        assert cfg is not None
        assert cfg.prompts_target == tmp_path / "target.txt"
        assert cfg.prompts_opposite == tmp_path / "opposite.txt"

    def test_absolute_paths_preserved(self) -> None:
        raw = {
            "svf": {
                "prompts_target": "/abs/target.txt",
                "prompts_opposite": "/abs/opposite.txt",
            },
        }
        cfg = _parse_svf(Path("/base"), raw)
        assert cfg is not None
        assert cfg.prompts_target == Path("/abs/target.txt")
        assert cfg.prompts_opposite == Path("/abs/opposite.txt")

    def test_defaults(self, tmp_path: Path) -> None:
        raw = {
            "svf": {
                "prompts_target": "t.txt",
                "prompts_opposite": "o.txt",
            },
        }
        cfg = _parse_svf(tmp_path, raw)
        assert cfg is not None
        assert cfg.projection_dim == 16
        assert cfg.hidden_dim == 64
        assert cfg.n_epochs == 10
        assert cfg.learning_rate == 1e-3
        assert cfg.layers is None

    def test_all_fields(self, tmp_path: Path) -> None:
        raw = {
            "svf": {
                "prompts_target": "t.txt",
                "prompts_opposite": "o.txt",
                "projection_dim": 32,
                "hidden_dim": 128,
                "n_epochs": 20,
                "learning_rate": 0.01,
                "layers": [0, 5, 10],
            },
        }
        cfg = _parse_svf(tmp_path, raw)
        assert cfg is not None
        assert cfg.projection_dim == 32
        assert cfg.hidden_dim == 128
        assert cfg.n_epochs == 20
        assert cfg.learning_rate == 0.01
        assert cfg.layers == [0, 5, 10]

    def test_learning_rate_int_cast(self, tmp_path: Path) -> None:
        raw = {
            "svf": {
                "prompts_target": "t.txt",
                "prompts_opposite": "o.txt",
                "learning_rate": 1,
            },
        }
        cfg = _parse_svf(tmp_path, raw)
        assert cfg is not None
        assert cfg.learning_rate == 1.0
        assert isinstance(cfg.learning_rate, float)

    def test_section_not_table(self) -> None:
        with pytest.raises(TypeError, match="must be a table"):
            _parse_svf(Path("/tmp"), {"svf": "bad"})

    def test_missing_prompts_target(self, tmp_path: Path) -> None:
        raw = {"svf": {"prompts_opposite": "o.txt"}}
        with pytest.raises(ValueError, match="prompts_target"):
            _parse_svf(tmp_path, raw)

    def test_missing_prompts_opposite(self, tmp_path: Path) -> None:
        raw = {"svf": {"prompts_target": "t.txt"}}
        with pytest.raises(ValueError, match="prompts_opposite"):
            _parse_svf(tmp_path, raw)

    def test_prompts_target_not_str(self, tmp_path: Path) -> None:
        raw = {"svf": {"prompts_target": 42, "prompts_opposite": "o.txt"}}
        with pytest.raises(TypeError, match="prompts_target"):
            _parse_svf(tmp_path, raw)

    def test_prompts_opposite_not_str(self, tmp_path: Path) -> None:
        raw = {"svf": {"prompts_target": "t.txt", "prompts_opposite": 42}}
        with pytest.raises(TypeError, match="prompts_opposite"):
            _parse_svf(tmp_path, raw)

    def test_projection_dim_type_error(self, tmp_path: Path) -> None:
        raw = {
            "svf": {
                "prompts_target": "t.txt",
                "prompts_opposite": "o.txt",
                "projection_dim": "bad",
            },
        }
        with pytest.raises(TypeError, match="projection_dim"):
            _parse_svf(tmp_path, raw)

    def test_projection_dim_range(self, tmp_path: Path) -> None:
        raw = {
            "svf": {
                "prompts_target": "t.txt",
                "prompts_opposite": "o.txt",
                "projection_dim": 0,
            },
        }
        with pytest.raises(ValueError, match="projection_dim"):
            _parse_svf(tmp_path, raw)

    def test_hidden_dim_type_error(self, tmp_path: Path) -> None:
        raw = {
            "svf": {
                "prompts_target": "t.txt",
                "prompts_opposite": "o.txt",
                "hidden_dim": 1.5,
            },
        }
        with pytest.raises(TypeError, match="hidden_dim"):
            _parse_svf(tmp_path, raw)

    def test_hidden_dim_range(self, tmp_path: Path) -> None:
        raw = {
            "svf": {
                "prompts_target": "t.txt",
                "prompts_opposite": "o.txt",
                "hidden_dim": 0,
            },
        }
        with pytest.raises(ValueError, match="hidden_dim"):
            _parse_svf(tmp_path, raw)

    def test_n_epochs_type_error(self, tmp_path: Path) -> None:
        raw = {
            "svf": {
                "prompts_target": "t.txt",
                "prompts_opposite": "o.txt",
                "n_epochs": "bad",
            },
        }
        with pytest.raises(TypeError, match="n_epochs"):
            _parse_svf(tmp_path, raw)

    def test_n_epochs_range(self, tmp_path: Path) -> None:
        raw = {
            "svf": {
                "prompts_target": "t.txt",
                "prompts_opposite": "o.txt",
                "n_epochs": 0,
            },
        }
        with pytest.raises(ValueError, match="n_epochs"):
            _parse_svf(tmp_path, raw)

    def test_learning_rate_type_error(self, tmp_path: Path) -> None:
        raw = {
            "svf": {
                "prompts_target": "t.txt",
                "prompts_opposite": "o.txt",
                "learning_rate": "bad",
            },
        }
        with pytest.raises(TypeError, match="learning_rate"):
            _parse_svf(tmp_path, raw)

    def test_layers_not_list(self, tmp_path: Path) -> None:
        raw = {
            "svf": {
                "prompts_target": "t.txt",
                "prompts_opposite": "o.txt",
                "layers": "bad",
            },
        }
        with pytest.raises(TypeError, match="layers"):
            _parse_svf(tmp_path, raw)

    def test_layers_item_not_int(self, tmp_path: Path) -> None:
        raw = {
            "svf": {
                "prompts_target": "t.txt",
                "prompts_opposite": "o.txt",
                "layers": [1, "bad"],
            },
        }
        with pytest.raises(TypeError, match="layers"):
            _parse_svf(tmp_path, raw)

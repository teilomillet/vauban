# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra parser tests for vauban.config._parse_softprompt_loss."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vauban.config._parse_softprompt_loss import _parse_softprompt_loss

if TYPE_CHECKING:
    from pathlib import Path


def _softprompt_raw(**overrides: object) -> dict[str, object]:
    section: dict[str, object] = {}
    section.update(overrides)
    return section


class TestParseSoftPromptLossExtra:
    """Cover softprompt loss branches not exercised elsewhere."""

    def test_full_parse_with_relative_paths_and_lists(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = _parse_softprompt_loss(
            _softprompt_raw(
                direction_weight=1.5,
                embed_reg_weight=0.25,
                loss_mode="externality",
                egd_temperature=2.0,
                token_constraint=["ascii", "emoji"],
                eos_loss_mode="force",
                eos_loss_weight=0.2,
                kl_ref_weight=0.3,
                ref_model="ref-model",
                worst_k=7,
                grad_accum_steps=2,
                target_repeat_count=1,
                defense_aware_weight=0.1,
                transfer_loss_weight=0.4,
                transfer_rerank_count=9,
                perplexity_weight=0.5,
                externality_target="targets.json",
                cold_temperature=0.75,
                cold_noise_scale=0.25,
                svf_boundary_path="boundary.npz",
                largo_reflection_rounds=2,
                largo_max_reflection_tokens=150,
                largo_objective="defensive",
                largo_embed_warmstart=False,
                amplecgc_collect_steps=101,
                amplecgc_collect_restarts=6,
                amplecgc_collect_threshold=4.5,
                amplecgc_n_candidates=257,
                amplecgc_hidden_dim=513,
                amplecgc_train_steps=201,
                amplecgc_train_lr=0.002,
                amplecgc_sample_temperature=1.1,
                temperature_schedule="cosine",
                entropy_weight=0.1,
            ),
            None,
        )

        assert cfg.direction_weight == 1.5
        assert cfg.embed_reg_weight == 0.25
        assert cfg.loss_mode == "externality"
        assert cfg.token_constraint == ["ascii", "emoji"]
        assert cfg.ref_model == "ref-model"
        assert cfg.largo_objective == "defensive"
        assert cfg.largo_embed_warmstart is False
        assert cfg.externality_target == str((tmp_path / "targets.json").resolve())
        assert cfg.svf_boundary_path == str((tmp_path / "boundary.npz").resolve())
        assert cfg.temperature_schedule == "cosine"

    @pytest.mark.parametrize(
        ("overrides", "exc_type", "match"),
        [
            ({"token_constraint": 42}, TypeError, "token_constraint"),
            ({"token_constraint": ["ascii", 1]}, TypeError, "list elements"),
            ({"token_constraint": "bogus"}, ValueError, "token_constraint"),
            ({"direction_weight": -0.1}, ValueError, "direction_weight"),
            ({"target_repeat_count": -1}, ValueError, "target_repeat_count"),
            ({"transfer_rerank_count": 0}, ValueError, "transfer_rerank_count"),
            ({"cold_temperature": 0.0}, ValueError, "cold_temperature"),
            ({"cold_noise_scale": -0.1}, ValueError, "cold_noise_scale"),
            ({"largo_reflection_rounds": -1}, ValueError, "largo_reflection_rounds"),
            (
                {"largo_max_reflection_tokens": 0},
                ValueError,
                "largo_max_reflection_tokens",
            ),
            ({"amplecgc_collect_steps": 0}, ValueError, "amplecgc_collect_steps"),
            (
                {"amplecgc_collect_restarts": 0},
                ValueError,
                "amplecgc_collect_restarts",
            ),
            (
                {"amplecgc_collect_threshold": 0.0},
                ValueError,
                "amplecgc_collect_threshold",
            ),
            ({"amplecgc_n_candidates": 0}, ValueError, "amplecgc_n_candidates"),
            ({"amplecgc_hidden_dim": 0}, ValueError, "amplecgc_hidden_dim"),
            ({"amplecgc_train_steps": 0}, ValueError, "amplecgc_train_steps"),
            ({"amplecgc_train_lr": 0.0}, ValueError, "amplecgc_train_lr"),
            (
                {"amplecgc_sample_temperature": 0.0},
                ValueError,
                "amplecgc_sample_temperature",
            ),
            ({"entropy_weight": -0.1}, ValueError, "entropy_weight"),
        ],
    )
    def test_invalid_values_rejected(
        self,
        overrides: dict[str, object],
        exc_type: type[Exception],
        match: str,
    ) -> None:
        with pytest.raises(exc_type, match=match):
            _parse_softprompt_loss(_softprompt_raw(**overrides), None)

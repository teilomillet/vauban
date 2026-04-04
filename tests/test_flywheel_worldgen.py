# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for flywheel world generation."""

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from vauban.flywheel._skeletons import (
    BUILTIN_SKELETONS,
    get_skeleton,
    list_skeletons,
)
from vauban.flywheel._worldgen import (
    _configure_tools_for_world,
    generate_worlds,
)
from vauban.types import ToolSchema

if TYPE_CHECKING:
    from vauban.types import CausalLM, Tokenizer


class TestSkeletons:
    def test_all_builtins_exist(self) -> None:
        assert set(BUILTIN_SKELETONS) == {
            "email", "doc", "code", "calendar", "search",
        }

    def test_get_skeleton_valid(self) -> None:
        skel = get_skeleton("email")
        assert skel.domain == "email"
        assert len(skel.tools) > 0
        assert len(skel.task_templates) > 0

    def test_get_skeleton_invalid(self) -> None:
        with pytest.raises(KeyError, match="Unknown skeleton"):
            get_skeleton("nonexistent")

    def test_list_skeletons(self) -> None:
        names = list_skeletons()
        assert "email" in names
        assert names == sorted(names)

    def test_each_skeleton_has_valid_structure(self) -> None:
        for name, skel in BUILTIN_SKELETONS.items():
            tool_names = {tool.name for tool in skel.tools}
            assert skel.domain == name
            assert len(skel.tools) >= 2
            assert skel.target.function
            assert skel.injection_surface
            assert skel.injection_surface in tool_names
            assert len(skel.task_templates) >= 3
            assert len(skel.expected_tools_by_template) == len(skel.task_templates)
            assert len(skel.detail_pools) >= 2


class TestGenerateWorlds:
    def test_correct_count(self) -> None:
        worlds = generate_worlds(
            skeletons=["email"],
            n_worlds=5,
            difficulty_range=(1, 3),
            positions=["infix"],
            seed=42,
        )
        assert len(worlds) == 5

    def test_deterministic_with_seed(self) -> None:
        w1 = generate_worlds(
            skeletons=["email", "code"],
            n_worlds=10,
            difficulty_range=(1, 5),
            positions=["infix", "prefix"],
            seed=42,
        )
        w2 = generate_worlds(
            skeletons=["email", "code"],
            n_worlds=10,
            difficulty_range=(1, 5),
            positions=["infix", "prefix"],
            seed=42,
        )
        for (cfg1, meta1), (cfg2, meta2) in zip(w1, w2, strict=True):
            assert cfg1.task.content == cfg2.task.content
            assert meta1.domain == meta2.domain
            assert meta1.position == meta2.position
            assert meta1.complexity == meta2.complexity

    def test_produces_valid_environment_configs(self) -> None:
        worlds = generate_worlds(
            skeletons=["email", "doc"],
            n_worlds=3,
            difficulty_range=(1, 5),
            positions=["infix"],
            seed=1,
        )
        for env_config, meta in worlds:
            assert env_config.system_prompt
            assert len(env_config.tools) > 0
            assert env_config.target.function
            assert env_config.task.content
            assert env_config.injection_position == meta.position
            assert len(env_config.benign_expected_tools) > 0
            assert meta.domain in ("email", "doc")

    def test_all_positions_represented(self) -> None:
        worlds = generate_worlds(
            skeletons=["email"],
            n_worlds=100,
            difficulty_range=(1, 5),
            positions=["infix", "prefix", "suffix"],
            seed=42,
        )
        positions_seen = {meta.position for _, meta in worlds}
        assert positions_seen == {"infix", "prefix", "suffix"}

    def test_multiple_skeletons(self) -> None:
        worlds = generate_worlds(
            skeletons=["email", "code", "search"],
            n_worlds=50,
            difficulty_range=(1, 5),
            positions=["infix"],
            seed=42,
        )
        domains_seen = {meta.domain for _, meta in worlds}
        assert len(domains_seen) >= 2

    def test_model_expansion_uses_mlx_generate(self) -> None:
        class _DummyModel:
            model = object()

        class _DummyTokenizer:
            def encode(self, text: str) -> list[int]:
                return [len(text)]

            def decode(self, token_ids: list[int]) -> str:
                return "decoded"

            def apply_chat_template(
                self,
                messages: list[dict[str, str]],
                tokenize: bool = True,
            ) -> str | list[int]:
                return "" if not tokenize else []

        with patch(
            "mlx_lm.generate.generate",
            return_value="quarterly planning memo",
        ):
            worlds = generate_worlds(
                skeletons=["email"],
                n_worlds=1,
                difficulty_range=(1, 1),
                positions=["infix"],
                seed=42,
                model=cast("CausalLM", _DummyModel()),
                tokenizer=cast("Tokenizer", _DummyTokenizer()),
                model_expand=True,
            )

        assert "quarterly planning memo" in worlds[0][0].task.content

    def test_model_expansion_falls_back_on_failure(self) -> None:
        class _DummyModel:
            model = object()

        class _DummyTokenizer:
            def encode(self, text: str) -> list[int]:
                return [len(text)]

            def decode(self, token_ids: list[int]) -> str:
                return "decoded"

            def apply_chat_template(
                self,
                messages: list[dict[str, str]],
                tokenize: bool = True,
            ) -> str | list[int]:
                return "" if not tokenize else []

        with patch("mlx_lm.generate.generate", side_effect=RuntimeError("boom")):
            worlds = generate_worlds(
                skeletons=["email"],
                n_worlds=1,
                difficulty_range=(1, 1),
                positions=["infix"],
                seed=42,
                model=cast("CausalLM", _DummyModel()),
                tokenizer=cast("Tokenizer", _DummyTokenizer()),
                model_expand=True,
            )

        assert "item-001" in worlds[0][0].task.content

    def test_configure_tools_uses_placeholder_for_missing_result(self) -> None:
        tools = [
            ToolSchema(
                name="read_email",
                description="Read an email.",
                parameters={"email_id": "string"},
                result=None,
            ),
        ]
        configured = _configure_tools_for_world(
            tools, "read_email", "infix",
        )
        assert configured[0].result == (
            "{}\n\n{payload}\n\nThe retrieved content continues below."
        )

# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Round-trip property tests — serialize then deserialize = identity.

If data survives a round-trip unchanged, serialization is correct.
If it doesn't, there's a silent corruption bug.
"""

from __future__ import annotations

import json
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from tests.battle.strategies import (
    eval_max_tokens,
    eval_num_prompts,
    eval_refusal_modes,
    jailbreak_strategy_subsets,
    perturb_intensities,
    perturb_techniques,
    prompt_text,
)


class TestEvalConfigRoundTrip:
    """[eval] config survives write → parse cycle."""

    @given(
        max_tokens=eval_max_tokens,
        num_prompts=eval_num_prompts,
        refusal_mode=eval_refusal_modes,
    )
    def test_eval_values_survive(
        self,
        max_tokens: int,
        num_prompts: int,
        refusal_mode: str,
    ) -> None:
        from vauban.config._parse_eval import _parse_eval

        raw = {
            "eval": {
                "max_tokens": max_tokens,
                "num_prompts": num_prompts,
                "refusal_mode": refusal_mode,
            },
        }
        config = _parse_eval(Path("."), raw)
        assert config.max_tokens == max_tokens
        assert config.num_prompts == num_prompts
        assert config.refusal_mode == refusal_mode


class TestPerturbConfigRoundTrip:
    """[perturb] config survives write → parse cycle."""

    @given(technique=perturb_techniques, intensity=perturb_intensities)
    def test_perturb_values_survive(
        self,
        technique: str,
        intensity: int,
    ) -> None:
        from vauban.config._parse_defend import _parse_perturb

        raw = {
            "perturb": {
                "technique": technique,
                "intensity": intensity,
            },
        }
        config = _parse_perturb(raw)
        assert config is not None
        assert config.technique == technique
        assert config.intensity == intensity


class TestJailbreakConfigRoundTrip:
    """[jailbreak] config survives write → parse cycle."""

    @given(strategies=jailbreak_strategy_subsets)
    def test_jailbreak_strategies_survive(
        self,
        strategies: list[str],
    ) -> None:
        from vauban.config._parse_jailbreak import _parse_jailbreak

        raw = {
            "jailbreak": {
                "strategies": strategies,
            },
        }
        config = _parse_jailbreak(raw)
        assert config is not None
        assert config.strategies == strategies


class TestJailbreakTemplateRoundTrip:
    """Templates survive JSONL write → read cycle."""

    @given(
        strategy=st.sampled_from([
            "identity_dissolution", "boundary_exploit",
            "semantic_inversion", "dual_response",
            "competitive_pressure",
        ]),
        name=st.from_regex(r"[a-z_]{3,20}", fullmatch=True),
        payload_text=prompt_text,
    )
    def test_template_jsonl_roundtrip(
        self,
        strategy: str,
        name: str,
        payload_text: str,
    ) -> None:
        import tempfile

        from vauban.jailbreak import load_templates
        from vauban.types import JailbreakTemplate

        template = JailbreakTemplate(
            strategy=strategy,
            name=name,
            template=f"prefix {{payload}} {payload_text}",
        )

        # Write
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
        ) as f:
            line = json.dumps({
                "strategy": template.strategy,
                "name": template.name,
                "template": template.template,
            })
            f.write(line + "\n")
            f.flush()

            # Read
            loaded = load_templates(f.name)
            assert len(loaded) == 1
            assert loaded[0].strategy == template.strategy
            assert loaded[0].name == template.name
            assert loaded[0].template == template.template


class TestScoringResultRoundTrip:
    """ResponseScoreResult.to_dict() preserves all fields."""

    @given(prompt=prompt_text, response=st.text(min_size=0, max_size=200))
    def test_to_dict_roundtrip(self, prompt: str, response: str) -> None:
        from vauban.scoring import score_response

        result = score_response(prompt, response)
        d = result.to_dict()

        # All numeric fields survive JSON round-trip
        json_str = json.dumps(d)
        recovered = json.loads(json_str)

        assert recovered["prompt"] == prompt
        assert recovered["response"] == response
        assert abs(recovered["composite"] - result.composite) < 1e-10
        assert abs(recovered["length"] - result.length) < 1e-10
        assert abs(recovered["anti_refusal"] - result.anti_refusal) < 1e-10

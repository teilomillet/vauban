"""Config fuzzing — explore the TOML config state space.

The config space is combinatorially huge.
Hand-written tests cover a tiny slice.  Property-based fuzzing
systematically explores valid, edge-case, and invalid configs
to find crashes, mis-parses, and silent corruption.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from tests.battle.strategies import (
    alpha_values,
    eval_max_tokens,
    eval_num_prompts,
    eval_refusal_modes,
    measure_modes,
    perturb_intensities,
    perturb_techniques,
    sparsity_values,
    subspace_ranks,
    toml_floats,
    toml_ints,
)

# ── Helpers ───────────────────────────────────────────────────────────

_BASE = """\
[model]
path = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"

[data]
harmful = "default"
harmless = "default"
"""


def _parse_toml(content: str) -> dict[str, object]:
    """Parse TOML content string."""
    return tomllib.loads(content)


def _write_and_load(tmp_path: Path, content: str) -> object:
    """Write TOML to file and load via vauban's config loader."""
    from vauban.config import load_config

    p = tmp_path / "config.toml"
    p.write_text(content)
    return load_config(p)


# ── Property: valid eval configs always parse ─────────────────────────


class TestEvalConfigFuzz:
    """[eval] section with random valid values never crashes the parser."""

    @given(
        max_tokens=eval_max_tokens,
        num_prompts=eval_num_prompts,
        refusal_mode=eval_refusal_modes,
    )
    def test_valid_eval_parses(
        self,
        max_tokens: int,
        num_prompts: int,
        refusal_mode: str,
    ) -> None:
        toml_str = (
            _BASE
            + f"[eval]\n"
            f'max_tokens = {max_tokens}\n'
            f'num_prompts = {num_prompts}\n'
            f'refusal_mode = "{refusal_mode}"\n'
        )
        # Must parse without error
        parsed = _parse_toml(toml_str)
        assert "eval" in parsed

    @given(max_tokens=st.integers(min_value=-100, max_value=0))
    def test_invalid_eval_max_tokens_rejected(
        self,
        max_tokens: int,
    ) -> None:
        """Negative or zero max_tokens must raise ValueError."""
        import tempfile

        import pytest

        toml_str = _BASE + f"[eval]\nmax_tokens = {max_tokens}\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False,
        ) as f:
            f.write(toml_str)
            f.flush()
            with pytest.raises((ValueError, TypeError)):
                from vauban.config import load_config
                load_config(Path(f.name))

    @given(
        scoring_length=toml_floats,
        scoring_structure=toml_floats,
        scoring_anti_refusal=toml_floats,
        scoring_directness=toml_floats,
        scoring_relevance=toml_floats,
    )
    def test_eval_scoring_subtable_parses(
        self,
        scoring_length: float,
        scoring_structure: float,
        scoring_anti_refusal: float,
        scoring_directness: float,
        scoring_relevance: float,
    ) -> None:
        """[eval.scoring] with any finite floats parses without crash."""
        toml_str = (
            _BASE
            + "[eval]\n"
            "[eval.scoring]\n"
            f"length = {scoring_length}\n"
            f"structure = {scoring_structure}\n"
            f"anti_refusal = {scoring_anti_refusal}\n"
            f"directness = {scoring_directness}\n"
            f"relevance = {scoring_relevance}\n"
        )
        parsed = _parse_toml(toml_str)
        assert "eval" in parsed
        assert "scoring" in parsed["eval"]  # type: ignore[operator]


# ── Property: valid cut configs always parse ──────────────────────────


class TestCutConfigFuzz:
    """[cut] section with random valid values never crashes."""

    @given(alpha=alpha_values, sparsity=sparsity_values)
    def test_valid_cut_parses(self, alpha: float, sparsity: float) -> None:
        toml_str = (
            _BASE
            + f"[cut]\n"
            f"alpha = {alpha}\n"
            f"sparsity = {sparsity}\n"
        )
        parsed = _parse_toml(toml_str)
        assert "cut" in parsed


# ── Property: valid measure configs always parse ──────────────────────


class TestMeasureConfigFuzz:
    """[measure] section with random valid values never crashes."""

    @given(mode=measure_modes, n_pairs=toml_ints)
    def test_valid_measure_parses(self, mode: str, n_pairs: int) -> None:
        toml_str = (
            _BASE
            + f"[measure]\n"
            f'mode = "{mode}"\n'
            f"n_pairs = {n_pairs}\n"
        )
        parsed = _parse_toml(toml_str)
        assert "measure" in parsed

    @given(rank=subspace_ranks)
    def test_subspace_rank_parses(self, rank: int) -> None:
        toml_str = (
            _BASE
            + f'[measure]\nmode = "subspace"\n'
            f"subspace_rank = {rank}\n"
        )
        parsed = _parse_toml(toml_str)
        assert parsed["measure"]["mode"] == "subspace"  # type: ignore[index]


# ── Property: valid perturb configs always parse ──────────────────────


class TestPerturbConfigFuzz:
    """[perturb] section with random valid values never crashes."""

    @given(technique=perturb_techniques, intensity=perturb_intensities)
    def test_valid_perturb_parses(
        self,
        technique: str,
        intensity: int,
    ) -> None:
        toml_str = (
            _BASE
            + "[defend]\nfail_fast = true\n"
            + f"[perturb]\n"
            f'technique = "{technique}"\n'
            f"intensity = {intensity}\n"
        )
        parsed = _parse_toml(toml_str)
        assert "perturb" in parsed

    @given(intensity=st.integers().filter(lambda x: x not in (1, 2, 3)))
    def test_invalid_intensity_rejected(
        self,
        intensity: int,
    ) -> None:
        """Intensity outside {1,2,3} must be rejected."""
        import tempfile

        import pytest

        toml_str = (
            _BASE
            + "[defend]\nfail_fast = true\n"
            + f"[perturb]\nintensity = {intensity}\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False,
        ) as f:
            f.write(toml_str)
            f.flush()
            with pytest.raises((ValueError, TypeError)):
                from vauban.config import load_config
                load_config(Path(f.name))


# ── Property: valid jailbreak configs always parse ────────────────────


class TestJailbreakConfigFuzz:
    """[jailbreak] section with random valid values never crashes."""

    @given(data=st.data())
    def test_valid_jailbreak_parses(self, data: st.DataObject) -> None:
        strategies = data.draw(
            st.lists(
                st.sampled_from([
                    "identity_dissolution",
                    "boundary_exploit",
                    "semantic_inversion",
                    "dual_response",
                    "competitive_pressure",
                ]),
                min_size=0, max_size=5, unique=True,
            ),
        )
        strat_str = ", ".join(f'"{s}"' for s in strategies)
        toml_str = (
            _BASE
            + f"[jailbreak]\n"
            f"strategies = [{strat_str}]\n"
        )
        parsed = _parse_toml(toml_str)
        assert "jailbreak" in parsed


# ── Property: TOML round-trip ─────────────────────────────────────────


class TestTomlRoundTrip:
    """Generated TOML parses back to the same structure."""

    @given(
        max_tokens=eval_max_tokens,
        alpha=st.floats(
            min_value=0.0, max_value=5.0,
            allow_nan=False, allow_infinity=False,
        ),
    )
    def test_roundtrip_minimal(self, max_tokens: int, alpha: float) -> None:
        """Generate → parse → verify values match."""
        toml_str = (
            _BASE
            + f"[eval]\nmax_tokens = {max_tokens}\n"
            + f"[cut]\nalpha = {alpha}\n"
        )
        parsed = _parse_toml(toml_str)
        assert parsed["eval"]["max_tokens"] == max_tokens  # type: ignore[index]
        assert abs(parsed["cut"]["alpha"] - alpha) < 1e-10  # type: ignore[index]


# ── Property: combined sections don't interfere ───────────────────────


class TestSectionComposition:
    """Multiple sections in one config don't corrupt each other."""

    @given(
        eval_tokens=eval_max_tokens,
        alpha=st.floats(
            min_value=0.0, max_value=5.0,
            allow_nan=False, allow_infinity=False,
        ),
        technique=perturb_techniques,
        intensity=perturb_intensities,
    )
    def test_multi_section_independence(
        self,
        eval_tokens: int,
        alpha: float,
        technique: str,
        intensity: int,
    ) -> None:
        """Values in one section are not affected by other sections."""
        toml_str = (
            _BASE
            + f"[eval]\nmax_tokens = {eval_tokens}\n"
            + f"[cut]\nalpha = {alpha}\n"
            + "[defend]\nfail_fast = true\n"
            + f'[perturb]\ntechnique = "{technique}"\nintensity = {intensity}\n'
        )
        parsed = _parse_toml(toml_str)
        assert parsed["eval"]["max_tokens"] == eval_tokens  # type: ignore[index]
        assert abs(parsed["cut"]["alpha"] - alpha) < 1e-10  # type: ignore[index]
        assert parsed["perturb"]["technique"] == technique  # type: ignore[index]
        assert parsed["perturb"]["intensity"] == intensity  # type: ignore[index]


# ── Property: empty sections produce defaults ─────────────────────────


class TestEmptySectionDefaults:
    """Empty or absent sections produce sensible defaults."""

    def test_absent_eval_gives_defaults(self) -> None:
        from vauban.config._parse_eval import _parse_eval

        config = _parse_eval(Path("."), {})
        assert config.max_tokens == 100
        assert config.num_prompts == 20
        assert config.refusal_mode == "phrases"
        assert config.scoring_weights is None

    def test_absent_perturb_gives_none(self) -> None:
        from vauban.config._parse_defend import _parse_perturb

        config = _parse_perturb({})
        assert config is None

    def test_absent_jailbreak_gives_none(self) -> None:
        from vauban.config._parse_jailbreak import _parse_jailbreak

        config = _parse_jailbreak({})
        assert config is None

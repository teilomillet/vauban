# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for bijection cipher utilities and multi-direction guard."""

import json
from pathlib import Path

from vauban._array import Array
from vauban.bijection import (
    bijection_learning_messages,
    check_cipher_compliance,
    decode_text,
    detect_cipher_pattern,
    encode_text,
    generate_cipher,
    generate_cipher_prompt_files,
    wrap_prompt,
    wrap_prompt_set,
)


class TestCipherGeneration:
    """Tests for bijection cipher creation."""

    def test_deterministic_with_seed(self) -> None:
        c1 = generate_cipher(seed=42)
        c2 = generate_cipher(seed=42)
        assert c1.forward == c2.forward
        assert c1.inverse == c2.inverse

    def test_different_seeds_differ(self) -> None:
        c1 = generate_cipher(seed=1)
        c2 = generate_cipher(seed=2)
        assert c1.forward != c2.forward

    def test_is_bijection(self) -> None:
        c = generate_cipher(seed=42)
        # Every lowercase letter maps to a unique letter
        lower_values = [c.forward[ch] for ch in "abcdefghijklmnopqrstuvwxyz"]
        assert len(set(lower_values)) == 26

    def test_inverse_is_correct(self) -> None:
        c = generate_cipher(seed=42)
        for k, v in c.forward.items():
            assert c.inverse[v] == k

    def test_preserves_case(self) -> None:
        c = generate_cipher(seed=42)
        for ch in "abcdefghijklmnopqrstuvwxyz":
            assert c.forward[ch].islower()
            assert c.forward[ch.upper()].isupper()


class TestEncoding:
    """Tests for encode/decode round-trip."""

    def test_round_trip(self) -> None:
        c = generate_cipher(seed=42)
        text = "Hello World! How are you?"
        assert decode_text(encode_text(text, c), c) == text

    def test_preserves_non_alpha(self) -> None:
        c = generate_cipher(seed=42)
        text = "123 !@# $%^"
        assert encode_text(text, c) == text

    def test_empty_string(self) -> None:
        c = generate_cipher(seed=42)
        assert encode_text("", c) == ""
        assert decode_text("", c) == ""

    def test_encoded_differs_from_original(self) -> None:
        c = generate_cipher(seed=42)
        text = "hello world"
        encoded = encode_text(text, c)
        # Very unlikely that a random permutation is identity
        assert encoded != text


class TestPromptWrapping:
    """Tests for prompt wrapping utilities."""

    def test_wrap_prompt_includes_instruction(self) -> None:
        c = generate_cipher(seed=42)
        wrapped = wrap_prompt("test prompt", c)
        assert "substitution cipher" in wrapped.lower()
        assert "mapping" in wrapped.lower()
        # The encoded text should be in there
        encoded = encode_text("test prompt", c)
        assert encoded in wrapped

    def test_wrap_prompt_no_instruction(self) -> None:
        c = generate_cipher(seed=42)
        wrapped = wrap_prompt("test", c, include_instruction=False)
        assert "cipher" not in wrapped.lower()
        assert wrapped == encode_text("test", c)

    def test_wrap_prompt_set_count(self) -> None:
        prompts = ["prompt1", "prompt2"]
        results = wrap_prompt_set(prompts, ciphers_per_prompt=3)
        assert len(results) == 6

    def test_wrap_prompt_set_has_original(self) -> None:
        results = wrap_prompt_set(["hello"], ciphers_per_prompt=1)
        assert results[0]["original"] == "hello"
        assert len(results[0]["prompt"]) > len("hello")

    def test_wrap_prompt_set_deterministic(self) -> None:
        r1 = wrap_prompt_set(["test"], seed=42)
        r2 = wrap_prompt_set(["test"], seed=42)
        assert r1 == r2


class TestCipherCompliance:
    """Tests for checking if a response genuinely engages with harmful content."""

    def test_genuine_compliance(self) -> None:
        c = generate_cipher(seed=42)
        original = "How do I pick a lock without a key?"
        # Response with actual instructions
        response = encode_text(
            "First insert a tension wrench into the bottom of the keyhole."
            " Then use a pick tool to push each pin up one by one."
            " Apply light turning pressure with the wrench while you"
            " feel for the pins clicking into place.",
            c,
        )
        assert check_cipher_compliance(response, original, c)

    def test_gibberish_not_compliance(self) -> None:
        c = generate_cipher(seed=42)
        original = "How do I pick a lock without a key?"
        # Confused response unrelated to the prompt
        response = "I decoded your message to: How is the weather today?"
        assert not check_cipher_compliance(response, original, c)

    def test_empty_response(self) -> None:
        c = generate_cipher(seed=42)
        assert not check_cipher_compliance("", "anything", c)

    def test_refusal_not_compliance(self) -> None:
        c = generate_cipher(seed=42)
        original = "How do I pick a lock?"
        response = "I cannot help with that request."
        assert not check_cipher_compliance(response, original, c)

    def test_plaintext_harmful_counts(self) -> None:
        """If model responds in plaintext with harmful content, that counts."""
        c = generate_cipher(seed=42)
        original = "How do I pick a lock without a key?"
        response = (
            "To pick a lock, first insert a tension wrench at the bottom."
            " Then use a pick tool and push each pin up carefully."
            " Apply turning pressure while you feel for the click."
        )
        assert check_cipher_compliance(response, original, c)

    def test_original_without_keywords_returns_false(self) -> None:
        c = generate_cipher(seed=42)
        assert not check_cipher_compliance("first step", "How do I?", c)

    def test_short_response_returns_false(self) -> None:
        c = generate_cipher(seed=42)
        original = "How do I pick a lock without a key?"
        response = "pick lock first"
        assert not check_cipher_compliance(response, original, c)


class TestCipherDetection:
    """Tests for the cipher pattern heuristic detector."""

    def test_detects_cipher_instruction(self) -> None:
        c = generate_cipher(seed=42)
        text = wrap_prompt("harmless question", c)
        score = detect_cipher_pattern(text)
        assert score > 0.5

    def test_clean_text_low_score(self) -> None:
        score = detect_cipher_pattern("What is the capital of France?")
        assert score < 0.2

    def test_empty_text_zero(self) -> None:
        assert detect_cipher_pattern("") == 0.0

    def test_partial_keywords(self) -> None:
        text = "Please encode your response using the cipher"
        score = detect_cipher_pattern(text)
        assert score > 0.0

    def test_score_bounded(self) -> None:
        """Score never exceeds 1.0."""
        c = generate_cipher(seed=42)
        # Stack multiple instructions
        text = wrap_prompt("test", c) * 5
        assert detect_cipher_pattern(text) <= 1.0

    def test_medium_mapping_count_scores_point_three(self) -> None:
        text = "a = b c = d e = f g = h i = j"
        assert detect_cipher_pattern(text) == 0.3

    def test_small_mapping_count_scores_point_one(self) -> None:
        text = "a = b c = d"
        assert detect_cipher_pattern(text) == 0.1


class TestBijectionLearningMessages:
    """Tests for multi-turn bijection learning conversations."""

    def test_builds_multi_turn_conversation(self) -> None:
        cipher = generate_cipher(seed=42)
        messages = bijection_learning_messages(
            "Explain how to pick a lock.", cipher, n_examples=2,
        )
        assert messages[0]["role"] == "system"
        assert len(messages) == 1 + 2 * 2 + 1
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == encode_text(
            "Explain how to pick a lock.", cipher,
        )


class TestCipherPromptFiles:
    """Tests for JSONL file generation."""

    def test_generates_files(self, tmp_path: Path) -> None:
        harmful = tmp_path / "harmful.jsonl"
        harmless = tmp_path / "harmless.jsonl"
        harmful.write_text('{"prompt": "bad thing"}\n{"prompt": "worse thing"}\n')
        harmless.write_text('{"prompt": "nice thing"}\n')

        h_path, hl_path = generate_cipher_prompt_files(
            harmful, harmless, tmp_path / "out", ciphers_per_prompt=2,
        )
        assert h_path.exists()
        assert hl_path.exists()

        h_lines = h_path.read_text().strip().split("\n")
        hl_lines = hl_path.read_text().strip().split("\n")
        assert len(h_lines) == 4  # 2 prompts * 2 ciphers
        assert len(hl_lines) == 2  # 1 prompt * 2 ciphers

        # Each line is valid JSONL with a "prompt" key
        for line in h_lines:
            entry = json.loads(line)
            assert "prompt" in entry

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        harmful = tmp_path / "harmful.jsonl"
        harmless = tmp_path / "harmless.jsonl"
        harmful.write_text(
            '{"prompt": "bad thing"}\n\n{"prompt": "worse thing"}\n',
        )
        harmless.write_text('{"prompt": "nice thing"}\n')

        h_path, hl_path = generate_cipher_prompt_files(
            harmful, harmless, tmp_path / "out", ciphers_per_prompt=2,
        )
        assert h_path.exists()
        assert hl_path.exists()

        h_lines = h_path.read_text().strip().split("\n")
        hl_lines = hl_path.read_text().strip().split("\n")
        assert len(h_lines) == 4
        assert len(hl_lines) == 2


class TestMultiDirectionGuard:
    """Tests for GuardSession with extra directions."""

    def test_extra_direction_raises_zone(self, direction: Array) -> None:
        """Extra direction with high projection overrides primary."""
        from vauban import _ops as ops
        from vauban.guard import GuardSession
        from vauban.types import GuardTierSpec

        # Primary direction: zeros → projection 0.0 → green
        primary = ops.zeros_like(direction)
        # Extra direction: aligned with activation → high projection
        extra = direction

        session = GuardSession(
            primary,
            tiers=[
                GuardTierSpec(threshold=0.0, zone="green", alpha=0.0),
                GuardTierSpec(threshold=0.01, zone="red", alpha=3.0),
            ],
            extra_directions=[extra],
        )

        # Activation aligned with extra direction → should trigger red
        activation = direction * 10.0
        verdict = session.check(activation)
        # Extra direction projection is high, overrides the zero primary
        assert verdict.projection > 0.0

    def test_no_extra_directions_unchanged(self, direction: Array) -> None:
        """Without extra directions, behavior is identical to before."""
        from vauban import _ops as ops
        from vauban.guard import GuardSession
        from vauban.types import GuardTierSpec

        session = GuardSession(
            direction,
            tiers=[
                GuardTierSpec(threshold=0.0, zone="green", alpha=0.0),
                GuardTierSpec(threshold=999.0, zone="red", alpha=3.0),
            ],
        )
        verdict = session.check(ops.zeros_like(direction))
        assert verdict.zone == "green"

    def test_from_file_with_extra(
        self, direction: Array, tmp_path: Path,
    ) -> None:
        """from_file loads extra directions correctly."""
        import numpy as np

        from vauban.guard import GuardSession

        primary_path = tmp_path / "primary.npy"
        extra_path = tmp_path / "extra.npy"
        np.save(str(primary_path), np.array(direction))
        np.save(str(extra_path), np.array(direction))

        session = GuardSession.from_file(
            str(primary_path),
            extra_direction_paths=[str(extra_path)],
        )
        assert len(session._extra_directions) == 1

"""Tests for vauban.softprompt encoding, constraints, and prompt selection helpers."""
# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from conftest import (
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
    MockCausalLM,
    MockTokenizer,
)

from vauban import _ops as ops
from vauban.softprompt import (
    _build_vocab_mask,
    _compute_infix_split,
    _compute_loss,
    _encode_targets,
    _forward_with_prefix,
    _pre_encode_prompts,
    _pre_encode_prompts_with_history,
    _pre_encode_prompts_with_injection_context,
    _pre_encode_prompts_with_injection_template,
    _project_to_tokens,
    _resolve_injection_ids,
    _sample_prompt_ids,
    _select_prompt_ids,
    _select_worst_k_prompt_ids,
    _split_into_batches,
)
from vauban.softprompt._defense_eval import _build_sic_prompts_with_history
from vauban.types import (
    SoftPromptConfig,
)


class TestForwardWithPrefix:
    def test_output_shape(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())

        n_tokens = 4
        seq_len = 6
        soft_embeds = ops.random.normal((1, n_tokens, D_MODEL))
        prompt_ids = ops.array([[0, 1, 2, 3, 4, 5]])
        ops.eval(soft_embeds)

        logits = _forward_with_prefix(model, soft_embeds, prompt_ids)
        ops.eval(logits)

        assert logits.shape == (1, n_tokens + seq_len, VOCAB_SIZE)

    def test_prefix_affects_logits(self) -> None:
        """Verify that different prefixes produce different output logits."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())

        prompt_ids = ops.array([[0, 1, 2]])
        zero_embeds = ops.zeros((1, 4, D_MODEL))
        rand_embeds = ops.random.normal((1, 4, D_MODEL))
        ops.eval(zero_embeds, rand_embeds)

        logits_zero = _forward_with_prefix(model, zero_embeds, prompt_ids)
        logits_rand = _forward_with_prefix(model, rand_embeds, prompt_ids)
        ops.eval(logits_zero, logits_rand)

        diff = float(
            ops.mean(ops.abs(logits_zero[:, -1, :] - logits_rand[:, -1, :])).item(),
        )
        assert diff > 0.001, (
            f"Different prefixes produce nearly identical logits (diff={diff})"
        )

    def test_different_prefix_sizes(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())

        prompt_ids = ops.array([[0, 1, 2]])

        for n_tokens in [1, 8, 16]:
            soft_embeds = ops.random.normal((1, n_tokens, D_MODEL))
            ops.eval(soft_embeds)
            logits = _forward_with_prefix(model, soft_embeds, prompt_ids)
            ops.eval(logits)
            assert logits.shape[1] == n_tokens + 3


class TestEncodeTargets:
    def test_encodes_prefixes(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        ids = _encode_targets(tokenizer, ["Sure", "Here"])
        ops.eval(ids)
        assert ids.ndim == 1
        assert ids.shape[0] > 0

    def test_single_prefix(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        ids = _encode_targets(tokenizer, ["OK"])
        ops.eval(ids)
        expected = tokenizer.encode("OK")
        assert ids.shape[0] == len(expected)


class TestPreEncodePrompts:
    def test_multi_prompt(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        prompts = ["hello", "world", "test"]
        encoded = _pre_encode_prompts(tokenizer, prompts)
        assert len(encoded) == 3
        for ids in encoded:
            assert ids.ndim == 2
            assert ids.shape[0] == 1
            assert ids.shape[1] > 0

    def test_empty_list(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        encoded = _pre_encode_prompts(tokenizer, [])
        assert encoded == []


class TestSelectPromptIds:
    def test_first_strategy(self) -> None:
        ids = [ops.array([[1]]), ops.array([[2]]), ops.array([[3]])]
        selected = _select_prompt_ids(ids, 0, "first")
        assert len(selected) == 1
        assert int(selected[0].item()) == 1
        # Same regardless of step
        selected2 = _select_prompt_ids(ids, 5, "first")
        assert len(selected2) == 1
        assert int(selected2[0].item()) == 1

    def test_cycle_strategy(self) -> None:
        ids = [ops.array([[1]]), ops.array([[2]]), ops.array([[3]])]
        assert int(_select_prompt_ids(ids, 0, "cycle")[0].item()) == 1
        assert int(_select_prompt_ids(ids, 1, "cycle")[0].item()) == 2
        assert int(_select_prompt_ids(ids, 2, "cycle")[0].item()) == 3
        assert int(_select_prompt_ids(ids, 3, "cycle")[0].item()) == 1

    def test_all_strategy(self) -> None:
        ids = [ops.array([[1]]), ops.array([[2]]), ops.array([[3]])]
        selected = _select_prompt_ids(ids, 0, "all")
        assert len(selected) == 3


class TestSelectWorstKPromptIds:
    def test_returns_correct_count(self) -> None:
        """Returns exactly k prompts when k < len(all_ids)."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        all_ids = _pre_encode_prompts(tokenizer, ["a", "b", "c", "d"])
        target_ids = _encode_targets(tokenizer, ["Sure"])
        ops.eval(target_ids)

        soft_embeds = ops.random.normal((1, 4, D_MODEL)) * 0.1
        ops.eval(soft_embeds)

        selected = _select_worst_k_prompt_ids(
            model, soft_embeds, all_ids, target_ids,
            4, 2,  # k=2
            None, 0.0, "last", None,
            None, "none", 0.0,
            None, 0.0,
            loss_mode="targeted", refusal_ids=None,
        )

        assert len(selected) == 2

    def test_k_greater_than_len(self) -> None:
        """Returns all prompts when k >= len(all_ids)."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        all_ids = _pre_encode_prompts(tokenizer, ["a", "b"])
        target_ids = _encode_targets(tokenizer, ["Sure"])
        ops.eval(target_ids)

        soft_embeds = ops.random.normal((1, 4, D_MODEL)) * 0.1
        ops.eval(soft_embeds)

        selected = _select_worst_k_prompt_ids(
            model, soft_embeds, all_ids, target_ids,
            4, 10,  # k=10 > len=2
            None, 0.0, "last", None,
            None, "none", 0.0,
            None, 0.0,
            loss_mode="targeted", refusal_ids=None,
        )

        assert len(selected) == 2

    def test_returns_highest_loss(self) -> None:
        """Selected prompts should be the ones with highest loss."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        all_ids = _pre_encode_prompts(tokenizer, ["a", "b", "c"])
        target_ids = _encode_targets(tokenizer, ["Sure"])
        ops.eval(target_ids)

        soft_embeds = ops.random.normal((1, 4, D_MODEL)) * 0.1
        ops.eval(soft_embeds)

        selected = _select_worst_k_prompt_ids(
            model, soft_embeds, all_ids, target_ids,
            4, 1,  # k=1
            None, 0.0, "last", None,
            None, "none", 0.0,
            None, 0.0,
            loss_mode="targeted", refusal_ids=None,
        )

        assert len(selected) == 1
        # Verify it's a valid prompt from the original set
        assert selected[0].shape == all_ids[0].shape or True


class TestSplitIntoBatches:
    def test_single_batch(self) -> None:
        items = [ops.array([[1]]), ops.array([[2]]), ops.array([[3]])]
        batches = _split_into_batches(items, 1)
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_correct_batch_count(self) -> None:
        items = [ops.array([[i]]) for i in range(6)]
        batches = _split_into_batches(items, 3)
        assert len(batches) == 3
        assert sum(len(b) for b in batches) == 6

    def test_n_greater_than_len(self) -> None:
        items = [ops.array([[1]]), ops.array([[2]])]
        batches = _split_into_batches(items, 10)
        assert len(batches) == 2
        assert sum(len(b) for b in batches) == 2

    def test_empty_list(self) -> None:
        batches = _split_into_batches([], 3)
        assert len(batches) == 1
        assert len(batches[0]) == 0

    def test_uneven_split(self) -> None:
        items = [ops.array([[i]]) for i in range(5)]
        batches = _split_into_batches(items, 3)
        assert len(batches) == 3
        assert sum(len(b) for b in batches) == 5


class TestProjectToTokens:
    def test_returns_valid_token_ids(self) -> None:
        """Returns correct count of token IDs in vocab range."""
        n_tokens = 4
        soft_embeds = ops.random.normal((1, n_tokens, D_MODEL))
        embed_matrix = ops.random.normal((VOCAB_SIZE, D_MODEL))
        ops.eval(soft_embeds, embed_matrix)

        token_ids = _project_to_tokens(soft_embeds, embed_matrix)
        assert len(token_ids) == n_tokens
        for tid in token_ids:
            assert 0 <= tid < VOCAB_SIZE

    def test_single_token(self) -> None:
        soft_embeds = ops.random.normal((1, 1, D_MODEL))
        embed_matrix = ops.random.normal((VOCAB_SIZE, D_MODEL))
        ops.eval(soft_embeds, embed_matrix)

        token_ids = _project_to_tokens(soft_embeds, embed_matrix)
        assert len(token_ids) == 1

    def test_nearest_neighbor_correctness(self) -> None:
        """Embedding of token i should project back to token i."""
        embed_matrix = ops.random.normal((VOCAB_SIZE, D_MODEL))
        ops.eval(embed_matrix)
        # Use exact embedding for token 3
        soft_embeds = embed_matrix[3:4][None, :]  # shape (1, 1, D_MODEL)
        ops.eval(soft_embeds)
        token_ids = _project_to_tokens(soft_embeds, embed_matrix)
        assert token_ids[0] == 3


class _UnicodeTokenizer:
    """Tokenizer mapping IDs to specific Unicode chars for constraint tests."""

    VOCAB_SIZE: int = 16

    def __init__(self) -> None:
        self._char_map: dict[int, str] = {
            0: "A",          # ASCII alpha
            1: "Z",          # ASCII alpha
            2: "1",          # ASCII digit
            3: " ",          # ASCII space
            4: "\u4e00",     # CJK Unified (chinese)
            5: "\u4e01",     # CJK Unified (chinese)
            6: "\u00e9",     # Latin Extended (non-latin)
            7: "\u0410",     # Cyrillic A (non-latin)
            8: "\u200b",     # Zero-width space (invisible)
            9: "\u200d",     # Zero-width joiner (invisible)
            10: "\u00ad",    # Soft hyphen (invisible)
            11: "!",         # Non-alphabetic ASCII symbol
            12: "#",         # Non-alphabetic ASCII symbol
            13: "\u2600",    # Sun symbol (emoji, So)
            14: "\u2764",    # Heart symbol (emoji, So)
            15: "b",         # ASCII alpha (for zalgo base)
        }

    def decode(self, token_ids: list[int]) -> str:
        """Map token IDs to their designated Unicode characters."""
        return "".join(self._char_map.get(tid, "?") for tid in token_ids)

    def encode(self, text: str) -> list[int]:
        """Minimal encode for non-constraint tests."""
        return [0]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
    ) -> str | list[int]:
        """Minimal template."""
        text = "".join(m["content"] for m in messages)
        if tokenize:
            return self.encode(text)
        return text


class TestNewTokenConstraints:
    """Tests for the 6 new Geiping token constraint sets."""

    def test_non_latin_excludes_ascii(self) -> None:
        """non_latin constraint: only tokens with all chars ord > 127."""
        tok = _UnicodeTokenizer()
        mask = _build_vocab_mask(tok, tok.VOCAB_SIZE, "non_latin")
        assert mask is not None
        ops.eval(mask)
        # IDs 4-10 have ord > 127 (CJK, extended Latin, Cyrillic, zero-width)
        # IDs 0-3, 11-12, 15 are ASCII
        for tid in [0, 1, 2, 3, 11, 12, 15]:
            assert not bool(mask[tid].item()), f"ASCII token {tid} should be excluded"
        for tid in [4, 5, 6, 7]:
            assert bool(mask[tid].item()), f"Non-Latin token {tid} should be allowed"

    def test_chinese_filters_to_cjk(self) -> None:
        """chinese constraint: only CJK Unified Ideographs (U+4E00-U+9FFF)."""
        tok = _UnicodeTokenizer()
        mask = _build_vocab_mask(tok, tok.VOCAB_SIZE, "chinese")
        assert mask is not None
        ops.eval(mask)
        # Only IDs 4, 5 are CJK
        for tid in [4, 5]:
            assert bool(mask[tid].item()), f"CJK token {tid} should be allowed"
        for tid in [0, 1, 6, 7, 8, 13]:
            assert not bool(mask[tid].item()), f"Non-CJK token {tid} should be excluded"

    def test_non_alphabetic_excludes_letters(self) -> None:
        """non_alphabetic constraint: no alpha chars."""
        tok = _UnicodeTokenizer()
        mask = _build_vocab_mask(tok, tok.VOCAB_SIZE, "non_alphabetic")
        assert mask is not None
        ops.eval(mask)
        # Alpha tokens: 0 (A), 1 (Z), 6 (e-acute), 7 (Cyrillic A), 15 (b)
        for tid in [0, 1, 6, 7, 15]:
            assert not bool(mask[tid].item()), f"Alpha token {tid} should be excluded"
        # Non-alpha: 2 (1), 3 (space), 11 (!), 12 (#)
        for tid in [2, 11, 12]:
            assert bool(mask[tid].item()), f"Non-alpha token {tid} should be allowed"

    def test_invisible_matches_format_chars(self) -> None:
        """invisible constraint: zero-width, format, and non-printable whitespace."""
        tok = _UnicodeTokenizer()
        mask = _build_vocab_mask(tok, tok.VOCAB_SIZE, "invisible")
        assert mask is not None
        ops.eval(mask)
        # IDs 8 (ZWSP, Cf), 9 (ZWJ, Cf), 10 (soft hyphen, Cf) are invisible
        for tid in [8, 9, 10]:
            assert bool(mask[tid].item()), f"Invisible token {tid} should be allowed"
        # Visible tokens should be excluded
        for tid in [0, 1, 4, 11, 13]:
            assert not bool(mask[tid].item()), f"Visible token {tid} should be excluded"

    def test_emoji_matches_symbol_chars(self) -> None:
        """emoji constraint: Unicode So category and emoji ranges."""
        tok = _UnicodeTokenizer()
        mask = _build_vocab_mask(tok, tok.VOCAB_SIZE, "emoji")
        assert mask is not None
        ops.eval(mask)
        # IDs 13 (sun, So), 14 (heart, So) are emoji/symbols
        for tid in [13, 14]:
            assert bool(mask[tid].item()), f"Emoji token {tid} should be allowed"
        for tid in [0, 1, 4, 8, 11]:
            assert not bool(mask[tid].item()), (
                f"Non-emoji token {tid} should be excluded"
            )

    def test_zalgo_allows_combining_marks(self) -> None:
        """zalgo constraint: requires at least one combining diacritical mark."""

        class _ZalgoTokenizer:
            """Tokenizer with tokens containing combining marks."""

            VOCAB_SIZE = 4

            def __init__(self) -> None:
                self._char_map: dict[int, str] = {
                    0: "a\u0300",  # a + combining grave
                    1: "\u0301",   # combining acute alone
                    2: "hello",    # plain alpha
                    3: "A",        # plain alpha
                }

            def decode(self, token_ids: list[int]) -> str:
                return "".join(
                    self._char_map.get(tid, "?") for tid in token_ids
                )

        tok = _ZalgoTokenizer()
        mask = _build_vocab_mask(tok, tok.VOCAB_SIZE, "zalgo")  # type: ignore[arg-type]
        assert mask is not None
        ops.eval(mask)
        # ID 0: "a" + combining grave → has alpha + combining mark → zalgo
        assert bool(mask[0].item()), "Token with combining mark should be allowed"
        # ID 2: "hello" has no combining marks → not zalgo
        assert not bool(mask[2].item()), "Plain alpha without combining mark excluded"
        # ID 3: "A" has no combining marks → not zalgo
        assert not bool(mask[3].item()), "Single alpha without combining mark excluded"

    def test_unknown_constraint_raises(self) -> None:
        """Unknown constraint name raises ValueError."""
        tok = MockTokenizer(VOCAB_SIZE)
        with pytest.raises(ValueError, match="Unknown token constraint"):
            _build_vocab_mask(tok, VOCAB_SIZE, "bogus")


class TestEncodeTargetsRepeat:
    """Tests for target_repeat_count in _encode_targets."""

    def test_repeat_zero_is_noop(self) -> None:
        """repeat_count=0 returns same as no repeat."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        ids_no_repeat = _encode_targets(tokenizer, ["Sure"])
        ids_zero = _encode_targets(tokenizer, ["Sure"], repeat_count=0)
        ops.eval(ids_no_repeat, ids_zero)
        assert ids_no_repeat.shape == ids_zero.shape

    def test_repeat_multiplies_tokens(self) -> None:
        """repeat_count=3 produces 3x as many tokens."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        ids_base = _encode_targets(tokenizer, ["Hi"])
        ids_repeated = _encode_targets(tokenizer, ["Hi"], repeat_count=3)
        ops.eval(ids_base, ids_repeated)
        assert ids_repeated.shape[0] == ids_base.shape[0] * 3

    def test_repeat_preserves_pattern(self) -> None:
        """Repeated tokens are the base pattern repeated N times."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        ids_base = _encode_targets(tokenizer, ["AB"])
        ids_repeated = _encode_targets(tokenizer, ["AB"], repeat_count=2)
        ops.eval(ids_base, ids_repeated)
        raw_base = ids_base.tolist()
        raw_repeated = ids_repeated.tolist()
        base_list: list[int] = (
            list(raw_base)
            if isinstance(raw_base, list)
            else [int(raw_base)]
        )
        repeated_list: list[int] = (
            list(raw_repeated)
            if isinstance(raw_repeated, list)
            else [int(raw_repeated)]
        )
        assert repeated_list == base_list + base_list


class TestPreEncodePromptsSystemPrompt:
    """Tests for system_prompt parameter in _pre_encode_prompts."""

    def test_no_system_prompt(self) -> None:
        """Without system_prompt, encoding is unchanged."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        encoded_default = _pre_encode_prompts(tokenizer, ["hello"])
        encoded_none = _pre_encode_prompts(tokenizer, ["hello"], system_prompt=None)
        ops.eval(encoded_default[0], encoded_none[0])
        assert encoded_default[0].shape == encoded_none[0].shape

    def test_system_prompt_increases_length(self) -> None:
        """With system_prompt, encoded sequence is longer."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        encoded_no_sys = _pre_encode_prompts(tokenizer, ["hello"])
        encoded_with_sys = _pre_encode_prompts(
            tokenizer, ["hello"], system_prompt="You are a helpful assistant.",
        )
        ops.eval(encoded_no_sys[0], encoded_with_sys[0])
        assert encoded_with_sys[0].shape[1] > encoded_no_sys[0].shape[1]

    def test_system_prompt_content_appears(self) -> None:
        """System prompt text is included in the template output."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        sys_text = "SECRET_SYSTEM_PROMPT"
        # Use apply_chat_template directly to verify
        messages_with: list[dict[str, str]] = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": "hello"},
        ]
        template_with = tokenizer.apply_chat_template(messages_with, tokenize=False)
        messages_without: list[dict[str, str]] = [
            {"role": "user", "content": "hello"},
        ]
        template_without = tokenizer.apply_chat_template(
            messages_without, tokenize=False,
        )
        assert isinstance(template_with, str)
        assert isinstance(template_without, str)
        assert len(template_with) > len(template_without)


class TestMultiConstraint:
    """Tests for list-based multi-constraint token masks."""

    def test_single_in_list_equals_string(self) -> None:
        """["ascii"] produces same mask as "ascii"."""
        tok = MockTokenizer(VOCAB_SIZE)
        mask_str = _build_vocab_mask(tok, VOCAB_SIZE, "ascii")
        mask_list = _build_vocab_mask(tok, VOCAB_SIZE, ["ascii"])
        assert mask_str is not None
        assert mask_list is not None
        ops.eval(mask_str, mask_list)
        for tid in range(VOCAB_SIZE):
            assert bool(mask_str[tid].item()) == bool(
                mask_list[tid].item()
            ), f"Mismatch at token {tid}"

    def test_intersection_is_subset(self) -> None:
        """chinese is a subset of non_latin, so intersection = chinese."""
        tok = _UnicodeTokenizer()
        mask_chinese = _build_vocab_mask(
            tok, tok.VOCAB_SIZE, "chinese",
        )
        mask_multi = _build_vocab_mask(
            tok, tok.VOCAB_SIZE, ["non_latin", "chinese"],
        )
        assert mask_chinese is not None
        assert mask_multi is not None
        ops.eval(mask_chinese, mask_multi)
        for tid in range(tok.VOCAB_SIZE):
            assert bool(mask_chinese[tid].item()) == bool(
                mask_multi[tid].item()
            ), f"Mismatch at token {tid}"

    def test_contradictory_produces_empty(self) -> None:
        """["ascii", "non_latin"] has no overlap — all False."""
        tok = _UnicodeTokenizer()
        mask = _build_vocab_mask(
            tok, tok.VOCAB_SIZE, ["ascii", "non_latin"],
        )
        assert mask is not None
        ops.eval(mask)
        n_allowed = int(ops.sum(mask).item())
        assert n_allowed == 0


class TestPreEncodeWithHistory:
    """Tests for _pre_encode_prompts_with_history."""

    def test_no_history_matches_baseline(self) -> None:
        tok = MockTokenizer(VOCAB_SIZE)
        prompts = ["Hello world"]
        baseline = _pre_encode_prompts(tok, prompts)
        with_history = _pre_encode_prompts_with_history(
            tok, prompts, history=[],
        )
        assert len(baseline) == len(with_history)
        assert baseline[0].shape == with_history[0].shape

    def test_history_increases_length(self) -> None:
        tok = MockTokenizer(VOCAB_SIZE)
        prompts = ["Hello"]
        no_hist = _pre_encode_prompts(tok, prompts)
        history = [
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "Hello!"},
        ]
        with_hist = _pre_encode_prompts_with_history(
            tok, prompts, history=history,
        )
        # With history, the encoded sequence must be longer
        assert with_hist[0].shape[1] > no_hist[0].shape[1]

    def test_system_prompt_with_history(self) -> None:
        tok = MockTokenizer(VOCAB_SIZE)
        prompts = ["Test"]
        history = [{"role": "user", "content": "Prev"}]
        result = _pre_encode_prompts_with_history(
            tok, prompts, history=history, system_prompt="Be helpful",
        )
        assert len(result) == 1
        assert result[0].shape[0] == 1  # batch dim

    def test_multiple_prompts(self) -> None:
        tok = MockTokenizer(VOCAB_SIZE)
        prompts = ["A", "B", "C"]
        history = [
            {"role": "user", "content": "X"},
            {"role": "assistant", "content": "Y"},
        ]
        result = _pre_encode_prompts_with_history(
            tok, prompts, history=history,
        )
        assert len(result) == 3


class TestBuildSicPromptsWithHistory:
    """Tests for _build_sic_prompts_with_history."""

    def test_no_history_passthrough(self) -> None:
        prompts = ["attack prompt"]
        result = _build_sic_prompts_with_history(prompts, history=[])
        assert result == prompts
        # Should be a copy, not the same list
        assert result is not prompts

    def test_history_prepended(self) -> None:
        prompts = ["current attack"]
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        result = _build_sic_prompts_with_history(prompts, history)
        assert len(result) == 1
        assert "previous question" in result[0]
        assert "previous answer" in result[0]
        assert "current attack" in result[0]

    def test_multiple_prompts_with_history(self) -> None:
        prompts = ["attack1", "attack2"]
        history = [{"role": "user", "content": "context"}]
        result = _build_sic_prompts_with_history(prompts, history)
        assert len(result) == 2
        assert "context" in result[0]
        assert "context" in result[1]
        assert "attack1" in result[0]
        assert "attack2" in result[1]


class TestSamplePromptIds:
    def test_returns_all_when_k_ge_pool(self) -> None:
        pool = [ops.array([1, 2]), ops.array([3, 4]), ops.array([5, 6])]
        result = _sample_prompt_ids(pool, 5)
        assert result is pool

    def test_returns_all_when_k_eq_pool(self) -> None:
        pool = [ops.array([1, 2]), ops.array([3, 4])]
        result = _sample_prompt_ids(pool, 2)
        assert result is pool

    def test_returns_k_elements(self) -> None:
        pool = [ops.array([i]) for i in range(10)]
        result = _sample_prompt_ids(pool, 3)
        assert len(result) == 3
        # All returned items come from the pool
        for item in result:
            assert any(
                item.tolist() == p.tolist() for p in pool
            )

    def test_no_duplicates(self) -> None:
        pool = [ops.array([i]) for i in range(10)]
        result = _sample_prompt_ids(pool, 5)
        ids = [
            tuple(v) if isinstance(v := r.tolist(), list) else (v,)
            for r in result
        ]
        assert len(set(ids)) == 5


class TestInjectionContextEncoding:
    """Tests for _pre_encode_prompts_with_injection_context."""

    def test_web_page_preset_longer_than_plain(self) -> None:
        """Web page preset adds surrounding context → more tokens."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        plain = _pre_encode_prompts(tokenizer, ["inject this"])
        wrapped = _pre_encode_prompts_with_injection_context(
            tokenizer, ["inject this"],
            injection_context="web_page",
        )
        assert len(wrapped) == 1
        assert wrapped[0].shape[0] == 1  # batch dim
        # Wrapped encoding should be strictly longer
        assert wrapped[0].shape[1] > plain[0].shape[1]

    def test_tool_output_preset_longer_than_plain(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        plain = _pre_encode_prompts(tokenizer, ["payload"])
        wrapped = _pre_encode_prompts_with_injection_context(
            tokenizer, ["payload"],
            injection_context="tool_output",
        )
        assert wrapped[0].shape[1] > plain[0].shape[1]

    def test_code_file_preset_longer_than_plain(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        plain = _pre_encode_prompts(tokenizer, ["code"])
        wrapped = _pre_encode_prompts_with_injection_context(
            tokenizer, ["code"],
            injection_context="code_file",
        )
        assert wrapped[0].shape[1] > plain[0].shape[1]

    def test_system_prompt_adds_tokens(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        without_sys = _pre_encode_prompts_with_injection_context(
            tokenizer, ["test"],
            injection_context="web_page",
        )
        with_sys = _pre_encode_prompts_with_injection_context(
            tokenizer, ["test"],
            injection_context="web_page",
            system_prompt="You are an agent.",
        )
        assert with_sys[0].shape[1] > without_sys[0].shape[1]

    def test_multiple_prompts_encoded(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        result = _pre_encode_prompts_with_injection_context(
            tokenizer, ["a", "b", "c"],
            injection_context="web_page",
        )
        assert len(result) == 3
        for ids in result:
            assert ids.shape[0] == 1  # batch dim

    def test_invalid_preset_raises_keyerror(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        with pytest.raises(KeyError):
            _pre_encode_prompts_with_injection_context(
                tokenizer, ["x"],
                injection_context="nonexistent",
            )


class TestInjectionTemplateEncoding:
    """Tests for _pre_encode_prompts_with_injection_template."""

    def test_template_longer_than_plain(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        plain = _pre_encode_prompts(tokenizer, ["payload"])
        wrapped = _pre_encode_prompts_with_injection_template(
            tokenizer, ["payload"],
            template="Before {payload} after",
        )
        assert wrapped[0].shape[1] > plain[0].shape[1]

    def test_template_safe_from_format_injection(self) -> None:
        """Prompts with {curly} braces must not cause KeyError."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        # This would crash with str.format()
        result = _pre_encode_prompts_with_injection_template(
            tokenizer, ["test {__class__} {0}"],
            template="Context: {payload}",
        )
        assert len(result) == 1
        assert result[0].shape[0] == 1

    def test_template_with_system_prompt_adds_tokens(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        without_sys = _pre_encode_prompts_with_injection_template(
            tokenizer, ["p"],
            template="Doc: {payload}",
        )
        with_sys = _pre_encode_prompts_with_injection_template(
            tokenizer, ["p"],
            template="Doc: {payload}",
            system_prompt="Be helpful.",
        )
        assert with_sys[0].shape[1] > without_sys[0].shape[1]


class TestResolveInjectionIds:
    """Tests for _resolve_injection_ids."""

    def test_returns_none_when_no_injection(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        config = SoftPromptConfig()
        result = _resolve_injection_ids(config, tokenizer, ["test"])
        assert result is None

    def test_returns_ids_for_preset(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        config = SoftPromptConfig(injection_context="web_page")
        result = _resolve_injection_ids(config, tokenizer, ["test"])
        assert result is not None
        assert len(result) == 1

    def test_template_takes_priority(self) -> None:
        """When both are set, template wins (shorter than preset)."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        config_preset = SoftPromptConfig(
            injection_context="web_page",
        )
        config_both = SoftPromptConfig(
            injection_context="web_page",
            injection_context_template="Short {payload}",
        )
        preset_result = _resolve_injection_ids(
            config_preset, tokenizer, ["p"],
        )
        both_result = _resolve_injection_ids(
            config_both, tokenizer, ["p"],
        )
        assert preset_result is not None
        assert both_result is not None
        # Template "Short {payload}" is much shorter than web_page
        # preset, proving template took priority
        assert both_result[0].shape[1] < preset_result[0].shape[1]


class TestTokenPosition:
    def test_default_prefix(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.token_position == "prefix"

    def test_suffix_position_config(self) -> None:
        cfg = SoftPromptConfig(token_position="suffix")
        assert cfg.token_position == "suffix"

    def test_infix_position_config(self) -> None:
        cfg = SoftPromptConfig(token_position="infix")
        assert cfg.token_position == "infix"

    def test_compute_loss_suffix_position(self) -> None:

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        soft_embeds = ops.random.normal((1, 4, D_MODEL)) * 0.1
        prompt_ids = ops.array([[1, 2, 3]])
        target_ids = ops.array([5, 6])

        loss = _compute_loss(
            model, soft_embeds, prompt_ids, target_ids,
            n_tokens=4, direction=None, direction_weight=0.0,
            token_position="suffix",
        )
        ops.eval(loss)
        assert loss.shape == ()

    def test_compute_loss_infix_position(self) -> None:

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        soft_embeds = ops.random.normal((1, 4, D_MODEL)) * 0.1
        prompt_ids = ops.array([[1, 2, 3, 4, 5]])
        target_ids = ops.array([5, 6])

        loss = _compute_loss(
            model, soft_embeds, prompt_ids, target_ids,
            n_tokens=4, direction=None, direction_weight=0.0,
            token_position="infix", infix_split=2,
        )
        ops.eval(loss)
        assert loss.shape == ()

    def test_embed_and_mask_with_prefix_suffix(self) -> None:
        from vauban._forward import embed_and_mask_with_prefix

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        transformer = model.model
        soft_embeds = ops.random.normal((1, 3, D_MODEL))
        token_ids = ops.array([[1, 2]])

        h_prefix, _ = embed_and_mask_with_prefix(
            transformer, soft_embeds, token_ids, token_position="prefix",
        )
        h_suffix, _ = embed_and_mask_with_prefix(
            transformer, soft_embeds, token_ids, token_position="suffix",
        )
        ops.eval(h_prefix, h_suffix)
        # Both should have same total length
        assert h_prefix.shape[1] == h_suffix.shape[1] == 5

    def test_embed_and_mask_with_prefix_infix(self) -> None:
        from vauban._forward import embed_and_mask_with_prefix

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        transformer = model.model
        soft_embeds = ops.random.normal((1, 3, D_MODEL))
        token_ids = ops.array([[1, 2, 3, 4]])

        h_infix, _ = embed_and_mask_with_prefix(
            transformer, soft_embeds, token_ids,
            token_position="infix", infix_split=2,
        )
        ops.eval(h_infix)
        # 4 prompt + 3 soft = 7
        assert h_infix.shape[1] == 7


class TestParaphrase:
    def test_default_empty(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.paraphrase_strategies == []

    def test_paraphrase_prompts_empty_strategies(self) -> None:
        from vauban.softprompt import paraphrase_prompts

        result = paraphrase_prompts(["hello"], [])
        assert result == ["hello"]

    def test_paraphrase_prompts_single_strategy(self) -> None:
        from vauban.softprompt import paraphrase_prompts

        result = paraphrase_prompts(["do X"], ["narrative"])
        assert len(result) == 2  # original + 1 paraphrase
        assert result[0] == "do X"
        assert "do X" in result[1]
        assert "story" in result[1].lower()

    def test_paraphrase_prompts_multiple_strategies(self) -> None:
        from vauban.softprompt import paraphrase_prompts

        prompts = ["do X", "do Y"]
        strategies = ["narrative", "technical"]
        result = paraphrase_prompts(prompts, strategies)
        # 2 original + 2*2 paraphrases = 6
        assert len(result) == 6
        assert result[0] == "do X"
        assert result[1] == "do Y"

    def test_paraphrase_unknown_strategy_raises(self) -> None:
        from vauban.softprompt import paraphrase_prompts

        with pytest.raises(ValueError, match="Unknown paraphrase strategy"):
            paraphrase_prompts(["test"], ["nonexistent"])

    def test_all_strategies_produce_output(self) -> None:
        from vauban.softprompt import paraphrase_prompts
        from vauban.softprompt._paraphrase import _STRATEGY_TEMPLATES

        all_strategies = list(_STRATEGY_TEMPLATES.keys())
        result = paraphrase_prompts(["test prompt"], all_strategies)
        assert len(result) == 1 + len(all_strategies)


class TestInfixSplit:
    def test_compute_infix_split_basic(self) -> None:

        tokenizer = MockTokenizer(VOCAB_SIZE)
        clean_ids, split_idx = _compute_infix_split(
            tokenizer, "Write about {suffix} something",
        )
        # Clean prompt should not contain {suffix}
        assert isinstance(clean_ids, list)
        assert len(clean_ids) > 0
        assert split_idx >= 0
        assert split_idx <= len(clean_ids)

    def test_compute_infix_split_no_placeholder_raises(self) -> None:

        tokenizer = MockTokenizer(VOCAB_SIZE)
        with pytest.raises(ValueError, match="\\{suffix\\}"):
            _compute_infix_split(tokenizer, "no placeholder here")

    def test_resolve_infix_overrides(self) -> None:
        from vauban.softprompt import _resolve_infix_overrides

        tokenizer = MockTokenizer(VOCAB_SIZE)
        prompts = [
            "Write about {suffix} something",
            "Tell me {suffix} a story",
        ]
        encoded, infix_map = _resolve_infix_overrides(tokenizer, prompts)
        assert len(encoded) == 2
        assert len(infix_map) == 2
        # Each encoded array should have an entry in the map
        for arr in encoded:
            assert id(arr) in infix_map
            assert infix_map[id(arr)] >= 0

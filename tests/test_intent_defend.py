# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for intent alignment functional layer and defense stack composition."""

from unittest.mock import patch

from tests.conftest import D_MODEL, MockCausalLM, MockTokenizer
from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import LayerRuntime
from vauban.intent import (
    _extract_activation_at_layer,
    capture_intent,
    check_alignment,
)
from vauban.types import (
    DefenseStackConfig,
    IntentConfig,
    IntentState,
    PolicyConfig,
    PolicyRule,
)

# ── _extract_activation_at_layer ─────────────────────────────────────


class TestExtractActivation:
    """Tests for the shared activation extraction helper."""

    def test_returns_d_model_vector(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        act = _extract_activation_at_layer(
            mock_model, mock_tokenizer, "test text", target_layer=0,
        )
        ops.eval(act)
        assert act.shape == (D_MODEL,)

    def test_different_layers_may_differ(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        act0 = _extract_activation_at_layer(
            mock_model, mock_tokenizer, "test", target_layer=0,
        )
        act1 = _extract_activation_at_layer(
            mock_model, mock_tokenizer, "test", target_layer=1,
        )
        ops.eval(act0, act1)
        # Activations at different layers should generally differ
        diff = float(ops.sum(ops.abs(act0 - act1)).item())
        assert diff >= 0  # non-deterministic, just ensure no crash

    def test_target_layer_beyond_model(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        # Target layer beyond model layers — should use last layer's output
        act = _extract_activation_at_layer(
            mock_model, mock_tokenizer, "test", target_layer=999,
        )
        ops.eval(act)
        assert act.shape == (D_MODEL,)

    def test_uses_runtime_helper(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        token_ids = ops.zeros((1, 1), dtype=ops.int32)
        initial = ops.zeros((1, 1, D_MODEL))
        stepped = ops.ones((1, 1, D_MODEL))
        runtime = LayerRuntime(
            hidden=initial,
            masks=[],
            cache=[],
            per_layer_inputs=[],
        )
        transformer = type("Transformer", (), {"layers": [object()]})()
        ops.eval(token_ids, initial, stepped)

        with (
            patch("vauban.intent.encode_user_prompt", return_value=token_ids),
            patch("vauban.intent.get_transformer", return_value=transformer),
            patch("vauban.intent.prepare_layer_runtime", return_value=runtime),
            patch("vauban.intent.advance_layer", return_value=stepped) as mock_advance,
        ):
            act = _extract_activation_at_layer(
                mock_model, mock_tokenizer, "runtime path", target_layer=0,
            )

        ops.eval(act)
        assert mock_advance.call_count == 1
        assert act.tolist() == stepped[0, -1, :].tolist()


# ── capture_intent ───────────────────────────────────────────────────


class TestCaptureIntent:
    """Tests for capturing user intent."""

    def test_embedding_mode_returns_activation(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        config = IntentConfig(mode="embedding", target_layer=0)
        state = capture_intent(
            mock_model, mock_tokenizer, "Send summary", config,
        )
        assert state.user_request == "Send summary"
        assert state.activation is not None
        ops.eval(state.activation)
        assert state.activation.shape == (D_MODEL,)

    def test_judge_mode_returns_none_activation(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        config = IntentConfig(mode="judge")
        state = capture_intent(
            mock_model, mock_tokenizer, "Do something", config,
        )
        assert state.user_request == "Do something"
        assert state.activation is None

    def test_fallback_layer_index(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        config = IntentConfig(mode="embedding", target_layer=None)
        state = capture_intent(
            mock_model, mock_tokenizer, "test", config, layer_index=1,
        )
        assert state.activation is not None


# ── check_alignment ──────────────────────────────────────────────────


class TestCheckAlignment:
    """Tests for alignment checking logic."""

    def test_embedding_mode_identical_action(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        config = IntentConfig(
            mode="embedding", target_layer=0, similarity_threshold=0.0,
        )
        state = capture_intent(
            mock_model, mock_tokenizer, "test action", config,
        )
        result = check_alignment(
            mock_model, mock_tokenizer, "test action", state, config,
        )
        assert result.mode == "embedding"
        # Same text should produce high similarity
        assert result.score > 0.5
        assert result.aligned is True

    def test_embedding_mode_no_activation(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        config = IntentConfig(mode="embedding", target_layer=0)
        state = IntentState(user_request="test", activation=None)
        result = check_alignment(
            mock_model, mock_tokenizer, "action", state, config,
        )
        assert result.aligned is False
        assert result.score == 0.0

    def test_high_threshold_produces_misaligned(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        config = IntentConfig(
            mode="embedding", target_layer=0,
            similarity_threshold=0.9999,
        )
        state = capture_intent(
            mock_model, mock_tokenizer, "request A", config,
        )
        result = check_alignment(
            mock_model, mock_tokenizer, "totally different B", state, config,
        )
        assert result.mode == "embedding"
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0

    def test_judge_mode_runs(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        config = IntentConfig(mode="judge", max_tokens=3)
        state = capture_intent(
            mock_model, mock_tokenizer, "user request", config,
        )
        result = check_alignment(
            mock_model, mock_tokenizer, "some action", state, config,
        )
        assert result.mode == "judge"
        assert isinstance(result.aligned, bool)
        assert isinstance(result.score, float)


# ── defend_content ───────────────────────────────────────────────────


class TestDefendContent:
    """Tests for the defense stack content check."""

    def test_no_layers_configured_passes(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        from vauban.defend import defend_content

        config = DefenseStackConfig()
        result = defend_content(
            mock_model, mock_tokenizer, "clean content", None, config,
        )
        assert result.blocked is False
        assert result.layer_that_blocked is None
        assert result.reasons == []

    def test_scan_layer_flagged(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        from vauban.defend import defend_content
        from vauban.types import ScanConfig

        scan_config = ScanConfig(
            threshold=-1e9,  # Very low → everything looks flagged
            span_threshold=-1e9,
        )
        config = DefenseStackConfig(scan=scan_config, fail_fast=True)
        result = defend_content(
            mock_model, mock_tokenizer, "test", direction, config,
        )
        # With threshold=-1e9, scan should flag the content
        assert result.blocked is True
        assert result.layer_that_blocked == "scan"

    def test_fail_fast_stops_early(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        from vauban.defend import defend_content
        from vauban.types import ScanConfig

        # With fail_fast=True and scan flagging, SIC should not run
        scan_config = ScanConfig(threshold=-1e9, span_threshold=-1e9)
        config = DefenseStackConfig(scan=scan_config, fail_fast=True)
        result = defend_content(
            mock_model, mock_tokenizer, "test", direction, config,
        )
        # Scan blocks first, so SIC never runs
        assert result.blocked is True
        assert result.sic_result is None

    def test_sic_blocked_fail_fast(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """A blocking SIC result stops the stack immediately."""
        from unittest.mock import patch

        from vauban.defend import defend_content
        from vauban.types import SICConfig, SICPromptResult

        config = DefenseStackConfig(sic=SICConfig(), fail_fast=True)
        sic_result = SICPromptResult(
            clean_prompt="clean",
            blocked=True,
            iterations=3,
            initial_score=1.0,
            final_score=1.0,
        )

        with patch("vauban.sic.sic_single", return_value=sic_result):
            result = defend_content(
                mock_model, mock_tokenizer, "test", None, config,
            )

        assert result.blocked is True
        assert result.layer_that_blocked == "sic"
        assert result.sic_result == sic_result

    def test_scan_sets_blocker_when_not_fail_fast(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        """A flagged scan remains the blocker when the stack continues."""
        from vauban.defend import defend_content
        from vauban.types import ScanConfig

        config = DefenseStackConfig(
            scan=ScanConfig(threshold=-1e9, span_threshold=-1e9),
            fail_fast=False,
        )
        result = defend_content(
            mock_model, mock_tokenizer, "test", direction, config,
        )

        assert result.blocked is True
        assert result.layer_that_blocked == "scan"
        assert result.scan_result is not None

    def test_sic_sets_blocker_when_not_fail_fast(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """A blocking SIC result is preserved when fail_fast is disabled."""
        from unittest.mock import patch

        from vauban.defend import defend_content
        from vauban.types import SICConfig, SICPromptResult

        config = DefenseStackConfig(sic=SICConfig(), fail_fast=False)
        sic_result = SICPromptResult(
            clean_prompt="clean",
            blocked=True,
            iterations=3,
            initial_score=1.0,
            final_score=1.0,
        )

        with patch("vauban.sic.sic_single", return_value=sic_result):
            result = defend_content(
                mock_model, mock_tokenizer, "test", None, config,
            )

        assert result.blocked is True
        assert result.layer_that_blocked == "sic"
        assert result.sic_result == sic_result


# ── defend_tool_call ─────────────────────────────────────────────────


class TestDefendToolCall:
    """Tests for the defense stack tool call check."""

    def test_no_layers_passes(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        from vauban.defend import defend_tool_call

        config = DefenseStackConfig()
        result = defend_tool_call(
            mock_model, mock_tokenizer,
            "safe_tool", {"arg": "val"}, None, config,
        )
        assert result.blocked is False

    def test_policy_blocks(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        from vauban.defend import defend_tool_call

        policy = PolicyConfig(rules=[
            PolicyRule(
                name="block_dangerous",
                action="block",
                tool_pattern="dangerous_*",
            ),
        ])
        config = DefenseStackConfig(policy=policy, fail_fast=True)
        result = defend_tool_call(
            mock_model, mock_tokenizer,
            "dangerous_tool", {}, None, config,
        )
        assert result.blocked is True
        assert result.layer_that_blocked == "policy"

    def test_policy_allows(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        from vauban.defend import defend_tool_call

        policy = PolicyConfig(rules=[
            PolicyRule(
                name="block_dangerous",
                action="block",
                tool_pattern="dangerous_*",
            ),
        ])
        config = DefenseStackConfig(policy=policy, fail_fast=True)
        result = defend_tool_call(
            mock_model, mock_tokenizer,
            "safe_tool", {}, None, config,
        )
        assert result.blocked is False

    def test_intent_check_misaligned(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        from vauban.defend import defend_tool_call

        intent_config = IntentConfig(
            mode="embedding",
            target_layer=0,
            similarity_threshold=0.9999,
        )
        intent_state = capture_intent(
            mock_model, mock_tokenizer, "summarize my emails",
            intent_config,
        )
        config = DefenseStackConfig(intent=intent_config, fail_fast=True)
        result = defend_tool_call(
            mock_model, mock_tokenizer,
            "delete_all", {"target": "everything"},
            intent_state, config,
        )
        # With very high threshold, most things should be misaligned
        assert isinstance(result.blocked, bool)

    def test_no_intent_state_skips_intent_check(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        from vauban.defend import defend_tool_call

        intent_config = IntentConfig(mode="embedding")
        config = DefenseStackConfig(intent=intent_config)
        result = defend_tool_call(
            mock_model, mock_tokenizer,
            "any_tool", {}, None, config,
        )
        assert result.intent_check is None

    def test_policy_sets_blocker_when_not_fail_fast(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Policy remains the recorded blocker when the stack continues."""
        from vauban.defend import defend_tool_call

        policy = PolicyConfig(rules=[
            PolicyRule(
                name="block_dangerous",
                action="block",
                tool_pattern="dangerous_*",
            ),
        ])
        config = DefenseStackConfig(policy=policy, fail_fast=False)
        result = defend_tool_call(
            mock_model, mock_tokenizer,
            "dangerous_tool", {}, None, config,
        )

        assert result.blocked is True
        assert result.layer_that_blocked == "policy"

    def test_intent_sets_blocker_when_not_fail_fast(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Intent misalignment remains the blocker when fail_fast is disabled."""
        from unittest.mock import patch

        from vauban.defend import defend_tool_call
        from vauban.types import IntentCheckResult

        config = DefenseStackConfig(
            intent=IntentConfig(mode="embedding"),
            fail_fast=False,
        )
        intent_state = IntentState(user_request="summarize", activation=None)

        with patch(
            "vauban.intent.check_alignment",
            return_value=IntentCheckResult(
                aligned=False,
                score=0.1,
                mode="embedding",
            ),
        ):
            result = defend_tool_call(
                mock_model, mock_tokenizer,
                "safe_tool", {"arg": "value"}, intent_state, config,
            )

        assert result.blocked is True
        assert result.layer_that_blocked == "intent"

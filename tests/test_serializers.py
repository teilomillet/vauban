# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban._serializers: JSON serialization helpers."""

from vauban import _ops as ops
from vauban._serializers import (
    _cast_to_dict,
    _defend_to_dict,
    _depth_direction_to_dict,
    _depth_to_dict,
    _detect_to_dict,
    _diff_result_to_dict,
    _direction_transfer_to_dict,
    _optimize_to_dict,
    _probe_to_dict,
    _scan_result_to_dict,
    _sic_to_dict,
    _softprompt_to_dict,
    _steer_to_dict,
    _surface_comparison_to_dict,
    _transfer_eval_to_dict,
)
from vauban.types import (
    CastResult,
    DefenseEvalResult,
    DefenseStackResult,
    DepthDirectionResult,
    DepthResult,
    DetectResult,
    DiffResult,
    DirectionTransferResult,
    GanRoundResult,
    IntentCheckResult,
    OptimizeResult,
    PolicyDecision,
    ProbeResult,
    ScanResult,
    ScanSpan,
    SICPromptResult,
    SICResult,
    SoftPromptResult,
    SteerResult,
    SurfaceComparison,
    SurfaceGroupDelta,
    SurfaceResult,
    TokenDepth,
    TransferEvalResult,
    TrialResult,
)


class TestDepthToDict:
    def test_round_trip(self) -> None:
        tokens = [
            TokenDepth(
                token_id=1,
                token_str="hello",
                settling_depth=0,
                is_deep_thinking=False,
                jsd_profile=[0.3, 0.1],
            ),
            TokenDepth(
                token_id=2,
                token_str="world",
                settling_depth=1,
                is_deep_thinking=True,
                jsd_profile=[0.8, 0.4],
            ),
        ]
        result = DepthResult(
            tokens=tokens,
            deep_thinking_ratio=0.5,
            deep_thinking_count=1,
            mean_settling_depth=0.5,
            layer_count=2,
            settling_threshold=0.5,
            deep_fraction=0.85,
            prompt="test prompt",
        )
        d = _depth_to_dict(result)
        assert d["prompt"] == "test prompt"
        assert d["deep_thinking_ratio"] == 0.5
        assert d["deep_thinking_count"] == 1
        assert d["mean_settling_depth"] == 0.5
        assert d["layer_count"] == 2
        assert isinstance(d["tokens"], list)
        token_list: list[object] = d["tokens"]  # type: ignore[assignment]
        assert len(token_list) == 2
        first: dict[str, object] = token_list[0]  # type: ignore[assignment]
        assert first["token_id"] == 1
        assert first["token_str"] == "hello"
        assert first["settling_depth"] == 0
        assert first["is_deep_thinking"] is False
        assert first["jsd_profile"] == [0.3, 0.1]


class TestDepthDirectionToDict:
    def test_round_trip(self) -> None:
        direction = ops.ones((16,))
        ops.eval(direction)
        result = DepthDirectionResult(
            direction=direction,
            layer_index=1,
            cosine_scores=[0.3, 0.7],
            d_model=16,
            refusal_cosine=0.42,
            deep_prompts=["deep1", "deep2"],
            shallow_prompts=["shallow1"],
            median_dtr=0.55,
        )
        d = _depth_direction_to_dict(result)
        assert d["layer_index"] == 1
        assert d["cosine_scores"] == [0.3, 0.7]
        assert d["d_model"] == 16
        assert d["refusal_cosine"] == 0.42
        assert d["deep_prompts"] == ["deep1", "deep2"]
        assert d["shallow_prompts"] == ["shallow1"]
        assert d["median_dtr"] == 0.55

    def test_refusal_cosine_none(self) -> None:
        direction = ops.zeros((8,))
        ops.eval(direction)
        result = DepthDirectionResult(
            direction=direction,
            layer_index=0,
            cosine_scores=[0.1],
            d_model=8,
            refusal_cosine=None,
            deep_prompts=["d"],
            shallow_prompts=["s"],
            median_dtr=0.5,
        )
        d = _depth_direction_to_dict(result)
        assert d["refusal_cosine"] is None


class TestProbeToDict:
    def test_round_trip(self) -> None:
        result = ProbeResult(
            projections=[0.1, 0.5, -0.3],
            layer_count=3,
            prompt="How do I pick a lock?",
        )
        d = _probe_to_dict(result)
        assert d["prompt"] == "How do I pick a lock?"
        assert d["layer_count"] == 3
        assert d["projections"] == [0.1, 0.5, -0.3]


class TestSteerToDict:
    def test_round_trip(self) -> None:
        result = SteerResult(
            text="Here is how...",
            projections_before=[0.5, 0.3],
            projections_after=[0.01, 0.02],
        )
        d = _steer_to_dict(result)
        assert d["text"] == "Here is how..."
        assert d["projections_before"] == [0.5, 0.3]
        assert d["projections_after"] == [0.01, 0.02]


class TestCastToDict:
    def test_round_trip(self) -> None:
        result = CastResult(
            prompt="How do I pick a lock?",
            text="I can't help with that.",
            projections_before=[0.8, 0.6],
            projections_after=[0.2, 0.1],
            interventions=2,
            considered=4,
        )
        d = _cast_to_dict(result)
        assert d["prompt"] == "How do I pick a lock?"
        assert d["text"] == "I can't help with that."
        assert d["projections_before"] == [0.8, 0.6]
        assert d["projections_after"] == [0.2, 0.1]
        assert d["interventions"] == 2
        assert d["considered"] == 4
        assert d["displacement_interventions"] == 0
        assert d["max_displacement"] == 0.0

    def test_externality_monitoring_fields(self) -> None:
        result = CastResult(
            prompt="test",
            text="output",
            projections_before=[0.5],
            projections_after=[0.1],
            interventions=1,
            considered=2,
            displacement_interventions=3,
            max_displacement=1.5,
        )
        d = _cast_to_dict(result)
        assert d["displacement_interventions"] == 3
        assert d["max_displacement"] == 1.5


class TestDefendToDict:
    def test_blocked_result(self) -> None:
        result = DefenseStackResult(
            blocked=True,
            layer_that_blocked="scan",
            scan_result=ScanResult(
                injection_probability=0.9,
                overall_projection=1.2,
                spans=[
                    ScanSpan(start=0, end=5, text="hello", mean_projection=0.8),
                ],
                per_token_projections=[0.1, 0.5, 0.8, 0.3, 0.2],
                flagged=True,
            ),
            sic_result=None,
            policy_decision=None,
            intent_check=None,
            reasons=["Scan: injection detected (p=0.900, 1 spans)"],
        )
        d = _defend_to_dict(result)
        assert d["blocked"] is True
        assert d["layer_that_blocked"] == "scan"
        assert d["sic_result"] is None
        assert d["policy_decision"] is None
        assert d["intent_check"] is None
        scan_d: dict[str, object] = d["scan_result"]  # type: ignore[assignment]
        assert scan_d["injection_probability"] == 0.9
        assert scan_d["flagged"] is True
        spans_list: list[object] = scan_d["spans"]  # type: ignore[assignment]
        assert len(spans_list) == 1

    def test_unblocked_result(self) -> None:
        result = DefenseStackResult(
            blocked=False,
            layer_that_blocked=None,
            scan_result=ScanResult(
                injection_probability=0.1,
                overall_projection=0.2,
                spans=[],
                per_token_projections=[0.1],
                flagged=False,
            ),
            sic_result=SICPromptResult(
                clean_prompt="safe content",
                blocked=False,
                iterations=1,
                initial_score=0.1,
                final_score=0.05,
            ),
            policy_decision=PolicyDecision(
                action="allow",
                matched_rules=[],
                reasons=[],
            ),
            intent_check=IntentCheckResult(
                aligned=True,
                score=0.95,
                mode="embedding",
            ),
            reasons=[],
        )
        d = _defend_to_dict(result)
        assert d["blocked"] is False
        assert d["layer_that_blocked"] is None
        sic_d: dict[str, object] = d["sic_result"]  # type: ignore[assignment]
        assert sic_d["blocked"] is False
        policy_d: dict[str, object] = d["policy_decision"]  # type: ignore[assignment]
        assert policy_d["action"] == "allow"
        intent_d: dict[str, object] = d["intent_check"]  # type: ignore[assignment]
        assert intent_d["aligned"] is True
        assert intent_d["score"] == 0.95


class TestDiffResultToDict:
    def test_round_trip(self) -> None:
        basis = ops.ones((2, 16))
        ops.eval(basis)
        per_layer_bases = [ops.ones((2, 16)), ops.ones((2, 16))]
        ops.eval(*per_layer_bases)
        result = DiffResult(
            basis=basis,
            singular_values=[0.9, 0.5],
            explained_variance=[0.8, 0.2],
            best_layer=3,
            d_model=16,
            source_model="base-model",
            target_model="aligned-model",
            per_layer_bases=per_layer_bases,
            per_layer_singular_values=[[0.9, 0.5], [0.7, 0.3]],
        )
        d = _diff_result_to_dict(result)
        assert d["singular_values"] == [0.9, 0.5]
        assert d["explained_variance"] == [0.8, 0.2]
        assert d["best_layer"] == 3
        assert d["d_model"] == 16
        assert d["source_model"] == "base-model"
        assert d["target_model"] == "aligned-model"
        assert d["per_layer_singular_values"] == [[0.9, 0.5], [0.7, 0.3]]
        # mx.array fields should NOT be in the dict
        assert "basis" not in d
        assert "per_layer_bases" not in d


class TestSurfaceComparisonToDict:
    def test_round_trip(self) -> None:
        before = SurfaceResult(
            points=[],
            groups_by_label=[],
            groups_by_category=[],
            threshold=0.5,
            total_scanned=100,
            total_refused=20,
        )
        after = SurfaceResult(
            points=[],
            groups_by_label=[],
            groups_by_category=[],
            threshold=0.3,
            total_scanned=100,
            total_refused=5,
        )
        cat_delta = SurfaceGroupDelta(
            name="harmful",
            count=50,
            refusal_rate_before=0.4,
            refusal_rate_after=0.1,
            refusal_rate_delta=-0.3,
            mean_projection_before=0.8,
            mean_projection_after=0.2,
            mean_projection_delta=-0.6,
        )
        label_delta = SurfaceGroupDelta(
            name="violence",
            count=10,
            refusal_rate_before=0.6,
            refusal_rate_after=0.0,
            refusal_rate_delta=-0.6,
            mean_projection_before=1.0,
            mean_projection_after=0.1,
            mean_projection_delta=-0.9,
        )
        comparison = SurfaceComparison(
            before=before,
            after=after,
            refusal_rate_before=0.2,
            refusal_rate_after=0.05,
            refusal_rate_delta=-0.15,
            threshold_before=0.5,
            threshold_after=0.3,
            threshold_delta=-0.2,
            category_deltas=[cat_delta],
            label_deltas=[label_delta],
        )
        d = _surface_comparison_to_dict(comparison)
        summary: dict[str, object] = d["summary"]  # type: ignore[assignment]
        assert summary["refusal_rate_before"] == 0.2
        assert summary["refusal_rate_after"] == 0.05
        assert summary["refusal_rate_delta"] == -0.15
        assert summary["threshold_before"] == 0.5
        assert summary["total_scanned"] == 100
        assert summary["coverage_score_before"] == 0.0  # default
        cats: list[object] = d["category_deltas"]  # type: ignore[assignment]
        assert len(cats) == 1
        cat_d: dict[str, object] = cats[0]  # type: ignore[assignment]
        assert cat_d["name"] == "harmful"
        assert cat_d["count"] == 50
        assert cat_d["refusal_rate_delta"] == -0.3
        labels: list[object] = d["label_deltas"]  # type: ignore[assignment]
        assert len(labels) == 1

    def test_empty_optional_deltas(self) -> None:
        before = SurfaceResult(
            points=[],
            groups_by_label=[],
            groups_by_category=[],
            threshold=0.5,
            total_scanned=0,
            total_refused=0,
        )
        comparison = SurfaceComparison(
            before=before,
            after=before,
            refusal_rate_before=0.0,
            refusal_rate_after=0.0,
            refusal_rate_delta=0.0,
            threshold_before=0.5,
            threshold_after=0.5,
            threshold_delta=0.0,
            category_deltas=[],
            label_deltas=[],
        )
        d = _surface_comparison_to_dict(comparison)
        assert d["style_deltas"] == []
        assert d["language_deltas"] == []
        assert d["turn_depth_deltas"] == []
        assert d["framing_deltas"] == []
        assert d["cell_deltas"] == []


class TestDetectToDict:
    def test_round_trip(self) -> None:
        result = DetectResult(
            hardened=True,
            confidence=0.85,
            effective_rank=3.2,
            cosine_concentration=0.95,
            silhouette_peak=0.72,
            hdd_red_distance=0.15,
            residual_refusal_rate=0.3,
            mean_refusal_position=2.5,
            evidence=["High effective rank", "Strong cosine concentration"],
        )
        d = _detect_to_dict(result)
        assert d["hardened"] is True
        assert d["confidence"] == 0.85
        assert d["effective_rank"] == 3.2
        assert d["cosine_concentration"] == 0.95
        assert d["silhouette_peak"] == 0.72
        assert d["hdd_red_distance"] == 0.15
        assert d["residual_refusal_rate"] == 0.3
        assert d["mean_refusal_position"] == 2.5
        assert d["evidence"] == ["High effective rank", "Strong cosine concentration"]

    def test_nullable_fields(self) -> None:
        result = DetectResult(
            hardened=False,
            confidence=0.1,
            effective_rank=1.0,
            cosine_concentration=0.3,
            silhouette_peak=0.1,
            hdd_red_distance=None,
            residual_refusal_rate=None,
            mean_refusal_position=None,
            evidence=[],
        )
        d = _detect_to_dict(result)
        assert d["hardened"] is False
        assert d["hdd_red_distance"] is None
        assert d["residual_refusal_rate"] is None
        assert d["mean_refusal_position"] is None
        assert d["evidence"] == []


class TestOptimizeToDict:
    def _make_trial(self, num: int) -> TrialResult:
        return TrialResult(
            trial_number=num,
            alpha=0.5,
            sparsity=0.1,
            norm_preserve=True,
            layer_strategy="top_k",
            layer_top_k=5,
            target_layers=[0, 1, 2],
            refusal_rate=0.05,
            perplexity_delta=0.1,
            kl_divergence=0.02,
        )

    def test_round_trip(self) -> None:
        trial = self._make_trial(1)
        result = OptimizeResult(
            all_trials=[trial],
            pareto_trials=[trial],
            baseline_refusal_rate=0.8,
            baseline_perplexity=15.0,
            n_trials=1,
            best_refusal=trial,
            best_balanced=trial,
        )
        d = _optimize_to_dict(result)
        assert d["n_trials"] == 1
        assert d["baseline_refusal_rate"] == 0.8
        assert d["baseline_perplexity"] == 15.0
        best_r: dict[str, object] = d["best_refusal"]  # type: ignore[assignment]
        assert best_r["trial_number"] == 1
        assert best_r["alpha"] == 0.5
        assert best_r["norm_preserve"] is True
        assert best_r["layer_strategy"] == "top_k"
        assert best_r["layer_top_k"] == 5
        assert best_r["target_layers"] == [0, 1, 2]
        all_t: list[object] = d["all_trials"]  # type: ignore[assignment]
        assert len(all_t) == 1

    def test_none_bests(self) -> None:
        result = OptimizeResult(
            all_trials=[],
            pareto_trials=[],
            baseline_refusal_rate=0.5,
            baseline_perplexity=10.0,
            n_trials=0,
            best_refusal=None,
            best_balanced=None,
        )
        d = _optimize_to_dict(result)
        assert d["best_refusal"] is None
        assert d["best_balanced"] is None
        assert d["all_trials"] == []
        assert d["pareto_trials"] == []


class TestSicToDict:
    def test_round_trip(self) -> None:
        result = SICResult(
            prompts_clean=["safe prompt", "another safe"],
            prompts_blocked=[False, True],
            iterations_used=[1, 3],
            initial_scores=[0.1, 0.9],
            final_scores=[0.05, 0.8],
            total_blocked=1,
            total_sanitized=1,
            total_clean=1,
            calibrated_threshold=0.5,
        )
        d = _sic_to_dict(result)
        assert d["prompts_clean"] == ["safe prompt", "another safe"]
        assert d["prompts_blocked"] == [False, True]
        assert d["iterations_used"] == [1, 3]
        assert d["initial_scores"] == [0.1, 0.9]
        assert d["final_scores"] == [0.05, 0.8]
        assert d["total_blocked"] == 1
        assert d["total_sanitized"] == 1
        assert d["total_clean"] == 1
        assert d["calibrated_threshold"] == 0.5

    def test_calibrated_threshold_none(self) -> None:
        result = SICResult(
            prompts_clean=[],
            prompts_blocked=[],
            iterations_used=[],
            initial_scores=[],
            final_scores=[],
            total_blocked=0,
            total_sanitized=0,
            total_clean=0,
        )
        d = _sic_to_dict(result)
        assert d["calibrated_threshold"] is None


class TestTransferEvalToDict:
    def test_round_trip(self) -> None:
        result = TransferEvalResult(
            model_id="mlx-community/Qwen2.5-1.5B-Instruct-bf16",
            success_rate=0.75,
            eval_responses=["Sure, here is", "I cannot help"],
        )
        d = _transfer_eval_to_dict(result)
        assert d["model_id"] == "mlx-community/Qwen2.5-1.5B-Instruct-bf16"
        assert d["success_rate"] == 0.75
        assert d["eval_responses"] == ["Sure, here is", "I cannot help"]


class TestSoftpromptToDict:
    def test_round_trip_gcg(self) -> None:
        result = SoftPromptResult(
            mode="gcg",
            success_rate=0.6,
            final_loss=2.5,
            loss_history=[5.0, 3.0, 2.5],
            n_steps=100,
            n_tokens=8,
            embeddings=None,
            token_ids=[1, 2, 3],
            token_text="abc",
            eval_responses=["Sure"],
            accessibility_score=0.08,
            per_prompt_losses=[2.4, 2.6],
            early_stopped=True,
        )
        d = _softprompt_to_dict(result)
        assert d["mode"] == "gcg"
        assert d["success_rate"] == 0.6
        assert d["final_loss"] == 2.5
        assert d["loss_history"] == [5.0, 3.0, 2.5]
        assert d["n_steps"] == 100
        assert d["n_tokens"] == 8
        assert d["token_ids"] == [1, 2, 3]
        assert d["token_text"] == "abc"
        assert d["eval_responses"] == ["Sure"]
        assert d["accessibility_score"] == 0.08
        assert d["per_prompt_losses"] == [2.4, 2.6]
        assert d["early_stopped"] is True
        assert d["transfer_results"] == []
        assert d["defense_eval"] is None
        assert d["gan_history"] == []
        # embeddings should NOT be in the dict
        assert "embeddings" not in d

    def test_with_defense_eval(self) -> None:
        defense = DefenseEvalResult(
            sic_blocked=1,
            sic_sanitized=2,
            sic_clean=3,
            sic_bypass_rate=0.5,
            cast_interventions=10,
            cast_refusal_rate=0.2,
            cast_responses=["blocked"],
        )
        result = SoftPromptResult(
            mode="gcg",
            success_rate=0.5,
            final_loss=3.0,
            loss_history=[3.0],
            n_steps=50,
            n_tokens=4,
            embeddings=None,
            token_ids=[1],
            token_text="x",
            eval_responses=[],
            defense_eval=defense,
        )
        d = _softprompt_to_dict(result)
        de: dict[str, object] = d["defense_eval"]  # type: ignore[assignment]
        assert de["sic_blocked"] == 1
        assert de["cast_refusal_rate"] == 0.2

    def test_with_transfer_results(self) -> None:
        tr = TransferEvalResult(
            model_id="other-model",
            success_rate=0.3,
            eval_responses=["resp"],
        )
        result = SoftPromptResult(
            mode="egd",
            success_rate=0.4,
            final_loss=4.0,
            loss_history=[4.0],
            n_steps=20,
            n_tokens=6,
            embeddings=None,
            token_ids=None,
            token_text=None,
            eval_responses=[],
            transfer_results=[tr],
        )
        d = _softprompt_to_dict(result)
        transfers: list[object] = d["transfer_results"]  # type: ignore[assignment]
        assert len(transfers) == 1
        first: dict[str, object] = transfers[0]  # type: ignore[assignment]
        assert first["model_id"] == "other-model"
        assert first["success_rate"] == 0.3

    def test_with_gan_history(self) -> None:
        inner_result = SoftPromptResult(
            mode="gcg",
            success_rate=0.3,
            final_loss=5.0,
            loss_history=[5.0],
            n_steps=10,
            n_tokens=4,
            embeddings=None,
            token_ids=[1, 2],
            token_text="ab",
            eval_responses=["resp"],
        )
        defense = DefenseEvalResult(
            sic_blocked=0,
            sic_sanitized=0,
            sic_clean=1,
            sic_bypass_rate=1.0,
            cast_interventions=2,
            cast_refusal_rate=0.0,
            cast_responses=["ok"],
        )
        gan_round = GanRoundResult(
            round_index=0,
            attack_result=inner_result,
            defense_result=defense,
            attacker_won=True,
            config_snapshot={"lr": 0.01},
        )
        result = SoftPromptResult(
            mode="gcg",
            success_rate=0.6,
            final_loss=3.0,
            loss_history=[3.0],
            n_steps=50,
            n_tokens=4,
            embeddings=None,
            token_ids=[1],
            token_text="x",
            eval_responses=[],
            gan_history=[gan_round],
        )
        d = _softprompt_to_dict(result)
        history: list[object] = d["gan_history"]  # type: ignore[assignment]
        assert len(history) == 1
        r0: dict[str, object] = history[0]  # type: ignore[assignment]
        assert r0["round_index"] == 0
        assert r0["attacker_won"] is True
        assert r0["config_snapshot"] == {"lr": 0.01}
        attack_d: dict[str, object] = r0["attack_result"]  # type: ignore[assignment]
        assert attack_d["mode"] == "gcg"
        assert attack_d["success_rate"] == 0.3
        defense_d: dict[str, object] = r0["defense_result"]  # type: ignore[assignment]
        assert defense_d["sic_bypass_rate"] == 1.0
        assert defense_d["cast_interventions"] == 2


class TestDirectionTransferToDict:
    def test_round_trip(self) -> None:
        result = DirectionTransferResult(
            model_id="target-model",
            cosine_separation=0.6,
            best_native_separation=0.8,
            transfer_efficiency=0.75,
            per_layer_cosines=[0.5, 0.6, 0.7],
        )
        d = _direction_transfer_to_dict(result)
        assert d["model_id"] == "target-model"
        assert d["cosine_separation"] == 0.6
        assert d["best_native_separation"] == 0.8
        assert d["transfer_efficiency"] == 0.75
        assert d["per_layer_cosines"] == [0.5, 0.6, 0.7]


class TestScanResultToDict:
    def test_round_trip(self) -> None:
        result = ScanResult(
            injection_probability=0.85,
            overall_projection=1.1,
            spans=[
                ScanSpan(start=0, end=3, text="bad", mean_projection=0.9),
                ScanSpan(start=5, end=8, text="txt", mean_projection=0.7),
            ],
            per_token_projections=[0.1, 0.5, 0.9, 0.3, 0.2, 0.7, 0.6, 0.4],
            flagged=True,
        )
        d = _scan_result_to_dict(result)
        assert d["injection_probability"] == 0.85
        assert d["overall_projection"] == 1.1
        assert d["flagged"] is True
        assert d["per_token_projections"] == [0.1, 0.5, 0.9, 0.3, 0.2, 0.7, 0.6, 0.4]
        spans_list: list[object] = d["spans"]  # type: ignore[assignment]
        assert len(spans_list) == 2
        first: dict[str, object] = spans_list[0]  # type: ignore[assignment]
        assert first["start"] == 0
        assert first["end"] == 3
        assert first["text"] == "bad"
        assert first["mean_projection"] == 0.9

    def test_no_spans(self) -> None:
        result = ScanResult(
            injection_probability=0.05,
            overall_projection=0.1,
            spans=[],
            per_token_projections=[0.05],
            flagged=False,
        )
        d = _scan_result_to_dict(result)
        assert d["flagged"] is False
        assert d["spans"] == []

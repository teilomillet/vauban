# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for Vauban runtime primitive contracts."""

from __future__ import annotations

from typing import Protocol, cast

import pytest

from vauban.runtime import (
    BackendCapabilities,
    DeviceRef,
    ForwardRequest,
    ForwardTrace,
    InterventionRecord,
    LoadedModel,
    ModelRef,
    StageProfile,
    TokenizeRequest,
    Trace,
    TraceRequest,
    access_boundary_for_capabilities,
    access_level_for_capabilities,
    access_policy_for_capabilities,
    access_policy_for_trace,
    available_runtime_backends,
    create_runtime,
    declared_capabilities,
    forward_trace_summary,
    max_capabilities,
    mlx_capabilities,
    profile_stage,
    runtime_capabilities,
    runtime_capability_snapshot,
    runtime_evidence_refs,
    runtime_report_evidence,
    torch_capabilities,
)

mx = pytest.importorskip("mlx.core")


class HasShape(Protocol):
    """Object with tensor-like shape metadata."""

    @property
    def shape(self) -> tuple[int, ...]: ...


class FakeEmbedding:
    """Small embedding surface for MLX runtime contract tests."""

    def __call__(self, token_ids: object) -> object:
        """Embed token IDs as deterministic MLX activations."""
        shape = cast("HasShape", token_ids).shape
        batch = int(shape[0])
        seq_len = int(shape[1])
        return mx.ones((batch, seq_len, 4))

    def as_linear(self, h: object) -> object:
        """Use hidden states directly as fake logits."""
        return h


class FakeLayer:
    """Small transformer layer that shifts hidden states."""

    def __init__(self, increment: float) -> None:
        """Initialize the fake layer."""
        self.increment = increment

    def __call__(
        self,
        h: object,
        mask: object | None,
        *,
        cache: object | None = None,
    ) -> object:
        """Apply a deterministic activation shift."""
        return h + self.increment


class FakeNorm:
    """Identity norm layer."""

    def __call__(self, h: object) -> object:
        """Return hidden states unchanged."""
        return h


class FakeIntervention:
    """Deterministic fake activation intervention."""

    name = "shift"
    layer_index = 0

    def apply(self, activation: object) -> object:
        """Shift fake activations."""
        return activation + 10.0


class FakeTransformer:
    """Minimal transformer surface consumed by MlxRuntime."""

    def __init__(self) -> None:
        """Initialize fake transformer components."""
        self.embed_tokens = FakeEmbedding()
        self.layers = [FakeLayer(1.0), FakeLayer(2.0)]
        self.norm = FakeNorm()


class FakeModel:
    """Minimal model surface consumed by MlxRuntime."""

    def __init__(self) -> None:
        """Initialize fake model."""
        self.model = FakeTransformer()


class FakeTokenizer:
    """Minimal tokenizer surface consumed by MlxRuntime."""

    def encode(self, text: str) -> list[int]:
        """Encode text into deterministic token IDs."""
        return [ord(char) % 32 for char in text]

    def decode(self, token_ids: list[int]) -> str:
        """Decode fake token IDs."""
        return "".join(str(token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
    ) -> str | list[int]:
        """Render a minimal chat template."""
        rendered = "\n".join(message["content"] for message in messages)
        if tokenize:
            return self.encode(rendered)
        return rendered


class FakeTorchEmbedding:
    """Small Torch embedding surface for optional Torch runtime tests."""

    def __init__(self, torch_module: object) -> None:
        """Initialize fake Torch embedding."""
        self._torch = torch_module
        self.weight = torch_module.ones((1,))

    def __call__(self, token_ids: object) -> object:
        """Embed token IDs as deterministic Torch activations."""
        shape = cast("HasShape", token_ids).shape
        batch = int(shape[0])
        seq_len = int(shape[1])
        return self._torch.ones((batch, seq_len, 4))


class FakeTorchLayer:
    """Small Torch layer that shifts hidden states."""

    def __init__(self, increment: float) -> None:
        """Initialize the fake layer."""
        self.increment = increment

    def __call__(
        self,
        h: object,
        mask: object | None = None,
        cache: object | None = None,
    ) -> object:
        """Apply a deterministic activation shift."""
        return h + self.increment


class FakeTorchNorm:
    """Identity Torch norm."""

    def __call__(self, h: object) -> object:
        """Return hidden states unchanged."""
        return h


class FakeTorchHead:
    """Identity Torch LM head."""

    def __call__(self, h: object) -> object:
        """Return fake logits."""
        return h


class FakeTorchTransformer:
    """Minimal Torch transformer surface consumed by TorchRuntime."""

    def __init__(self, torch_module: object) -> None:
        """Initialize fake Torch transformer."""
        self.embed_tokens = FakeTorchEmbedding(torch_module)
        self.layers = [FakeTorchLayer(1.0), FakeTorchLayer(2.0)]
        self.norm = FakeTorchNorm()


class FakeTorchModel:
    """Minimal Torch model surface consumed by TorchRuntime."""

    def __init__(self, torch_module: object) -> None:
        """Initialize fake Torch model."""
        self.model = FakeTorchTransformer(torch_module)
        self.lm_head = FakeTorchHead()
        self.device = "cpu"


def _loaded_fake_model() -> LoadedModel:
    """Build a loaded fake MLX model handle."""
    return LoadedModel(
        ref=ModelRef("fake-model"),
        backend="mlx",
        capabilities=mlx_capabilities(),
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
    )


def _loaded_fake_torch_model(torch_module: object) -> LoadedModel:
    """Build a loaded fake Torch model handle."""
    return LoadedModel(
        ref=ModelRef("fake-torch-model"),
        backend="torch",
        capabilities=torch_capabilities(),
        model=FakeTorchModel(torch_module),
        tokenizer=FakeTokenizer(),
    )


def _artifact_shape_map(trace: Trace) -> dict[str, tuple[int, ...] | None]:
    """Return artifact shapes keyed by backend-independent artifact ID."""
    return {
        artifact.artifact_id: artifact.shape
        for artifact in trace.artifacts
    }


class TestRuntimeCapabilities:
    """Capability declarations must be explicit and epistemic."""

    def test_known_backends_are_declared(self) -> None:
        assert available_runtime_backends() == ("torch", "mlx", "max")

    def test_mlx_capabilities_support_activation_diagnostics(self) -> None:
        caps = mlx_capabilities()
        assert caps.supports("activations")
        assert access_level_for_capabilities(caps) == "activations"
        boundary = access_boundary_for_capabilities(caps)
        assert boundary.claim_strength == "activation_diagnostic"

    def test_capability_access_policy_lists_available_and_missing_evidence(
        self,
    ) -> None:
        policy = access_policy_for_capabilities(mlx_capabilities())
        assert policy.level == "activations"
        assert "Runtime activation traces (full)" in policy.available_evidence
        assert policy.missing_evidence == ()
        assert policy.notes == ()

    def test_torch_runtime_contract_declares_full_primitive_support(self) -> None:
        caps = torch_capabilities()
        assert caps.device_kinds == ("cpu", "cuda", "mps")
        assert caps.support_level("logits") == "full"
        assert caps.support_level("logprobs") == "full"
        assert caps.support_level("activations") == "full"
        assert caps.support_level("interventions") == "full"
        assert caps.support_level("kv_cache") == "full"
        assert caps.support_level("weight_access") == "full"
        assert caps.support_level("mutable_weights") == "full"
        assert access_level_for_capabilities(caps) == "activations"

    def test_torch_full_capabilities_do_not_emit_partial_notes(self) -> None:
        policy = access_policy_for_capabilities(torch_capabilities())
        assert policy.level == "activations"
        assert "Runtime activation traces (full)" in policy.available_evidence
        assert policy.missing_evidence == ()
        assert policy.notes == ()

    def test_declared_capabilities_reject_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="Unknown runtime backend"):
            declared_capabilities("not-real")

    def test_runtime_capabilities_uses_explicit_name(self) -> None:
        assert runtime_capabilities("torch").name == "torch"

    def test_capability_validation_requires_device(self) -> None:
        with pytest.raises(ValueError, match="device_kinds"):
            BackendCapabilities(
                name="mlx",
                device_kinds=(),
                logits="full",
                logprobs="full",
                activations="full",
                interventions="full",
                kv_cache="full",
                weight_access="full",
                mutable_weights="full",
            )


class TestRuntimeTypes:
    """Runtime dataclasses must reject inconsistent evidence."""

    def test_forward_request_rejects_empty_prompt(self) -> None:
        with pytest.raises(ValueError, match="prompt_ids"):
            ForwardRequest(prompt_ids=())

    def test_forward_request_rejects_logprobs_without_logits(self) -> None:
        with pytest.raises(ValueError, match="return_logprobs requires"):
            ForwardRequest(prompt_ids=(1,), return_logits=False, return_logprobs=True)

    def test_loaded_model_backend_must_match_capabilities(self) -> None:
        with pytest.raises(ValueError, match="backend must match"):
            LoadedModel(
                ref=ModelRef("fake-model"),
                backend="torch",
                capabilities=mlx_capabilities(),
                model=FakeModel(),
            )

    def test_forward_trace_rejects_logprobs_without_logits(self) -> None:
        with pytest.raises(ValueError, match="logprobs require logits"):
            ForwardTrace(
                logits=None,
                logprobs=mx.ones((1, 1)),
                activations={},
                device=DeviceRef(kind="gpu", label="mlx-gpu"),
            )

    def test_intervention_record_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="name"):
            InterventionRecord(name="", layer_index=0)

    def test_stage_timer_records_profile(self) -> None:
        with profile_stage("unit") as timer:
            value = 1 + 1
        assert value == 2
        assert isinstance(timer.profile, StageProfile)
        assert timer.profile.name == "unit"
        assert timer.profile.duration_s >= 0.0

    def test_stage_timer_records_usl_ready_profile_fields(self) -> None:
        with profile_stage(
            "unit",
            batch_size=2,
            token_count=8,
            input_bytes=16,
            output_bytes=32,
            host_device_copies=1,
            sync_points=2,
            queue_depth=3,
        ) as timer:
            value = 1 + 1

        assert value == 2
        assert timer.profile.batch_size == 2
        assert timer.profile.token_count == 8
        assert timer.profile.input_bytes == 16
        assert timer.profile.output_bytes == 32
        assert timer.profile.host_device_copies == 1
        assert timer.profile.sync_points == 2
        assert timer.profile.queue_depth == 3


class TestMlxRuntime:
    """The MLX adapter should satisfy the primitive contract on fake models."""

    def test_create_runtime_returns_mlx_runtime(self) -> None:
        runtime = create_runtime("mlx")
        assert runtime.capabilities.name == "mlx"

    def test_create_runtime_returns_torch_runtime(self) -> None:
        runtime = create_runtime("torch")
        assert runtime.capabilities.name == "torch"
        assert runtime.capabilities.support_level("activations") == "full"

    def test_tokenize_uses_loaded_tokenizer(self) -> None:
        runtime = create_runtime("mlx")
        result = runtime.tokenize(_loaded_fake_model(), TokenizeRequest("abc"))
        assert result.token_ids == (1, 2, 3)
        assert result.profile[0].name == "tokenize"

    def test_forward_returns_logits_and_selected_activations(self) -> None:
        runtime = create_runtime("mlx")
        trace = runtime.forward(
            _loaded_fake_model(),
            ForwardRequest(
                prompt_ids=(1, 2, 3),
                collect_layers=(0,),
                interventions=(FakeIntervention(),),
                return_logits=True,
                return_logprobs=True,
            ),
        )

        assert trace.logits is not None
        assert trace.logprobs is not None
        assert trace.logits.shape == (1, 3, 4)
        assert trace.activations[0].shape == (1, 3, 4)
        assert trace.interventions == (
            InterventionRecord(name="shift", layer_index=0),
        )
        assert tuple(profile.name for profile in trace.profile) == (
            "prepare_batch",
            "forward",
            "lm_head",
        )

    def test_runtime_evidence_summary_has_stable_shape_keys(self) -> None:
        runtime = create_runtime("mlx")
        trace = runtime.forward(
            _loaded_fake_model(),
            ForwardRequest(
                prompt_ids=(1, 2),
                collect_layers=(0,),
                interventions=(FakeIntervention(),),
                return_logits=True,
                return_logprobs=True,
            ),
        )

        summary = forward_trace_summary(trace)

        assert set(summary) == {
            "activation_shapes",
            "artifacts",
            "device",
            "interventions",
            "logits_shape",
            "logprobs_shape",
            "profile",
            "profile_summary",
            "spans",
        }
        assert summary["logits_shape"] == [1, 2, 4]
        assert summary["logprobs_shape"] == [1, 2, 4]
        assert summary["activation_shapes"] == {"0": [1, 2, 4]}
        assert summary["interventions"] == [
            {"name": "shift", "layer_index": 0},
        ]
        profile = summary["profile"]
        assert isinstance(profile, list)
        first_profile = profile[0]
        assert isinstance(first_profile, dict)
        assert first_profile["token_count"] == 2
        assert first_profile["host_device_copies"] == 1
        profile_summary = summary["profile_summary"]
        assert isinstance(profile_summary, dict)
        assert profile_summary["profiled_spans"] == 3
        assert profile_summary["token_count"] == 2

    def test_runtime_evidence_refs_are_stable(self) -> None:
        runtime = create_runtime("mlx")
        trace = runtime.forward(
            _loaded_fake_model(),
            ForwardRequest(
                prompt_ids=(1, 2),
                collect_layers=(0,),
                return_logits=True,
                return_logprobs=True,
            ),
        )

        refs = runtime_evidence_refs(trace, prefix="mlx")

        assert tuple(ref.evidence_id for ref in refs) == (
            "mlx.trace",
            "mlx.logprobs",
            "mlx.activations",
        )
        assert tuple(ref.kind for ref in refs) == (
            "trace",
            "logprobs",
            "activation",
        )

    def test_trace_access_policy_uses_collected_evidence(self) -> None:
        runtime = create_runtime("mlx")
        trace = runtime.forward(
            _loaded_fake_model(),
            ForwardRequest(
                prompt_ids=(1, 2),
                return_logits=True,
                return_logprobs=False,
            ),
        )

        policy = access_policy_for_trace(mlx_capabilities(), trace)

        assert policy.level == "logprobs"
        assert "Runtime logits" in policy.available_evidence
        assert "Runtime activation traces not collected" in policy.missing_evidence

    def test_trace_access_policy_rejects_unsupported_evidence(self) -> None:
        trace = ForwardTrace(
            logits=mx.ones((1, 1, 4)),
            logprobs=None,
            activations={},
            device=DeviceRef(kind="gpu", label="mlx-gpu"),
        )

        with pytest.raises(ValueError, match="logits unsupported"):
            access_policy_for_trace(max_capabilities(), trace)

    def test_runtime_report_evidence_packages_report_inputs(self) -> None:
        runtime = create_runtime("mlx")
        trace = runtime.forward(
            _loaded_fake_model(),
            ForwardRequest(
                prompt_ids=(1, 2),
                collect_layers=(0,),
                return_logits=True,
                return_logprobs=True,
            ),
        )

        package = runtime_report_evidence(
            mlx_capabilities(),
            trace,
            prefix="behavior_trace.prompt_a",
        )
        payload = package.to_dict()

        assert package.access.level == "activations"
        assert tuple(ref.evidence_id for ref in package.evidence) == (
            "behavior_trace.prompt_a.capabilities",
            "behavior_trace.prompt_a.trace",
            "behavior_trace.prompt_a.logprobs",
            "behavior_trace.prompt_a.activations",
        )
        assert payload["access"]["claim_strength"] == "activation_diagnostic"
        assert payload["trace"]["activation_shapes"] == {"0": [1, 2, 4]}
        assert payload["capabilities"]["name"] == "mlx"

    def test_mlx_trace_method_emits_token_and_forward_artifacts(self) -> None:
        runtime = create_runtime("mlx")
        result = runtime.trace(
            _loaded_fake_model(),
            TraceRequest(
                trace_id="mlx.unit",
                input_text="abc",
                requested_artifacts=(
                    "tokens",
                    "logits",
                    "logprobs",
                    "activation",
                    "intervention_result",
                ),
                collect_layers=(0,),
                interventions=(FakeIntervention(),),
            ),
        )

        assert result.tokenized is not None
        assert result.tokenized.token_ids == (1, 2, 3)
        assert result.forward.logits is not None
        assert result.forward.logprobs is not None
        assert tuple(span.name for span in result.trace.spans) == (
            "tokenize",
            "prepare_batch",
            "forward",
            "lm_head",
        )
        assert result.trace.spans[1].input_artifact_ids == ("tokens",)
        assert tuple(artifact.kind for artifact in result.trace.artifacts) == (
            "tokens",
            "logits",
            "logprobs",
            "activation",
            "intervention_result",
        )
        package = runtime_report_evidence(
            runtime.capabilities,
            result.forward,
            prefix="mlx.unit",
            runtime_trace=result.trace,
        )

        assert tuple(ref.evidence_id for ref in package.evidence) == (
            "mlx.unit.capabilities",
            "mlx.unit.trace",
            "mlx.unit.tokens",
            "mlx.unit.logprobs",
            "mlx.unit.activations",
        )
        assert package.trace["profile_summary"]["profiled_spans"] == 4

    def test_trace_records_missing_requested_artifacts(self) -> None:
        runtime = create_runtime("mlx")
        result = runtime.trace(
            _loaded_fake_model(),
            TraceRequest(
                trace_id="mlx.missing",
                input_text="abc",
                requested_artifacts=("text", "tokens"),
            ),
        )

        assert result.trace.artifacts_by_kind("tokens")
        assert result.trace.metadata["requested_artifacts"] == ["text", "tokens"]
        assert result.trace.metadata["missing_requested_artifacts"] == ["text"]
        assert result.trace.metadata["unsupported_requested_artifacts"] == ["text"]

    def test_mlx_and_torch_trace_artifact_shapes_match_on_fake_models(
        self,
    ) -> None:
        torch = pytest.importorskip("torch")
        request = TraceRequest(
            trace_id="parity",
            input_text="ab",
            requested_artifacts=("tokens", "logits", "logprobs", "activation"),
            collect_layers=(0,),
        )
        mlx_result = create_runtime("mlx").trace(_loaded_fake_model(), request)
        torch_result = create_runtime("torch").trace(
            _loaded_fake_torch_model(torch),
            request,
        )

        assert _artifact_shape_map(mlx_result.trace) == _artifact_shape_map(
            torch_result.trace,
        )
        assert tuple(span.name for span in mlx_result.trace.spans) == tuple(
            span.name for span in torch_result.trace.spans
        )
        assert mlx_result.tokenized is not None
        assert torch_result.tokenized is not None
        assert mlx_result.tokenized.token_ids == torch_result.tokenized.token_ids

    def test_runtime_capability_snapshot_is_stable(self) -> None:
        assert runtime_capability_snapshot(mlx_capabilities()) == {
            "name": "mlx",
            "device_kinds": ["cpu", "gpu"],
            "logits": "full",
            "logprobs": "full",
            "activations": "full",
            "interventions": "full",
            "kv_cache": "full",
            "weight_access": "full",
            "mutable_weights": "full",
            "artifact_support": {
                "text": "unsupported",
                "tokens": "full",
                "logits": "full",
                "logprobs": "full",
                "activation": "full",
                "intervention_result": "full",
                "weight_snapshot": "full",
                "metric": "unsupported",
                "report_evidence": "unsupported",
                "profile": "full",
                "other": "unsupported",
            },
        }

    def test_torch_runtime_forward_when_torch_is_available(self) -> None:
        torch = pytest.importorskip("torch")
        runtime = create_runtime("torch")

        trace = runtime.forward(
            _loaded_fake_torch_model(torch),
            ForwardRequest(
                prompt_ids=(1, 2),
                collect_layers=(0,),
                interventions=(FakeIntervention(),),
                return_logits=True,
                return_logprobs=True,
            ),
        )

        assert trace.logits is not None
        assert trace.logprobs is not None
        assert trace.logits.shape == (1, 2, 4)
        assert trace.activations[0].shape == (1, 2, 4)
        assert trace.interventions == (
            InterventionRecord(name="shift", layer_index=0),
        )
        assert trace.profile[0].input_bytes is not None
        assert trace.profile[0].output_bytes is not None
        assert trace.profile[2].input_bytes is not None
        assert trace.profile[2].output_bytes is not None
        assert trace.profile[2].sync_points == 0

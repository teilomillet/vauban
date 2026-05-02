# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Small profiling helpers for runtime primitive stages."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from vauban.runtime._types import DeviceRef, RuntimeValue, StageProfile

if TYPE_CHECKING:
    from types import TracebackType


class StageTimer:
    """Context manager that records one runtime stage duration."""

    def __init__(
        self,
        name: str,
        *,
        device: DeviceRef | None = None,
        memory_bytes: int | None = None,
        batch_size: int | None = None,
        token_count: int | None = None,
        input_bytes: int | None = None,
        output_bytes: int | None = None,
        host_device_copies: int = 0,
        sync_points: int = 0,
        queue_depth: int | None = None,
        metadata: dict[str, RuntimeValue] | None = None,
    ) -> None:
        """Initialize a stage timer."""
        self._name = name
        self._device = device
        self._memory_bytes = memory_bytes
        self._batch_size = batch_size
        self._token_count = token_count
        self._input_bytes = input_bytes
        self._output_bytes = output_bytes
        self._host_device_copies = host_device_copies
        self._sync_points = sync_points
        self._queue_depth = queue_depth
        self._metadata = metadata or {}
        self._started_s = 0.0
        self._profile: StageProfile | None = None

    def __enter__(self) -> StageTimer:
        """Start the stage timer."""
        self._started_s = perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the stage timer."""
        duration_s = perf_counter() - self._started_s
        self._profile = StageProfile(
            name=self._name,
            duration_s=duration_s,
            device=self._device,
            memory_bytes=self._memory_bytes,
            batch_size=self._batch_size,
            token_count=self._token_count,
            input_bytes=self._input_bytes,
            output_bytes=self._output_bytes,
            host_device_copies=self._host_device_copies,
            sync_points=self._sync_points,
            queue_depth=self._queue_depth,
            metadata=dict(self._metadata),
        )

    @property
    def profile(self) -> StageProfile:
        """Return the completed stage profile."""
        if self._profile is None:
            msg = "stage timer has not completed"
            raise RuntimeError(msg)
        return self._profile


def profile_stage(
    name: str,
    *,
    device: DeviceRef | None = None,
    memory_bytes: int | None = None,
    batch_size: int | None = None,
    token_count: int | None = None,
    input_bytes: int | None = None,
    output_bytes: int | None = None,
    host_device_copies: int = 0,
    sync_points: int = 0,
    queue_depth: int | None = None,
    metadata: dict[str, RuntimeValue] | None = None,
) -> StageTimer:
    """Create a stage timer for one runtime primitive."""
    return StageTimer(
        name,
        device=device,
        memory_bytes=memory_bytes,
        batch_size=batch_size,
        token_count=token_count,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        host_device_copies=host_device_copies,
        sync_points=sync_points,
        queue_depth=queue_depth,
        metadata=metadata,
    )

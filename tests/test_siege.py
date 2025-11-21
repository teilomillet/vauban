import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import vauban
from vauban.interfaces import Target
from vauban.engine import SiegeEngine, SiegeResult
from vauban.strategies.gepa import GEPAStrategy
from vauban.judge import LLMJudge
from vauban.intel import IntelSystem


class AsyncMockTarget(Target):
    async def invoke_async(self, prompt: str) -> str:
        return "Refusal"


@pytest.mark.asyncio
@patch("vauban.api.GEPAStrategy")
@patch("vauban.api.SiegeEngine")
@patch("vauban.api.LLMJudge")
@patch("vauban.api.create_default_intel")
async def test_siege_wiring(
    mock_create_intel, mock_judge_cls, mock_engine_cls, mock_gepa_cls
):
    """
    Test that vauban.siege correctly instantiates and wires up components.
    """
    # Setup
    mock_intel = MagicMock(spec=IntelSystem)
    mock_create_intel.return_value = mock_intel

    mock_judge = MagicMock(spec=LLMJudge)
    mock_judge_cls.return_value = mock_judge

    mock_strategy = MagicMock(spec=GEPAStrategy)
    mock_gepa_cls.return_value = mock_strategy

    mock_engine = MagicMock(spec=SiegeEngine)
    mock_engine.run = AsyncMock(
        return_value=SiegeResult(success=False, max_score=1.0, history=[], stats={})
    )
    mock_engine_cls.return_value = mock_engine

    target = AsyncMockTarget()

    # Action
    result = await vauban.siege(goal="test", generations=2, squad_size=2, target=target)

    # Assert
    # Verify components were created
    mock_create_intel.assert_called_once()
    mock_judge_cls.assert_called_once()
    mock_gepa_cls.assert_called_once()

    # Verify Engine was initialized with correct components
    mock_engine_cls.assert_called_once()
    _, kwargs = mock_engine_cls.call_args
    assert kwargs["strategy"] == mock_strategy
    assert kwargs["judge"] == mock_judge
    assert kwargs["intel_system"] == mock_intel
    assert kwargs["target"] == target
    assert "active_scenario" in kwargs  # Should be wired even when None

    # Verify Engine.run was called
    mock_engine.run.assert_called_once()
    assert result.max_score == 1.0

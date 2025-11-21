import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from vauban.intel import IntelSystem
from vauban.interfaces import Embedder, VectorDB, Target


@pytest.fixture
def mock_embedder():
    embedder = MagicMock(spec=Embedder)
    embedder.embed.return_value = [0.1] * 1536
    return embedder


@pytest.fixture
def mock_vectordb():
    db = MagicMock(spec=VectorDB)
    db.get_all_refusals = MagicMock(return_value=[])
    return db


@pytest.fixture
def mock_detector():
    detector = MagicMock()
    detector.score_samples.return_value = [-0.5]
    detector.predict.return_value = [1]  # 1 for inlier, -1 for outlier
    detector.fit = MagicMock()
    return detector


@pytest.fixture
def intel_system(mock_embedder, mock_vectordb, mock_detector):
    # Pass a dummy path
    sys = IntelSystem(
        embedder=mock_embedder,
        vector_db=mock_vectordb,
        detector_path="dummy_detector.pkl",
    )
    sys.detector = mock_detector
    sys._fitted = True  # Pretend baseline already ran for unit tests
    return sys


def test_assess_deflected(intel_system):
    # Setup
    intel_system.detector.predict.return_value = [1]  # Inlier (Deflected)

    # Action
    score, is_breach = intel_system.assess_response("Refusal response")

    # Assert
    assert is_breach is False
    assert isinstance(score, float)
    intel_system.embedder.embed.assert_called_with("Refusal response")


def test_assess_breach(intel_system):
    # Setup
    intel_system.detector.predict.return_value = [-1]  # Outlier (Breach)

    # Action
    score, is_breach = intel_system.assess_response("Breach response")

    # Assert
    assert is_breach is True


def test_assess_raises_when_uncalibrated(intel_system):
    intel_system._fitted = False
    intel_system._bootstrap_attempted = True  # Skip bootstrap so guard raises
    with pytest.raises(RuntimeError):
        intel_system.assess_response("Any response")


@pytest.mark.asyncio
async def test_establish_baseline(intel_system, mock_detector):
    # Setup
    mock_target = MagicMock(spec=Target)
    mock_target.invoke_async = AsyncMock(return_value="Refusal")

    # Mock LocalOutlierFactor to return our mock detector
    with patch("vauban.intel.LocalOutlierFactor", return_value=mock_detector):
        # Mock pickle dump to avoid writing to disk
        with patch("pickle.dump"):
            # Action
            await intel_system.establish_baseline_async(target=mock_target)

    # Assert
    assert mock_target.invoke_async.called
    # Verify fit was called on our mock detector
    assert mock_detector.fit.called
    assert intel_system.vector_db.add.called
    assert intel_system._fitted is True

from unittest.mock import patch, MagicMock, AsyncMock
import vauban
from vauban.interfaces import Target


@patch("vauban.api.ModelTarget")
def test_scout_default(mock_openai_class):
    # Setup
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = "Response"
    mock_openai_class.return_value = mock_instance

    # Action
    response = vauban.scout("prompt")

    # Assert
    assert response == "Response"
    mock_instance.invoke.assert_called_with("prompt")


def test_scout_custom_target():
    # Setup
    target = MagicMock(spec=Target)
    target.invoke_async = AsyncMock(return_value="Custom Response")

    # Action
    # vauban.scout should fallback to invoke_async because spec=Target doesn't have invoke
    response = vauban.scout("prompt", target=target)

    # Assert
    assert response == "Custom Response"
    target.invoke_async.assert_called_with("prompt")


@patch("vauban.api._get_intel")
def test_assess_api(mock_get_intel):
    # Setup
    mock_sys = MagicMock()
    mock_sys.assess_response.return_value = (-0.5, False)
    mock_get_intel.return_value = mock_sys

    # Action
    score, is_breach = vauban.assess("Response")

    # Assert
    assert score == -0.5
    assert is_breach is False
    mock_sys.assess_response.assert_called_once_with("Response")

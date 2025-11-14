import os
import pytest
from unittest.mock import patch, MagicMock
from atlasmind.core.tools.audio_tool import AudioTool
from atlasmind.core.planning.base_plan import PlanTemplate, ToolType
from atlasmind.core.tools.base_tool import ExecutionStatus

@pytest.fixture(autouse=True)
def mock_deepgram():
    """Patch DeepgramClient for ALL tests so AudioTool() never calls real constructor."""
    with patch("atlasmind.core.tools.audio_tool.DeepgramClient", autospec=True) as mock_client:
        instance = mock_client.return_value
        # Pre-stub Deepgram call chain
        instance.listen.v1.media.transcribe_file = MagicMock()
        yield mock_client


@pytest.fixture
def sample_plan(tmp_path):
    """Create a minimal plan template for testing."""
    audio_file = tmp_path / "sample_audio.mp3"
    audio_file.write_bytes(b"fake audio data")
    return PlanTemplate(
        question="Transcribe the homework audio file",
        tool=ToolType.AUDIO_TRANSCRIBER,
        plan_steps=["Fetch audio file", "Transcribe via Deepgram"],
        file_path=str(audio_file)
    )


def test_audio_tool_success(mock_deepgram, sample_plan):
    mock_instance = mock_deepgram.return_value

    mock_response = MagicMock()
    mock_response.results.channels[0].alternatives[0].transcript = "Mock transcription result"

    mock_instance.listen.v1.media.transcribe_file.return_value = mock_response

    tool = AudioTool()
    result = tool.execute(sample_plan)

    assert result.status == ExecutionStatus.SUCCESS
    assert result.data is not None
    assert result.data["transcript"] == "Mock transcription result"
    mock_instance.listen.v1.media.transcribe_file.assert_called_once()


def test_audio_tool_invalid_file_path():
    """Test fail when file_path is missing or does not exist."""

    bad_plan = PlanTemplate(
        question="Transcribe something",
        tool=ToolType.AUDIO_TRANSCRIBER,
        file_path=None,   # no file provided
        plan_steps=[]
    )

    tool = AudioTool()

    with pytest.raises(RuntimeError, match="Invalid file_path"):
        tool.execute(bad_plan)

def test_audio_tool_transcription_failure(mock_deepgram, sample_plan):
    """Test failure when Deepgram throws exception."""

    mock_instance = mock_deepgram.return_value
    mock_instance.listen.v1.media.transcribe_file.side_effect = RuntimeError("Deepgram error")

    tool = AudioTool()

    with pytest.raises(RuntimeError, match="AudioTool execution failed"):
        tool.execute(sample_plan)

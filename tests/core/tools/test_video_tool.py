import pytest
from unittest.mock import patch, MagicMock
from atlasmind.core.tools.video_tool import VideoTool
from atlasmind.core.tools.base_tool import ExecutionStatus, ToolType
from atlasmind.core.planning.base_plan import PlanTemplate


@pytest.fixture
def sample_plan():
    """Fixture: Example PlanTemplate with a YouTube question."""
    return PlanTemplate(
        question="What does Teal'c say in https://www.youtube.com/watch?v=1htKBjuUWec?",
        tool=ToolType.VIDEO_TRANSCRIPT,
        plan_steps=["Fetch transcript from YouTube video."],
        metadata={"intent": "extract transcript"}
    )


def test_execute_success(sample_plan):
    """Test: Successful transcript retrieval."""
    tool = VideoTool()

    # Mock youtube_transcript_api
    mock_snippets = [MagicMock(text="Hello"), MagicMock(text="world")]
    with patch("atlasmind.core.tools.video_tool.YouTubeTranscriptApi.fetch", return_value=mock_snippets):
        result = tool.execute(sample_plan)

    assert result.status == ExecutionStatus.SUCCESS
    assert result.tool == ToolType.VIDEO_TRANSCRIPT
    assert result.data is not None
    assert "1htKBjuUWec" in result.data["video_id"]
    assert "Hello world" in result.data["transcript"]
    assert result.message == "YouTube transcript fetched successfully"


def test_execute_transcript_disabled(sample_plan):
    """Test: Handles TranscriptsDisabled exception."""
    tool = VideoTool()

    with patch("atlasmind.core.tools.video_tool.YouTubeTranscriptApi.fetch") as mock_fetch:
        from youtube_transcript_api import TranscriptsDisabled
        mock_fetch.side_effect = TranscriptsDisabled("Transcript disabled")

        with pytest.raises(RuntimeError, match="Transcript unavailable"):
            tool.execute(sample_plan)


def test_execute_invalid_url():
    """Test: Raises error for invalid YouTube URL."""
    tool = VideoTool()
    invalid_plan = PlanTemplate(
        question="Video link missing or invalid",
        tool=ToolType.VIDEO_TRANSCRIPT,
        plan_steps=[],
        metadata={}
    )

    with pytest.raises(RuntimeError, match="Invalid or missing YouTube video URL"):
        tool.execute(invalid_plan)

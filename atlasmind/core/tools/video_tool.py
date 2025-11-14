import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from atlasmind.core.tools.base_tool import BaseTool, ToolResult, ExecutionStatus, ToolType
from atlasmind.core.planning.base_plan import PlanTemplate
from atlasmind.utils.logger import get_logger


class VideoTool(BaseTool):
    """Tool for extracting transcripts from YouTube videos."""

    def __init__(self):
        super().__init__()
        self.logger = get_logger(self.__class__.__name__)

    def _extract_video_id(self, text: str) -> str:
        """Extracts YouTube video ID from a given string or URL."""
        match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", text)
        if not match:
            raise ValueError(f"Invalid or missing YouTube video URL in: {text}")
        return match.group(1)

    def execute(self, plan: PlanTemplate) -> ToolResult:
        """Fetches transcript text from a YouTube video based on the provided plan."""
        question = plan.question

        try:
            # Extract video ID 
            self.logger.info(f"[YouTubeTranscriptTool] Starting transcript extraction for question {question}")
            video_id = self._extract_video_id(question)
            self.logger.info(f"[YouTubeTranscriptTool] Extracted video ID: {video_id}")

            # Fetch transcript
            yt = YouTubeTranscriptApi()
            transcript_snippets = yt.fetch(video_id=video_id)
            transcript_text = " ".join([t.text for t in transcript_snippets])

            self.logger.info(f"[YouTubeTranscriptTool] Transcript fetched successfully for {video_id}")

            # Step 3: Build and return ToolResult
            return ToolResult(
                question=question,
                tool=ToolType.VIDEO_TRANSCRIPT,
                message="YouTube transcript fetched successfully",
                data={"video_id": video_id, "transcript": transcript_text},
                query=None,
                status=ExecutionStatus.SUCCESS,
            )

        except (TranscriptsDisabled, NoTranscriptFound) as e:
            self.logger.error(f"[YouTubeTranscriptTool] Transcript not available for video: {e}")
            raise RuntimeError(f"Transcript unavailable: {e}")

        except Exception as e:
            self.logger.error(f"[YouTubeTranscriptTool] Failed for task {question}: {e}")
            raise RuntimeError(f"YouTubeTranscriptTool execution failed: {e}")

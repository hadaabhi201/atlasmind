from atlasmind.core.planning.base_plan import BasePlanner, PlanTemplate, ToolType


class VideoPlanner(BasePlanner):
    """Planner for VIDEO_REASONING tasks."""

    def __init__(self):
        super().__init__()

    def build_plan(self, question: str, file_path: str | None) -> PlanTemplate:
        """Build structured plan for video analysis and reasoning."""
        plan = self.create_plan(question, ToolType.VIDEO_TRANSCRIPT, file_path)

        self.add_step(plan, "Load the video input or video URL.")
        self.add_step(plan, "Extract transcript or frame-level metadata using YouTubeTranscriptTool.")
        self.add_step(plan, "Perform text-based reasoning or summarize the transcript.")
        self.add_step(plan, "Return synthesized insights, timestamps, or detected events.")

        self.add_metadata(plan, "source", "Video File or Streaming Platform")
        self.add_metadata(plan, "intent", "Video Content Understanding and Summarization")

        return self.finalize(plan)

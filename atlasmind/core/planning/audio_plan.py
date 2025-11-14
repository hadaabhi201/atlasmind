from atlasmind.core.planning.base_plan import BasePlanner, PlanTemplate, ToolType


class AudioPlanner(BasePlanner):
    """Planner for AUDIO_REASONING tasks."""

    def __init__(self):
        super().__init__()

    def build_plan(self, question: str, file_path: str | None) -> PlanTemplate:
        """Build structured plan for audio analysis and reasoning."""
        plan = self.create_plan(question, ToolType.AUDIO_TRANSCRIBER, file_path)

        self.add_step(plan, "Load the provided audio file or stream input.")
        self.add_step(plan, "Use AudioTranscriberTool to convert speech to text.")
        self.add_step(plan, "Extract entities or sentiment from transcribed text if applicable.")
        self.add_step(plan, "Summarize main findings and return insights.")

        self.add_metadata(plan, "source", "Audio File")
        self.add_metadata(plan, "intent", "Speech Transcription and Acoustic Analysis")


        return self.finalize(plan)

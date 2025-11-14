from atlasmind.core.planning.base_plan import BasePlanner, PlanTemplate, ToolType


class VisualPlanner(BasePlanner):
    """Planner for VISUAL_REASONING tasks."""

    def __init__(self):
        super().__init__()

    def build_plan(self, question: str, file_path: str | None) -> PlanTemplate:
        """Build structured plan for visual (image) reasoning."""
        plan = self.create_plan(question, ToolType.IMAGE_ANALYZER, file_path)

        self.add_step(plan, "Load the image input and validate format.")
        self.add_step(plan, "Use ImageAnalyzerTool to detect objects, text, or visual features.")
        self.add_step(plan, "If applicable, correlate detected elements with query context.")
        self.add_step(plan, "Generate reasoning-based description or visual answer.")

        self.add_metadata(plan, "source", "Image or Screenshot")
        self.add_metadata(plan, "intent", "Visual Recognition and Scene Reasoning")

        return self.finalize(plan)

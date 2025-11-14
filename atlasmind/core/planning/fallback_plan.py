from atlasmind.core.planning.base_plan import BasePlanner, PlanTemplate, ToolType


class FallbackPlanner(BasePlanner):
    """Fallback planner for unrecognized or unsupported task types."""

    def __init__(self):
        super().__init__()

    def build_plan(self, question: str, file_path: str | None) -> PlanTemplate:
        """Builds a generic fallback plan for unknown task types."""
        plan = self.create_plan(question, ToolType.WEB_SEARCH, file_path)

        self.add_step(plan, "Parse input and identify key keywords or entities.")
        self.add_step(plan, "Search web or relevant index using WebSearchTool.")
        self.add_step(plan, "Retrieve and summarize top-ranked results.")
        self.add_step(plan, "Provide a concise synthesized answer based on results.")

        self.add_metadata(plan, "source", "Fallback Web Search")
        self.add_metadata(plan, "intent", "Fallback Information Retrieval")

        return self.finalize(plan)

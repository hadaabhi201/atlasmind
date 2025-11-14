from atlasmind.core.planning.base_plan import BasePlanner, PlanTemplate, ToolType


class SemanticPlanner(BasePlanner):
    """Planner for semantic reasoning and categorization tasks."""

    def __init__(self):
        super().__init__()

    def build_plan(self, question: str, file_path: str | None) -> PlanTemplate:
        # Initialize the plan structure
        plan = self.create_plan(question, ToolType.SEMANTIC_ANALYZER, file_path)

        # Define semantic reasoning steps
        self.add_step(plan, "Parse the question to identify entities and their relationships.")
        self.add_step(plan, "Use SemanticTool to analyze meanings and relationships between entities.")
        self.add_step(plan, "Categorize or classify entities based on semantic context.")
        self.add_step(plan, "Organize and format results logically (e.g., grouped or sorted lists).")
        self.add_step(plan, "Synthesize the final structured response emphasizing accuracy of classification.")

        # Add metadata for reasoning trace
        self.add_metadata(plan, "source", "SemanticTool")
        self.add_metadata(plan, "intent", "Semantic Categorization or Logical Reasoning")

        return self.finalize(plan)

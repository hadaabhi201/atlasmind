from atlasmind.core.planning.base_plan import BasePlanner, PlanTemplate, ToolType


class KnowledgePlanner(BasePlanner):
    """Planner for knowledge retrieval and factual queries using Wikipedia."""

    def __init__(self):
        super().__init__()
    
    def build_plan(self, question: str, file_path: str | None) -> PlanTemplate:
        # Initialize the plan structure
        plan = self.create_plan(question, ToolType.WIKIPEDIA, file_path)
        
        # Define the logical steps
        self.add_step(plan, "Parse the question to extract main topic or entity.")
        self.add_step(plan, "Use WikipediaTool to fetch article summary for that topic.")
        self.add_step(plan, "Select the most relevant section (intro, summary, infobox).")
        self.add_step(plan, "Generate a concise synthesized answer.")
        
        # Optional metadata
        self.add_metadata(plan, "source", "Wikipedia")
        self.add_metadata(plan, "intent", "Factual Information Retrieval")
        
        return self.finalize(plan)

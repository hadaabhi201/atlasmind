from atlasmind.core.planning.base_plan import BasePlanner, PlanTemplate, ToolType


class StructuredDataPlanner(BasePlanner):
    """Planner for analyzing and reasoning over structured or tabular data (Excel, CSV)."""

    def __init__(self):
        super().__init__()

    def build_plan(self, question: str, file_path: str | None) -> PlanTemplate:
        # Initialize the plan structure
        plan = self.create_plan(question, ToolType.EXCEL_READER, file_path)

        # Define structured reasoning steps
        self.add_step(plan, "Load the provided Excel or CSV file using ExcelTool.")
        self.add_step(plan, "Identify relevant sheets, columns, and data ranges.")
        self.add_step(plan, "Parse the data and detect numerical or categorical fields.")
        self.add_step(plan, "Apply necessary filters, aggregations, or computations.")
        self.add_step(plan, "Summarize the findings or generate a formatted output table.")

        # Add metadata for traceability
        self.add_metadata(plan, "source", "ExcelTool")
        self.add_metadata(plan, "intent", "Structured Data Analysis")

        # Finalize and return the plan
        return self.finalize(plan)

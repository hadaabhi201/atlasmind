from atlasmind.core.planning.base_plan import BasePlanner, PlanTemplate, ToolType


class CodePlanner(BasePlanner):
    """Planner for CODE_EXECUTION tasks."""

    def __init__(self):
        super().__init__()

    def build_plan(self, question: str, file_path: str | None) -> PlanTemplate:
        """Build structured plan for executing or analyzing code."""
        plan = self.create_plan(question,ToolType.CODE_RUNNER, file_path)

        self.add_step(plan, "Parse the programming language and syntax of the provided code.")
        self.add_step(plan, "Validate and sandbox the code for secure execution.")
        self.add_step(plan, "Use CodeRunnerTool to execute or simulate the code.")
        self.add_step(plan, "Capture outputs, logs, and error traces.")
        self.add_step(plan, "Summarize or explain the results of the execution.")

        self.add_metadata(plan, "source", "User Code Snippet")
        self.add_metadata(plan, "intent", "Code Execution and Debugging")

        return self.finalize(plan)

from atlasmind.core.tools.base_tool import BaseTool, ToolResult, ExecutionStatus, ToolType
from atlasmind.core.planning.base_plan import PlanTemplate
from atlasmind.utils.logger import get_logger
from atlasmind.utils.llm_registry import llm_registry
from atlasmind.core.tools.semantic_helper.seamntic_reasoning_prompt import build_semantic_reasoning_prompt


class SemanticTool(BaseTool):
    """Tool for semantic categorization, classification, and text reasoning."""

    def __init__(self):
        super().__init__()
        self.logger = get_logger(self.__class__.__name__)
        self.llm = llm_registry.get_gemini_model()

    def execute(self, plan: PlanTemplate) -> ToolResult:
        """Executes semantic reasoning based on structured plan."""
        question = plan.question

        try:
            # --- Step 1: Build reasoning prompt ---
            prompt = build_semantic_reasoning_prompt(plan)
            self.logger.info(f"[SemanticTool] Executing semantic reasoning for question {question}")

            # --- Step 2: Run model ---
            response = self.llm.generate_content(prompt)
            answer = response.text

            # --- Step 3: Return result ---
            return ToolResult(
                question=question,
                tool=ToolType.SEMANTIC_ANALYZER,
                message="Semantic reasoning completed successfully",
                data={"answer": answer},
                query=None,
                status=ExecutionStatus.SUCCESS,
            )

        except Exception as e:
            self.logger.error(f"[SemanticTool] Failed for question {question}: {e}")
            raise RuntimeError(f"SemanticTool execution failed: {e}")

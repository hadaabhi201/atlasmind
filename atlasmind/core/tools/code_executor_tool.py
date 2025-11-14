import os
from atlasmind.core.tools.base_tool import BaseTool, ToolResult, ExecutionStatus
from atlasmind.core.tools.code_executor_svc.code_runner import CodeRunner
from atlasmind.core.planning.base_plan import PlanTemplate, ToolType
from atlasmind.utils.logger import get_logger

logger = get_logger(__name__)


class CodeExecutorTool(BaseTool):
    """Fetches Python code from Hugging Face and executes it via Judge0 (RapidAPI)."""

    def __init__(self):
        self.runner = CodeRunner()

    def execute(self, plan: PlanTemplate) -> ToolResult:
        """
        Execute the code execution plan.

        Args:
            plan (PlanTemplate): Contains question, and metadata.
        Returns:
            ToolResult: Structured result with stdout, stderr, exit_code, etc.
        """
        question = plan.question
        file_path = plan.file_path

        if not file_path or not os.path.exists(file_path):
            msg = (
                "[CodeExecutorTool] Invalid file_path. "
                f"Provided: {file_path}. "
                "A valid file must be supplied for code execution."
            )
            logger.error(f"{msg} Question: {question}")
            raise RuntimeError(msg)

        try:
            # Execute via Judge0
            result = self.runner.run(file_path)

            # Step 3: Construct ToolResult
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            status = ExecutionStatus.SUCCESS if not stderr else ExecutionStatus.FAILED

            logger.info(f"[CodeExecutorTool] Execution completed for question tt_id")

            return ToolResult(
                question=question,
                tool=ToolType.CODE_RUNNER,
                message="Code execution completed successfully",
                data={
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": result.get("status", {}).get("id"),
                    "execution_time": result.get("time"),
                },
                query=None,
                status=status,
            )

        except Exception as e:
            logger.error(f"[CodeExecutorTool] Failed for question {question}: {e}")
            raise RuntimeError(f"CodeExecutorTool execution failed: {e}")

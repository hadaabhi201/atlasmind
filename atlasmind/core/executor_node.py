from atlasmind.core.tools.video_tool import VideoTool
from atlasmind.core.tools.semantic_tool import SemanticTool
from atlasmind.core.tools.image_tool import ImageTool
from atlasmind.core.tools.excel_tool import ExcelTool
from atlasmind.core.tools.audio_tool import AudioTool
from atlasmind.core.tools.base_tool import ExecutionStatus, ToolResult
from atlasmind.utils.logger import get_logger
from atlasmind.core.planning.base_plan import PlanTemplate, ToolType
from atlasmind.core.tools.wikipedia_tool import WikipediaTool
from atlasmind.core.tools.web_search_tool import WebSearchTool
from atlasmind.core.tools.code_executor_tool import CodeExecutorTool


class ExecutorNode:
    """Runs the tool associated with a plan, with automatic fallback to WebSearchTool on failure."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.tools = {
            ToolType.WIKIPEDIA: WikipediaTool(),
            ToolType.WEB_SEARCH: WebSearchTool(),
            ToolType.CODE_RUNNER: CodeExecutorTool(),
            ToolType.AUDIO_TRANSCRIBER: AudioTool(),
            ToolType.EXCEL_READER: ExcelTool(),
            ToolType.IMAGE_ANALYZER: ImageTool(),
            ToolType.SEMANTIC_ANALYZER: SemanticTool(),
            ToolType.VIDEO_TRANSCRIPT: VideoTool(),
        }

    def execute(self, plan: PlanTemplate) -> ToolResult:
        """Execute the tool defined in the plan. Falls back to WebSearchTool if the primary tool fails."""
        tool_type = plan.tool
        self.logger.info(f"[Executor] Executing tool: {tool_type}")
        self.logger.info(f"[Executor] Plan : {plan}")

        # Initialize a default result (used in all error/fallback cases)
        result = ToolResult(
            question=plan.question,
            status=ExecutionStatus.ERROR,
            tool=tool_type,
            message="Tool execution not started.",
            data=None,
            query=None,
        )

        try:
            tool_instance = self.tools.get(tool_type)
            if not tool_instance:
                raise ValueError(f"No registered tool found for: {tool_type}")

            return tool_instance.execute(plan)

        # --- Handle failure and fallback ---
        except Exception as primary_error:
            self.logger.warning(f"[Executor] Primary tool '{tool_type}' failed: {primary_error}")
            result.status = ExecutionStatus.FAILED
            result.fallback_used = True
            result.message = str(primary_error)

            self.logger.info("[Executor] Attempting fallback: WebSearchTool")

            try:
                web_tool = self.tools.get(ToolType.WEB_SEARCH)
                if not web_tool:
                    raise ValueError("WebSearchTool not registered in executor.")
                fallback_result = web_tool.execute(plan)
                fallback_result.fallback_used = True
                self.logger.info("[Executor] WebSearchTool executed successfully as fallback.")
                return fallback_result

            except Exception as fallback_error:
                self.logger.error(f"[Executor] WebSearchTool fallback failed: {fallback_error}")
                result.tool = ToolType.WEB_SEARCH
                result.message = f"Both primary and fallback tools failed: {fallback_error}"
                result.status = ExecutionStatus.ERROR
                return result

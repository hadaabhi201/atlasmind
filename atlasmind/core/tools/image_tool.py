import os
from PIL import Image

from atlasmind.core.tools.base_tool import BaseTool, ToolResult, ExecutionStatus, ToolType
from atlasmind.utils.logger import get_logger
from atlasmind.core.planning.base_plan import PlanTemplate
from atlasmind.utils.llm_registry import llm_registry
from atlasmind.core.tools.image_tool_helper.image_reasoning_prompt import build_image_reasoning_prompt


class ImageTool(BaseTool):
    """Tool for multimodal image reasoning using Gemini."""

    def __init__(self):
        super().__init__()
        self.logger = get_logger(self.__class__.__name__)
        self.gemini_model = llm_registry.get_gemini_model()

    def execute(self,plan: PlanTemplate) -> ToolResult:
        """Executes image reasoning using Gemini multimodal API."""
        question = plan.question
        image_path = plan.file_path

        if not image_path or not os.path.exists(image_path):
            msg = (
                "[ImageTool] Invalid file_path. "
                f"Provided: {image_path}. "
                "A valid file must be supplied for image."
            )
            self.logger.error(f"{msg} Question: {question}")
            raise RuntimeError(msg)

        try:
            self.logger.info(f"[ImageAnalyzerTool] Fetched image for question {question}: {image_path}")

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            image = Image.open(image_path)

            prompt = build_image_reasoning_prompt(plan)
            self.logger.info(f"[ImageTool] Executing Gemini vision reasoning for question {question}")

            response = self.gemini_model.generate_content([prompt, image])
            output_text = response.text

            return ToolResult(
                question=question,
                tool=ToolType.IMAGE_ANALYZER,
                message="Image reasoning completed successfully",
                data={"analysis": output_text},
                query=None,
                status=ExecutionStatus.SUCCESS,
            )

        except Exception as e:
            self.logger.error(f"[ImageAnalyzerTool] Failed for question {question}: {e}")
            raise RuntimeError(f"ImageAnalyzerTool execution failed: {e}")
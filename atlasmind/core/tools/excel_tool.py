import os
import pandas as pd
import openpyxl
from atlasmind.utils.logger import get_logger
from atlasmind.core.tools.base_tool import BaseTool, ToolResult, ExecutionStatus
from atlasmind.core.planning.base_plan import PlanTemplate, ToolType


class ExcelTool(BaseTool):
    """
    ExcelTool — loads Excel data, extracts structure & previews,
    and delegates reasoning to the model via synthesizer.
    """

    def __init__(self):
        super().__init__()
        self.logger = get_logger(self.__class__.__name__)
        

    def execute(self, plan: PlanTemplate) -> ToolResult:
        """
        Load Excel file, summarize its content, and return
        structured metadata for reasoning by the synthesizer.
        """
        question = plan.question
        file_path = plan.file_path

        if not file_path or not os.path.exists(file_path):
            msg = (
                "[CodeExecutorTool] Invalid file_path. "
                f"Provided: {file_path}. "
                "A valid file must be supplied for code execution."
            )
            self.logger.error(f"{msg} Question: {question}")
            raise RuntimeError(msg)

        try:        
            # Load Excel safely
            df = pd.read_excel(file_path, engine="openpyxl")
            self.logger.info(f"[ExcelTool] Loaded {len(df)} rows and {len(df.columns)} columns.")

            if df.empty:
                raise ValueError("Excel file is empty or contains no readable data.")

            #  Prepare structured summary
            preview_rows = df.head(10).to_dict(orient="records")
            columns = list(df.columns)
            numeric_summary = df.describe(include="number").to_dict() if not df.empty else {} # type: ignore

            structured_payload = {
                "columns": columns,
                "preview_rows": preview_rows,
                "numeric_summary": numeric_summary,
                "row_count": len(df),
            }

            self.logger.info(f"[ExcelTool] Generated structured preview for LLM reasoning.")

            # Return ToolResult for synthesis layer to reason
            return ToolResult(
                question=question,
                tool=ToolType.EXCEL_READER,
                message="Excel file summarized successfully",
                data=structured_payload,
                query=None,
                status=ExecutionStatus.SUCCESS,
            )

        except Exception as e:
            self.logger.error(f"[ExcelTool] Failed for question {question}: {e}")
            raise RuntimeError(f"ExcelTool execution failed: {e}")

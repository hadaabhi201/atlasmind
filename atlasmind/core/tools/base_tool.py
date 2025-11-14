from abc import ABC, abstractmethod
from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Optional

from atlasmind.core.planning.base_plan import PlanTemplate, ToolType

class ExecutionStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    FAILED = "failed"

class ToolResult(BaseModel):
    """Standard structured result returned by all tools."""
    question: str = Field(..., description="The original user question that initiated this tool execution.")
    status: ExecutionStatus = Field(..., description="Execution status: success, failed, or error.")
    tool: ToolType = Field(..., description="The tool enum that produced this result.")
    message: Optional[str] = Field(None, description="Short message or summary of the result.")
    data: Optional[Any] = Field(None, description="Structured data or content returned by the tool.")
    query: Optional[str] = Field(None, description="Original input query that triggered this execution.")
    fallback_used: bool = Field(
        default=False,
        description="True if a fallback tool (e.g., WebSearchTool) was used after the primary tool failed."
    )

class BaseTool(ABC):
    """Abstract base class for all tools."""

    @abstractmethod
    def execute(self, plan: PlanTemplate) -> ToolResult:
        """Execute the tool and return a structured ToolResult."""
        pass

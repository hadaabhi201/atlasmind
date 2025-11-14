from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from atlasmind.utils.logger import get_logger

class ToolType(Enum):
    WIKIPEDIA = "WikipediaTool"
    AUDIO_TRANSCRIBER = "AudioTranscriberTool"
    VIDEO_TRANSCRIPT = "YouTubeTranscriptTool"
    IMAGE_ANALYZER = "ImageAnalyzerTool"
    EXCEL_READER = "ExcelTool"
    CODE_RUNNER = "CodeRunnerTool"
    WEB_SEARCH = "WebSearchTool"
    SEMANTIC_ANALYZER = "SemanticTool"

class PlanTemplate(BaseModel):
    """Defines a structured plan for executing a classified task."""
    question: str
    tool: ToolType
    plan_steps: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    file_path: Optional[str] = None
    

    class Config:
        arbitrary_types_allowed = True


class BasePlanner:
    """Base class for all planners. Handles shared logic like logging and step building."""
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def create_plan(self, question: str, tool: ToolType, file_path: str | None) -> PlanTemplate:
        """Initialize a generic plan structure."""
        return PlanTemplate(
            question=question,
            tool=tool,
            file_path=file_path,
            plan_steps=[],
            metadata={}
        )

    def add_step(self, plan: PlanTemplate, description: str):
        """Append a new step to the plan and log it."""
        plan.plan_steps.append(description)
        self.logger.debug(f"Added step: {description}")

    def add_metadata(self, plan: PlanTemplate, key: str, value: Any):
        """Attach contextual metadata to the plan."""
        if plan.metadata is None:
            plan.metadata = {}
        plan.metadata[key] = value
        self.logger.debug(f"Added metadata: {key}={value}")

    def finalize(self, plan: PlanTemplate) -> PlanTemplate:
        """Validate and finalize before returning."""
        return plan

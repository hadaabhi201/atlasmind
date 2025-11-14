from atlasmind.core.planning.refiner.llm_refiner import LLMPlanRefiner
from atlasmind.utils.logger import get_logger
from atlasmind.core.classification.base import ClassificationResult, ReasoningCategory
from atlasmind.core.planning.audio_plan import AudioPlanner
from atlasmind.core.planning.code_plan import CodePlanner
from atlasmind.core.planning.fallback_plan import FallbackPlanner
from atlasmind.core.planning.knowledge_plan import KnowledgePlanner
from atlasmind.core.planning.semantic_plan import SemanticPlanner
from atlasmind.core.planning.structured_data_plan import StructuredDataPlanner
from atlasmind.core.planning.video_plan import VideoPlanner
from atlasmind.core.planning.visual_plan import VisualPlanner
from atlasmind.core.planning.base_plan import PlanTemplate

class PlannerNode:
    """PlannerNode generates a step-by-step execution plan for each classified task."""
    
    def __init__(self, llm=None):
        self.logger = get_logger(__name__)
        self.llm = llm
        self.refiner = LLMPlanRefiner(llm, self.logger) if llm else None
        
        self.planners = {
            ReasoningCategory.KNOWLEDGE_RETRIEVAL: KnowledgePlanner(),
            ReasoningCategory.AUDIO_REASONING: AudioPlanner(),
            ReasoningCategory.VIDEO_REASONING: VideoPlanner(),
            ReasoningCategory.VISUAL_REASONING: VisualPlanner(),
            ReasoningCategory.CODE_EXECUTION: CodePlanner(),
            ReasoningCategory.STRUCTURED_DATA: StructuredDataPlanner(),
            ReasoningCategory.FALLBACK_SEARCH: FallbackPlanner(),
            ReasoningCategory.SEMANTIC_CATEGORIZATION: SemanticPlanner(),
        }

        self.fallback_planner = FallbackPlanner()

    async def plan(self, classification_result: ClassificationResult) -> PlanTemplate:
        """Generate an executable plan using the first matching planner.
        If no match or reasoning_type is missing, use the fallback plan.
        """
        reasoning_type = getattr(classification_result, "reasoning_type", None)
        question = classification_result.question
        file_path = classification_result.file_path


        base_plan = None

        # Loop through planners to find a matching reasoning category
        for category, planner in self.planners.items():
            if reasoning_type == category:
                
                base_plan = planner.build_plan(question, file_path)
                self.logger.info(f"Plan Tempate: {base_plan}")
               # Fallback if no matching planner found

        if base_plan is None:
            base_plan = self.fallback_planner.build_plan(question, file_path)

        if self.refiner and self.llm:
            try:
                base_plan = await self.refiner.refine(question, base_plan)
            except Exception as e:
                self.logger.error(f"[PlannerNode] LLM refinement failed for question={question}: {e}")

        return base_plan   

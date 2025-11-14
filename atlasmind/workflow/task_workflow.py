from llama_index.core.workflow import Workflow, step, Event, Context, StartEvent, StopEvent
from atlasmind.utils.llm_registry import llm_registry
from atlasmind.core.trace import ReasoningTrace
from atlasmind.core.executor_node import ExecutorNode
from atlasmind.core.synthesizer_node import SynthesizerNode
from atlasmind.core.synthesizer.base_synthesizre import SynthesisResult
from atlasmind.core.tools.base_tool import ToolResult
from atlasmind.core.planner_node import PlannerNode
from atlasmind.core.classification.base import ClassificationResult
from atlasmind.utils.logger import get_logger
from atlasmind.core.classifier_node import ClassifierNode
from atlasmind.core.planning.base_plan import PlanTemplate


# ---- EVENTS ----
class ClassificationEvent(Event):
    classification: ClassificationResult


class PlanEvent(Event):
    plan: PlanTemplate


class ExecutionEvent(Event):
    execution: ToolResult


class SynthesisEvent(Event):
    synthesis: SynthesisResult


class TaskWorkflow(Workflow):
    """AtlasMind workflow for planning and later execution."""

    def __init__(self, timeout: int = 90, verbose: bool = True):
        super().__init__(timeout=timeout, verbose=verbose)
        self.timeout = timeout
        self.verbose = verbose
        self.logger = get_logger(__name__)
        planning_llm = llm_registry.get_planning_llm()
        reasoning_llm = llm_registry.get_reasoning_llm()
        
        self.classifier = ClassifierNode()
        self.planner = PlannerNode(planning_llm)
        self.executor = ExecutorNode()
        self.synthesizer = SynthesizerNode(summarize_llm=planning_llm, synthesize_llm=reasoning_llm)
        self.trace = ReasoningTrace()

    @step
    async def classify_step(self, ev: StartEvent, ctx: Context) -> ClassificationEvent:
        """Run the classifier on the given question."""
        try:
            task = await ctx.store.get("task")
            self.logger.info(f"Classifying task: {task}")
            if not isinstance(task, dict):
                raise TypeError(f"Expected dict in context store, got {type(task)}")

            classification = self.classifier.classify(task)
            self.trace.log(f"[Classifier] Identified reasoning type: {classification.reasoning_type}")
            return ClassificationEvent(classification=classification)
        except Exception as e:
            self.logger.error(f"Execution failed on Classify Workflow: {e}")
            self.trace.log(f"[Classifier] Failed: {e}")
            raise

    @step
    async def planning_step(self, ev: ClassificationEvent, ctx: Context) -> PlanEvent:
        """Generate plan from classification result."""
        try:
            plan = await self.planner.plan(ev.classification)
            self.trace.log(f"[Planner] Created a plan using tool: {plan.tool}")
            return PlanEvent(plan=plan)
        except Exception as e:
            self.logger.error(f"Execution failed on Planning Step Workflow: {e}")
            self.trace.log(f"[Planner] Failed: {e}")
            raise

    @step
    async def execute_tools(self, ev: PlanEvent, ctx: Context) -> ExecutionEvent:
        """Execute the generated plan."""
        try:
            result = self.executor.execute(ev.plan)
            self.trace.log(f"[Executor] Executed tool {ev.plan.tool} successfully.")
            if result.fallback_used:
                self.trace.log(f"[Executor] Primary tool {ev.plan.tool.name} failed. ")

            return ExecutionEvent(execution=result)
        except Exception as e:
            self.trace.log(f"[Executor] Failed: {e}")
            self.logger.error(f"Execution failed on Execute Tool Workflow: {e}")
            raise

    @step
    async def synthesize(self, ev: ExecutionEvent, ctx: Context) -> StopEvent:
        """Generate the final synthesized answer."""
        try:
            synthesis_result = await self.synthesizer.synthesize(ev.execution)
            self.trace.log("[Synthesizer] Generated the final model answer.")
        except Exception as e:
            self.trace.log(f"[Synthesizer] Failed: {e}")
            raise


        return StopEvent(result= synthesis_result.model_answer)

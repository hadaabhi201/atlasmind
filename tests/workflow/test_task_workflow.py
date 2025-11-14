import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from llama_index.core.workflow import Context
from atlasmind.workflow.task_workflow import TaskWorkflow
from atlasmind.core.classification.base import ReasoningCategory, ClassificationResult
from atlasmind.core.planning.base_plan import PlanTemplate, ToolType
from atlasmind.core.tools.base_tool import ExecutionStatus, ToolResult
from atlasmind.core.synthesizer.base_synthesizre import SynthesisResult


def make_task(question: str, file_path=None):
    return {"question": question, "file_path": file_path}


def make_classification(question: str, file_path, category):
    return ClassificationResult(
        question=question,
        file_path=file_path,
        reasoning_type=category
    )


def make_plan(question, tool):
    return PlanTemplate(
        question=question,
        tool=tool,
        plan_steps=["Step 1", "Step 2"],
        metadata={"source": tool.value}
    )


def make_tool_result(question, tool, data):
    return ToolResult(
        question=question,
        status=ExecutionStatus.SUCCESS,
        tool=tool,
        message="ok",
        data=data,
        query=""
    )


def set_up_workflow(mock_classification, mock_plan, mock_tool_result=None, synth_result=None):
    with patch.object(TaskWorkflow, "__init__", return_value=None):
        workflow = TaskWorkflow()
        workflow.logger = MagicMock()
        workflow.classifier = MagicMock()
        workflow.planner = MagicMock()
        workflow.executor = MagicMock()
        workflow.synthesizer = MagicMock()

        from atlasmind.core.trace import ReasoningTrace
        workflow.trace = ReasoningTrace()
        workflow.timeout = 30
        workflow.verbose = True

        workflow.classifier.classify.return_value = mock_classification
        workflow.planner.plan = AsyncMock(return_value=mock_plan)

        if mock_tool_result:
            workflow.executor.execute.return_value = mock_tool_result

        if synth_result:
            workflow.synthesizer.synthesize = AsyncMock(return_value=synth_result)

        return workflow
    
@pytest.mark.asyncio
@pytest.mark.parametrize("file_path", [None, "samples/test.mp3"])
async def test_task_workflow_full_chain(file_path):
    question = "What is the capital of France?"
    task = make_task(question, file_path)

    classification = make_classification(
        question, file_path, ReasoningCategory.KNOWLEDGE_RETRIEVAL
    )

    plan = make_plan(question, ToolType.WIKIPEDIA)

    tool_result = make_tool_result(question, ToolType.WIKIPEDIA, {"answer": "Paris"})

    synth_result = SynthesisResult(
        model_answer="Paris is the capital of France.",
        reasoning_trace="trace ok",
    )

    workflow = set_up_workflow(classification, plan, tool_result, synth_result)

    ctx = Context(workflow)
    await ctx.store.set("task", task)

    c_event = await workflow.classify_step(None, ctx)
    p_event = await workflow.planning_step(c_event, ctx)
    e_event = await workflow.execute_tools(p_event, ctx)
    s_event = await workflow.synthesize(e_event, ctx)

    assert c_event.classification.file_path == file_path
    assert p_event.plan == plan
    assert e_event.execution == tool_result
    assert s_event.result == "Paris is the capital of France."

    assert isinstance(workflow.classifier.classify, MagicMock)
    workflow.classifier.classify.assert_called_once_with(task)

@pytest.mark.asyncio
@pytest.mark.parametrize("file_path", [None, "samples/test.mp3"])
async def test_task_workflow_executor_exception(file_path):
    question = "What is the capital of France?"
    task = make_task(question, file_path)

    classification = make_classification(
        question, file_path, ReasoningCategory.KNOWLEDGE_RETRIEVAL
    )

    plan = make_plan(question, ToolType.WIKIPEDIA)

    workflow = set_up_workflow(classification, plan)
    
    workflow.executor.execute = MagicMock()
    workflow.executor.execute.side_effect = RuntimeError("Executor node failed")

    ctx = Context(workflow)
    await ctx.store.set("task", task)

    c_event = await workflow.classify_step(None, ctx)
    p_event = await workflow.planning_step(c_event, ctx)

    with pytest.raises(RuntimeError):
        await workflow.execute_tools(p_event, ctx)

    trace = str(workflow.trace)
    assert "[Executor]" in trace
    assert "Executor node failed" in trace
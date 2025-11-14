import pytest
from atlasmind.core.executor_node import ExecutorNode
from atlasmind.core.tools.base_tool import ToolResult, ExecutionStatus
from atlasmind.core.planning.base_plan import PlanTemplate, ToolType


@pytest.fixture
def sample_plan():
    return PlanTemplate(
        question="What is Python?",
        tool=ToolType.WIKIPEDIA,
        plan_steps=["Fetch Wikipedia page about Python"],
        metadata={}
    )

class MockFailingTool:
    def execute(self, plan):
        raise RuntimeError("Primary tool failed")


class MockSucceedingTool:
    def execute(self, plan):
        return ToolResult(
            question=plan.question,
            status=ExecutionStatus.SUCCESS,
            tool=ToolType.WEB_SEARCH,
            message="WebSearchTool executed successfully (mock).",
            data={"result": "fallback-success"},
            query=None,
            fallback_used=True,
        )


class MockFailingFallback:
    def execute(self, plan):
        raise RuntimeError("Fallback tool failed")

@pytest.mark.parametrize(
    "primary_tool, fallback_tool, expected_status, expected_fallback_used, expected_tool_type, expected_msg_part",
    [
        # 1. Primary FAILS → Fallback SUCCEEDS
        (
            MockFailingTool(),
            MockSucceedingTool(),
            ExecutionStatus.SUCCESS,
            True,
            ToolType.WEB_SEARCH,
            "websearchtool executed successfully"
        ),
        # 2. Primary FAILS → Fallback FAILS
        (
            MockFailingTool(),
            MockFailingFallback(),
            ExecutionStatus.ERROR,
            True,
            ToolType.WEB_SEARCH,
            "Both primary and fallback tools failed"
        ),
        # 3. Primary MISSING → Fallback SUCCEEDS
        (
            None,  # primary missing
            MockSucceedingTool(),
            ExecutionStatus.SUCCESS,
            True,
            ToolType.WEB_SEARCH,
            "websearchtool executed successfully"
        ),
    ]
)
def test_executor_node_behavior(
    sample_plan,
    primary_tool,
    fallback_tool,
    expected_status,
    expected_fallback_used,
    expected_tool_type,
    expected_msg_part,
):
    executor = ExecutorNode()

    # Install fallback tool
    executor.tools[ToolType.WEB_SEARCH] = fallback_tool

    # Install or remove primary tool
    if primary_tool is not None:
        executor.tools[ToolType.WIKIPEDIA] = primary_tool
    else:
        executor.tools.pop(ToolType.WIKIPEDIA, None)

    # Execute
    result = executor.execute(sample_plan)

    # Assertions
    assert result.status == expected_status
    assert result.fallback_used == expected_fallback_used
    assert result.tool == expected_tool_type
    assert expected_msg_part.lower() in result.message.lower()

    if expected_status == ExecutionStatus.SUCCESS:
        assert result.data is not None
        assert "result" in result.data
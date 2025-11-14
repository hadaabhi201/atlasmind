import pytest
from unittest.mock import Mock, patch
from atlasmind.core.tools.code_executor_tool import CodeExecutorTool
from atlasmind.core.planning.base_plan import PlanTemplate, ToolType
from atlasmind.core.tools.base_tool import ToolResult, ExecutionStatus

@pytest.fixture
def sample_plan(tmp_path):
    """Creates a temporary Python file for execution."""
    code_file = tmp_path / "sample_code.py"
    code_file.write_text("print('Hello Test')")

    return PlanTemplate(
        question="Run this code",
        tool=ToolType.CODE_RUNNER,
        file_path=str(code_file)
    )

@patch("atlasmind.core.tools.code_executor_tool.CodeRunner")
def test_execute_success(mock_runner_class, sample_plan):
    mock_runner = mock_runner_class.return_value

    mock_runner.run.return_value = {
        "stdout": "Hello Test\n",
        "stderr": "",
        "status": {"id": 3},
        "time": "0.01"
    }

    tool = CodeExecutorTool()
    result = tool.execute(sample_plan)

    mock_runner.run.assert_called_once_with(sample_plan.file_path)

    assert result.status == ExecutionStatus.SUCCESS
    assert result.data is not None
    assert result.data["stdout"] == "Hello Test\n"

@patch("atlasmind.core.tools.code_executor_tool.CodeRunner")
def test_execute_with_stderr(mock_runner_class, sample_plan):
    mock_runner = mock_runner_class.return_value

    mock_runner.run.return_value = {
        "stdout": "",
        "stderr": "NameError: x not defined",
        "status": {"id": 6},
        "time": "0.01"
    }

    tool = CodeExecutorTool()
    result = tool.execute(sample_plan)

    assert result.status == ExecutionStatus.FAILED
    assert result.data is not None
    assert "NameError" in result.data["stderr"]

def test_execute_missing_file():
    tool = CodeExecutorTool()

    bad_plan = PlanTemplate(
        question="Run code",
        tool=ToolType.CODE_RUNNER,
        file_path=None
    )

    with pytest.raises(RuntimeError, match="Invalid file_path"):
        tool.execute(bad_plan)


def test_execute_invalid_file():
    tool = CodeExecutorTool()

    bad_plan = PlanTemplate(
        question="Run code",
        tool=ToolType.CODE_RUNNER,
        file_path="/tmp/does_not_exist.py"
    )

    with pytest.raises(RuntimeError, match="Invalid file_path"):
        tool.execute(bad_plan)

@patch("atlasmind.core.tools.code_executor_tool.CodeRunner")
def test_execute_runner_failure(mock_runner_class, sample_plan):
    mock_runner = mock_runner_class.return_value
    mock_runner.run.side_effect = RuntimeError("Judge0 error")

    tool = CodeExecutorTool()

    with pytest.raises(RuntimeError, match="CodeExecutorTool execution failed"):
        tool.execute(sample_plan)
import pytest
from unittest.mock import patch, MagicMock

from atlasmind.core.tools.semantic_tool import SemanticTool
from atlasmind.core.planning.base_plan import PlanTemplate
from atlasmind.core.tools.base_tool import ToolType, ExecutionStatus


@pytest.fixture
def sample_plan():
    """Fixture: Provide a minimal semantic reasoning plan."""
    return PlanTemplate(
        question="Categorize grocery list to include only vegetables, alphabetized.",
        tool=ToolType.SEMANTIC_ANALYZER,
        plan_steps=[
            "Parse the grocery list.",
            "Identify items that are vegetables (exclude fruits and seeds).",
            "Alphabetize the vegetable names.",
            "Return them as a comma-separated list."
        ],
        metadata={"source": "unit-test"}
    )


def test_semantic_tool_success(sample_plan):
    """Test successful execution using mocked Gemini model."""
    tool = SemanticTool()

    with (
        patch("atlasmind.core.tools.semantic_tool.build_semantic_reasoning_prompt",
              return_value="Mocked prompt for testing.") as mock_prompt,
        patch.object(tool.llm, "generate_content") as mock_generate,
    ):
        mock_resp = MagicMock()
        mock_resp.text = "broccoli, celery, fresh basil, lettuce, sweet potatoes"
        mock_generate.return_value = mock_resp

        result = tool.execute(sample_plan)

        # --- Assertions ---
        mock_prompt.assert_called_once_with(sample_plan)
        mock_generate.assert_called_once_with("Mocked prompt for testing.")
        assert result.status == ExecutionStatus.SUCCESS
        assert result.tool == ToolType.SEMANTIC_ANALYZER
        assert result.data is not None
        assert "broccoli" in result.data["answer"]
        assert result.message is not None
        assert "Semantic reasoning completed successfully" in result.message


def test_semantic_tool_failure(sample_plan):
    """Test when Gemini model raises an exception."""
    tool = SemanticTool()

    with patch.object(tool.llm, "generate_content", side_effect=Exception("Gemini API failed")):
        with pytest.raises(RuntimeError) as excinfo:
            tool.execute(sample_plan)
        assert "SemanticTool execution failed" in str(excinfo.value)

import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from atlasmind.core.tools.image_tool import ImageTool
from atlasmind.core.planning.base_plan import PlanTemplate
from atlasmind.core.tools.base_tool import ToolType, ExecutionStatus


@pytest.fixture
def dummy_image(tmp_path):
    """Fixture: Create a valid PNG image."""
    image_path = tmp_path / "test_image.png"
    img = Image.new("RGB", (64, 64), color="white")
    img.save(image_path)
    return str(image_path)


@pytest.fixture
def sample_plan(dummy_image):
    """Provide a plan *with file_path*, required by new ImageTool."""
    return PlanTemplate(
        question="What does this image show?",
        tool=ToolType.IMAGE_ANALYZER,
        plan_steps=["Analyze image"],
        metadata={"source": "unittest"},
        file_path=dummy_image   # IMPORTANT
    )

def test_image_tool_success(sample_plan):
    tool = ImageTool()

    mock_response = MagicMock()
    mock_response.text = "This image appears to show a white square."

    with patch.object(tool.gemini_model, "generate_content", return_value=mock_response) as mock_generate:
        result = tool.execute(sample_plan)

        assert result.status == ExecutionStatus.SUCCESS
        assert result.tool == ToolType.IMAGE_ANALYZER
        assert result.data is not None
        assert "white square" in result.data["analysis"].lower()
        mock_generate.assert_called_once()

def test_image_tool_missing_file():
    tool = ImageTool()

    bad_plan = PlanTemplate(
        question="Test missing image",
        tool=ToolType.IMAGE_ANALYZER,
        file_path="does_not_exist.png"
    )

    with pytest.raises(RuntimeError, match="Invalid file_path"):
        tool.execute(bad_plan)

def test_image_tool_exception(sample_plan):
    tool = ImageTool()

    with patch.object(tool.gemini_model, "generate_content", side_effect=Exception("Gemini API failed")):
        with pytest.raises(RuntimeError, match="Gemini API failed"):
            tool.execute(sample_plan)
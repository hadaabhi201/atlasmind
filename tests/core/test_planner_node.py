import pytest
from unittest.mock import AsyncMock, patch
from atlasmind.core.planner_node import PlannerNode
from atlasmind.core.classification.base import ReasoningCategory
from atlasmind.core.planning.base_plan import PlanTemplate, ToolType


@pytest.fixture
def planner_node():
    """Fixture to initialize PlannerNode without real LLM refiner."""
    with patch("atlasmind.core.planner_node.LLMPlanRefiner", autospec=True) as mock_refiner_cls:
        mock_refiner = AsyncMock()
        mock_refiner.refine = AsyncMock(side_effect=lambda q, p: p)  # Return unchanged plan
        mock_refiner_cls.return_value = mock_refiner
        node = PlannerNode(llm="fake-llm")
        return node


@pytest.fixture
def classification_result_factory():
    """Factory to create mock classification_result objects."""
    class MockResult:
        def __init__(self, reasoning_type, file_path=None):
            self.reasoning_type = reasoning_type
            self.question = "Mock question for testing."
            self.file_path = file_path
    return MockResult


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,reasoning_type,expect_fallback,file_path",
    [
        ("Knowledge Retrieval", ReasoningCategory.KNOWLEDGE_RETRIEVAL, False, None),
        ("Audio Reasoning", ReasoningCategory.AUDIO_REASONING, False, "samples/audio.mp3"),
        ("Video Reasoning", ReasoningCategory.VIDEO_REASONING, False, None),
        ("Visual Reasoning", ReasoningCategory.VISUAL_REASONING, False, "samples/image.png"),
        ("Code Execution", ReasoningCategory.CODE_EXECUTION, False, "sampes/code.py"),
        ("Structured Data", ReasoningCategory.STRUCTURED_DATA, False, "samples/data.csv"),
        ("Semantic Categorization", ReasoningCategory.SEMANTIC_CATEGORIZATION, False, None),
        ("Fallback - Known Category", ReasoningCategory.FALLBACK_SEARCH, True, None),
        ("Fallback - None Category", None, True, None),
        ("Fallback - Unknown Category", "UnknownCategory", True, None),
    ]
)
async def test_planner_node_dispatch(
    name, reasoning_type, expect_fallback,file_path,
    planner_node, classification_result_factory
):
    """Test PlannerNode.plan() dispatch for all reasoning categories including fallback."""
    classification_result = classification_result_factory(reasoning_type, file_path)

    # --- Act ---
    result = await planner_node.plan(classification_result)

    # --- Validate PlanTemplate structure ---
    assert isinstance(result, PlanTemplate), f"{name} failed: Did not return PlanTemplate"
    assert result.plan_steps and isinstance(result.plan_steps, list), f"{name} failed: Missing plan steps"

    # --- Validate ToolType ---
    assert result.tool is not None, f"{name} failed: Tool is None"
    if isinstance(result.tool, ToolType):
        assert result.tool in ToolType, f"{name} failed: Invalid ToolType {result.tool}"
    elif isinstance(result.tool, str):
        valid_values = [t.value for t in ToolType]
        assert result.tool in valid_values, f"{name} failed: Unknown tool string {result.tool}"
    else:
        pytest.fail(f"{name} failed: Tool must be ToolType or string, got {type(result.tool)}")
    
     # --- File path propagation tests ---
    if reasoning_type in {
        ReasoningCategory.AUDIO_REASONING,
        ReasoningCategory.VIDEO_REASONING,
        ReasoningCategory.VISUAL_REASONING,
        ReasoningCategory.CODE_EXECUTION,
        ReasoningCategory.STRUCTURED_DATA,
    }:
        assert result.file_path == classification_result.file_path, (
            f"{name} failed: file_path not propagated in PlanTemplate"
        )
    else:
        assert result.file_path is None, (
            f"{name} failed: file_path should be None for reasoning type {reasoning_type}"
        )

    # --- Fallback checks ---
    if expect_fallback:
        src = (result.metadata or {}).get("source", "").lower()
        steps = " ".join(result.plan_steps).lower()
        assert "fallback" in src or "fallback" in steps, f"{name} failed: Fallback plan not triggered"
    else:
        assert result.tool, f"{name} failed: No tool assigned for reasoning type {reasoning_type}"

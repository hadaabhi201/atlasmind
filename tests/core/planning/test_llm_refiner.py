import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from atlasmind.core.planning.refiner.llm_refiner import LLMPlanRefiner
from atlasmind.core.planning.base_plan import PlanTemplate, ToolType


@pytest.mark.asyncio
async def test_llm_refiner_success():
    """Ensure LLMPlanRefiner enriches plan with metadata and refined steps."""
    fake_response = {
        "entity": "Python",
        "intent": "Summarize features",
        "specialized_steps": ["Fetch docs", "Summarize"],
    }

    fake_llm = AsyncMock()
    fake_llm.acomplete.return_value = MagicMock(text=json.dumps(fake_response))
    fake_logger = MagicMock()

    base_plan = PlanTemplate(
        question="What is Python?",
        plan_steps=["Step 1"],
        tool=ToolType.WIKIPEDIA,
        metadata={},  # ensure dict
    )

    refiner = LLMPlanRefiner(fake_llm, fake_logger)
    refined = await refiner.refine("What is Python?", base_plan)

    # Metadata and steps updated
    assert refined.metadata is not None
    assert refined.metadata["entity"] == "Python"
    assert "Fetch docs" in refined.plan_steps
    fake_logger.info.assert_called()


@pytest.mark.asyncio
async def test_llm_refiner_handles_invalid_json():
    """If LLM returns non-JSON output, refiner should skip enrichment."""
    fake_llm = AsyncMock()
    fake_llm.acomplete.return_value = MagicMock(text="INVALID_JSON")
    fake_logger = MagicMock()

    base_plan = PlanTemplate(
        question="Invalid test",
        plan_steps=["Step 1"],
        tool=ToolType.WIKIPEDIA,
        metadata={"original": True},
    )

    refiner = LLMPlanRefiner(fake_llm, fake_logger)
    refined = await refiner.refine("Invalid test", base_plan)

    # Metadata remains unchanged
    assert refined.metadata is not None
    assert refined.metadata["original"] is True
    fake_logger.warning.assert_called()


@pytest.mark.asyncio
async def test_llm_refiner_handles_llm_exception():
    """If LLM throws exception, refiner returns plan unchanged."""
    fake_llm = AsyncMock()
    fake_llm.acomplete.side_effect = RuntimeError("LLM failed")
    fake_logger = MagicMock()

    base_plan = PlanTemplate(
        question="Test fallback",
        plan_steps=["Step 1"],
        tool=ToolType.WIKIPEDIA,
        metadata={"safe": True},
    )

    refiner = LLMPlanRefiner(fake_llm, fake_logger)
    refined = await refiner.refine("Test fallback", base_plan)

    # Should not modify metadata or crash
    assert refined.metadata is not None
    assert refined.metadata["safe"] is True
    fake_logger.error.assert_called()

import pytest
from unittest.mock import AsyncMock, patch
from atlasmind.core.synthesizer.summarizer_agent import SummarizerAgent


@pytest.mark.asyncio
async def test_combine_summaries_builds_prompt_and_calls_llm():
    """Verify combine_summaries builds correct prompt and calls LLM."""
    mock_llm = AsyncMock()
    mock_llm.acomplete.return_value = "   combined summary   "

    with patch("atlasmind.core.synthesizer.summarizer_agent.build_combine_prompt", return_value="combine-prompt"):
        agent = SummarizerAgent(mock_llm)
        result = await agent.combine_summaries("query", "question", "summaries")

    mock_llm.acomplete.assert_awaited_once_with("combine-prompt")
    assert result == "combined summary"

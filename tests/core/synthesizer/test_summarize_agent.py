import pytest
from unittest.mock import AsyncMock, patch
from atlasmind.core.synthesizer.summarizer_agent import SummarizerAgent


@pytest.mark.asyncio
async def test_summarize_chunk_builds_prompt_and_calls_llm():
    """Verify summarize_chunk builds prompt and calls LLM asynchronously."""
    mock_llm = AsyncMock()
    mock_llm.acomplete.return_value = "   partial summary   "

    with patch("atlasmind.core.synthesizer.summarizer_agent.build_summarize_prompt", return_value="prompt-text"):
        agent = SummarizerAgent(mock_llm)
        result = await agent.summarize_chunk("query", "question", "text")

    mock_llm.acomplete.assert_awaited_once_with("prompt-text")
    assert result == "partial summary"

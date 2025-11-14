import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from atlasmind.core.synthesizer.summarize_guard import (
    SummarizerGuard,
    _rough_count_tokens,
    _split_text_hard,
    HARD_TOKEN_LIMIT,
)


# ----------- Utility function tests -----------

def test_rough_count_tokens_basic():
    """_rough_count_tokens should approximate 1 token per 4 characters."""
    assert _rough_count_tokens("abcd") == 1
    assert _rough_count_tokens("a" * 20) == 5
    assert _rough_count_tokens(None) == 0


def test_split_text_hard_splits_correctly():
    """_split_text_hard should split text into chunks ~ max_tokens * 4 characters each."""
    text = "a" * 50
    chunks = list(_split_text_hard(text, max_tokens=5))  # 5*4=20 chars per chunk
    assert len(chunks) == 3
    assert all(isinstance(c, str) for c in chunks)


# ----------- SummarizerGuard tests -----------

@pytest.mark.asyncio
async def test_process_within_limit():
    """If text is within token limit, summarization is skipped."""

    mock_llm = MagicMock()
    guard = SummarizerGuard(mock_llm, token_limit=HARD_TOKEN_LIMIT)

    small_text = "Short text"
    result = await guard.process(query="Q", question="What?", data=small_text)

    assert result == small_text


@pytest.mark.asyncio
async def test_process_exceeds_limit(monkeypatch):
    """When text exceeds token limit, it should summarize chunks and combine them."""

    mock_llm = MagicMock()
    mock_agent = AsyncMock()
    mock_agent.summarize_chunk.side_effect = ["Summary 1", "Summary 2"]
    mock_agent.combine_summaries.return_value = "Final Combined Summary"

    with patch("atlasmind.core.synthesizer.summarize_guard.SummarizerAgent", return_value=mock_agent):
        guard = SummarizerGuard(mock_llm, token_limit=10)  # small to force summarization
        long_text = "x" * 30000  # large enough to exceed limit
        result = await guard.process(query="Q", question="What?", data=long_text)

    assert result == "Final Combined Summary"
    assert mock_agent.summarize_chunk.await_count >= 1
    assert mock_agent.combine_summaries.await_count == 1

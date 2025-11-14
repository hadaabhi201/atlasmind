import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from atlasmind.core.synthesizer_node import SynthesizerNode
from atlasmind.core.synthesizer.base_synthesizre import SynthesisResult
from atlasmind.core.tools.base_tool import ToolResult, ExecutionStatus, ToolType


@pytest.mark.asyncio
async def test_synthesize_success(monkeypatch):
    """Test successful synthesis when tool execution succeeds."""

    # Mock summarizer
    mock_summarizer = AsyncMock()
    mock_summarizer.process.return_value = "short summary"

    # Mock LLMs
    mock_summarize_llm = MagicMock()
    mock_synthesize_llm = MagicMock()
    mock_synthesize_llm.complete.return_value = MagicMock(text="Final synthesized answer")

    # Patch SummarizerGuard to use mock
    with patch("atlasmind.core.synthesizer_node.SummarizerGuard", return_value=mock_summarizer):
        node = SynthesizerNode(mock_summarize_llm, mock_synthesize_llm)

    # Create a successful tool result
    execution = ToolResult(
        question="Who is Ada Lovelace?",
        query="Ada Lovelace biography",
        tool=ToolType.WIKIPEDIA,
        status=ExecutionStatus.SUCCESS,
        message="OK",
        data="Full Wikipedia content about Ada Lovelace",
    )

    result = await node.synthesize(execution)

    # Assertions
    assert isinstance(result, SynthesisResult)
    assert result.model_answer == "Final synthesized answer"
    assert "Synthesized final answer" in result.reasoning_trace
    mock_summarizer.process.assert_awaited_once_with(
        "Ada Lovelace biography", "Who is Ada Lovelace?", "Full Wikipedia content about Ada Lovelace"
    )
    mock_synthesize_llm.complete.assert_called_once()


@pytest.mark.asyncio
async def test_synthesize_tool_failure(monkeypatch):
    """Test that SynthesizerNode returns fallback answer when tool fails."""

    mock_summarize_llm = MagicMock()
    mock_synthesize_llm = MagicMock()

    with patch("atlasmind.core.synthesizer_node.SummarizerGuard"):
        node = SynthesizerNode(mock_summarize_llm, mock_synthesize_llm)

    execution = ToolResult(
        question="What is the capital of France?",
        query="France capital",
        tool=ToolType.WIKIPEDIA,
        status=ExecutionStatus.ERROR,
        message="Request failed",
        data=None,
    )

    result = await node.synthesize(execution)

    assert isinstance(result, SynthesisResult)
    assert result.model_answer == "I'm sorry, I couldn't generate an answer."
    assert "failed" in result.reasoning_trace.lower()


@pytest.mark.asyncio
async def test_synthesize_exception(monkeypatch):
    """Test that SynthesizerNode handles unexpected exceptions gracefully."""

    mock_summarizer = AsyncMock()
    mock_summarizer.process.side_effect = RuntimeError("LLM crashed")

    mock_summarize_llm = MagicMock()
    mock_synthesize_llm = MagicMock()

    with patch("atlasmind.core.synthesizer_node.SummarizerGuard", return_value=mock_summarizer):
        node = SynthesizerNode(mock_summarize_llm, mock_synthesize_llm)

    execution = ToolResult(
        question="What is AI?",
        query="AI definition",
        tool=ToolType.WIKIPEDIA,
        status=ExecutionStatus.SUCCESS,
        message="OK",
        data="Artificial intelligence info",
    )

    result = await node.synthesize(execution)

    assert isinstance(result, SynthesisResult)
    assert "couldn't generate" in result.model_answer
    assert "failed" in result.reasoning_trace.lower()

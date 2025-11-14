import traceback
from atlasmind.core.synthesizer.summarize_guard import SummarizerGuard
from atlasmind.core.synthesizer.prompts.synthesis_prompt_template import build_synthesis_prompt
from atlasmind.utils.logger import get_logger
from atlasmind.core.synthesizer.base_synthesizre import SynthesisResult
from atlasmind.core.tools.base_tool import ExecutionStatus, ToolResult


class SynthesizerNode:
    """SynthesizerNode generates the final natural language answer."""

    def __init__(self, summarize_llm, synthesize_llm):
        self.logger = get_logger(self.__class__.__name__)
        self.summarize_llm = summarize_llm
        self.synthesize_llm = synthesize_llm
        self.summarizer = SummarizerGuard(llm=self.summarize_llm)

    async def synthesize(self, execution: ToolResult) -> SynthesisResult:
        """Generate the final synthesized answer from a tool execution result."""
        try:
            question = getattr(execution, "question", "")
            query = getattr(execution, "query", "")
            tool = execution.tool
            status = execution.status
            message = getattr(execution, "message", "")
            data = getattr(execution, "data", "")

            # Handle unsuccessful tool execution
            if status != ExecutionStatus.SUCCESS:
                self.logger.warning(f"[Synthesizer] Tool execution failed: {status}")
                self.logger.warning(f"[Synthesizer] Error Message: {message}")
                return SynthesisResult(
                    model_answer="I'm sorry, I couldn't generate an answer.",
                    reasoning_trace=f"[Synthesizer] Tool {tool} failed.",
                )

            # Summarize data if available
            summarized_data = await self.summarizer.process(query, question, data)

            # Build synthesis prompt
            prompt = build_synthesis_prompt(
                question=question,
                tool=tool,
                query=query,
                message=message,
                data=summarized_data,
            )

            # Generate response using synthesize LLM
            response = self.synthesize_llm.complete(prompt)
            answer_text = getattr(response, "text", str(response)).strip()

            return SynthesisResult(
                model_answer=answer_text,
                reasoning_trace=f"[Synthesizer] Synthesized final answer from {tool}.",
            )

        except Exception as e:
            self.logger.error(f"[Synthesizer] Failed: {e}")
            self.logger.error("Full traceback:\n" + traceback.format_exc())
            return SynthesisResult(
                model_answer="I'm sorry, I couldn't generate an answer.",
                reasoning_trace=f"[Synthesizer] Failed due to error: {e}",
            )

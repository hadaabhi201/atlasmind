from atlasmind.core.synthesizer.prompts.summarize_prompt import build_prompt as build_summarize_prompt
from atlasmind.core.synthesizer.prompts.combine_prompt import build_prompt as build_combine_prompt
from atlasmind.utils.logger import get_logger

logger = get_logger(__name__)

class SummarizerAgent:
    def __init__(self, llm):
        self.llm = llm  # expects OpenAILike compatible with .acomplete(prompt: str) -> str-like

    async def summarize_chunk(self, query: str, question: str, text: str) -> str:
        prompt = build_summarize_prompt(query=query, question=question, text=text)
        logger.debug("[SummarizerAgent] summarize_chunk request")
        resp = await self.llm.acomplete(prompt)
        return resp.strip() if hasattr(resp, "strip") else str(resp)

    async def combine_summaries(self, query: str, question: str, summaries: str) -> str:
        prompt = build_combine_prompt(query=query, question=question, summaries=summaries)
        logger.debug("[SummarizerAgent] combine_summaries request")
        resp = await self.llm.acomplete(prompt)
        return resp.strip() if hasattr(resp, "strip") else str(resp)

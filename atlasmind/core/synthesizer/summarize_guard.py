from typing import Any, Iterable, List, Optional
from atlasmind.utils.logger import get_logger
from atlasmind.core.synthesizer.summarizer_agent import SummarizerAgent

logger = get_logger(__name__)

HARD_TOKEN_LIMIT = 6000
CHUNK_TARGET = 5000 

def _rough_count_tokens(text: Optional[str]) -> int:
    if text is None:
        return 0
    return max(1, int(len(str(text)) / 4))

def _split_text_hard(text: str, max_tokens: int) -> Iterable[str]:
    s = str(text)
    approx_chars = max_tokens * 4
    for i in range(0, len(s), approx_chars):
        yield s[i:i + approx_chars]

class SummarizerGuard:
    def __init__(self, llm, token_limit: int = HARD_TOKEN_LIMIT):
        self.llm = llm
        self.token_limit = token_limit
        self.agent = SummarizerAgent(llm)

    async def process(self, query: str, question: str, data: Any) -> str:
        text = str(data) if data is not None else ""
        total = _rough_count_tokens(text)
        logger.info(f"[SummarizerGuard] Approx tokens: {total}")

        if total <= self.token_limit:
            logger.info("[SummarizerGuard] Within limit. Skipping summarization.")
            return text

        chunks: List[str] = list(_split_text_hard(text, CHUNK_TARGET))
        partials: List[str] = []
        for idx, chunk in enumerate(chunks, 1):
            logger.info(f"[SummarizerGuard] Summarizing chunk {idx}/{len(chunks)}")
            partial = await self.agent.summarize_chunk(query=query, question=question, text=chunk)
            partials.append(partial)

        combined_input = "\n\n".join(partials)
        logger.info("[SummarizerGuard] Combining partial summaries")
        final_summary = await self.agent.combine_summaries(query=query, question=question, summaries=combined_input)
        return final_summary

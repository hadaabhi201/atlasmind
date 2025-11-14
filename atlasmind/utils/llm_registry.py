from llama_index.llms.openai_like import OpenAILike
import google.generativeai as genai
from atlasmind.utils.config import settings


class LLMRegistry:
    """Centralized registry for all LLM clients used in Voyager-MM."""

    def __init__(self):
       # Main reasoning model
        self.reasoning_llm = OpenAILike(
            model="openai/gpt-4o",
            api_base="https://models.github.ai/inference",
            api_key=settings.GITHUB_TOKEN,
            is_chat_model=True,
        )

        # Lightweight model for retrieval/summarization
        self.planning_llm = OpenAILike(
            model="openai/gpt-4o-mini",
            api_base="https://models.github.ai/inference",
            api_key=settings.GITHUB_TOKEN,
            is_chat_model=True,
        )
        
        self._gemini_model = None
            

        
    
    def get_reasoning_llm(self):
        return self.reasoning_llm

    def get_planning_llm(self):
        return self.planning_llm
    
    def get_gemini_model(self):
        if self._gemini_model is None:
            genai.configure(api_key=settings.GEMINI_API_KEY) # type: ignore
            self._gemini_model = genai.GenerativeModel('gemini-2.5-flash') # type: ignore
        return self._gemini_model



# Singleton instance for all nodes to share
llm_registry = LLMRegistry()
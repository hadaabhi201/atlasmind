from llama_index.llms.openai_like import OpenAILike
from google import genai
from atlasmind.utils.config import settings


class GeminiResponseWrapper:
    """Ensures response.text always exists."""

    def __init__(self, raw_response):
        self.raw = raw_response
        self.text = self._extract_text()

    def _extract_text(self):
        try:
            cands = self.raw.candidates
            if not cands:
                return ""
            parts = cands[0].content.parts
            if not parts:
                return ""
            # First text part
            for p in parts:
                if hasattr(p, "text"):
                    return p.text
            return ""  # no text part found
        except Exception:
            return ""

    def __str__(self):
        return self.text or ""

class GeminiModelWrapper:
    """
    Wrapper so AtlasMind tools can continue calling:
        model.generate_content(...)
    while internally using the new Google Client API.
    """

    def __init__(self, client: genai.Client, model_name: str):
        self.client = client
        self.model_name = model_name

    def generate_content(self, contents):
        # supports string, list, multimodal
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents
        )
        return response
    

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
        self._gemini_client = None    
        
    
    def get_reasoning_llm(self):
        return self.reasoning_llm

    def get_planning_llm(self):
        return self.planning_llm
    
    def get_gemini_model(self):
        """Return a generate_content()-compatible Gemini model wrapper."""
        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)

        if self._gemini_model is None:
            # configure wrapper around the new Client API
            self._gemini_model = GeminiModelWrapper(
                client=self._gemini_client,
                model_name="gemini-2.5-flash"
            )

        return self._gemini_model



# Singleton instance for all nodes to share
llm_registry = LLMRegistry()
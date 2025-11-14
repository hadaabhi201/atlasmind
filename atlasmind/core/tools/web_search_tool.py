from serpapi import GoogleSearch
from atlasmind.core.tools.base_tool import BaseTool, ToolResult, ExecutionStatus
from atlasmind.core.planning.base_plan import PlanTemplate
from atlasmind.utils.logger import get_logger
from atlasmind.utils.config import settings


class WebSearchTool(BaseTool):
    """Placeholder implementation for WebSearchTool."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.api_key = settings.SERPAPI_KEY

    def execute(self, plan: PlanTemplate) -> ToolResult:
        """Execute a Google search using SerpApi."""
        query = plan.question
        self.logger.info(f"[WebSearchTool] Executing search for: {query}")

        try:
            params = {
                "q": query,
                "api_key": self.api_key,
                "num": 5,
            }

            search = GoogleSearch(params)
            results = search.get_dict()

            # Basic validation of API response
            if "error" in results:
                raise RuntimeError(f"SerpApi returned an error: {results['error']}")

            organic_results = results.get("organic_results")
            if not organic_results:
                raise ValueError("No search results returned from SerpApi.")

            top_results = [
                {
                    "title": r.get("title"),
                    "link": r.get("link"),
                    "snippet": r.get("snippet"),
                }
                for r in organic_results
            ]

            return ToolResult(
                question=query,
                status=ExecutionStatus.SUCCESS,
                tool=plan.tool,
                message=f"Fetched {len(top_results)} web search results.",
                data=top_results,
                query=query,
            )

        except Exception as e:
            self.logger.error(f"[WebSearchTool] Search failed: {e}")
            raise RuntimeError(f"WebSearchTool execution failed: {e}")

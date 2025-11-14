from bs4 import BeautifulSoup
import requests
from typing import Optional, Dict, Any
from atlasmind.core.tools.base_tool import BaseTool, ExecutionStatus, ToolResult
from atlasmind.core.planning.base_plan import PlanTemplate
from atlasmind.utils.logger import get_logger
from atlasmind.utils.config import settings


class WikipediaTool(BaseTool):
    """Fetches detailed Wikipedia page content for a given query."""

    HEADERS = {
        "User-Agent": "AtlasMindBot/1.0 (atlasmind.ai)"
    }

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.base_url = settings.WIKI_API

    # -------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------
    def _search_topic(self, query: str) -> Optional[str]:
        """Return the most relevant Wikipedia page title for a given query."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
        }
        try:
            resp = requests.get(self.base_url, params=params, headers=self.HEADERS, timeout=10)
            resp.raise_for_status()
            results = resp.json().get("query", {}).get("search", [])
            return results[0]["title"] if results else None
        except Exception as e:
            self.logger.error(f"[WikipediaTool] Search failed for '{query}': {e}")
            raise RuntimeError(f"Wikipedia search request failed: {e}")

    def _fetch_full_page(self, title: str) -> str:
        """Return the full plain-text content of a Wikipedia page, including all sections."""
        params = {
            "action": "parse",
            "page": title,
            "prop": "text",
            "format": "json",
            "redirects": 1,
        }
        try:
            resp = requests.get(self.base_url, params=params, headers=self.HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            # Extract HTML safely from the response
            page_html = data.get("parse", {}).get("text", {}).get("*", "")
            if not page_html:
                self.logger.warning(f"[WikipediaTool] Empty content for '{title}'")
                return "No content found."

            soup = BeautifulSoup(page_html, "html.parser")
            page_text = soup.get_text(separator="\n").strip()

            return page_text

        except Exception as e:
            self.logger.error(f"[WikipediaTool] Fetch failed for '{title}': {e}")
            raise RuntimeError(f"Wikipedia fetch failed for '{title}': {e}")


    # -------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------
    def execute(self, plan: PlanTemplate) -> ToolResult:
        """Execute Wikipedia search and fetch based on plan metadata."""
        metadata: Dict[str, Any] = plan.metadata or {}
        question = metadata.get("question", plan.question)
        entity = metadata.get("entity", "unknown")
        filters = metadata.get("filters", {})
        intent = metadata.get("intent", "Fetch detailed content from Wikipedia")

        query = f"{entity} {filters.get('category', '')}".strip()
        self.logger.info(f"[WikipediaTool] Query: {query}")

        # ---- search and fetch ----
        title = self._search_topic(query)
        if not title:
            message = f"No Wikipedia page found for '{query}'."
            self.logger.warning(f"[WikipediaTool] Execution failed: {message}")
            raise ValueError(message)

        content = self._fetch_full_page(title)

        if filters.get("start_year") and filters.get("end_year"):
            self.logger.info(
                f"[WikipediaTool] Time range applied: {filters['start_year']}-{filters['end_year']}"
            )

        return ToolResult(
            question=question,
            status=ExecutionStatus.SUCCESS,
            tool=plan.tool,
            message="Wikipedia full page content fetched successfully.",
            data={
                "title": title,
                "content": content,
                "filters": filters,
                "intent": intent,
            },
            query=query,
        )

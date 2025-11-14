import pytest
from atlasmind.core.tools.web_search_tool import WebSearchTool
from atlasmind.core.tools.base_tool import ExecutionStatus
from atlasmind.core.planning.base_plan import PlanTemplate, ToolType


@pytest.fixture
def mock_settings(monkeypatch):
    """Mock settings.SERPAPI_KEY before creating the tool."""
    monkeypatch.setattr("atlasmind.core.tools.web_search_tool.settings.SERPAPI_KEY", "mock_api_key")


@pytest.fixture
def sample_plan():
    """Return a sample PlanTemplate for testing."""
    return PlanTemplate(
        question="What is the capital of France?",
        tool=ToolType.WEB_SEARCH,
        plan_steps=["Perform a web search"]
    )


class MockGoogleSearch:
    """Mock replacement for serpapi.GoogleSearch."""

    def __init__(self, params):
        self.params = params

    def get_dict(self):
        return {
            "organic_results": [
                {
                    "title": "Paris - Wikipedia",
                    "link": "https://en.wikipedia.org/wiki/Paris",
                    "snippet": "Capital of France"
                }
            ]
        }


def test_execute_success(monkeypatch, mock_settings, sample_plan):
    """Test successful search execution with mock SerpApi."""
    monkeypatch.setattr("atlasmind.core.tools.web_search_tool.GoogleSearch", MockGoogleSearch)

    tool = WebSearchTool()
    result = tool.execute(sample_plan)

    assert result.status == ExecutionStatus.SUCCESS
    assert result.data is not None
    assert "Paris" in result.data[0]["title"]
    assert len(result.data) == 1
    assert result.message is not None
    assert "Fetched" in result.message


def test_execute_serpapi_error(monkeypatch, mock_settings, sample_plan):
    """Test handling of SerpApi returning an error field."""
    class MockErrorSearch(MockGoogleSearch):
        def get_dict(self):
            return {"error": "Invalid query"}

    monkeypatch.setattr("atlasmind.core.tools.web_search_tool.GoogleSearch", MockErrorSearch)
    tool = WebSearchTool()

    with pytest.raises(RuntimeError, match="SerpApi returned an error"):
        tool.execute(sample_plan)


def test_execute_no_results(monkeypatch, mock_settings, sample_plan):
    """Test handling of empty or missing organic_results list."""
    class MockNoResultsSearch:
        def __init__(self, params):
            self.params = params

        def get_dict(self):
            return {"organic_results": []}

    monkeypatch.setattr("atlasmind.core.tools.web_search_tool.GoogleSearch", MockNoResultsSearch)
    tool = WebSearchTool()

    # Expect RuntimeError because the ValueError is caught and re-raised
    with pytest.raises(RuntimeError, match="WebSearchTool execution failed: No search results returned from SerpApi."):
        tool.execute(sample_plan)

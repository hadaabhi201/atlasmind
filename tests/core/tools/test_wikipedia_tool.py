import pytest
import requests
from atlasmind.core.tools.wikipedia_tool import WikipediaTool
from atlasmind.core.tools.base_tool import ExecutionStatus
from atlasmind.core.planning.base_plan import PlanTemplate, ToolType


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def sample_plan():
    return PlanTemplate(
        question="Who is Ada Lovelace?",
        tool=ToolType.WIKIPEDIA,
        plan_steps=["Search Wikipedia for Ada Lovelace"],
        metadata={"entity": "Ada Lovelace"},
    )


# ----------------------------------------------------------------------
# Mocking helpers
# ----------------------------------------------------------------------
class MockResponse:
    """Simple mock for requests.Response."""
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError("Bad request")

    def json(self):
        return self._json


# ----------------------------------------------------------------------
# TEST 1: Successful fetch (normal flow)
# ----------------------------------------------------------------------
def test_execute_success(monkeypatch, sample_plan):
    """Should return ToolResult with SUCCESS when Wikipedia API responds normally."""

    def mock_get(url, params, headers, timeout):
        if params.get("list") == "search":
            # Mock the search endpoint
            return MockResponse({"query": {"search": [{"title": "Ada Lovelace"}]}})
        else:
            html_content = "<div class='mw-parser-output'><p>Ada Lovelace was a mathematician.</p></div>"
            return MockResponse({
                "parse": {
                    "text": {"*": html_content}
                }
            })
    monkeypatch.setattr(requests, "get", mock_get)

    tool = WikipediaTool()
    result = tool.execute(sample_plan)

    assert result.status == ExecutionStatus.SUCCESS
    assert result.tool == ToolType.WIKIPEDIA
    assert result.data is not None
    assert result.data["title"] == "Ada Lovelace"
    assert "mathematician" in result.data["content"]
    assert result.fallback_used is False 



# ----------------------------------------------------------------------
# TEST 2: No Wikipedia page found -> raises ValueError
# ----------------------------------------------------------------------
def test_execute_no_results(monkeypatch, sample_plan):
    """Should raise ValueError when no Wikipedia page found."""

    def mock_search(url, params, headers, timeout):
        return MockResponse({"query": {"search": []}})

    monkeypatch.setattr(requests, "get", mock_search)

    tool = WikipediaTool()
    with pytest.raises(ValueError) as exc_info:
        tool.execute(sample_plan)

    assert "No Wikipedia page found" in str(exc_info.value)


# ----------------------------------------------------------------------
# TEST 3: Wikipedia search request failure -> raises RuntimeError
# ----------------------------------------------------------------------
def test_execute_search_failure(monkeypatch, sample_plan):
    """Should raise RuntimeError if search API call fails."""

    def mock_fail(*args, **kwargs):
        raise requests.RequestException("Network error")

    monkeypatch.setattr(requests, "get", mock_fail)

    tool = WikipediaTool()
    with pytest.raises(RuntimeError) as exc_info:
        tool._search_topic("Ada Lovelace")

    assert "Wikipedia search request failed" in str(exc_info.value)


# ----------------------------------------------------------------------
# TEST 4: Wikipedia fetch failure -> raises RuntimeError
# ----------------------------------------------------------------------
def test_execute_fetch_failure(monkeypatch, sample_plan):
    """Should raise RuntimeError if fetch API call fails."""

    def mock_fetch(*args, **kwargs):
        raise requests.RequestException("Timeout error")

    monkeypatch.setattr(requests, "get", mock_fetch)

    tool = WikipediaTool()
    with pytest.raises(RuntimeError) as exc_info:
        tool._fetch_full_page("Ada Lovelace")

    assert "Wikipedia fetch failed" in str(exc_info.value)

import pytest
from atlasmind.core.planning.refiner.schema import PlanRefinementSchema


def test_validate_full_data():
    """Valid LLM JSON should normalize all fields and keep structure intact."""
    data = {
        "entity": "Python",
        "intent": "Summarize features",
        "filters": {"year": 2024},
        "topic": "Programming",
        "answer_type": "list",
        "language": "ENGLISH ",
        "confidence": "0.95",
        "specialized_steps": ["Fetch docs", "Extract key points"],
    }

    validated = PlanRefinementSchema.validate(data)

    assert validated["entity"] == "Python"
    assert validated["intent"] == "Summarize features"
    assert isinstance(validated["filters"], dict)
    assert validated["language"] == "ENGLISH"
    assert isinstance(validated["confidence"], float)
    assert "Fetch docs" in validated["specialized_steps"]


def test_validate_missing_fields_defaults():
    """Missing keys should fallback to defaults safely."""
    data = {"entity": "World War II"}

    validated = PlanRefinementSchema.validate(data)

    assert validated["entity"] == "World War II"
    assert validated["intent"] == ""
    assert validated["filters"] == {}
    assert validated["language"] == "English"
    assert validated["specialized_steps"] == ["No refined steps provided."]


def test_validate_invalid_data_types():
    """Non-dict or malformed data should not crash and should return defaults."""
    validated = PlanRefinementSchema.validate("not-a-dict")

    assert isinstance(validated, dict)
    assert validated == {} or validated == {}


def test_validate_specialized_steps_string():
    """If specialized_steps is a single string, convert it to a list."""
    data = {"specialized_steps": "Single step"}
    validated = PlanRefinementSchema.validate(data)

    assert validated["specialized_steps"] == ["Single step"]

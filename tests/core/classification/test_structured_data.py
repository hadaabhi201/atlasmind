import pytest
from atlasmind.core.classification.structured_data import StructuredDataClassifier
from atlasmind.core.classification.base import ReasoningCategory
@pytest.fixture
def classifier():
    return StructuredDataClassifier()

def test_detects_structured_data(classifier):
    q = "The attached Excel file contains the menu sales. What is the total revenue?"
    r = classifier.classify(q)
    assert r is not None
    assert r == ReasoningCategory.STRUCTURED_DATA

def test_ignores_non_tabular(classifier):
    q = "Listen to the mp3 file."
    assert classifier.classify(q) is None

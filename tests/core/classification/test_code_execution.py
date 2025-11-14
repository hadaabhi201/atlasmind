import pytest
from atlasmind.core.classification.code_execution import CodeExecutionClassifier
from atlasmind.core.classification.base import ReasoningCategory

@pytest.fixture
def classifier():
    return CodeExecutionClassifier()

def test_detects_code_execution(classifier):
    q = "Run the attached Python code and give the result."
    r = classifier.classify(q)
    assert r is not None
    assert r == ReasoningCategory.CODE_EXECUTION

def test_ignores_non_code(classifier):
    q = "What does the image show?"
    assert classifier.classify(q) is None

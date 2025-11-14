import pytest
from atlasmind.core.classification.knowledge_retrieval import KnowledgeRetrievalClassifier
from atlasmind.core.classification.base import ReasoningCategory

@pytest.fixture
def classifier():
    return KnowledgeRetrievalClassifier()

def test_detects_explicit_wikipedia_reference(classifier):
    question = "How many albums were published by Mercedes Sosa? You can use English Wikipedia."
    result = classifier.classify(question)
    assert result is not None
    assert result == ReasoningCategory.KNOWLEDGE_RETRIEVAL

def test_detects_factual_terms(classifier):
    question = "Where were the Vietnamese specimens described by Kuznetzov in Nedoshivina's 2010 paper eventually deposited?"
    result = classifier.classify(question)
    assert result is not None
    assert result == ReasoningCategory.KNOWLEDGE_RETRIEVAL

def test_detects_interrogative_pattern(classifier):
    question = "Who nominated the only Featured Article on English Wikipedia about a dinosaur?"
    result = classifier.classify(question)
    assert result is not None
    assert result == ReasoningCategory.KNOWLEDGE_RETRIEVAL

def test_excludes_non_text_tasks(classifier):
    question = "The attached Excel file contains sales data. Compute total sales."
    result = classifier.classify(question)
    assert result is None

def test_returns_none_for_unrelated_question(classifier):
    question = "Listen to the attached mp3 and summarize it."
    result = classifier.classify(question)
    assert result is None
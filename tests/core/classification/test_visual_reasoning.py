import pytest
from atlasmind.core.classification.visual_reasoning import VisualReasoningClassifier
from atlasmind.core.classification.base import ReasoningCategory

@pytest.fixture
def classifier():
    return VisualReasoningClassifier()

def test_detects_image_question(classifier):
    q = "Review the chess position provided in the attached image."
    r = classifier.classify(q)
    assert r is not None, "Expected classifier to detect visual reasoning question"
    assert r == ReasoningCategory.VISUAL_REASONING

def test_detects_photo_or_diagram_reference(classifier):
    q = "Analyze the diagram and describe what it shows."
    r = classifier.classify(q)
    assert r is not None, "Expected classifier to detect diagram-related visual question"
    assert r == ReasoningCategory.VISUAL_REASONING

def test_ignores_non_visual(classifier):
    q = "Listen to the attached mp3 file."
    r = classifier.classify(q)
    assert r is None, "Expected classifier to ignore non-visual queries"

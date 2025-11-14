import pytest
from atlasmind.core.classification.audio_reasoning import AudioReasoningClassifier
from atlasmind.core.classification.base import ReasoningCategory

@pytest.fixture
def classifier():
    return AudioReasoningClassifier()

def test_detects_audio_reference(classifier):
    q = "Please listen to the attached mp3 recording and summarize the recipe."
    r = classifier.classify(q)
    assert r is not None
    assert r == ReasoningCategory.AUDIO_REASONING

def test_excludes_non_audio(classifier):
    q = "The attached Excel file contains data."
    assert classifier.classify(q) is None

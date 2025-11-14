import pytest
from atlasmind.core.classification.video_reasoning import VideoReasoningClassifier
from atlasmind.core.classification.base import ReasoningCategory

@pytest.fixture
def classifier():
    return VideoReasoningClassifier()

def test_detects_youtube_reference(classifier):
    q = "In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of birds?"
    r = classifier.classify(q)
    assert r is not None
    assert r == ReasoningCategory.VIDEO_REASONING

def test_non_video_question_returns_none(classifier):
    q = "Read the attached PDF and summarize."
    assert classifier.classify(q) is None

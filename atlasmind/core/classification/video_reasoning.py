import re
from atlasmind.utils.logger import get_logger
from atlasmind.core.classification.base import BaseClassifier, ReasoningCategory

class VideoReasoningClassifier(BaseClassifier):
    def __init__(self):
        self.logger = get_logger(__name__)
        self.patterns = {
            "youtube": re.compile(r"(youtube\.com|youtu\.be)", re.IGNORECASE),
            "video_terms": re.compile(r"(video|clip|footage|on\s+camera)", re.IGNORECASE),
        }

    def classify(self, question: str, file_path: str | None = None) -> ReasoningCategory | None:
        text = question.strip()

        if self.patterns["youtube"].search(text) or self.patterns["video_terms"].search(text):
            return ReasoningCategory.VIDEO_REASONING
        return None

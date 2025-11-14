import os
import re
from atlasmind.utils.logger import get_logger
from atlasmind.core.classification.base import (
    BaseClassifier, ReasoningCategory
)

class VisualReasoningClassifier(BaseClassifier):
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

    def __init__(self):
        self.logger = get_logger(__name__)
        self.patterns = {
            "visual_terms": re.compile(
                r"\b(see|look at|shown in|refer to)?\s*(image|picture|photo|diagram|screenshot|figure(?!\s*out))\b",
                re.IGNORECASE,
            ),
            "exclude": re.compile(r"(video|audio|mp3|excel|code|script)", re.IGNORECASE),
        }

    def classify(self, question: str, file_path: str | None = None) -> ReasoningCategory | None:
        text = question.strip()

        self.logger.warning(f"VisualReasoning file path:{file_path}")
        if self.patterns["exclude"].search(text):
            self.logger.debug("Skipped VisualReasoning due to exclusion pattern.")
            return None

        has_image_ext = False
        if file_path:
            _, ext = os.path.splitext(file_path.lower())
            has_image_ext = ext in self.IMAGE_EXTENSIONS

        if has_image_ext or self.patterns["visual_terms"].search(text):
            return ReasoningCategory.VISUAL_REASONING

        return None

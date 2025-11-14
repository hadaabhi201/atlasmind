import re
from atlasmind.utils.logger import get_logger
from atlasmind.core.classification.base import (
    BaseClassifier,
    ReasoningCategory,
)


class SemanticReasoningClassifier(BaseClassifier):
    """Classifier for semantic reasoning and categorization tasks."""

    def __init__(self):
        self.logger = get_logger(__name__)

        self.patterns = {
            "categorize": re.compile(
                r"\b(categorize|categorizing|classify|classification|group|grouping|organize|sort|arrange|separate|filter|headings?)\b",
                re.IGNORECASE,
            ),
            "logical": re.compile(
                r"\b(difference|compare|relation|relationship|belongs to|similar|same type|type of)\b",
                re.IGNORECASE,
            ),
            "semantic": re.compile(
                r"\b(fruit|fruits|vegetable|vegetables|animal|animals|object|objects|category|categories|botanical|species|taxonomy|list of)\b",
                re.IGNORECASE,
            ),
        }

    def classify(self, question: str, file_path: str | None = None) -> ReasoningCategory | None:
        text = question.strip().lower()

        # Match at least one reasoning signal
        if (
            self.patterns["categorize"].search(text)
            or self.patterns["logical"].search(text)
            or self.patterns["semantic"].search(text)
        ):
            return ReasoningCategory.SEMANTIC_CATEGORIZATION

        self.logger.debug("No semantic reasoning patterns matched.")
        return None

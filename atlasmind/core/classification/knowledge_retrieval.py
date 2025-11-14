import re
from atlasmind.utils.logger import get_logger
from atlasmind.core.classification.base import BaseClassifier, ReasoningCategory


class KnowledgeRetrievalClassifier(BaseClassifier):
    """Classifier for factual or knowledge-based questions."""

    def __init__(self):
        self.logger = get_logger(__name__)

        # Precompile regex patterns once
        self.media_skip_pattern = re.compile(
            r"(attached|mp3|excel|video|image|code)",
            re.IGNORECASE,
        )
        self.factual_pattern = re.compile(
            r"\b(article|paper|research|published|award|team|olympics|century|specimens?|actor|actress|cast|role|played|voice|dub|version|language|character|film|movie|tv|series|season|episode|imdb|pitcher|player|yankees?|walks|at\s*bats)\b",
            re.IGNORECASE,
        )

    def classify(self, question: str, file_path: str | None = None) -> ReasoningCategory | None:
        """Detect factual or research-related questions."""
        text = question.strip().lower()

        # Skip non-text media
        if self.media_skip_pattern.search(text):
            self.logger.debug("Skipped KnowledgeRetrieval due to non-text pattern.")
            return None

        # Apply detection rules
        if "wikipedia" in text:
            self.logger.info("Detected explicit 'wikipedia' keyword.")
        elif self.factual_pattern.search(text):
            self.logger.info("Detected factual or research-related term.")
        else:
            return None

        # Return structured classification result
        return ReasoningCategory.KNOWLEDGE_RETRIEVAL

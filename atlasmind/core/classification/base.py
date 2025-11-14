from enum import Enum
from dataclasses import dataclass
from typing import Optional

class ReasoningCategory(Enum):
    KNOWLEDGE_RETRIEVAL = "KnowledgeRetrieval"
    AUDIO_REASONING = "AudioReasoning"
    VIDEO_REASONING = "VideoReasoning"
    VISUAL_REASONING = "VisualReasoning"
    STRUCTURED_DATA = "StructuredDataReasoning"
    CODE_EXECUTION = "CodeExecution"
    SEMANTIC_CATEGORIZATION = "SemanticCategorization"
    FALLBACK_SEARCH = "FallbackSearch"


@dataclass
class ClassificationResult:
    question: str
    file_path: Optional[str]
    reasoning_type: ReasoningCategory    

class BaseClassifier:
    """Base class for all specific classifiers."""
    def classify(self, question: str, file_path: str | None) -> ReasoningCategory:
        raise NotImplementedError("Subclasses must implement classify()")

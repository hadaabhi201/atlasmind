from atlasmind.core.classification.visual_reasoning import VisualReasoningClassifier
from atlasmind.core.classification.audio_reasoning import AudioReasoningClassifier
from atlasmind.core.classification.video_reasoning import VideoReasoningClassifier
from atlasmind.core.classification.structured_data import StructuredDataClassifier
from atlasmind.core.classification.code_execution import CodeExecutionClassifier
from atlasmind.core.classification.semanitc_reasoning import SemanticReasoningClassifier
from atlasmind.utils.logger import get_logger
from atlasmind.core.classification.knowledge_retrieval import KnowledgeRetrievalClassifier
from atlasmind.core.classification.base import ClassificationResult, ReasoningCategory

class ClassifierNode:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.classifiers = [
            KnowledgeRetrievalClassifier(),
            AudioReasoningClassifier(),
            VideoReasoningClassifier(),
            StructuredDataClassifier(),
            CodeExecutionClassifier(),
            VisualReasoningClassifier(),
            SemanticReasoningClassifier(),
        ]

    def classify(self, task)  -> ClassificationResult:
        question = task.get("question", "")
        file_path = task.get("file_path", None) 

        for classifier in self.classifiers:
            reasoning_type = classifier.classify(question, file_path)
            if reasoning_type:
                self.logger.info(f"Detected {reasoning_type.value} for question {question}")
                return ClassificationResult(
                    question=question,
                    file_path=file_path,
                    reasoning_type=reasoning_type
                )

        # Fallback if no classification matches
        self.logger.warning(f"No classifier matched for question {question}")
        return ClassificationResult(
            file_path=file_path,
            question=question,           
            reasoning_type=ReasoningCategory.FALLBACK_SEARCH,
        )

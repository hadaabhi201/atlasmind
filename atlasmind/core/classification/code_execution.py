import os
import re
from atlasmind.utils.logger import get_logger
from atlasmind.core.classification.base import (
    BaseClassifier, ReasoningCategory
)

class CodeExecutionClassifier(BaseClassifier):
    PYTHON_EXTENSIONS = {".py"}

    def __init__(self):
        self.logger = get_logger(__name__)
        self.patterns = {
            "code_terms": re.compile(
                r"\b("
                r"python|java|c\+\+|script|snippet|"
                r"run\s+(this|the)?\s*code|"
                r"execute\s+(the\s*)?code|"
                r"output\s+of\s+the\s+code|"
                r"source\s*code|program"
                r")\b",
                re.IGNORECASE,
            ),
            "file_ref": re.compile(
                r"\b(attached\s+file|\.py\b|\.ipynb|code\s+snippet|source\s+file)\b",
                re.IGNORECASE,
            ),
            "exclude": re.compile(
                r"(audio|mp3|video|excel|xlsx|image|ioc\s+code|country\s+code|language\s+code|postal\s+code)",
                re.IGNORECASE,
            ),
        }

    def classify(self, question: str, file_path: str | None = None) -> ReasoningCategory | None:
        text = question.strip()

        if self.patterns["exclude"].search(text):
            self.logger.debug("Skipped CodeExecution due to exclusion pattern.")
            return None

        has_python_ext = False
        if file_path:
            _, ext = os.path.splitext(file_path.lower())
            has_python_ext = ext in self.PYTHON_EXTENSIONS
        
        if has_python_ext or self.patterns["code_terms"].search(text):
            return ReasoningCategory.CODE_EXECUTION
        
        return None

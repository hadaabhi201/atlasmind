import os
import re
from atlasmind.utils.logger import get_logger
from atlasmind.core.classification.base import (
    BaseClassifier, ReasoningCategory
)

class StructuredDataClassifier(BaseClassifier):
    EXCEL_EXTENSIONS = {".xls", ".xlsx", ".xlsm"}

    def __init__(self):
        self.logger = get_logger(__name__)
        self.patterns = {
            "spreadsheet": re.compile(r"\b(excel|xlsx|spreadsheet|csv|table|sheet)\b", re.IGNORECASE),
            "file_ref": re.compile(r"(attached|file|upload|data)", re.IGNORECASE),
            "exclude": re.compile(r"(audio|mp3|video|image|python|code|script)", re.IGNORECASE),
        }

    def classify(self, question: str, file_path: str | None = None) -> ReasoningCategory | None:
        text = question.strip()

        if self.patterns["exclude"].search(text):
            self.logger.debug("Skipped StructuredData due to exclusion pattern.")
            return None

        has_excel_ext = False
        if file_path:
            _, ext = os.path.splitext(file_path.lower())
            has_excel_ext = ext in self.EXCEL_EXTENSIONS
        
        if has_excel_ext or self.patterns["spreadsheet"].search(text):
            return ReasoningCategory.STRUCTURED_DATA

        return None

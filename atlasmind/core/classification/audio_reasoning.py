import os
import re
from atlasmind.utils.logger import get_logger
from atlasmind.core.classification.base import BaseClassifier, ReasoningCategory

class AudioReasoningClassifier(BaseClassifier):
    AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}

    def __init__(self):
        self.logger = get_logger(__name__)
        self.patterns = {
            "audio_ref": re.compile(r"(audio|recording|mp3|listen|voice\s?memo|transcribe|sound)", re.IGNORECASE),
            "exclude": re.compile(r"(video|image|excel|code|attached\s+excel|attached\s+image)", re.IGNORECASE),
        }

    def classify(self, question: str, file_path: str | None = None) -> ReasoningCategory | None:
        text = question.strip()

        if self.patterns["exclude"].search(text):
            self.logger.debug("Skipped AudioReasoning due to non-audio attachment.")
            return 
        
        has_audio_ext = False
        if file_path:
            _, ext = os.path.splitext(file_path.lower())
            has_audio_ext = ext in self.AUDIO_EXTENSIONS

        if has_audio_ext or self.patterns["audio_ref"].search(text):
            return ReasoningCategory.AUDIO_REASONING
        
        return None
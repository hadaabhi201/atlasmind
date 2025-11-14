import requests
from atlasmind.utils.config import code_execution_settings
from atlasmind.utils.logger import get_logger




class CodeRunner:
    """Executes Python code through Judge0 (RapidAPI)."""

    def __init__(self):
        super().__init__()
        self.url = code_execution_settings.RAPIDAPI_URL
        self.host = code_execution_settings.RAPIDAPI_HOST
        self.key = code_execution_settings.RAPIDAPI_KEY
        self.logger = get_logger(self.__class__.__name__)

    def run(self, code_path: str) -> dict:
        """Execute the given Python file and return the result JSON."""
        try:
            with open(code_path, "r", encoding="utf-8") as f:
                source_code = f.read()

            headers = {
                "Content-Type": "application/json",
                "X-RapidAPI-Key": self.key,
                "X-RapidAPI-Host": self.host,
            }

            payload = {
                "language_id": 71,  # Python 3
                "source_code": source_code,
                "stdin": "",
            }

            self.logger.info(f"[CodeRunner] Submitting {code_path} to Judge0")
            r = requests.post(
                f"{self.url}?base64_encoded=false&wait=true",
                json=payload,
                headers=headers,
                timeout=30,
            )
            r.raise_for_status()

            result = r.json()
            self.logger.info(f"[CodeRunner] Execution completed: {result.get('status', {}).get('description')}")
            return result

        except Exception as e:
            self.logger.error(f"[CodeRunner] Execution failed: {e}")
            raise RuntimeError(f"CodeRunner failed: {e}")

import os
from deepgram import DeepgramClient
from atlasmind.utils.logger import get_logger
from atlasmind.utils.config import settings
from atlasmind.core.planning.base_plan import PlanTemplate, ToolType
from atlasmind.core.tools.base_tool import BaseTool, ExecutionStatus, ToolResult


class AudioTool(BaseTool):
    """Tool for audio transcription using Deepgram API."""
    def __init__(self):
        # self.dg_client = DeepgramClient(api_key="e6304ccbd0c75181fe1ee3baa75bc6541fd35407")
        self.dg_client = DeepgramClient(api_key=settings.DEEPGRAM_API_KEY)
        self.logger = get_logger(__name__)
    
    def execute(self, plan: PlanTemplate) -> ToolResult:
        """
        Execute the audio transcription plan.
      
        """

        question = plan.question
        audio_path = plan.file_path

        if not audio_path or not os.path.exists(audio_path):
            msg = (
                "[CodeExecutorTool] Invalid file_path. "
                f"Provided: {audio_path}. "
                "A valid file must be supplied/"
            )
            self.logger.error(f"{msg} Question: {question}")
            raise RuntimeError(msg)

        try:
            # Read binary data
            with open(audio_path, "rb") as audio_file:
                response = self.dg_client.listen.v1.media.transcribe_file(
                    request=audio_file.read(),
                    model="nova-3",
                    smart_format=True
                )

            # Transcribe via Deepgram
            transcript = response.results.channels[0].alternatives[0].transcript # type: ignore

            self.logger.info(f"[AudioTool] Transcription successful for task tt_id")

            return ToolResult(
                question=question,
                tool=ToolType.AUDIO_TRANSCRIBER,
                message="Audio transcription completed successfully",
                data={"transcript": transcript},
                query=None,
                status=ExecutionStatus.SUCCESS,
            )

        except Exception as e:
            self.logger.error(f"[AudioTool] Failed for question {question}: {e}")
            raise RuntimeError(f"AudioTool execution failed: {e}")


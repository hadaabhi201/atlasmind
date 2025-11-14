import asyncio
from llama_index.core.workflow import Context
from atlasmind.utils.logger import get_logger
from atlasmind.workflow.task_workflow import TaskWorkflow


class BasicAgent:
    """Basic callable agent that runs the AtlasMind TaskWorkflow and returns its plan."""

    def __init__(self, timeout: int = 90, verbose: bool = True):
        self.logger = get_logger(__name__)
        self.timeout = timeout
        self.verbose = verbose
        self.workflow = TaskWorkflow(timeout=timeout, verbose=verbose)
        self.ctx = Context(self.workflow)

    async def _arun(self, question: str, file_path: str | None = None):
        """Internal async runner for TaskWorkflow."""
        task = {
            "question": question,
            "file_path": file_path
        }
        
        await self.ctx.store.set("task", task)

        self.logger.info(f"\n[Agent] Starting TaskWorkflow for question: {question}")
        result = await self.workflow.run(ctx=self.ctx)
       

        self.logger.info(f"[Agent] Workflow finished for {question} using tool: {result}")
        return result 

    def __call__(self, question: str, file_path: str | None = None):
        """Call agent like: agent(question, file_path)."""
        return asyncio.run(self._arun(question, file_path))

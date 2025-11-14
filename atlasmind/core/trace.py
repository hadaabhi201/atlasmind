from typing import List

class ReasoningTrace:
    """Tracks reasoning steps across all workflow nodes, preventing duplicate consecutive entries."""

    def __init__(self):
        self.steps: List[str] = []

    def log(self, step: str):
        """Record a new reasoning step, avoiding consecutive duplicates."""
        if not step or not step.strip():
            return

        step = step.strip()
        if not self.steps or self.steps[-1] != step:
            self.steps.append(step)

    def extend(self, steps: List[str]):
        """Add multiple steps at once."""
        for step in steps:
            self.log(step)

    def __len__(self) -> int:
        return len(self.steps)

    def __bool__(self) -> bool:
        return bool(self.steps)

    def __str__(self) -> str:
        if not self.steps:
            return "No reasoning trace recorded."
        formatted = "\n".join(f"{i+1}. {s}" for i, s in enumerate(self.steps))
        return f"The model reached its answer through the following reasoning steps:\n{formatted}"

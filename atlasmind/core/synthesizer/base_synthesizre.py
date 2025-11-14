from pydantic import BaseModel, Field


class SynthesisResult(BaseModel):
    """Represents the final synthesized output after reasoning and execution."""

    model_answer: str = Field(..., description="Final answer produced by the model or synthesis process.")
    reasoning_trace: str = Field(..., description="Explanation or reasoning trace that led to the final answer.")

    def to_dict(self) -> dict:
        """Convert to dictionary for downstream compatibility."""
        return self.model_dump()

    def __str__(self) -> str:
        return (
            f"model_answer={self.model_answer[:40]}..., "
            f"reasoning_trace={self.reasoning_trace[:40]}...)"
        )

"""
Prompt builder for multimodal image reasoning tasks using Gemini.
"""

from atlasmind.core.planning.base_plan import PlanTemplate


def build_image_reasoning_prompt(plan: PlanTemplate) -> str:
    """
    Build a structured prompt for Gemini image reasoning based on the PlanTemplate.

    Args:
        plan (PlanTemplate): Contains question, and plan_steps.

    Returns:
        str: The full reasoning prompt string for Gemini.
    """
    steps = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan.plan_steps))

    prompt = f"""
You are a vision reasoning model capable of analyzing complex images such as chess positions, diagrams, or photographs.

Follow these steps carefully to complete the task.

Task:
{plan.question}

Reasoning Steps:
{steps}

Return your reasoning as a detailed explanation and provide the final structured output in JSON format:
{{
  "analysis_steps": [
    "summary of step 1",
    "summary of step 2",
    "summary of step 3"
  ],
  "final_answer": "<concise final answer>"
}}
"""
    return prompt.strip()

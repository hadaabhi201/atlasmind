"""
Prompt builder for semantic reasoning and categorization tasks.
"""

from atlasmind.core.planning.base_plan import PlanTemplate


def build_semantic_reasoning_prompt(plan: PlanTemplate) -> str:
    """
    Build a structured text reasoning prompt for semantic classification or categorization.

    Args:
        plan (PlanTemplate): Plan with question and step instructions.

    Returns:
        str: Full formatted prompt string for the LLM.
    """
    steps = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan.plan_steps))

    prompt = f"""
You are a precise and detail-oriented language reasoning model.

Task:
{plan.question}

Follow these reasoning steps exactly:
{steps}

Guidelines:
- Follow the listed steps faithfully without adding assumptions.
- Rely only on explicit information from the plan and task.
- Keep the output concise and limited to the requested format.
- When producing lists or structured data, preserve clarity and ordering.

Output Format:
Return only the final result that satisfies the plan.
"""
    return prompt.strip()
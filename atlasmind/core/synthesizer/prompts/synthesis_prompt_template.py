"""
Prompt template and builder for generating synthesized answers from ToolResult data.
"""

from typing import Any

from atlasmind.core.planning.base_plan import ToolType


SYNTHESIS_PROMPT_TEMPLATE = """
You are an intelligent synthesis model.

Your goal is to provide a clear and concise final answer to the user's question
based on the information provided by a specific tool's output.

---

User Question:
{question}

Tool Used:
{tool}

Query Executed:
{query}

Tool Message:
{message}

Tool Data (summary, structured results, or output):
{data}

---

Instructions:
1. Read the question carefully and understand what the user is asking.
2. Use the tool data to construct a factually correct and direct answer.
3. Avoid repeating the tool’s metadata or process details.
4. Keep the response natural, helpful, and to the point.
5. If a number or count is required, reason through the text to extract it.
6. Limit your answer to 2–4 sentences maximum.

Now, generate the final synthesized answer below:
"""


def build_synthesis_prompt(
    question: str,
    tool: ToolType,
    query: str,
    message: str,
    data: Any,
) -> str:
    """
    Build the synthesis prompt dynamically for the LLM synthesizer.

    Args:
        question (str): The original user question.
        tool (str): Name of the tool used to fetch or process the data.
        query (str): The internal query executed by the tool.
        message (str): Summary or status message from the tool.
        data (Any): Structured or textual data returned by the tool.

    Returns:
        str: Fully formatted prompt string for the LLM.
    """
    # Ensure that None values don't break the format
    safe_question = question or "N/A"
    safe_tool = tool or "Unknown Tool"
    safe_query = query or "N/A"
    safe_message = message or "No message available."
    safe_data = str(data) if data is not None else "No data returned."

    return SYNTHESIS_PROMPT_TEMPLATE.format(
        question=safe_question.strip(),
        tool=safe_tool,
        query=safe_query.strip(),
        message=safe_message.strip(),
        data=safe_data.strip(),
    )

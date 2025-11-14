def build_prompt(*, query: str, question: str, summaries: str) -> str:
    return f"""
You are a synthesis assistant.

User Query: {query}
User Question: {question}

Combine the partial summaries below into a single coherent summary.
Preserve key facts, remove duplicates, and keep it focused on the question.
Target about 400 tokens.

Partial Summaries:
{summaries}
"""

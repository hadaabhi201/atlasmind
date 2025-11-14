def build_prompt(*, query: str, question: str, text: str) -> str:
    return f"""
You are a concise summarizer.

User Query: {query}
User Question: {question}

Summarize the following content accurately. Keep information relevant to the query and question.
Be precise and avoid redundancy. Target about 300 tokens.

Content:
{text}
"""

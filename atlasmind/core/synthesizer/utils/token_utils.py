import tiktoken

def count_tokens(text: str, model="gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def split_into_chunks(text: str, max_tokens: int):
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = enc.encode(text)
    for i in range(0, len(tokens), max_tokens):
        yield enc.decode(tokens[i:i + max_tokens])

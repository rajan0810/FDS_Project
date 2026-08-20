def count_tokens_fallback(text: str) -> int:
    """
    A simple fallback token counter using whitespace splitting.
    For more accuracy, a proper tokenizer (like tiktoken) should be used.
    """
    return int(len(text.split()) * 1.3)

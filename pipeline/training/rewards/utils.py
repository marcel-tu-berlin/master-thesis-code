
def extract_content(completion) -> str:
    # TRL may pass either a chat-message list (`[{"role": ..., "content": ...}]`)
    # or, for some configurations, a raw string. Handle the string form
    # explicitly so the silent `""` fallback only fires on truly malformed
    # input rather than swallowing legitimate plain-string completions.
    if isinstance(completion, str):
        return completion
    try:
        return completion[0]["content"]
    except (IndexError, KeyError, TypeError):
        return ""

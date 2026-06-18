import json


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


def model_token_count(completion, tokenizer) -> int:
    """Count the model's own (assistant) tokens in a completion.

    TRL hands reward functions the full multi-turn stream, with the game/tool
    feedback it injects between turns interleaved as `role == "tool"` messages.
    The efficiency reward must measure the model's output, not that feedback, so:

    - a string completion (single-turn plain text) is encoded whole;
    - a message-list completion sums only `role == "assistant"` content, plus the
      serialized arguments of any tool calls the model emitted, and skips tool
      messages;
    - a lone message dict encodes its content.

    Re-encoding assistant content is a documented approximation of the exact
    generated-token count (chat-template framing differs slightly); it counts the
    model's actual content tokens, which is the quantity of interest.
    """
    def _enc(text) -> int:
        if not text:
            return 0
        return len(tokenizer.encode(text, add_special_tokens=False))

    if isinstance(completion, str):
        return _enc(completion)

    if isinstance(completion, dict):
        return _enc(completion.get("content") or "")

    if isinstance(completion, list):
        total = 0
        for msg in completion:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            total += _enc(msg.get("content") or "")
            for call in (msg.get("tool_calls") or []):
                fn = (call or {}).get("function") or {}
                args = fn.get("arguments")
                if args is None:
                    continue
                total += _enc(args if isinstance(args, str) else json.dumps(args))
        return total

    return 0

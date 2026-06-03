
def _require_answers(answer, n: int, reward_name: str) -> list[str]:
    """Validate the ground-truth `answer` column TRL passes as a keyword arg.

    Shared by every reward that needs ground truth (AnswerReward, NumericReward,
    CosineLengthReward) so the contract and error message stay in one place.
    """
    if answer is None:
        raise ValueError(
            f"{reward_name} requires the dataset to expose an 'answer' column. "
            "TRL passes dataset columns as keyword args to reward functions; "
            "ensure the loader's map() output includes 'answer'."
        )
    if len(answer) != n:
        raise ValueError(
            f"{reward_name}: len(answer)={len(answer)} does not match len(completions)={n}"
        )
    return answer


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

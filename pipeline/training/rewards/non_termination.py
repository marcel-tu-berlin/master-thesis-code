class NonTerminationPenalty:
    """Penalty for an episode that ran out of budget without finishing (E3).

    Returns -1.0 for an episode the environment never reported `done` for AND
    that ended because a budget cut it off: the turn cap (the trajectory ends
    after a tool result, or after a turn whose calls were never dispatched) or
    the completion budget (the trajectory filled `max_completion_tokens`).
    Returns 0.0 otherwise. Composed with `naive_sum` at weight lambda this is
    the shaped reward of the thesis:

        R_lambda = R_task - lambda * C_target,   C_target = 1[ran out of budget]

    Why "ran out of budget" and not "not done". An episode can also end because
    the model stops emitting tool calls before the env is done - premature
    stopping. The thesis predicts that a non-termination penalty invites exactly
    that substitute (the cheapest way to never run out of steps is to stop
    early), and the off-target panel measures it. A cost that penalized every
    not-done episode penalized the substitute too, so the predicted substitution
    could not occur by construction. The cost is therefore the budget exhaustion
    the reward targets, and stopping early is left to the task reward alone
    (it earns 0 there). The eval loop draws the same three-way line
    (env_done | no_tool_call | max_turns, hit_generation_cap), see
    eval.agentic_eval._run_multiturn_episodes.

    The sign lives in the component, not in the weight, on purpose: a config
    author writing `weight: 4.0` is expressing lambda = 4, and a component that
    returned +1 for the bad case would silently turn that into a *reward* for
    running out of budget. Weights stay positive across every reward in the
    registry.

    Reads two things TRL passes every reward function on the environment path:
    `environments` (the live adapter instances, `done` set by whichever action
    makes the env report done) and `completion_ids` (the trajectory's token
    ids, which TRL caps at max_completion_length, tool results included). The
    completion itself is the multi-turn message stream: it ends with an
    assistant message carrying no tool call exactly when the model stopped on
    its own.

    Caveat, and the reason the E3 sweep runs under `naive_sum`: under
    `advantage_weighted` each component is z-scored per prompt-group, and a
    component with no within-group variance contributes exactly 0. Once a policy
    stays inside its budget reliably, most groups are all-zero, so the penalty
    would go silent precisely where behavior is already good and speak only in
    mixed groups. That is defensible GRPO behavior but it is not the thesis's
    formula. Under naive_sum the weight is a dose only if the trainer does not
    rescale advantages by the group std: in a group with constant task reward
    the std of R_lambda is lambda * std(C) and lambda cancels (DIET, App. B).
    `training.scale_rewards: none | batch` keeps it a dose.
    """

    def __init__(self, max_completion_tokens: int) -> None:
        if int(max_completion_tokens) < 1:
            raise ValueError(
                f"max_completion_tokens must be positive, got {max_completion_tokens}"
            )
        self.max_completion_tokens = int(max_completion_tokens)

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        environments = kwargs.get("environments")
        completion_ids = kwargs.get("completion_ids")
        if environments is None or completion_ids is None:
            raise ValueError(
                "NonTerminationPenalty requires kwargs['environments'] and "
                "kwargs['completion_ids'] from the agentic environment_factory "
                "path (TRL GRPOTrainer)."
            )
        if len(environments) != len(completions) or len(completion_ids) != len(completions):
            raise ValueError(
                f"environments {len(environments)} / completion_ids {len(completion_ids)} "
                f"!= completions {len(completions)}"
            )
        out = []
        for env, completion, ids in zip(environments, completions, completion_ids):
            if getattr(env, "done", False):
                out.append(0.0)
                continue
            if len(ids) >= self.max_completion_tokens:
                out.append(-1.0)            # filled the completion budget
                continue
            out.append(0.0 if _stopped_on_its_own(completion) else -1.0)
        return out


def _stopped_on_its_own(completion) -> bool:
    """True when the trajectory ends with an assistant turn that requested no
    tool call: the model chose to stop. A trajectory ending after a tool result,
    or after a turn whose calls were never answered, was cut by the turn cap.
    A plain-string completion is the dataset path, where E3 is undefined."""
    if not isinstance(completion, list):
        raise ValueError(
            "NonTerminationPenalty expects the multi-turn message list TRL passes on "
            f"the environment path, got {type(completion).__name__}"
        )
    if not completion:
        return False
    last = completion[-1]
    return (isinstance(last, dict) and last.get("role") == "assistant"
            and not last.get("tool_calls"))

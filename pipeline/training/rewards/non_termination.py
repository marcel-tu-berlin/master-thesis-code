class NonTerminationPenalty:
    """Penalty for an episode that never finished the task (reward condition E3).

    Returns -1.0 for an episode whose environment never reported `done`, and 0.0
    otherwise. Composed with `naive_sum` at weight lambda this is exactly the
    shaped reward of the expose:

        R_lambda = R_task - lambda * C_target,   C_target = 1[not terminated]

    The sign lives in the component, not in the weight, on purpose: a config
    author writing `weight: 4.0` is expressing lambda = 4, and a component that
    returned +1 for the bad case would silently turn that into a *reward* for
    not finishing. Weights stay positive across every reward in the registry.

    `done` is the adapter's termination flag - set by the terminal tool
    (reasoning_gym `answer`, finqa `submit_answer`, repl's FINAL, textarena's
    solving move). An episode that ran out of turns, stopped emitting tool calls,
    or filled the token budget leaves it False.

    Caveat, and the reason the E3 sweep runs under `naive_sum`: under
    `advantage_weighted` each component is z-scored per prompt-group, and a
    component with no within-group variance contributes exactly 0. Once a policy
    terminates reliably, most groups are all-terminated, so the penalty would go
    silent precisely where behavior is already good and speak only in mixed
    groups. That is defensible GRPO behavior but it is not the expose's formula.
    """

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        environments = kwargs.get("environments")
        if environments is None:
            raise ValueError(
                "NonTerminationPenalty requires kwargs['environments'] from the "
                "agentic environment_factory path (TRL GRPOTrainer)."
            )
        if len(environments) != len(completions):
            raise ValueError(
                f"environments length {len(environments)} != completions {len(completions)}"
            )
        return [0.0 if getattr(env, "done", False) else -1.0 for env in environments]

class EnvReward:
    """Task-success reward computed by the OpenEnv environment.

    Under TRL's environment_factory path the trainer passes the live env
    instances to every reward function as kwargs['environments']. Each env has
    stored its episode reward on `env.reward` (set by the adapter's answer tool
    when the model submitted its answer). This component surfaces that
    per-completion reward to the composer so it is z-scored and weighted exactly
    like any code-computed reward.
    """

    def __call__(self, prompts, completions, **kwargs):
        environments = kwargs.get("environments")
        if environments is None:
            raise ValueError(
                "EnvReward requires kwargs['environments'] from the agentic "
                "environment_factory path (TRL GRPOTrainer)."
            )
        if len(environments) != len(completions):
            raise ValueError(
                f"environments length {len(environments)} != completions {len(completions)}"
            )
        return [float(env.reward) for env in environments]

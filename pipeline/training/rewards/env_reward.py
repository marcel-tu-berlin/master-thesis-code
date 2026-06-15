class EnvReward:
    """Task-success reward computed by the OpenEnv environment.

    In agentic mode the rollout_func attaches the per-completion environment
    reward as kwargs['env_reward']; this component surfaces it to the composer
    so it is z-scored and weighted exactly like any code-computed reward.
    """

    def __call__(self, prompts, completions, env_reward=None, **kwargs):
        if env_reward is None:
            raise ValueError(
                "EnvReward requires kwargs['env_reward'] from the agentic rollout_func"
            )
        if len(env_reward) != len(completions):
            raise ValueError(
                f"env_reward length {len(env_reward)} != completions {len(completions)}"
            )
        return [float(x) for x in env_reward]

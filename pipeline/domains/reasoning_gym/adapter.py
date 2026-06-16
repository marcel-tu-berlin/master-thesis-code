class ReasoningGymEnvAdapter:
    """TRL environment_factory adapter for the OpenEnv reasoning_gym env.

    TRL's GRPOTrainer creates one instance per rollout slot, calls reset(**row)
    (whose return becomes the prompt), exposes every *other* public method as a
    tool the model may call, and reads the task reward off `self.reward`. The
    public surface is therefore deliberately exactly {reset, answer}: any extra
    public method would be registered as a tool and shown to the model.

    The adapter wraps a *synchronous* OpenEnv client (EnvClient.sync()). The
    client and action class are injectable so the unit tests exercise the
    adapter logic without the OpenEnv install or a running server. reasoning_gym
    is single-step: one answer ends the episode.
    """

    def __init__(self, base_url, env_config=None, client=None, action_cls=None):
        self._env_config = dict(env_config or {})
        self._client = client if client is not None else self._connect(base_url)
        self._action_cls = action_cls
        self.reward = 0.0

    @staticmethod
    def _connect(base_url):
        # Lazy: the reasoning_gym client lives in the cloned OpenEnv repo's
        # envs/ package (not on PyPI), and EnvClient is async by default - the
        # .sync() wrapper is required because TRL rejects async tools.
        from reasoning_gym_env import ReasoningGymEnv

        return ReasoningGymEnv(base_url).sync()

    def _action(self, answer):
        if self._action_cls is None:
            from reasoning_gym_env import ReasoningGymAction

            self._action_cls = ReasoningGymAction
        return self._action_cls(answer=answer)

    def reset(self, *, seed=None, prompt=None, **_row):
        """Reset to a fresh reasoning_gym question; return the question text.

        TRL appends the returned string to the prompt's last user message, then
        applies the tool-calling chat template. `prompt` and any other dataset
        columns arrive here as kwargs and are ignored - only `seed` selects the
        question (deterministic), with dataset name/config from env_config. size
        is pinned to 1: one question per episode for this single-step env.
        """
        reset_kwargs = {"size": 1}
        name = self._env_config.get("dataset_name") or self._env_config.get("dataset")
        if name is not None:
            reset_kwargs["dataset_name"] = name
        dataset_config = self._env_config.get("dataset_config")
        if dataset_config is not None:
            reset_kwargs["dataset_config"] = dataset_config
        if seed is not None:
            reset_kwargs["seed"] = seed
        result = self._client.reset(**reset_kwargs)
        self.reward = 0.0
        return result.observation.question

    def answer(self, answer: str) -> str:
        """Tool: submit the final answer. Scores via the env, stores the reward."""
        result = self._client.step(self._action(answer))
        self.reward = float(result.observation.score)
        return f"Recorded answer: {answer}"

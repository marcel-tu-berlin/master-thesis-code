class TextArenaEnvAdapter:
    """TRL environment_factory adapter for the OpenEnv textarena env (multi-turn).

    TRL's GRPOTrainer creates one instance per rollout slot, calls reset(**row)
    (whose return becomes the prompt), exposes every *other* public method as a
    tool the model may call, and reads the task reward off `self.reward`. The
    public surface is therefore exactly {reset, move}: any extra public method
    would be registered as a tool. `reward` and `done` are public *attributes*
    (a float and a bool, not methods), so TRL does not turn them into tools; the
    eval loop reads `done` to end an episode.

    Multi-turn by construction: the model calls move() repeatedly under env
    feedback until the game ends (done) or the iteration cap. The env reward is
    terminal, so self.reward after the final step is the episode outcome that
    EnvReward reads via [e.reward for e in environments].

    The adapter wraps a *synchronous* OpenEnv client (EnvClient.sync()). The
    client and action class are injectable so unit tests exercise the adapter
    logic without the OpenEnv install or a running server.
    """

    def __init__(self, base_url, env_config=None, client=None, action_cls=None):
        self._env_config = dict(env_config or {})
        self._client = client if client is not None else self._connect(base_url)
        self._action_cls = action_cls
        self.reward = 0.0
        self.done = False

    @staticmethod
    def _connect(base_url):
        # Lazy: the textarena client lives in the cloned OpenEnv repo's envs/
        # package (not on PyPI). EnvClient is async by default; .sync() is
        # required because TRL rejects async tools.
        from textarena_env import TextArenaEnv

        return TextArenaEnv(base_url).sync()

    def _action(self, message):
        if self._action_cls is None:
            from textarena_env import TextArenaAction

            self._action_cls = TextArenaAction
        return self._action_cls(message=message)

    def reset(self, *, seed=None, **_row):
        """Reset to a fresh game instance; return the opening prompt text.

        TRL appends the returned string to the prompt's last user message, then
        applies the tool-calling chat template. Only `seed` selects the instance
        (deterministic); other dataset columns arrive as kwargs and are ignored.
        """
        result = self._client.reset(seed=seed) if seed is not None else self._client.reset()
        self.reward = 0.0
        self.done = False
        return self._prompt_text(result.observation)

    def move(self, message: str) -> str:
        """Make one move in the current text game and read the result.

        TRL renders this method into the tool spec the model sees, so this
        summary and the Args description below are the tool's model-facing
        documentation. transformers' get_json_schema requires a Google-style
        Args entry for every parameter.

        Args:
            message: The move to play, in the game's expected format (for Wordle,
                a single five-letter word guess).
        """
        if self.done:
            # Done-guard: once the game is over, further moves change nothing, so
            # a model that keeps calling move cannot corrupt the terminal reward.
            return "The game is already over; no further moves are accepted."
        result = self._client.step(self._action(message))
        obs = result.observation
        self.reward = float(getattr(obs, "reward", 0.0) or 0.0)
        self.done = bool(getattr(obs, "done", False))
        return self._feedback_text(obs)

    @staticmethod
    def _prompt_text(obs) -> str:
        return str(getattr(obs, "prompt", "") or "")

    @staticmethod
    def _feedback_text(obs) -> str:
        # The game's feedback for the move just played, for the model to read on
        # its next turn. Verified against the live env: the textarena observation
        # carries `messages`, a list of TextArenaMessage (pydantic) objects whose
        # `.content` holds the text. Read that attribute; fall back to a dict's
        # ["content"] or the running prompt.
        msgs = getattr(obs, "messages", None)
        if msgs:
            last = msgs[-1]
            content = getattr(last, "content", None)        # TextArenaMessage.content
            if content is None and isinstance(last, dict):
                content = last.get("content")
            if content is not None:
                return str(content)
            return str(last)
        return str(getattr(obs, "prompt", "") or "")

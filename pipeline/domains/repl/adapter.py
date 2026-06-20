class REPLEnvAdapter:
    """TRL environment_factory adapter for the OpenEnv repl env (multi-turn).

    TRL's GRPOTrainer creates one instance per rollout slot, calls reset(**row)
    (whose return becomes the prompt), exposes every *other* public method as a
    tool the model may call, and reads the task reward off `self.reward`. The
    public surface is exactly {reset, execute}: one tool, `execute(code)`, since
    repl is a Python REPL the model drives until it prints its final answer.

    repl is a bare execution sandbox with no task source of its own, so unlike
    reasoning_gym the TASK is supplied by us: reset(**row) forwards `context`,
    `task_prompt`, and `expected_answer` (the ground truth) from the dataset row
    into the server, which sets them on its exact-match rubric. The reward is
    binary (1.0 if the model's FINAL(answer) matches expected_answer, else 0.0;
    a small negative on a no-answer timeout), populated as the episode ends.

    Multi-turn: the model calls execute() repeatedly under stdout feedback until
    it prints FINAL(answer) (or the iteration cap), so the adapter keeps
    `self.done` like textarena and refreshes `self.reward` every step (non-final
    steps carry process reward, the final step carries the outcome).

    The client is injectable so the unit tests exercise the adapter logic without
    the OpenEnv install or a running server.
    """

    def __init__(self, base_url, env_config=None, client=None):
        self._env_config = dict(env_config or {})
        self._client = client if client is not None else self._connect(base_url)
        self.reward = 0.0
        self.done = False

    @staticmethod
    def _connect(base_url):
        # Lazy: the repl client lives in the cloned OpenEnv repo's envs/ package
        # (not on PyPI). REPLEnv is an EnvClient; .sync() wraps it for TRL (which
        # rejects async tools), matching the textarena/reasoning_gym pattern.
        from repl_env import REPLEnv

        return REPLEnv(base_url).sync()

    def reset(self, *, seed=None, prompt=None, **_row):
        """Derive this seed's task, send it to the server, return the task text.

        TRL appends the returned string to the prompt's last user message, then
        applies the tool-calling chat template. Only `seed` matters: the task is
        a pure function of the seed (make_task), so training and eval both call
        reset(seed=N) and get the identical task - no extra dataset columns, and
        the server-side exact-match rubric is armed with the seed's
        expected_answer. `prompt` and other columns arrive as kwargs and are
        ignored.
        """
        from domains.repl.tasks import make_task

        context, task_prompt, expected_answer = make_task(seed if seed is not None else 0)
        self._client.reset(
            context=context, task_prompt=task_prompt, expected_answer=expected_answer,
            seed=seed,
        )
        self.reward = 0.0
        self.done = False
        return task_prompt

    def execute(self, code: str) -> str:
        """Run Python in the REPL and read stdout. Finish by printing FINAL(answer).

        State persists across calls (variables stay defined). To submit your
        final answer, print it as FINAL(<answer>) - e.g. print("FINAL(42)") -
        which ends the episode.

        TRL renders this method into the tool spec the model sees, so this
        summary and the Args description below are the tool's model-facing
        documentation. transformers' get_json_schema requires a Google-style Args
        entry for every parameter.

        Args:
            code: A snippet of Python to execute in the persistent REPL.
        """
        if self.done:
            # Done-guard: once a final answer is recorded, further code cannot
            # corrupt the terminal reward.
            return "The task is already answered; no further code is accepted."
        # step(REPLAction) is the canonical path (matches textarena/finqa); the
        # client's execute() convenience wraps the same call.
        from repl_env import REPLAction

        result = self._client.step(REPLAction(code=code))
        self.reward = float(getattr(result, "reward", 0.0) or 0.0)
        self.done = bool(getattr(result, "done", False))
        return self._feedback_text(result)

    @staticmethod
    def _feedback_text(result) -> str:
        # Return the REPL's stdout for the model's next turn, plus the error if
        # the snippet raised. The observation carries a CodeBlockResult (stdout /
        # stderr / exception); fall back gracefully if shapes differ.
        obs = getattr(result, "observation", None)
        block = getattr(obs, "result", None)
        if block is None:
            return str(getattr(result, "observation", "") or "")
        out = str(getattr(block, "stdout", "") or "")
        exc = getattr(block, "exception", None)
        err = str(getattr(block, "stderr", "") or "")
        parts = [p for p in (out, exc or err) if p]
        return "\n".join(parts) if parts else "(no output)"

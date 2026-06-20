from domains.env_base import EnvDomain
from domains.repl.adapter import REPLEnvAdapter

# Brief task framing prepended to each prompt's user message. The tool spec is
# injected by the model's native tool-calling template (tools=...); this nudges
# the model to use the REPL and end with FINAL(...). reset() appends the specific
# task after this lead-in (TRL appends to the last user message).
_LEAD_IN = (
    "You solve the task by writing and running Python with the execute tool. "
    "Variables persist across calls. When you have the answer, print it as "
    "FINAL(<answer>) - for example print(\"FINAL(42)\") - to finish.\n\n"
)

class REPLDomain(EnvDomain):
    """Fourth agentic domain: a Python REPL served through OpenEnv.

    Multi-turn code execution: the model writes Python, reads stdout, iterates,
    and prints FINAL(answer) to finish. The efficiency reward counts the model's
    assistant tokens across all turns; the env reward is binary exact-match on
    the final answer.

    repl is a bare sandbox with no built-in tasks, so this domain owns the task
    distribution: build_seed_dataset mints a deterministic (context, task_prompt,
    expected_answer) per seed, and the adapter threads expected_answer into the
    server's rubric at reset. The smoke task family is arithmetic-over-a-list;
    it is intentionally simple - the point is a real automatic reward and real
    length slack, not task difficulty.
    """

    # `python -m <server_module>` launches the OpenEnv env server (no Docker).
    server_module = "repl_env.server.app"
    multi_turn = True

    def make_env_factory(self, base_url, env_config=None, client_factory=None):
        env_config = dict(env_config or {})
        if client_factory is None:
            return lambda: REPLEnvAdapter(base_url, env_config)
        # Test/injection path: build the adapter around a supplied client.
        return lambda: REPLEnvAdapter(base_url, env_config, client=client_factory())

    def build_seed_dataset(self, env_config=None, n=500, seed_base=0):
        # Each row is one training prompt: the lead-in plus a distinct seed. The
        # adapter derives (and arms) the seed's task at reset. TRL repeats each
        # row num_generations times to form a GRPO group; the task is a pure
        # function of `seed`, so every rollout in a group solves the same task.
        from datasets import Dataset

        rows = [
            {"prompt": [{"role": "user", "content": _LEAD_IN}], "seed": seed_base + i}
            for i in range(int(n))
        ]
        return Dataset.from_list(rows)

    def episode_messages(self, task_prompt):
        """Eval-time prompt for one episode. Mirrors the training framing: the
        same lead-in the seed rows carry, with the task text appended (in
        training, TRL appends the task to the row's lead-in via reset)."""
        return [{"role": "user", "content": _LEAD_IN + task_prompt}]

    def eval_tools(self, env):
        """Single tool for the REPL: run code."""
        return [env.execute]

    def server_env(self, env_config=None):
        # The repl server reads its iteration cap from a process env var.
        cfg = dict(env_config or {})
        env = {}
        if cfg.get("max_iterations") is not None:
            env["REPL_MAX_ITERATIONS"] = str(int(cfg["max_iterations"]))
        return env

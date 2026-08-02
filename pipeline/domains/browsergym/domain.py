from domains.env_base import EnvDomain
from domains.browsergym.adapter import BrowserGymEnvAdapter

# Brief task framing prepended to each prompt's user message. The tool spec is
# injected by the model's native tool-calling template (tools=...); this only
# frames the setting. reset() appends the goal and the page after this lead-in
# (TRL appends to the last user message).
#
# It explains the interface and stops there. An earlier version ended "Most tasks
# need a selection followed by clicking Submit", which states the solution to
# both original training families outright. Removing it is the fair measurement,
# not a fix: on paired seeds it changed nothing (click-option 1.00 -> 1.00,
# click-checkboxes 0.95 -> 1.00). What actually inflated base accuracy was the
# adapter exposing two tools where the first probe used five. See
# runs/browsergym_difficulty_correction.md.
_LEAD_IN = (
    "You are controlling a web page. Read the goal and the accessibility tree, "
    "then call one tool per turn to act on the page. Element ids are the numbers "
    "in square brackets.\n\n"
)


class BrowserGymDomain(EnvDomain):
    """MiniWoB web tasks served through OpenEnv's browsergym env.

    Selected over finqa by the feasibility probe
    (`pipeline/runs/browsergym_feasibility_findings.md`). The two properties that
    decided it, neither of which finqa had:

    1. Difficulty is selectable. MiniWoB spans saturated (click-test 1.00) to
       unreachable (click-tab-2 0.00) for Qwen3-1.7B, so a task set can be chosen
       inside the 40-80% band where GRPO gets within-group variance and the E1
       baseline has headroom. finqa was fixed at 1% with no dial.
    2. Task performance and the off-target axis can be decoupled. The default set
       mixes click-option (0.50 success, headroom) with click-checkboxes (0.35
       success against 0.50 termination, so the axes separate). On a set where
       every terminated episode is correct, a non-termination penalty would be
       indistinguishable from a task reward and RQ2 would have nothing to measure.

    Multi-turn: the model clicks its way through a page across several turns, so
    the efficiency reward counts assistant tokens across all turns, excluding the
    page observations the env injects between them. The dominant base-model
    failure is `no_tool_call` - stopping without acting - which is premature
    stopping measured directly rather than inferred.
    """

    # `python -m <server_module>` launches the OpenEnv env server (no Docker).
    server_module = "browsergym_env.server.app"
    multi_turn = True

    def make_env_factory(self, base_url, env_config=None, client_factory=None):
        env_config = dict(env_config or {})
        if client_factory is None:
            return lambda: BrowserGymEnvAdapter(base_url, env_config)
        # Test/injection path: build the adapter around a supplied client.
        return lambda: BrowserGymEnvAdapter(base_url, env_config, client=client_factory())

    def build_seed_dataset(self, env_config=None, n=500, seed_base=0):
        # Each row is one training prompt: a fixed lead-in plus a distinct seed.
        # TRL repeats each row num_generations times to form a GRPO group, and
        # reset(seed=...) turns the seed into a deterministic (task, page) pair -
        # so every rollout in a group sees the same page, and the mixed task set
        # is interleaved evenly by `tasks[seed % len(tasks)]`.
        from datasets import Dataset

        rows = [
            {"prompt": [{"role": "user", "content": _LEAD_IN}], "seed": seed_base + i}
            for i in range(int(n))
        ]
        return Dataset.from_list(rows)

    def episode_messages(self, observation):
        """Eval-time prompt for one episode. Mirrors the training framing: the
        same lead-in the seed rows carry, with the env's goal + page appended (in
        training, TRL appends reset's return to the row's lead-in)."""
        return [{"role": "user", "content": _LEAD_IN + str(observation)}]

    def eval_tools(self, env):
        """The adapter's two tools. `noop` is exposed so stalling is a countable
        tool call rather than an absence to infer."""
        return [env.click, env.noop]

    def server_env(self, env_config=None):
        # The browsergym server is configured entirely by process env vars.
        # BROWSERGYM_TASK_NAME is only the server's default: the adapter passes
        # task_name per reset, which is how one server serves the mixed set.
        #
        # No turn-cap var here on purpose. browsergym has no server-side step cap,
        # so `max_turns` is enforced client-side only - by TRL's
        # max_tool_calling_iterations in training and by the eval loop's turn
        # budget. Both read the same `max_turns` key, so the two agree; there is
        # simply no third place for them to disagree with.
        # The server picks its port from BROWSERGYM_PORT and ignores the --port
        # argv EnvServerProcess passes, so training.env_server.port must stay at
        # the server's own default of 8000. A mismatch is loud (nothing answers
        # the client), not silent.
        cfg = dict(env_config or {})
        tasks = cfg.get("tasks") or ["click-option", "click-checkboxes"]
        env = {
            "BROWSERGYM_BENCHMARK": str(cfg.get("benchmark", "miniwob")),
            "BROWSERGYM_TASK_NAME": str(tasks[0]),
            "BROWSERGYM_HEADLESS": "true",
        }
        # MiniWoB's HTML is not shipped with browsergym-miniwob; it is served from
        # a clone of miniwob-plusplus. Without this the env raises at the first
        # reset with "core is not defined".
        if cfg.get("miniwob_url"):
            env["MINIWOB_URL"] = str(cfg["miniwob_url"])
        return env

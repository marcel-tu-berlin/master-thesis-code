from domains.env_base import EnvDomain
from domains.finqa.adapter import FinQAEnvAdapter

# Brief task framing prepended to each prompt's user message. The tool spec is
# injected by the model's native tool-calling template (tools=...); this only
# nudges the model through the finqa workflow. reset() appends the question +
# company after this lead-in (TRL appends to the last user message).
_LEAD_IN = (
    "You are answering a financial question about a company's SEC 10-K filing. "
    "Work with the data tools: call get_descriptions to list the tables, "
    "get_table_info to inspect a table's columns, and sql_query to read values. "
    "When you have the number, call submit_answer with it.\n\n"
)


class FinQADomain(EnvDomain):
    """Third agentic domain: FinQA financial QA served through OpenEnv (MCP).

    Multi-turn (the model explores tables across several tool calls before
    submitting), so the efficiency reward counts the model's assistant tokens
    across all turns, excluding the tool results the env injects between them.
    The adapter exposes four tools; the env reward is binary (1 correct / 0
    wrong), set only after submit_answer.

    The finqa server selects questions non-deterministically as shipped (an
    unseeded shuffle), which breaks both the EnvDomain seed->question contract
    and GRPO's same-question-per-group invariant on a shared server. We carry a
    small server patch (see decisions.md / openenv_patches) making reset(seed=N)
    select questions[N % len] deterministically; this domain assumes it applied.
    """

    # `python -m <server_module>` launches the OpenEnv env server (no Docker).
    server_module = "finqa_env.server.app"
    multi_turn = True

    def make_env_factory(self, base_url, env_config=None, client_factory=None):
        env_config = dict(env_config or {})
        if client_factory is None:
            return lambda: FinQAEnvAdapter(base_url, env_config)
        # Test/injection path: build the adapter around a supplied client.
        return lambda: FinQAEnvAdapter(base_url, env_config, client=client_factory())

    def build_seed_dataset(self, env_config=None, n=500, seed_base=0):
        # Each row is one training prompt: a fixed lead-in plus a distinct seed.
        # TRL repeats each row num_generations times to form a GRPO group, and
        # reset(seed=...) turns the seed into a deterministic question (one shared
        # question per group, given the server seed-patch).
        from datasets import Dataset

        rows = [
            {"prompt": [{"role": "user", "content": _LEAD_IN}], "seed": seed_base + i}
            for i in range(int(n))
        ]
        return Dataset.from_list(rows)

    def episode_messages(self, question):
        """Eval-time prompt for one episode. Mirrors the training framing: the
        same lead-in the seed rows carry, with the env question text appended (in
        training, TRL appends the question to the row's lead-in via reset)."""
        return [{"role": "user", "content": _LEAD_IN + question}]

    def eval_tools(self, env):
        """The four finqa tools, in workflow order."""
        return [env.get_descriptions, env.get_table_info, env.sql_query, env.submit_answer]

    def server_env(self, env_config=None):
        # The finqa server reads its data dir and step cap from process env vars.
        # FINQA_DATA_PATH default is a Docker path; our no-Docker launch must
        # point it at the dataset downloaded into the OpenEnv clone.
        cfg = dict(env_config or {})
        env = {
            "FINQA_DATA_PATH": str(cfg.get("data_path", "/workspace/OpenEnv/envs/finqa_env/data")),
            "FINQA_TASK": "finqa",
        }
        if cfg.get("max_steps") is not None:
            env["FINQA_MAX_STEPS"] = str(int(cfg["max_steps"]))
        return env

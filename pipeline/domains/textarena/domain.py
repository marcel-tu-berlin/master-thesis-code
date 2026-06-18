from domains.env_base import EnvDomain
from domains.textarena.adapter import TextArenaEnvAdapter

# Brief task framing prepended to each prompt's user message. The tool spec is
# injected by the model's native tool-calling template (tools=...); this only
# nudges the model to read the game state and call move. reset() appends the
# game's opening prompt after this lead-in (TRL appends to the last user message).
_LEAD_IN = (
    "You are playing a text game. Read the latest game state, then call the move "
    "tool with your next move. Think briefly before moving.\n\n"
)


class TextArenaDomain(EnvDomain):
    """Second agentic domain: TextArena text games served through OpenEnv.

    Multi-turn (a sequence of moves under feedback per episode), so the efficiency
    reward counts the model's assistant tokens across all turns, excluding the
    game feedback TRL injects between them. The model is driven through its native
    tool-calling template; the adapter exposes exactly one tool (move).
    """

    # `python -m <server_module>` launches the OpenEnv env server (no Docker).
    server_module = "textarena_env.server.app"
    multi_turn = True

    def make_env_factory(self, base_url, env_config=None, client_factory=None):
        env_config = dict(env_config or {})
        if client_factory is None:
            return lambda: TextArenaEnvAdapter(base_url, env_config)
        # Test/injection path: build the adapter around a supplied client.
        return lambda: TextArenaEnvAdapter(
            base_url, env_config, client=client_factory()
        )

    def build_seed_dataset(self, env_config=None, n=500, seed_base=0):
        # Each row is one training prompt: a fixed lead-in plus a distinct seed.
        # TRL repeats each row num_generations times to form a GRPO group, and
        # reset(seed=...) turns the seed into a deterministic game instance.
        from datasets import Dataset

        rows = [
            {"prompt": [{"role": "user", "content": _LEAD_IN}], "seed": seed_base + i}
            for i in range(int(n))
        ]
        return Dataset.from_list(rows)

    def episode_messages(self, observation):
        """Eval-time prompt for the first turn of an episode. Mirrors the training
        framing: the same lead-in the seed rows carry, with the game's opening
        prompt appended."""
        return [{"role": "user", "content": _LEAD_IN + str(observation)}]

    def eval_tools(self, env):
        """Single tool for the multi-turn move env."""
        return [env.move]

    def server_env(self, env_config=None):
        # The textarena server reads its game and settings from process env vars.
        cfg = dict(env_config or {})
        env = {
            "TEXTARENA_ENV_ID": str(cfg.get("env_id", "Wordle-v0")),
            "TEXTARENA_NUM_PLAYERS": str(int(cfg.get("num_players", 1))),
        }
        if cfg.get("max_turns") is not None:
            env["TEXTARENA_MAX_TURNS"] = str(int(cfg["max_turns"]))
        return env

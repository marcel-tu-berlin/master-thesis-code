from domains.env_base import EnvDomain
from domains.reasoning_gym.adapter import ReasoningGymEnvAdapter

# Brief task framing prepended to each prompt's user message. The tool spec
# itself is injected by the model's native chat template (tools=...), so this
# only nudges the model to actually call the answer tool. reset() appends the
# reasoning_gym question after this lead-in.
_LEAD_IN = (
    "Solve the following problem. When you have the final answer, "
    "call the answer tool with it.\n\n"
)


class ReasoningGymDomain(EnvDomain):
    """First agentic domain: Reasoning Gym tasks served through OpenEnv.

    Single-step (one answer per episode), so the completion is entirely
    model-generated and the efficiency rewards apply without a multi-turn token
    mask. The model is driven through its native tool-calling template; the
    adapter exposes exactly one tool (answer).
    """

    def make_env_factory(self, base_url, env_config=None, client_factory=None):
        env_config = dict(env_config or {})
        if client_factory is None:
            return lambda: ReasoningGymEnvAdapter(base_url, env_config)
        # Test/injection path: build the adapter around a supplied client.
        return lambda: ReasoningGymEnvAdapter(
            base_url, env_config, client=client_factory()
        )

    def build_seed_dataset(self, env_config=None, n=500, seed_base=0):
        # Each row is one training prompt: a fixed user lead-in plus a distinct
        # seed. TRL repeats each row num_generations times to form a GRPO group,
        # and reset(seed=...) turns the seed into a deterministic question.
        from datasets import Dataset

        rows = [
            {"prompt": [{"role": "user", "content": _LEAD_IN}], "seed": seed_base + i}
            for i in range(int(n))
        ]
        return Dataset.from_list(rows)

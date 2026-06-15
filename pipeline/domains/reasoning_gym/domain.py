from domains.env_base import EnvDomain

SYSTEM_PROMPT = (
    "You are given a reasoning problem.\n"
    "Think about the problem and provide your working out.\n"
    "Place it between <start_working_out> and <end_working_out>.\n"
    "Then, provide your solution between <SOLUTION></SOLUTION>"
)


class ReasoningGymDomain(EnvDomain):
    """First agentic domain: Reasoning Gym tasks served through OpenEnv.

    reasoning_gym is single-step (one action per episode), so the completion is
    entirely model-generated and the efficiency rewards apply without a
    multi-turn token mask.
    """

    system_prompt = SYSTEM_PROMPT

    def make_client(self, env_config: dict | None = None):
        # Lazy import so the unit tests exercise the domain logic with a fake
        # StepResult and need neither the OpenEnv install nor a running server.
        # The exact client class and constructor are pinned against the installed
        # package during the rollout-wiring task (B6) before any live call.
        from envs.reasoning_gym_env import ReasoningGymEnv  # noqa: E402 (verified in B6)

        env_config = env_config or {}
        return ReasoningGymEnv.from_docker_image(
            env_config.get("image", "reasoning-gym-env:latest")
        )

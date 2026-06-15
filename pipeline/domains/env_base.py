from domains.base import build_reasoning_chat_template


class EnvDomain:
    """Base for OpenEnv-backed agentic domains.

    The dataset abstractions of `Domain` (load_dataset, extract_answer) do not
    apply here; an env domain instead provides a client factory and reads the
    environment-computed reward off the OpenEnv StepResult. The reasoning-tag
    chat template is shared with `Domain` via build_reasoning_chat_template.
    """

    system_prompt: str = ""
    reasoning_start: str = "<start_working_out>"

    def make_client(self, env_config: dict | None = None):
        """Return a connected OpenEnv client for this environment."""
        raise NotImplementedError

    def episode_reward(self, step_result) -> float:
        """The environment's task reward for a finished episode."""
        return float(step_result.reward)

    def is_correct(self, step_result) -> bool:
        """Binary success for evaluation: a positive env reward counts as solved."""
        return float(step_result.reward) > 0.0

    def difficulty(self, task) -> float | None:
        return None

    def build_chat_template(self, tokenizer) -> None:
        build_reasoning_chat_template(tokenizer, self.system_prompt, self.reasoning_start)

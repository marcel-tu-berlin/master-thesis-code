class EnvDomain:
    """Base for OpenEnv-backed agentic domains.

    The dataset / answer-extraction abstractions of `Domain` do not apply here.
    An env domain instead provides a TRL environment_factory (one fresh env per
    rollout slot) and a seed-row train dataset, and reads the
    environment-computed reward off the OpenEnv StepResult. Concrete env domains
    implement `make_env_factory` and `build_seed_dataset`; the eval-side reward
    helpers below are shared. The agentic path uses the model's native
    tool-calling chat template, so there is no reasoning-tag template here.
    """

    def make_env_factory(self, base_url, env_config=None, client_factory=None):
        """Return a zero-arg callable building one fresh env adapter per call."""
        raise NotImplementedError

    def build_seed_dataset(self, env_config=None, n=500, seed_base=0):
        """Return a HF Dataset of reset-kwarg rows (distinct seed per row)."""
        raise NotImplementedError

    def episode_reward(self, step_result) -> float:
        """The environment's task reward for a finished episode."""
        return float(step_result.reward)

    def is_correct(self, step_result) -> bool:
        """Binary success for evaluation: a positive env reward counts as solved."""
        return float(step_result.reward) > 0.0

    def difficulty(self, task) -> float | None:
        return None

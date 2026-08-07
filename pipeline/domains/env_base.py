# Minimum env reward that counts as a solved episode. reasoning_gym scorers are
# graded, not binary: countdown hands 0.05 to a near-miss and 0.01 to garbage,
# polynomial_equations decays exponentially with distance to the root (an
# off-by-one answer still scores 4.5e-5). A bare `> 0` therefore counts almost
# any parseable answer as correct. 0.5 admits only near-exact solutions under
# every scorer in use while staying scorer-agnostic.
CORRECT_REWARD_THRESHOLD = 0.5


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

    # Single-step by default; multi-turn domains (browsergym) set True so the
    # eval loop runs the turn loop instead of a single tool call.
    multi_turn = False

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
        """Binary success for evaluation: near-exact reward counts as solved."""
        return float(step_result.reward) >= CORRECT_REWARD_THRESHOLD

    def difficulty(self, task) -> float | None:
        return None

    def server_env(self, env_config=None) -> dict:
        """Extra process env vars for the env-server subprocess.

        Default none: reasoning_gym takes dataset/seed via request kwargs, so it
        needs no server-side env. Domains whose server is configured by env vars
        (browsergym: BROWSERGYM_BENCHMARK, ...) override this.
        """
        return {}

    def eval_tools(self, env):
        """The bound tool method(s) to expose to the model during eval.

        Concrete domains override (reasoning_gym -> [env.answer]; browsergym ->
        [env.click, env.noop]). The eval loop passes these to
        apply_chat_template(tools=...).
        """
        raise NotImplementedError

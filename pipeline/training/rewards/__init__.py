"""
Reward registry. Maps a config key (under `rewards:`) to a default weight and
a builder. `train.build_reward_components` iterates this registry instead of
hard-coding branches; new signals only require a builder + entry here plus the
matching key in `config_schema._KNOWN_REWARD_KEYS`.

Each builder has the signature:
    build(domain, runner, training_cfg, reward_cfg) -> Callable
where reward_cfg is the dict under `rewards.<key>` from the YAML.

Agentic-only: `env_reward` is the task-success signal; `token_length` (E2) and
`non_termination` (E3) are the efficiency signals. All default off and are
enabled explicitly per config.
"""
from training.rewards.cosine_length import CosineLengthReward
from training.rewards.env_reward import EnvReward
from training.rewards.non_termination import NonTerminationPenalty


def _build_token_length(domain, runner, training_cfg, cfg):
    # Cosine length reward (Wu/Yeo 2025): correct -> prefer shorter, wrong ->
    # prefer longer. Non-linear and correctness-gated (correctness comes from the
    # env), so it survives the advantage_weighted per-group z-scoring.
    return CosineLengthReward(
        runner.tokenizer,
        max_len=int(cfg.get("max_len", 256)),
        r_correct_short=cfg.get("r_correct_short", 1.0),
        r_correct_long=cfg.get("r_correct_long", 0.5),
        r_wrong_short=cfg.get("r_wrong_short", -1.0),
        r_wrong_long=cfg.get("r_wrong_long", -0.5),
    )


def _build_env_reward(domain, runner, training_cfg, cfg):
    # Task-success reward from the OpenEnv environment. TRL's environment_factory
    # path passes the live env instances as kwargs['environments']; EnvReward
    # reads env.reward off each.
    return EnvReward()


def _build_non_termination(domain, runner, training_cfg, cfg):
    # E3: -1 per episode that never reached the terminal tool. Reads env.done off
    # the same live env instances. No knobs - lambda is the component `weight`.
    return NonTerminationPenalty()


# key -> (default_enabled, default_weight, builder). All default off; agentic
# configs enable env_reward + the efficiency signals explicitly.
REWARD_REGISTRY: dict[str, tuple[bool, float, callable]] = {
    "token_length":     (False, 1.0, _build_token_length),
    "env_reward":       (False, 1.0, _build_env_reward),
    "non_termination":  (False, 1.0, _build_non_termination),
}

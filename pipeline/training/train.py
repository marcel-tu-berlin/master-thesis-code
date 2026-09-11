import argparse
import os
import random
import sys
import time

# Set the GPU before importing Transformers, TRL, or vLLM. Those imports can
# initialize CUDA, after which changing visibility is too late.
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import yaml

# Allow running as: python -m training.train
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from transformers import TrainerCallback, set_seed

from domains import build_domain
from eval.agentic_eval import seed_block
from training.batch import mark_smoke_checkpoint
from training.config_schema import (
    DEFAULT_N_ROLLOUTS,
    resolve_checkpoint_steps,
    validate_config,
    warn_inert_scalars,
)
from training.env_server import build_env_server
from training.env_stamp import write_env_stamp
from training.grpo_runner import GRPORunner
from training.rewards import REWARD_REGISTRY
from training.rewards.compose import build_composer
from training.rewards.placebo import maybe_placebo


class _ComponentMetricsCallback(TrainerCallback):
    """Drain the composer's per-component reward metrics into the trainer log.

    TRL only sees one composed reward function, so it logs a single
    rewards/<composer>/{mean,std}; the individual env/length/entropy
    contributions are invisible. transformers.Trainer.log() appends the log
    entry to state.log_history BEFORE calling on_log, so updating `logs` alone
    is too late — the persisted entry is already a copy. We update that
    just-appended entry in place (state.log_history[-1]) so the metrics land in
    train_log.json next to reward/kl/loss, and also update `logs` for any live
    logger (wandb/trackio). Purely observational: it never touches the composed
    reward or the advantage math.
    """

    def __init__(self, composer) -> None:
        self.composer = composer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not hasattr(self.composer, "pop_step_metrics"):
            return
        metrics = self.composer.pop_step_metrics()
        if not metrics:
            return
        # state.log_history[-1] is this step's entry (appended just before this
        # hook fires); update it in place so the metrics persist to train_log.
        if logs is not None:
            logs.update(metrics)
        if state is not None and getattr(state, "log_history", None):
            state.log_history[-1].update(metrics)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_reward_components(config: dict, domain, runner: GRPORunner) -> list:
    """Build (reward_fn, weight) pairs from config using REWARD_REGISTRY."""
    rewards_cfg = config.get("rewards", {}) or {}
    training_cfg = config.get("training", {}) or {}

    method = rewards_cfg.get("compose_method", "advantage_weighted")
    scale_rewards = str(training_cfg.get("scale_rewards", "group"))
    for w in warn_inert_scalars(rewards_cfg, method, scale_rewards):
        print(f"⚠  {w}")

    components = []

    for key, (_reg_enabled, default_weight, builder) in REWARD_REGISTRY.items():
        cfg = rewards_cfg.get(key) or {}
        if not cfg.get("enabled", _reg_enabled):
            continue
        weight = float(cfg.get("weight", default_weight))
        # Placebo arm: uniformly shuffle the same term within each prompt-group.
        # Its expected centered contribution at every group position is zero,
        # but total-reward variance and gradient noise need not match the real arm.
        component = maybe_placebo(
            builder(domain, runner, training_cfg, cfg),
            cfg,
            training_cfg,
            config.get("seed", 42),
        )
        components.append((component, weight))

    return components


def apply_smoke_overrides(config: dict) -> dict:
    """Patch config for fast smoke testing: 3 steps, 2 rollouts, short seq.

    Sets `_smoke=True` so downstream eval also caps to 4 episodes per split.
    """
    checkpoint_steps = resolve_checkpoint_steps(config, smoke=True)
    config.setdefault("model", {})
    config.setdefault("training", {})
    # Pick a smoke context that clears the env's prompt but fits the L4. A 512 cap
    # rejected finqa's ~700-token tool-rich prompt outright; the full 4096 OOMs the
    # policy-grad backward under vLLM colocate. 2048 holds the prompt plus a few
    # turns and leaves room for the step. (Don't go below the config when it is
    # already smaller, e.g. a 1024-context model.)
    seq = min(int(config["model"].get("max_seq_length", 2048) or 2048), 2048)
    config["model"]["max_seq_length"] = seq
    # Under colocate the backward competes with vLLM's KV pool for the 24 GB, so
    # give training headroom in smoke (0.6 util OOMs a full agentic rollout here).
    # Set before --vllm's setdefault so this wins. Not under sleep mode: there
    # the engine releases its pool during the backward, and a smoke of that
    # knob has to run the share the config asks for.
    if not config["model"].get("vllm_enable_sleep_mode"):
        config["model"]["gpu_memory_utilization"] = min(
            float(config["model"].get("gpu_memory_utilization", 0.45) or 0.45), 0.45
        )
    # Safety: the completion budget is max_seq - max_prompt_length, so keep the
    # prompt cap at half the context (a config with max_prompt_length == max_seq
    # would otherwise leave zero room to generate).
    config["training"]["max_prompt_length"] = min(
        int(config["training"].get("max_prompt_length", seq // 2) or seq // 2), seq // 2
    )
    config["training"]["max_steps"] = 3
    config["training"]["save_steps"] = checkpoint_steps[0] if checkpoint_steps else 3
    config["training"]["n_rollouts"] = 2
    # Cap eval generation: a multi-turn agentic eval runs many model.generate
    # calls per episode, and the default budget (max_seq - max_prompt) lets each
    # produce ~1k tokens, so a smoke eval can take 20+ minutes. 256 is plenty to
    # emit a tool call for a sanity check.
    config.setdefault("eval", {})
    config["eval"]["max_new_tokens"] = 256
    config["_smoke"] = True
    print(
        f"⚠  Smoke mode: max_steps=3, n_rollouts=2, max_seq_length={seq} (<=2048), "
        f"gpu_memory_utilization={config['model']['gpu_memory_utilization']}, "
        f"max_prompt_length={config['training']['max_prompt_length']}, eval=4/split"
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval", action="store_true", help="Run eval after training")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Override config for fast smoke test (3 steps, 2 rollouts, 512 seq)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing run directory",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Route GRPO rollouts through vLLM fast inference",
    )
    parser.add_argument(
        "--readiness-capture",
        action="store_true",
        help="Capture one full rollout batch and its first optimizer update",
    )
    args = parser.parse_args()
    if args.readiness_capture and args.smoke:
        parser.error("--readiness-capture requires the real geometry, not --smoke")
    readiness_started = time.perf_counter() if args.readiness_capture else None

    config = load_config(args.config)
    if args.smoke:
        apply_smoke_overrides(config)
    if args.vllm:
        config.setdefault("model", {})
        config["model"]["use_vllm"] = True
        config["model"].setdefault("gpu_memory_utilization", 0.6)
        print("⚠  vLLM fast inference ON (--vllm): gpu_memory_utilization=0.6")
    validate_config(config)
    checkpoint_steps = resolve_checkpoint_steps(config)
    if checkpoint_steps:
        config["training"]["save_steps"] = checkpoint_steps[0]
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)  # covers torch + cuda RNGs

    exp_id = config["experiment_id"]
    run_dir = os.path.join("runs", exp_id)
    # Refuse to clobber an existing run unless --overwrite is given. Without
    # this guard, a re-invocation with the same experiment_id silently
    # overwrites the prior frozen config and (later) trampling checkpoints.
    existing_final = os.path.join(run_dir, "checkpoint-final")
    if os.path.isdir(existing_final) and not args.overwrite:
        raise FileExistsError(
            f"Run directory {run_dir!r} already has checkpoint-final/. "
            "Pass --overwrite to replace, or change experiment_id."
        )
    os.makedirs(run_dir, exist_ok=True)

    # Persist config alongside run artifacts. Strip the runtime-only `_smoke`
    # marker so re-running eval against the frozen config does not silently
    # cap each split to 10 samples.
    frozen = {k: v for k, v in config.items() if k != "_smoke"}
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(frozen, f)

    domain = build_domain(config)
    runner = GRPORunner(config)

    # Build composed reward function (shared by both modes; the enabled set
    # differs per config — agentic configs enable env_reward + token_length).
    components = build_reward_components(config, domain, runner)
    if not components:
        raise ValueError("No reward components enabled. Check config rewards section.")

    method = config.get("rewards", {}).get("compose_method", "advantage_weighted")
    # The composer z-scores per GRPO group, cut positionally in blocks of
    # num_generations - the same resolution grpo_runner hands TRL.
    n_rollouts = int(config["training"].get("n_rollouts", DEFAULT_N_ROLLOUTS))
    reward_fn = build_composer(components, method, n_rollouts)
    trainer_reward_fn = reward_fn
    readiness_recorder = None
    if args.readiness_capture:
        from probes.readiness import ReadinessRecorder

        readiness_recorder = ReadinessRecorder(
            run_dir, frozen, started_monotonic=readiness_started
        )
        rewards_cfg = config.get("rewards") or {}
        training_cfg = config.get("training") or {}
        diagnostic_components = {
            key: builder(
                domain,
                runner,
                training_cfg,
                rewards_cfg.get(key) or {},
            )
            for key, (_enabled, _weight, builder) in REWARD_REGISTRY.items()
        }
        trainer_reward_fn = readiness_recorder.wrap_reward(
            reward_fn, diagnostic_components
        )

    # The callback holds the same composer instance passed as the reward fn, so
    # it drains the very buffer the trainer's reward calls populate (T2.1).
    callbacks: list[TrainerCallback] = (
        [_ComponentMetricsCallback(reward_fn)]
        if hasattr(reward_fn, "pop_step_metrics")
        else []
    )

    print(f"Experiment: {exp_id}  (agentic)")
    print(f"Reward components: {[type(fn).__name__ for fn, _ in components]}")
    print(f"Compose method: {method}")

    checkpoint_dir = os.path.join(run_dir, "checkpoint-final")

    # Native tool-calling template (NOT a reasoning-tag one). Each seed-row is a
    # distinct reasoning_gym question; the runner owns the env-server subprocess
    # and builds the TRL environment_factory against its base_url.
    env_config = config["training"].get("env_config", {}) or {}
    n_prompts = int(env_config.get("size", 500))
    # Each seed trains on its own block of the seed -> question mapping. Passing
    # the raw seed through made --seeds 42 43 44 share 499 of 500 questions.
    dataset = domain.build_seed_dataset(
        env_config, n=n_prompts, seed_base=seed_block(seed)
    )
    server = build_env_server(config, domain, python=sys.executable)
    # The frozen config records what the run asked for; this records what the box
    # actually had installed while it trained. A hand-installed package between
    # two arms is otherwise invisible.
    write_env_stamp(run_dir, "train", server.repo_envs_path)

    def make_factory(base_url):
        return domain.make_env_factory(base_url, env_config)

    print(
        f"Agentic env: {config['training']['env']}  seed-rows: {len(dataset)}  "
        f"server: {server.base_url} (max_concurrent={server.max_concurrent})"
    )
    try:
        runner.train(
            dataset,
            trainer_reward_fn,
            output_dir=run_dir,
            callbacks=callbacks or None,
            server=server,
            make_factory=make_factory,
            readiness_recorder=readiness_recorder,
        )

        runner.save_lora(checkpoint_dir)
        mark_smoke_checkpoint(checkpoint_dir, bool(config.get("_smoke")))

        if args.eval:
            if checkpoint_steps:
                if readiness_recorder is not None:
                    readiness_recorder.finish("complete")
                # Replace training's process so its policy and vLLM allocations are gone.
                cmd = [
                    sys.executable,
                    "-m",
                    "eval.runner",
                    "--config",
                    os.path.join(run_dir, "config.yaml"),
                ]
                if args.smoke:
                    cmd.append("--smoke")
                os.execv(sys.executable, cmd)
            from eval.agentic_eval import run_agentic_eval

            run_agentic_eval(config, checkpoint_dir, domain, run_dir)
        if readiness_recorder is not None:
            readiness_recorder.finish("complete")
    except BaseException as exc:
        if readiness_recorder is not None:
            readiness_recorder.finish("failed", str(exc))
        raise


if __name__ == "__main__":
    main()

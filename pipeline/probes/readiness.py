"""Admission gate and bounded evidence capture for the experiment campaign.

Run this command between every diagnostic arm. It validates the proposed
configs, checks any run artifacts already present, writes one JSON report, and
names the only next phase that is admitted. A failed or stale contract exits
non-zero so it cannot be used as an automatic green light for another job.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from training.config_schema import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_N_ROLLOUTS,
    validate_config,
    warn_inert_scalars,
)
from training.env_stamp import collect_env_stamp

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "pipeline"
CAPTURE_DIR = "readiness"
CAPTURE_SCHEMA = 1
EXPECTED_CONDITIONS = ("E1", "E2", "E3")

RECIPE_FIELDS = {
    "model.slug": "qwen3-1.7b",
    "model.lora_r": 16,
    "model.lora_alpha": 32,
    "model.load_in_4bit": False,
    "model.max_seq_length": 8192,
    "model.use_vllm": True,
    "model.gpu_memory_utilization": 0.3,
    "model.vllm_enable_sleep_mode": False,
    "training.mode": "agentic",
    "training.env": "browsergym",
    "training.env_config.benchmark": "miniwob",
    "training.env_config.tasks": ["click-menu-2"],
    "training.env_config.miniwob_url": "http://localhost:8080/miniwob/",
    "training.env_config.max_turns": 8,
    "training.env_config.size": 4,
    "training.env_server.repo_path": "/workspace/OpenEnv/envs",
    "training.env_server.port": 8000,
    "training.max_prompt_length": 4096,
    "training.batch_size": 4,
    "training.n_rollouts": 8,
    "training.micro_batch_size": 1,
    "training.max_steps": 3,
    "training.save_steps": 3,
    "training.learning_rate": 5e-5,
    "training.kl_beta": 0.0,
    "training.optim": "adamw_torch_fused",
    "training.lr_scheduler_type": "constant_with_warmup",
    "training.warmup_ratio": 0.1,
    "training.weight_decay": 0.1,
    "training.temperature": 1.0,
    "training.vllm_importance_sampling_mode": "token_truncate",
    "training.num_iterations": 1,
    "training.use_liger_kernel": False,
    "training.scale_rewards": "none",
    "training.loss_type": "dapo",
    "rewards.compose_method": "naive_sum",
    "eval.temperature": 0.0,
    "eval.do_sample": False,
    "eval.max_new_tokens": 4096,
    "eval.agentic.n_episodes": 4,
}


def _canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(root: Path = ROOT) -> dict[str, str]:
    """Hash every source file that can change training or evaluation semantics."""
    paths: list[Path] = []
    for directory in (
        root / "pipeline/training",
        root / "pipeline/domains",
        root / "pipeline/eval",
    ):
        paths.extend(directory.rglob("*.py"))
    paths.extend(
        root / name
        for name in ("requirements.lock.txt", "setup.sh", "pipeline/OPENENV_COMMIT")
    )
    paths.append(root / "pipeline/probes/readiness.py")
    return {
        str(path.relative_to(root)): _file_hash(path)
        for path in sorted(set(paths))
        if path.is_file() and "__pycache__" not in path.parts
    }


def _get(config: dict, dotted: str):
    value = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _load_config(path: Path) -> dict:
    with path.open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: config must be a mapping")
    return value


def _condition(config: dict) -> str | None:
    rewards = config.get("rewards") or {}
    enabled = {
        name
        for name in ("env_reward", "token_length", "non_termination")
        if (rewards.get(name) or {}).get("enabled")
    }
    if enabled == {"env_reward"}:
        return "E1"
    if enabled == {"env_reward", "token_length"}:
        return "E2"
    if enabled == {"env_reward", "non_termination"}:
        return "E3"
    return None


def _check_config(path: Path, config: dict, expected_condition: str) -> list[str]:
    errors = []
    try:
        validate_config(config)
    except ValueError as exc:
        errors.append(str(exc))

    for dotted, expected in RECIPE_FIELDS.items():
        actual = _get(config, dotted)
        if actual != expected:
            errors.append(f"{dotted}: expected {expected!r}, got {actual!r}")

    if "checkpoint_schedule" in (config.get("eval") or {}):
        errors.append("diagnostic configs must use final-only evaluation")

    condition = _condition(config)
    if condition != expected_condition:
        errors.append(
            f"expected {expected_condition}, got reward condition {condition!r}"
        )

    rewards = config.get("rewards") or {}
    warnings = warn_inert_scalars(
        rewards,
        str(rewards.get("compose_method", "advantage_weighted")),
        str((config.get("training") or {}).get("scale_rewards", "group")),
    )
    errors.extend(f"campaign dose warning: {warning}" for warning in warnings)

    token_cfg = rewards.get("token_length") or {}
    expected_endpoints = {
        "max_len": 4096,
        "r_correct_short": 1.0,
        "r_correct_long": 0.5,
        "r_wrong_short": -1.0,
        "r_wrong_long": -0.5,
        "placebo": False,
    }
    for key, expected in expected_endpoints.items():
        if token_cfg.get(key) != expected:
            errors.append(
                f"rewards.token_length.{key}: expected {expected!r}, "
                f"got {token_cfg.get(key)!r}"
            )

    for name in ("env_reward", "token_length", "non_termination"):
        reward = rewards.get(name)
        if (
            not isinstance(reward, dict)
            or "enabled" not in reward
            or "weight" not in reward
        ):
            errors.append(f"rewards.{name} must state enabled and weight explicitly")
    if (rewards.get("env_reward") or {}).get("weight") != 1.0:
        errors.append("rewards.env_reward.weight must be 1.0")
    if (rewards.get("token_length") or {}).get("weight") != 1.0:
        errors.append("diagnostic E2 must exercise the highest proposed weight, 1.0")
    if (rewards.get("non_termination") or {}).get("weight") != 1.0:
        errors.append("diagnostic E3 must exercise the highest proposed weight, 1.0")
    if (rewards.get("non_termination") or {}).get("placebo") is not False:
        errors.append("diagnostic E3 placebo must be false")

    completion_budget = _get(config, "model.max_seq_length")
    prompt_budget = _get(config, "training.max_prompt_length")
    if (
        isinstance(completion_budget, int)
        and isinstance(prompt_budget, int)
        and completion_budget - prompt_budget != 4096
    ):
        errors.append("training completion budget must resolve to 4096")

    return [f"{path}: {error}" for error in errors]


def _normalized_non_reward(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.pop("experiment_id", None)
    value.pop("description", None)
    value.pop("rewards", None)
    return value


def _local_gate() -> dict:
    started = time.perf_counter()
    proc = subprocess.run(
        [str(ROOT / ".claude/check.sh")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = proc.stdout.strip().splitlines()
    return {
        "status": "pass" if proc.returncode == 0 else "fail",
        "returncode": proc.returncode,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "output_tail": output[-40:],
    }


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        tensor = value.detach().cpu()
        return tensor.item() if getattr(tensor, "ndim", 1) == 0 else tensor.tolist()
    return str(value)


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class ReadinessRecorder:
    """Capture one rollout batch and its first optimizer update without mutation."""

    def __init__(
        self,
        run_dir: str | Path,
        config: dict,
        *,
        started_monotonic: float | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / CAPTURE_DIR
        self.path.mkdir(parents=True, exist_ok=True)
        self.config = copy.deepcopy(config)
        training = config.get("training") or {}
        batch_size = int(training.get("batch_size", DEFAULT_BATCH_SIZE))
        n_rollouts = int(training.get("n_rollouts", DEFAULT_N_ROLLOUTS))
        micro = int(training.get("micro_batch_size", 1))
        self.max_loss_shards = batch_size * n_rollouts // micro
        self._loss_shards = 0
        self._reward_recorded = False
        self._batch_recorded = False
        self._started = (
            started_monotonic if started_monotonic is not None else time.perf_counter()
        )
        elapsed = time.perf_counter() - self._started

        hashes = source_hashes()
        repo_path = _get(config, "training.env_server.repo_path")
        stack = collect_env_stamp(repo_path)
        self._trl_version = (stack.get("packages") or {}).get("trl")
        self.metadata = {
            "schema_version": CAPTURE_SCHEMA,
            "status": "running",
            "started_at": (datetime.now(UTC) - timedelta(seconds=elapsed)).isoformat(),
            "config_sha256": _canonical_hash(config),
            "source_hashes": hashes,
            "source_sha256": _canonical_hash(hashes),
            "stack": stack,
            "stack_sha256": _canonical_hash(stack),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        _atomic_json(self.path / "metadata.json", self.metadata)

    def snapshot(self, value):
        return _jsonable(value)

    def wrap_reward(self, reward_fn, diagnostic_components: dict):
        def captured_reward(prompts, completions, **kwargs):
            total = reward_fn(prompts, completions, **kwargs)
            if not self._reward_recorded:
                raw = {
                    name: list(component(prompts, completions, **kwargs))
                    for name, component in diagnostic_components.items()
                }
                environments = kwargs.get("environments") or []
                columns = {
                    key: value
                    for key, value in kwargs.items()
                    if key
                    not in {
                        "environments",
                        "trainer_state",
                        "log_extra",
                        "log_metric",
                    }
                }
                records = []
                n_rollouts = int(
                    (self.config.get("training") or {}).get(
                        "n_rollouts", DEFAULT_N_ROLLOUTS
                    )
                )
                completion_ids = kwargs.get("completion_ids") or [[] for _ in total]
                seeds = columns.get("seed") or [None for _ in total]
                for index in range(len(total)):
                    env = environments[index] if index < len(environments) else None
                    records.append(
                        {
                            "index": index,
                            "group_index": index // n_rollouts,
                            "slot_index": index % n_rollouts,
                            "seed": seeds[index] if index < len(seeds) else None,
                            "prompt": _jsonable(prompts[index]),
                            "completion": _jsonable(completions[index]),
                            "completion_ids": _jsonable(completion_ids[index]),
                            "environment": {
                                "reward": _jsonable(getattr(env, "reward", None)),
                                "done": _jsonable(getattr(env, "done", None)),
                            },
                            "raw_rewards": {
                                name: _jsonable(values[index])
                                for name, values in raw.items()
                            },
                            "composed_reward": _jsonable(total[index]),
                        }
                    )
                _atomic_json(
                    self.path / "reward_batch.json",
                    {"columns": _jsonable(columns), "records": records},
                )
                self._reward_recorded = True
            return total

        captured_reward.__name__ = getattr(reward_fn, "__name__", "readiness_reward")
        return captured_reward

    def record_rollout_batch(self, inputs, prepared) -> None:
        if self._batch_recorded:
            return
        selected = {
            key: value
            for key, value in prepared.items()
            if key
            in {
                "prompt_ids",
                "prompt_mask",
                "completion_ids",
                "completion_mask",
                "tool_mask",
                "advantages",
                "num_items_in_batch",
                "old_per_token_logps",
                "sampling_per_token_logps",
                "importance_sampling_ratio",
                "ref_per_token_logps",
            }
        }
        _atomic_json(
            self.path / "rollout_batch.json",
            {"source_inputs": _jsonable(inputs), "prepared": _jsonable(selected)},
        )
        self._batch_recorded = True

    def record_loss(self, inputs, loss, per_token_logps, settings: dict) -> None:
        if self._loss_shards >= self.max_loss_shards:
            return
        selected = {
            key: value
            for key, value in inputs.items()
            if key
            in {
                "completion_ids",
                "completion_mask",
                "tool_mask",
                "advantages",
                "num_items_in_batch",
                "old_per_token_logps",
                "importance_sampling_ratio",
            }
        }
        record = {
            "shard": self._loss_shards,
            "inputs": _jsonable(selected),
            "per_token_logps": _jsonable(per_token_logps),
            "loss": _jsonable(loss),
            "settings": _jsonable(settings),
        }
        with (self.path / "loss_shards.jsonl").open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
        self._loss_shards += 1

    def record_log(self, logs: dict) -> None:
        with (self.path / "step_log.jsonl").open("a") as handle:
            handle.write(json.dumps(_jsonable(logs), sort_keys=True) + "\n")
            handle.flush()

    def record_gradients(self, model, accumulation_step: int) -> None:
        finite = True
        squared_norm = 0.0
        tensors = 0
        for parameter in model.parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            grad = parameter.grad.detach().float()
            finite = finite and bool(grad.isfinite().all().item())
            squared_norm += float((grad * grad).sum().item())
            tensors += 1
        record = {
            "accumulation_step": accumulation_step,
            "finite": finite,
            "l2_norm": math.sqrt(squared_norm),
            "tensors": tensors,
        }
        with (self.path / "gradients.jsonl").open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()

    def record_parameters(self, label: str, model) -> None:
        digest = hashlib.sha256()
        finite = True
        tensors = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            value = parameter.detach().float().cpu().contiguous()
            finite = finite and bool(value.isfinite().all().item())
            digest.update(name.encode())
            digest.update(value.numpy().tobytes())
            tensors += 1
        path = self.path / "parameters.json"
        records = json.loads(path.read_text()) if path.exists() else {}
        records[label] = {
            "sha256": digest.hexdigest(),
            "finite": finite,
            "tensors": tensors,
        }
        _atomic_json(path, records)

    def record_trainer_settings(self, trainer) -> None:
        args = trainer.args
        model_config = getattr(getattr(trainer, "model", None), "config", None)
        tokenizer = getattr(trainer, "processing_class", None)
        settings = {
            "trl_version": self._trl_version,
            "bf16": bool(args.bf16),
            "fp16": bool(args.fp16),
            "loss_type": trainer.loss_type,
            "scale_rewards": trainer.scale_rewards,
            "epsilon_low": trainer.epsilon_low,
            "epsilon_high": trainer.epsilon_high,
            "importance_sampling_level": trainer.importance_sampling_level,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "num_generations": trainer.num_generations,
            "max_completion_length": trainer.max_completion_length,
            "model_revision": getattr(model_config, "_commit_hash", None),
            "tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get(
                "_commit_hash"
            ),
        }
        try:
            import torch

            settings["cuda_device"] = torch.cuda.get_device_name(0)
        except (ImportError, RuntimeError):
            settings["cuda_device"] = None
        _atomic_json(self.path / "trainer_settings.json", settings)

    def finish(self, status: str, error: str | None = None) -> None:
        self.metadata["status"] = status
        self.metadata["finished_at"] = datetime.now(UTC).isoformat()
        self.metadata["duration_seconds"] = round(
            time.perf_counter() - self._started, 3
        )
        if error is not None:
            self.metadata["error"] = error
        _atomic_json(self.path / "metadata.json", self.metadata)


def _read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _centered(values: list[float], group_size: int) -> list[float]:
    result: list[float] = []
    for start in range(0, len(values), group_size):
        group = values[start : start + group_size]
        mean = sum(group) / len(group)
        result.extend(value - mean for value in group)
    return result


def _reference_dapo_loss(record: dict) -> float:
    inputs = record["inputs"]
    settings = record["settings"]
    if (
        settings["loss_type"] != "dapo"
        or settings["importance_sampling_level"] != "token"
    ):
        raise ValueError(
            "readiness reference supports DAPO with token-level sampling only"
        )
    if float(settings.get("beta", 0.0)) != 0.0:
        raise ValueError("readiness reference requires kl_beta=0")

    advantages = inputs["advantages"]
    if not isinstance(advantages, list):
        advantages = [advantages]
    current = record["per_token_logps"]
    old = inputs["old_per_token_logps"]
    completion_mask = inputs["completion_mask"]
    tool_mask = inputs.get("tool_mask")
    is_ratio = inputs.get("importance_sampling_ratio")
    normalizer = float(inputs["num_items_in_batch"])
    epsilon_low = float(settings["epsilon_low"])
    epsilon_high = float(settings["epsilon_high"])

    total = 0.0
    for row in range(len(current)):
        advantage = float(advantages[row])
        for col in range(len(current[row])):
            mask = float(completion_mask[row][col])
            if tool_mask is not None:
                mask *= float(tool_mask[row][col])
            if not mask:
                continue
            coefficient = math.exp(float(current[row][col]) - float(old[row][col]))
            clipped = min(max(coefficient, 1.0 - epsilon_low), 1.0 + epsilon_high)
            token_loss = -min(coefficient * advantage, clipped * advantage)
            if is_ratio is not None:
                token_loss *= float(is_ratio[row][col])
            total += token_loss
    return total / normalizer


def _assess_capture(
    run_dir: Path, config: dict, current_hashes: dict, current_stack: dict
) -> list[str]:
    errors = []
    capture = run_dir / CAPTURE_DIR
    required = (
        "metadata.json",
        "reward_batch.json",
        "rollout_batch.json",
        "loss_shards.jsonl",
        "gradients.jsonl",
        "step_log.jsonl",
        "parameters.json",
        "trainer_settings.json",
    )
    missing = [name for name in required if not (capture / name).is_file()]
    if missing:
        return [f"missing readiness evidence: {missing}"]

    metadata = _read_json(capture / "metadata.json")
    if metadata.get("status") != "complete":
        errors.append(
            f"capture status is {metadata.get('status')!r}, expected 'complete'"
        )
    if metadata.get("config_sha256") != _canonical_hash(config):
        errors.append("capture config hash does not match frozen config")
    if metadata.get("source_hashes") != current_hashes:
        errors.append(
            "capture source hashes are stale relative to the current checkout"
        )
    if metadata.get("stack_sha256") != _canonical_hash(current_stack):
        errors.append("capture stack is stale relative to the current runtime")

    stamp_path = run_dir / "env_stamp.json"
    if not stamp_path.is_file():
        errors.append("missing env_stamp.json")
    else:
        train_stamp = (_read_json(stamp_path) or {}).get("train")
        if _canonical_hash(train_stamp) != metadata.get("stack_sha256"):
            errors.append("capture stack hash does not match the run's train env stamp")

    settings = _read_json(capture / "trainer_settings.json")
    expected_settings = {
        "trl_version": "1.6.0",
        "bf16": True,
        "fp16": False,
        "loss_type": "dapo",
        "scale_rewards": "none",
        "importance_sampling_level": "token",
        "gradient_accumulation_steps": 32,
        "per_device_train_batch_size": 1,
        "num_generations": 8,
        "max_completion_length": 4096,
    }
    for key, expected in expected_settings.items():
        if settings.get(key) != expected:
            errors.append(
                f"trainer {key}: expected {expected!r}, got {settings.get(key)!r}"
            )
    if not settings.get("model_revision") or not settings.get("tokenizer_revision"):
        errors.append("model and tokenizer snapshot revisions must both be recorded")
    if metadata.get("cuda_visible_devices") != "1":
        errors.append("CUDA_VISIBLE_DEVICES must select physical GPU 1")
    if not settings.get("cuda_device"):
        errors.append("CUDA device name was not recorded")

    reward = _read_json(capture / "reward_batch.json")
    rollout = _read_json(capture / "rollout_batch.json")
    records = reward.get("records") or []
    prepared = rollout.get("prepared") or {}
    expected_n = 32
    if len(records) != expected_n:
        errors.append(f"captured {len(records)} rewards, expected {expected_n}")
    for key in (
        "completion_ids",
        "completion_mask",
        "tool_mask",
        "advantages",
        "old_per_token_logps",
        "sampling_per_token_logps",
        "importance_sampling_ratio",
    ):
        value = prepared.get(key)
        if not isinstance(value, list) or len(value) != expected_n:
            errors.append(f"prepared {key} must contain {expected_n} rows")

    if len(records) == expected_n:
        seeds = [record.get("seed") for record in records]
        for start in range(0, expected_n, 8):
            if len(set(seeds[start : start + 8])) != 1:
                errors.append(f"rollout group {start // 8} does not share one seed")
            prompts = [
                _canonical_hash(record.get("prompt"))
                for record in records[start : start + 8]
            ]
            if len(set(prompts)) != 1:
                errors.append(
                    f"rollout group {start // 8} does not share one task prompt"
                )
        if len(set(seeds[::8])) != 4:
            errors.append("the four prompt groups do not have four distinct seeds")

        for record in records:
            raw = record.get("raw_rewards") or {}
            if set(raw) != {"token_length", "env_reward", "non_termination"}:
                errors.append(
                    "diagnostic capture must include raw E1, E2, and E3 rewards"
                )
                break
            if raw["env_reward"] != (record.get("environment") or {}).get("reward"):
                errors.append(
                    "environment reward is misaligned with its captured rollout"
                )
                break

        condition = _condition(config)
        values = []
        for record in records:
            raw = record["raw_rewards"]
            value = raw["env_reward"]
            if condition == "E2":
                value += raw["token_length"]
            elif condition == "E3":
                value += raw["non_termination"]
            values.append(float(value))
            if not math.isclose(value, record["composed_reward"], abs_tol=1e-6):
                errors.append("composed reward does not match the independent raw sum")
                break
        advantages = prepared.get("advantages")
        if isinstance(advantages, list) and len(advantages) == expected_n:
            expected_advantages = _centered(values, 8)
            if any(
                not math.isclose(float(actual), expected, abs_tol=2e-5)
                for actual, expected in zip(
                    advantages, expected_advantages, strict=True
                )
            ):
                errors.append("trainer advantages do not match reward minus group mean")

        if not any(
            message.get("role") == "tool"
            for record in records
            for message in (record.get("completion") or [])
            if isinstance(message, dict)
        ):
            errors.append("captured batch contains no tool feedback")

    loss_records = _read_jsonl(capture / "loss_shards.jsonl")
    if len(loss_records) != 32:
        errors.append(f"captured {len(loss_records)} loss shards, expected 32")
    else:
        for record in loss_records:
            settings_with_beta = dict(record.get("settings") or {})
            settings_with_beta["beta"] = float(_get(config, "training.kl_beta"))
            record["settings"] = settings_with_beta
            try:
                expected_loss = _reference_dapo_loss(record)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                errors.append(f"could not recompute DAPO loss: {exc}")
                break
            if not math.isclose(
                float(record["loss"]),
                expected_loss,
                rel_tol=2e-4,
                abs_tol=2e-5,
            ):
                errors.append(
                    f"trainer loss {record['loss']} differs from reference "
                    f"{expected_loss}"
                )
                break

    gradients = _read_jsonl(capture / "gradients.jsonl")
    if not gradients or not all(record.get("finite") for record in gradients):
        errors.append("optimizer gradients are missing or non-finite")
    if gradients and not any(
        float(record.get("l2_norm", 0.0)) > 0 for record in gradients
    ):
        errors.append("all captured optimizer gradients are zero")

    parameters = _read_json(capture / "parameters.json")
    before, after = parameters.get("before"), parameters.get("after")
    if not before or not after:
        errors.append("missing before/after trainable-parameter fingerprints")
    elif not before.get("finite") or not after.get("finite"):
        errors.append("trainable parameters are non-finite")
    elif before.get("sha256") == after.get("sha256"):
        errors.append("trainable parameters did not change")

    return errors


def _assess_run(
    run_dir: Path, proposed: dict, current_hashes: dict, current_stack: dict
) -> list[str]:
    errors = []
    frozen_path = run_dir / "config.yaml"
    if not frozen_path.is_file():
        return ["run exists but has no frozen config.yaml"]
    frozen = _load_config(frozen_path)
    if _canonical_hash(frozen) != _canonical_hash(proposed):
        errors.append("frozen config differs from the admitted config")

    train_log_path = run_dir / "train_log.json"
    if not train_log_path.is_file():
        errors.append("missing train_log.json")
    else:
        log = _read_json(train_log_path)
        steps = [
            row
            for row in log
            if isinstance(row, dict) and "loss" in row and "grad_norm" in row
        ]
        if not steps or max(int(row.get("step", 0)) for row in steps) < 3:
            errors.append("training did not complete three optimizer steps")
        for row in steps:
            for key in ("loss", "grad_norm"):
                value = row.get(key)
                if value is None or not math.isfinite(float(value)):
                    errors.append(f"training log contains non-finite or missing {key}")
                    break

    checkpoint = run_dir / "checkpoint-final"
    if not checkpoint.is_dir():
        errors.append("missing checkpoint-final")
    if (checkpoint / ".smoke").exists():
        errors.append("diagnostic run used the smoke geometry")

    report_path = run_dir / "eval_report.json"
    if not report_path.is_file():
        errors.append("missing eval_report.json")
    else:
        report = _read_json(report_path)
        if report.get("smoke"):
            errors.append("evaluation report is marked smoke")
        results = report.get("results") or {}
        if not results or any(
            (result or {}).get("n_samples") != 4 for result in results.values()
        ):
            errors.append("evaluation report must contain four samples per split")
        for split in results:
            episodes_path = run_dir / f"episodes_{split}.jsonl"
            if not episodes_path.is_file():
                errors.append(f"missing {episodes_path.name}")
                continue
            episodes = _read_jsonl(episodes_path)
            if len(episodes) != 4:
                errors.append(
                    f"{episodes_path.name} contains {len(episodes)} episodes, "
                    "expected 4"
                )
            if any("initial_observation" not in episode for episode in episodes):
                errors.append(f"{episodes_path.name} lacks initial task observations")
            if any(
                "tool_results" not in turn
                for episode in episodes
                for turn in episode.get("turns", [])
            ):
                errors.append(f"{episodes_path.name} lacks tool feedback")

    errors.extend(_assess_capture(run_dir, proposed, current_hashes, current_stack))
    return errors


def build_report(
    config_paths: list[Path], runs_root: Path, *, run_local_gate: bool = True
) -> dict:
    started = time.perf_counter()
    configs = []
    config_errors = []
    for index, path in enumerate(config_paths):
        try:
            config = _load_config(path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            config_errors.append(str(exc))
            continue
        configs.append((path, config))
        expected = (
            EXPECTED_CONDITIONS[index]
            if index < len(EXPECTED_CONDITIONS)
            else "invalid"
        )
        config_errors.extend(_check_config(path, config, expected))

    if len(config_paths) != 3:
        config_errors.append(
            "Gate 3 requires exactly three configs in E1, E2, E3 order"
        )
    if len(configs) == 3:
        baseline = _normalized_non_reward(configs[0][1])
        if any(_normalized_non_reward(config) != baseline for _, config in configs[1:]):
            config_errors.append(
                "E1, E2, and E3 differ outside experiment metadata and rewards"
            )

    local_gate = _local_gate() if run_local_gate else {"status": "not_tested"}
    if local_gate["status"] == "fail":
        config_errors.append("the local project gate failed")

    current_hashes = source_hashes()
    repo_path = (
        _get(configs[0][1], "training.env_server.repo_path") if configs else None
    )
    current_stack = collect_env_stamp(repo_path)
    config_status = (
        "pass" if not config_errors and local_gate["status"] == "pass" else "fail"
    )
    if local_gate["status"] == "not_tested" and not config_errors:
        config_status = "not_tested"

    contracts = [
        {
            "name": "gate1_offline_contract",
            "status": config_status,
            "evidence": {
                "configs": [
                    {
                        "path": str(path),
                        "experiment_id": config.get("experiment_id"),
                        "condition": _condition(config),
                        "sha256": _canonical_hash(config),
                    }
                    for path, config in configs
                ],
                "errors": config_errors,
                "local_gate": local_gate,
            },
        }
    ]

    next_phase = "fix_gate1"
    blocked = config_status != "pass"
    total_gpu_seconds = 0.0
    for index, (path, config) in enumerate(configs):
        exp_id = config.get("experiment_id")
        run_dir = runs_root / str(exp_id)
        if blocked:
            status = "not_tested"
            errors = ["blocked by an earlier contract"]
        elif not run_dir.exists():
            status = "not_tested"
            errors = []
            next_phase = f"run_{EXPECTED_CONDITIONS[index].lower()}:{exp_id}"
            blocked = True
        else:
            errors = _assess_run(run_dir, config, current_hashes, current_stack)
            status = "fail" if errors else "pass"
            metadata_path = run_dir / CAPTURE_DIR / "metadata.json"
            if metadata_path.is_file():
                metadata = _read_json(metadata_path)
                total_gpu_seconds += float(metadata.get("duration_seconds", 0.0))
            if total_gpu_seconds > 7200:
                errors.append(
                    "the cumulative Gate 2-3 GPU time exceeded the 7200 second ceiling"
                )
                status = "fail"
            if errors:
                next_phase = f"diagnose_{EXPECTED_CONDITIONS[index].lower()}:{exp_id}"
                blocked = True
        contracts.append(
            {
                "name": f"gate3_{EXPECTED_CONDITIONS[index].lower()}_execution",
                "status": status,
                "evidence": {
                    "config": str(path),
                    "run_dir": str(run_dir),
                    "errors": errors,
                },
            }
        )

    if not blocked and len(configs) == 3:
        if total_gpu_seconds > 7200:
            contracts.append(
                {
                    "name": "gate2_3_gpu_time_ceiling",
                    "status": "fail",
                    "evidence": {
                        "seconds": total_gpu_seconds,
                        "ceiling_seconds": 7200,
                    },
                }
            )
            next_phase = "diagnose_gpu_time_ceiling"
        else:
            contracts.append(
                {
                    "name": "gate2_3_gpu_time_ceiling",
                    "status": "pass",
                    "evidence": {
                        "seconds": total_gpu_seconds,
                        "ceiling_seconds": 7200,
                    },
                }
            )
            next_phase = "gate4_e1_pilot"

    statuses = [contract["status"] for contract in contracts]
    overall = (
        "fail"
        if "fail" in statuses
        else "not_tested"
        if "not_tested" in statuses
        else "pass"
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": overall,
        "next_phase": next_phase,
        "measured_runtime_seconds": round(time.perf_counter() - started, 3),
        "gpu_capture_seconds": round(total_gpu_seconds, 3),
        "source_hashes": current_hashes,
        "source_sha256": _canonical_hash(current_hashes),
        "stack": current_stack,
        "stack_sha256": _canonical_hash(current_stack),
        "contracts": contracts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate E1/E2/E3 readiness and admit one next phase"
    )
    parser.add_argument(
        "configs", nargs=3, type=Path, help="E1 E2 E3 configs, in order"
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--report", type=Path, default=Path("runs/readiness_gate3.json")
    )
    args = parser.parse_args()

    report = build_report(args.configs, args.runs_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.report, report)
    print(f"Readiness: {report['status']}")
    print(f"Next phase: {report['next_phase']}")
    print(f"Report: {args.report}")
    if report["status"] == "fail":
        raise SystemExit(1)
    if report["status"] == "not_tested":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

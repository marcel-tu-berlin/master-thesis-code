"""CPU arithmetic replay through the installed trainer, without policy generation.

Uses saved loss inputs and labelled E3 endings. Only generation and the model
forward are fixtures; reward dispatch, centering, decoding and loss are native
TRL methods. Gradients are with respect to selected-token log probabilities,
not model parameters. The live diagnostic supplies the parameter-update proof.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from probes import readiness

LEGACY_READINESS_SHA = (
    "101d5df300e790bd1d4c0b65e8cd458b4757614dc292b927a84c1d1747868983"
)
LEGACY_CAPTURE_AST_SHA = (
    "fde7df6f07c43b1b9eb4cfc19e38ed62756a46315aa02c6e92775c845a596d0d"
)
LOSS_TOLERANCE = 2e-5
GRADIENT_TOLERANCE = 5e-7


def capture_sources_match(
    recorded: dict, current: dict, approved_replay_sha: str | None = None
) -> bool:
    """Accept the initial captures only for an assessor-only source change.

    Every other file and every AST node outside _assess_capture must still match
    the original capture implementation. No recorder or runtime code is exempted.
    """
    key = "pipeline/probes/readiness.py"
    replay_key = "pipeline/probes/readiness_replay.py"
    if (
        approved_replay_sha is None
        or current.get(replay_key) != approved_replay_sha
        or readiness._file_hash(Path(__file__)) != approved_replay_sha
    ):
        return False
    if recorded == current:
        return True
    if recorded.get(key) != LEGACY_READINESS_SHA:
        return False
    if {k: v for k, v in recorded.items() if k != key} != {
        k: v for k, v in current.items() if k not in (key, replay_key)
    }:
        return False
    path = Path(readiness.__file__)
    if current.get(key) != readiness._file_hash(path):
        return False
    tree = ast.parse(path.read_text())
    # The only other allowed delta is adding this review dependency to the
    # source manifest. Keep all the original hashing and capture nodes intact.
    addition = ast.parse(
        'paths.append(root / "pipeline/probes/readiness_replay.py")'
    ).body[0]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "source_hashes":
            node.body = [
                item for item in node.body if ast.dump(item) != ast.dump(addition)
            ]
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.FunctionDef) and node.name == "_assess_capture")
    ]
    digest = hashlib.sha256(ast.dump(tree, include_attributes=False).encode())
    return digest.hexdigest() == LEGACY_CAPTURE_AST_SHA


def _trainer(settings: dict):
    """Build a single-process CPU harness around unmodified TRL methods."""
    import torch
    from accelerate import Accelerator
    from transformers import AutoTokenizer, TrainerState
    from trl import GRPOConfig, GRPOTrainer

    from training.registry import get_model_config

    trainer = object.__new__(GRPOTrainer)
    trainer.accelerator = Accelerator(cpu=True)
    if trainer.accelerator.num_processes != 1:
        raise ValueError("readiness replay requires one process")
    trainer.model = SimpleNamespace(training=True, is_gradient_checkpointing=False)
    trainer.args = SimpleNamespace(
        delta=None,
        report_to=[],
        gradient_checkpointing_kwargs=None,
        steps_per_generation=32,
        gradient_accumulation_steps=32,
        per_device_train_batch_size=1,
    )
    trainer.state = TrainerState()
    for name in ("epsilon_low", "epsilon_high", "loss_type", "scale_rewards"):
        setattr(trainer, name, settings[name])
    trainer.importance_sampling_level = settings["importance_sampling_level"]
    trainer.beta = 0.0
    trainer.use_vllm = True
    trainer.vllm_importance_sampling_mode = "token_truncate"
    for name in (
        "top_entropy_quantile",
        "off_policy_mask_threshold",
        "vllm_importance_sampling_correction",
        "vllm_importance_sampling_clip_min",
        "vllm_importance_sampling_clip_max",
    ):
        setattr(trainer, name, GRPOConfig.__dataclass_fields__[name].default)
    trainer.num_iterations = 1
    trainer.num_generations = 8
    trainer.multi_objective_aggregation = "sum_then_normalize"
    trainer.reward_weights = torch.ones(1)
    trainer.reward_processing_classes = [None]
    trainer.reward_func_names = ["naive_sum_composer"]
    trainer._metrics = {"train": defaultdict(list)}
    trainer._logs = defaultdict(list)
    trainer._logs["rewards"] = defaultdict(list)
    trainer._logs["extra"] = defaultdict(list)
    trainer._pending_extra_logs = defaultdict(list)
    trainer._pending_metrics = defaultdict(list)
    trainer.processing_class = AutoTokenizer.from_pretrained(
        get_model_config(str(readiness.RECIPE_FIELDS["model.slug"]))["model_name"],
        revision=readiness.RECIPE_FIELDS["model.revision"],
        local_files_only=True,
    )
    trainer._tokenizer = trainer.processing_class
    trainer._is_vlm = False
    trainer.pad_to_multiple_of = None
    trainer.mask_truncated_completions = False
    trainer.tools = []
    return trainer


def _validate_token_arrays(arrays: dict, keys: tuple[str, ...], rows: int) -> None:
    """Reject misaligned token evidence before indexing or building tensors."""
    if rows < 1:
        raise ValueError("replay token arrays must contain at least one row")
    width = None
    for key in keys:
        value = arrays[key]
        if not isinstance(value, list) or len(value) != rows:
            raise ValueError(f"replay {key} must contain {rows} rows")
        for row in value:
            if not isinstance(row, list) or not row:
                raise ValueError(f"replay {key} contains an invalid token row")
            if width is None:
                width = len(row)
            if len(row) != width:
                raise ValueError(f"replay {key} has misaligned token rows")


def _loss_check(trainer, record: dict) -> dict:
    """Compare native loss/autograd to scalar DAPO and its analytic derivative."""
    import torch

    rows = len(record["per_token_logps"])
    _validate_token_arrays(
        dict(record["inputs"], per_token_logps=record["per_token_logps"]),
        (
            "completion_mask",
            "tool_mask",
            "old_per_token_logps",
            "importance_sampling_ratio",
            "per_token_logps",
        ),
        rows,
    )
    if len(record["inputs"]["advantages"]) != rows:
        raise ValueError("replay advantages do not align with token rows")
    inputs = {key: torch.tensor(value) for key, value in record["inputs"].items()}
    inputs["prompt_ids"] = torch.ones((rows, 1), dtype=torch.long)
    inputs["prompt_mask"] = torch.ones((rows, 1), dtype=torch.long)
    inputs["completion_ids"] = torch.ones_like(inputs["completion_mask"])
    logps = torch.tensor(record["per_token_logps"], requires_grad=True)
    trainer._get_per_token_logps_and_entropies = lambda *_a, **_k: (
        logps,
        torch.zeros_like(logps),
    )
    loss = trainer._compute_loss(trainer.model, inputs)
    loss.backward()
    gradient = logps.grad
    if gradient is None:
        raise ValueError("native loss is disconnected from selected-token logps")
    expected_loss = readiness._reference_dapo_loss(record)
    max_error = 0.0
    for row, advantage in enumerate(record["inputs"]["advantages"]):
        for col, current in enumerate(record["per_token_logps"][row]):
            old = record["inputs"]["old_per_token_logps"][row][col]
            coefficient = math.exp(current - old)
            active = record["inputs"]["completion_mask"][row][col]
            active *= record["inputs"]["tool_mask"][row][col]
            clipped = (advantage > 0 and coefficient > 1 + trainer.epsilon_high) or (
                advantage < 0 and coefficient < 1 - trainer.epsilon_low
            )
            expected = 0.0
            if active and not clipped:
                ratio = record["inputs"]["importance_sampling_ratio"][row][col]
                expected = (
                    -advantage
                    * coefficient
                    * ratio
                    / record["inputs"]["num_items_in_batch"]
                )
            actual = float(gradient[row, col])
            if not math.isfinite(actual):
                raise ValueError("non-finite native replay gradient")
            max_error = max(max_error, abs(actual - expected))
    error = abs(float(loss.detach()) - expected_loss)
    if not math.isfinite(error) or error > LOSS_TOLERANCE:
        raise ValueError(f"native replay loss error {error}")
    if max_error > GRADIENT_TOLERANCE:
        raise ValueError(f"native replay gradient error {max_error}")
    return {"loss_error": error, "gradient_max_error": max_error}


def _scripted_e3(trainer) -> list[dict]:
    """Exercise eight labelled endings per group at lambda 0, 0.5 and 1."""
    import torch

    from training.rewards.compose import NaiveSumComposer
    from training.rewards.env_reward import EnvReward
    from training.rewards.non_termination import NonTerminationPenalty

    # Explicit labels are independent of the implementation being tested.
    cases = [
        ("env_done", True, "tool", 7, 0.0),
        ("failed_env_done", True, "tool", 9, 0.0),
        ("voluntary_stop", False, "assistant", 11, 0.0),
        ("turn_cap", False, "tool", 13, -1.0),
        ("undispatched_tool", False, "call", 15, -1.0),
        ("below_token_cap", False, "assistant", 4095, 0.0),
        ("token_cap", False, "assistant", 4096, -1.0),
        ("done_at_cap", True, "tool", 4096, 0.0),
    ]
    completions, ids, masks, envs, expected_penalty, task_rewards = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for group in range(4):
        for slot, (_, done, ending, length, penalty) in enumerate(cases):
            message: dict = {"role": "assistant" if ending == "call" else ending}
            if ending == "call":
                message["tool_calls"] = [{"function": {"name": "click"}}]
            completions.append([message])
            ids.append([1] * length)
            masks.append([0 if i % 5 == 1 else 1 for i in range(length)])
            reward = float(done and slot != 1 and group != 0)
            task_rewards.append(reward)
            expected_penalty.append(penalty)
            envs.append(
                SimpleNamespace(done=done, reward=reward, reset=lambda **_: None)
            )
    trainer.environments = envs
    sampling = [[-1.5 if i % 3 else -2.5 for i in range(len(row))] for row in ids]
    normalizer = sum(sum(row) for row in masks)
    trainer._generate = lambda _: (
        [[1, 2]] * 32,
        ids,
        masks,
        completions,
        normalizer,
        sampling,
        {},
        None,
        None,
    )
    results = []
    for dose in (0.0, 0.5, 1.0):
        trainer.reward_funcs = [
            NaiveSumComposer([(EnvReward(), 1.0), (NonTerminationPenalty(4096), dose)])
        ]
        trainer._get_per_token_logps_and_entropies = lambda *_a, **_k: (
            torch.full((32, 4096), -1.0),
            None,
        )
        inputs = [
            {"prompt": [{"role": "user", "content": f"group {i // 8}"}]}
            for i in range(32)
        ]
        prepared = trainer._generate_and_score_completions(copy.deepcopy(inputs))
        expected_rewards = [
            r + dose * p for r, p in zip(task_rewards, expected_penalty, strict=True)
        ]
        expected_advantages = readiness._centered(expected_rewards, 8)
        actual = prepared["advantages"].tolist()
        if any(
            abs(a - b) > 2e-5 for a, b in zip(actual, expected_advantages, strict=True)
        ):
            raise ValueError(f"scripted E3 advantage mismatch at lambda {dose}")
        logged = trainer._logs["rewards"]["naive_sum_composer"][-32:]
        if logged != expected_rewards:
            raise ValueError(f"scripted E3 reward alignment mismatch at lambda {dose}")
        # Changed logps exercise both PPO clipping sides, not just ratio == 1.
        current = [[-1.0 + (-0.4, 0.1, 0.4)[i % 3] for i in range(4096)]] * 32
        selected = (
            "advantages",
            "completion_mask",
            "tool_mask",
            "old_per_token_logps",
            "importance_sampling_ratio",
            "num_items_in_batch",
        )
        record = {
            "inputs": {key: readiness._jsonable(prepared[key]) for key in selected},
            "per_token_logps": current,
            "settings": {
                "loss_type": "dapo",
                "importance_sampling_level": "token",
                "epsilon_low": 0.2,
                "epsilon_high": 0.2,
                "beta": 0.0,
            },
        }
        result = _loss_check(trainer, record)
        result.update(lambda_value=dose, advantages=actual, rewards=logged)
        results.append(result)
    return results


def _captured_doses(
    trainer, capture: Path, prepared: dict, records: list
) -> list[dict]:
    """Replay saved component rewards at fixed doses and a within-group placebo.

    The components are captured inputs here. Their calculation is checked by the
    ordinary reward contract; this checks their transport, centering and gradients.
    """
    import random

    import torch

    from training.rewards.compose import NaiveSumComposer

    rewards = readiness._read_json(capture / "reward_batch.json")["records"]
    ids = [row["completion_ids"] for row in rewards]
    masks = [
        row[: len(tokens)]
        for row, tokens in zip(prepared["tool_mask"], ids, strict=True)
    ]
    sampling = [
        row[: len(tokens)]
        for row, tokens in zip(prepared["sampling_per_token_logps"], ids, strict=True)
    ]
    trainer.environments = None
    trainer._generate = lambda _: (
        [[1]] * 32,
        ids,
        masks,
        [row["completion"] for row in rewards],
        prepared["num_items_in_batch"],
        sampling,
        {},
        None,
        None,
    )
    task = [row["raw_rewards"]["env_reward"] for row in rewards]
    results = []
    for component in ("token_length", "non_termination"):
        raw = [row["raw_rewards"][component] for row in rewards]
        for label, dose in (
            ("zero", 0.0),
            ("half", 0.5),
            ("full", 1.0),
            ("placebo", 1.0),
        ):
            assigned = list(raw)
            if label == "placebo":
                rng = random.Random(4001)
                for start in range(0, 32, 8):
                    group = assigned[start : start + 8]
                    rng.shuffle(group)
                    assigned[start : start + 8] = group
            trainer.reward_funcs = [
                NaiveSumComposer(
                    [
                        (lambda *_a, **_k: task, 1.0),
                        (lambda *_a, _assigned=assigned, **_k: _assigned, dose),
                    ]
                )
            ]
            trainer._get_per_token_logps_and_entropies = lambda *_a, **_k: (
                torch.tensor(prepared["old_per_token_logps"]),
                None,
            )
            output = trainer._generate_and_score_completions(
                [{"prompt": row["prompt"]} for row in rewards]
            )
            expected = readiness._centered(
                [a + dose * b for a, b in zip(task, assigned, strict=True)], 8
            )
            actual = output["advantages"].tolist()
            error = max(abs(a - b) for a, b in zip(actual, expected, strict=True))
            if not math.isfinite(error) or error > LOSS_TOLERANCE:
                raise ValueError(
                    f"captured {component}/{label} advantage error {error}"
                )
            # Loss shards have been shuffled by TRL; bind each to its original
            # row using IDs and masks, never its shuffled shard position.
            checks = []
            for record in records:
                matches = [
                    i
                    for i in range(32)
                    if prepared["completion_ids"][i]
                    == record["inputs"]["completion_ids"][0]
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "captured loss shard does not identify one rollout"
                    )
                changed = copy.deepcopy(record)
                changed["inputs"]["advantages"] = [actual[matches[0]]]
                checks.append(_loss_check(trainer, changed))
            results.append(
                {
                    "component": component,
                    "case": label,
                    "advantage_max_error": error,
                    "loss_max_error": max(c["loss_error"] for c in checks),
                    "gradient_max_error": max(c["gradient_max_error"] for c in checks),
                }
            )
    return results


def run_replay(capture: Path, condition: str | None) -> dict:
    """Recompute evidence now; never accept a stored replay's pass flag."""
    from trl import GRPOTrainer

    if condition not in readiness.EXPECTED_CONDITIONS:
        raise ValueError(f"unknown replay condition: {condition}")
    settings = readiness._read_json(capture / "trainer_settings.json")
    trainer = _trainer(settings)
    prepared = readiness._read_json(capture / "rollout_batch.json")["prepared"]
    _validate_token_arrays(
        prepared,
        (
            "completion_ids",
            "completion_mask",
            "tool_mask",
            "old_per_token_logps",
            "sampling_per_token_logps",
            "importance_sampling_ratio",
        ),
        32,
    )
    denominator = sum(
        completion * tool
        for completions, tools in zip(
            prepared["completion_mask"], prepared["tool_mask"], strict=True
        )
        for completion, tool in zip(completions, tools, strict=True)
    )
    if denominator != prepared["num_items_in_batch"]:
        raise ValueError("captured DAPO denominator differs from assistant-token mask")
    ratio_error = 0.0
    for row in range(32):
        for col, old in enumerate(prepared["old_per_token_logps"][row]):
            mask = (
                prepared["completion_mask"][row][col] * prepared["tool_mask"][row][col]
            )
            expected = math.exp(
                (old - prepared["sampling_per_token_logps"][row][col]) * mask
            )
            if trainer.vllm_importance_sampling_clip_min is not None:
                expected = max(expected, trainer.vllm_importance_sampling_clip_min)
            expected = min(expected, trainer.vllm_importance_sampling_clip_max)
            actual = prepared["importance_sampling_ratio"][row][col]
            if not math.isfinite(actual):
                raise ValueError("non-finite captured importance-sampling ratio")
            ratio_error = max(ratio_error, abs(actual - expected))
    if ratio_error > LOSS_TOLERANCE:
        raise ValueError(f"captured importance-sampling ratio error {ratio_error}")
    records = readiness._read_jsonl(capture / "loss_shards.jsonl")
    checks = [_loss_check(trainer, record) for record in records]
    report = {
        "kind": "installed_trainer_cpu_arithmetic_replay",
        "condition": condition,
        "replay_source_sha256": readiness._file_hash(Path(__file__)),
        "trainer_source_sha256": readiness._file_hash(
            Path(inspect.getfile(GRPOTrainer))
        ),
        "capture_loss_sha256": readiness._file_hash(capture / "loss_shards.jsonl"),
        "loss_tolerance": LOSS_TOLERANCE,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "importance_sampling_ratio_max_error": ratio_error,
        "importance_sampling_clip_max": trainer.vllm_importance_sampling_clip_max,
        "assistant_token_denominator": denominator,
        "captured_shards": checks,
        "captured_doses": _captured_doses(trainer, capture, prepared, records),
        "scripted_e3": _scripted_e3(trainer) if condition == "E3" else [],
    }
    return report

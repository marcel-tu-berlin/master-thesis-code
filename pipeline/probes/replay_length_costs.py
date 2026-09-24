"""CPU-only reward/tokenizer/native-advantage replay of decision 0019 candidates."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from probes.compare_length_costs import centered, reference_costs
from probes.readiness_replay import _loss_check, _trainer


def replay():
    import torch
    import yaml

    from training.rewards.compose import build_composer
    from training.rewards.utils import model_token_count
    from training.train import build_reward_components

    config = yaml.safe_load(Path("runs/table-pilot-e1-s4009/config.yaml").read_text())
    settings = json.loads(
        Path(
            "runs/readiness-g3-e2-s4010-feedback-v2/readiness/trainer_settings.json"
        ).read_text()
    )
    trainer = _trainer(settings)
    assert trainer.accelerator.device.type == "cpu" and not torch.cuda.is_available()
    runner = SimpleNamespace(
        tokenizer=trainer.processing_class, completion_budget=lambda: 4096
    )
    capture = Path("runs/table-pilot-e1-s4009/group_observations.jsonl")
    batches = [json.loads(line) for line in capture.read_text().splitlines()]
    for batch in batches:
        for row in batch["records"]:
            assert (
                model_token_count(row["completion"], runner.tokenizer)
                == row["model_tokens"]
            )
    checks = []
    for kind in ("linear", "relative"):
        for weight in (0.1, 0.2, 0.4):
            cfg = copy.deepcopy(config)
            cfg["rewards"]["successful_length"] = {
                "enabled": True,
                "kind": kind,
                "weight": weight,
                "max_len": 4096,
            }
            components = build_reward_components(cfg, None, runner)
            assert [(type(f).__name__, w) for f, w in components] == [
                ("EnvReward", 1.0),
                ("SuccessfulLengthPenalty", weight),
            ]
            trainer.reward_funcs = [build_composer(components, "naive_sum", 8)]
            for batch in batches:
                rows = batch["records"]
                trainer.environments = [
                    SimpleNamespace(
                        reward=float(r["correct"]),
                        done=r["done"],
                        reset=lambda **_: None,
                    )
                    for r in rows
                ]
                trainer._generate = lambda _, rs=rows: (
                    [[1]] * 32,
                    [[2, 3]] * 32,
                    [[1, 0]] * 32,
                    [r["completion"] for r in rs],
                    32,
                    [[0.0, 0.0]] * 32,
                    {},
                    None,
                    None,
                )
                trainer._get_per_token_logps_and_entropies = lambda *_a, **_k: (
                    torch.zeros((32, 2)),
                    None,
                )
                output = trainer._generate_and_score_completions(
                    [{"prompt": r["prompt"], "seed": r["seed"]} for r in rows]
                )
                expected = []
                for start in range(0, 32, 8):
                    group = rows[start : start + 8]
                    costs = reference_costs(
                        [r["model_tokens"] for r in group],
                        [r["correct"] for r in group],
                        kind,
                    )
                    expected.extend(
                        centered(
                            [
                                int(r["correct"]) - weight * c
                                for r, c in zip(group, costs, strict=True)
                            ]
                        )
                    )
                error = max(
                    abs(a - b)
                    for a, b in zip(
                        output["advantages"].tolist(), expected, strict=True
                    )
                )
                assert error < 3e-7
                loss = _loss_check(
                    trainer,
                    {
                        "settings": settings,
                        "per_token_logps": [[0.0, 0.0]] * 32,
                        "inputs": {
                            "completion_mask": [[1, 1]] * 32,
                            "tool_mask": [[1, 0]] * 32,
                            "old_per_token_logps": [[0.0, 0.0]] * 32,
                            "importance_sampling_ratio": [[1.0, 1.0]] * 32,
                            "advantages": output["advantages"].tolist(),
                            "num_items_in_batch": 32,
                        },
                    },
                )
                checks.append(
                    {
                        "kind": kind,
                        "weight": weight,
                        "step": batch["optimizer_step"],
                        "advantage_max_error": error,
                        **loss,
                    }
                )
    return {
        "status": "pass",
        "policy_loaded": False,
        "gpu_used": False,
        "capture_sha256": hashlib.sha256(capture.read_bytes()).hexdigest(),
        "trajectories_retokenized": sum(len(b["records"]) for b in batches),
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay()
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f"PASS: {len(result['checks'])} native batch replays, CPU only")


if __name__ == "__main__":
    main()

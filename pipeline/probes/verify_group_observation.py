"""Qualify only the observer delta against the admitted Gate 3 source and data."""

import ast
import hashlib
import json
import random
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

from probes.readiness import _canonical_hash, source_hashes
from probes.readiness_replay import _trainer
from training.group_observation import GroupObservation
from training.rewards.compose import NaiveSumComposer
from training.rewards.env_reward import EnvReward

BASELINE_TRAIN_AST_SHA = (
    "b4dd9ccb9104da9b7277c9003e85ae04ab64a00926744d87b4fbcb23ca5482b4"
)


def main():
    import numpy as np
    import torch

    admission = json.loads(Path("runs/readiness_gate3.json").read_text())
    assert admission["status"] == "pass"
    assert admission["next_phase"] == "gate4_e1_pilot"
    baseline = admission["source_hashes"]
    assert (
        _canonical_hash(baseline)
        == admission["source_sha256"]
        == ("9cb8ac80183cb2329b2ec789d61d3349110b8e1af823ed4f2eb6439436a062cd")
    )
    current = source_hashes()
    observer_key = "pipeline/training/group_observation.py"
    train_key = "pipeline/training/train.py"
    assert set(current) - set(baseline) == {observer_key}
    assert not set(baseline) - set(current)
    assert {k for k in baseline if baseline[k] != current[k]} == {train_key}
    additions = [
        """parser.add_argument(
            "--observe-groups", action="store_true",
            help="Save diagnostic prompt-groups at thirds and the last ten updates",
        )""",
        """if args.observe_groups:
            from training.group_observation import GroupObservation
            group_observation = GroupObservation(run_dir, config, domain, runner)
            trainer_reward_fn = group_observation.wrap_reward(trainer_reward_fn)
        """,
        """if args.observe_groups:
            group_observation.finish()
        """,
    ]
    expected = {ast.dump(ast.parse(s).body[0]): 0 for s in additions}

    class RemoveObserver(ast.NodeTransformer):
        def visit(self, node):
            key = ast.dump(node)
            if key in expected:
                expected[key] += 1
                return None
            return super().visit(node)

    projected = RemoveObserver().visit(ast.parse(Path("training/train.py").read_text()))
    assert all(count == 1 for count in expected.values())
    assert (
        hashlib.sha256(ast.dump(projected).encode()).hexdigest()
        == BASELINE_TRAIN_AST_SHA
    )

    capture = Path("runs/readiness-g3-e2-s4001-r2/readiness")
    records = json.loads((capture / "reward_batch.json").read_text())["records"]
    settings = json.loads((capture / "trainer_settings.json").read_text())
    trainer = _trainer(settings)
    trainer.state.global_step = (
        299  # Injected late step; these are saved Gate 3 inputs.
    )
    trainer.environments = [SimpleNamespace(**r["environment"]) for r in records]
    config = yaml.safe_load(
        Path("configs/archive/development-2026-09-24/readiness/g4-e1.yaml").read_text()
    )
    runner = SimpleNamespace(
        tokenizer=trainer.processing_class, completion_budget=lambda: 4096
    )
    prompts = [r["prompt"] for r in records]
    completions = [r["completion"] for r in records]
    ids = [r["completion_ids"] for r in records]
    inputs = [{"prompt": r["prompt"], "seed": r["seed"]} for r in records]
    arguments = (inputs, prompts, completions, ids)
    before_inputs = json.dumps(arguments, sort_keys=True)
    reward = NaiveSumComposer([(EnvReward(), 1.0)])
    trainer.reward_funcs = [reward]
    baseline_reward = trainer._calculate_rewards(*arguments)
    with tempfile.TemporaryDirectory(prefix="gate4-observer-proof-") as directory:
        observer = GroupObservation(directory, config, None, runner)
        trainer.reward_funcs = [observer.wrap_reward(reward)]
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        started = time.perf_counter()
        observed_reward = trainer._calculate_rewards(*arguments)
        elapsed = time.perf_counter() - started
        assert torch.equal(baseline_reward, observed_reward)
        assert json.dumps(arguments, sort_keys=True) == before_inputs
        assert random.getstate() == rng[0]
        assert np.array_equal(np.random.get_state()[1], rng[1][1])
        assert np.random.get_state()[2:] == rng[1][2:]
        assert torch.equal(torch.get_rng_state(), rng[2])
        observed = json.loads(observer.path.read_text())
        assert observed["optimizer_step"] == 300
        assert len(observed["records"]) == 32
        for actual, saved in zip(observed["records"], records, strict=True):
            assert actual["raw_rewards"] == saved["raw_rewards"]
            assert actual["seed"] == saved["seed"]
            assert actual["composed_reward"] == saved["raw_rewards"]["env_reward"]
    qualification_sources = {
        **current,
        "pipeline/probes/verify_group_observation.py": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "pipeline/tests/test_group_observation.py": hashlib.sha256(
            Path("tests/test_group_observation.py").read_bytes()
        ).hexdigest(),
    }
    print(
        json.dumps(
            {
                "status": "pass",
                "qualification": "Gate 3 baseline plus opt-in group observation; not a new Gate 3 capture",
                "baseline_source_sha256": admission["source_sha256"],
                "source_sha256": _canonical_hash(current),
                "config_sha256": _canonical_hash(config),
                "source_hashes": current,
                "qualification_source_hashes": qualification_sources,
                "qualification_sha256": _canonical_hash(qualification_sources),
                "unchanged_training_ast_without_observer": True,
                "captured_fixture": str(capture),
                "injected_optimizer_step": 300,
                "native_reward_equal": True,
                "diagnostic_rewards_equal_to_saved_capture": True,
                "inputs_and_rng_unchanged": True,
                "observed_batch_seconds": elapsed,
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

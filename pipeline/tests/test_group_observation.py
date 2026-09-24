"""Bounded pilot observation must not change the reward or its inputs."""

import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from training.group_observation import GroupObservation
from training.rewards.compose import NaiveSumComposer
from training.rewards.env_reward import EnvReward


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


def config():
    return yaml.safe_load(
        Path("configs/archive/development-2026-09-24/readiness/g4-e1.yaml").read_text()
    )


def batch(step):
    completions = [
        [{"role": "assistant", "content": "x" * (i % 8 + 1)}] for i in range(32)
    ]
    completions[7].append({"role": "tool", "content": "turn limit"})
    return {
        "prompts": [[{"role": "user", "content": "task"}]] * 32,
        "completions": completions,
        "completion_ids": [[1] * 20 for _ in range(32)],
        "environments": [
            SimpleNamespace(reward=float(i % 8 < 4), done=i % 8 < 4) for i in range(32)
        ],
        "seed": [4002000000 + i // 8 for i in range(32)],
        "trainer_state": SimpleNamespace(global_step=step - 1),
    }


def observer(path, cfg=None):
    runner = SimpleNamespace(tokenizer=Tokenizer(), completion_budget=lambda: 4096)
    return GroupObservation(path, cfg or config(), None, runner)


def test_observation_is_transparent_bounded_and_preserves_cost_variation(tmp_path):
    record = observer(tmp_path)
    returned = []
    composer = NaiveSumComposer([(EnvReward(), 1.0)])

    def reward(prompts, completions, **kwargs):
        result = composer(prompts, completions, **kwargs)
        returned.append(result)
        return result

    wrapped = record.wrap_reward(reward)
    assert wrapped.__name__ == reward.__name__
    rng = random.getstate()
    for step in range(1, 301):
        inputs = batch(step)
        before = copy.deepcopy(inputs)
        result = wrapped(**inputs)
        assert result is returned[-1]
        assert result == [env.reward for env in inputs["environments"]]
        assert inputs == before
    assert len(returned) == 300
    assert random.getstate() == rng
    record.finish()
    rows = [json.loads(line) for line in record.path.read_text().splitlines()]
    assert [r["optimizer_step"] for r in rows] == [1, 100, 200, *range(291, 301)]
    assert all(len(r["records"]) == 32 for r in rows)
    group = rows[-1]["records"][:8]
    assert [r["slot_index"] for r in group] == list(range(8))
    assert len({r["seed"] for r in group}) == 1
    assert [r["model_tokens"] for r in group] == list(range(1, 9))
    assert (
        group[0]["raw_rewards"]["token_length"]
        > group[3]["raw_rewards"]["token_length"]
    )
    assert group[6]["raw_rewards"]["non_termination"] == 0
    assert group[7]["raw_rewards"]["non_termination"] == -1


def test_observation_rejects_missing_duplicate_misaligned_and_old_evidence(tmp_path):
    record = observer(tmp_path)
    wrapped = record.wrap_reward(NaiveSumComposer([(EnvReward(), 1.0)]))
    bad = batch(1)
    bad["seed"][1] += 1
    with pytest.raises(ValueError, match="seed"):
        wrapped(**bad)
    assert record.path.read_text() == ""
    wrapped(**batch(1))
    with pytest.raises(ValueError, match="already observed"):
        wrapped(**batch(1))
    with pytest.raises(ValueError, match="Missing"):
        record.finish()
    with pytest.raises(FileExistsError):
        observer(tmp_path)


def test_native_observation_keeps_advantages_loss_gradients_and_rng(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    import numpy as np
    from probes.readiness_replay import _trainer

    trainer = _trainer(
        {
            "epsilon_low": 0.2,
            "epsilon_high": 0.2,
            "loss_type": "dapo",
            "scale_rewards": "none",
            "importance_sampling_level": "token",
        }
    )
    fixture = batch(300)
    trainer.state.global_step = 299
    trainer.environments = fixture["environments"]
    for environment in trainer.environments:
        environment.reset = lambda **_kwargs: "fixed task observation"
    ids = fixture["completion_ids"]
    trainer._generate = lambda _: (
        [[1]] * 32,
        ids,
        [[1] * 19 + [0]] * 32,
        fixture["completions"],
        32 * 19,
        [[-1.0] * 20] * 32,
        {},
        None,
        None,
    )
    inputs = [
        {"prompt": prompt, "seed": seed}
        for prompt, seed in zip(fixture["prompts"], fixture["seed"], strict=True)
    ]
    record = observer(tmp_path)
    outputs = []
    gradients = []
    losses = []
    for observe in (False, True):
        reward = NaiveSumComposer([(EnvReward(), 1.0)])
        trainer.reward_funcs = [record.wrap_reward(reward) if observe else reward]
        trainer._get_per_token_logps_and_entropies = lambda *_a, **_k: (
            torch.full((32, 20), -1.0),
            None,
        )
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        output = trainer._generate_and_score_completions(copy.deepcopy(inputs))
        assert random.getstate() == before[0]
        assert np.array_equal(np.random.get_state()[1], before[1][1])
        assert np.random.get_state()[2:] == before[1][2:]
        assert torch.equal(torch.get_rng_state(), before[2])
        outputs.append(output)
        logps = torch.full((32, 20), -0.9, requires_grad=True)
        trainer._get_per_token_logps_and_entropies = lambda *_a, **_k: (
            logps,
            torch.zeros_like(logps),
        )
        loss = trainer._compute_loss(trainer.model, output)
        losses.append(loss.detach())
        gradients.append(torch.autograd.grad(loss, logps)[0])
    for key in outputs[0]:
        if isinstance(outputs[0][key], torch.Tensor):
            assert torch.equal(outputs[0][key], outputs[1][key]), key
        else:
            assert outputs[0][key] == outputs[1][key], key
    assert torch.equal(losses[0], losses[1])
    assert torch.equal(gradients[0], gradients[1])
    assert torch.count_nonzero(gradients[1]) > 0
    saved = json.loads(record.path.read_text())
    assert saved["optimizer_step"] == 300
    assert any(r["raw_rewards"]["non_termination"] == -1 for r in saved["records"])

"""Observe fixed pilot batches without changing rewards or invoking generation.

The synchronous reward caller owns each observer and its recorded-step set.
Input/group mismatches, incomplete observation and existing output fail loudly.
Only selected batches are serialized; files are closed after each write.
"""

import json
from pathlib import Path

from domains.env_base import CORRECT_REWARD_THRESHOLD
from training.rewards import REWARD_REGISTRY
from training.rewards.utils import model_token_count

LATE_UPDATES = 10


class GroupObservation:
    """Save the first batch, thirds, and last ten updates of a declared pilot."""

    def __init__(self, run_dir, config, domain, runner) -> None:
        training = config["training"]
        steps = training["max_steps"]
        if steps < 3 or steps % 3 or training.get("num_iterations", 1) != 1:
            raise ValueError("Group observation requires thirds and num_iterations=1")
        self.group_size = training["n_rollouts"]
        self.batch_size = training["batch_size"] * self.group_size
        self.steps = {1, steps // 3, steps * 2 // 3} | set(
            range(max(1, steps - LATE_UPDATES + 1), steps + 1)
        )
        self.recorded: set[int] = set()
        self.tokenizer = runner.tokenizer
        self.components = {
            name: builder(domain, runner, training, config["rewards"].get(name) or {})
            for name, (_, _, builder) in REWARD_REGISTRY.items()
        }
        self.path = Path(run_dir) / "group_observations.jsonl"
        self.path.touch(exist_ok=False)

    def wrap_reward(self, reward_fn):
        """Return the original reward result after observing selected batches."""

        def observed_reward(prompts, completions, **kwargs):
            total = reward_fn(prompts, completions, **kwargs)
            step = kwargs["trainer_state"].global_step + 1
            if step in self.steps:
                self._record(step, prompts, completions, total, kwargs)
            return total

        observed_reward.__name__ = reward_fn.__name__
        return observed_reward

    def _record(self, step, prompts, completions, total, kwargs) -> None:
        if step in self.recorded:
            raise ValueError(f"Optimizer step {step} already observed")
        seeds = kwargs["seed"]
        environments = kwargs["environments"]
        ids = kwargs["completion_ids"]
        if any(
            len(values) != self.batch_size
            for values in (prompts, completions, total, seeds, environments, ids)
        ):
            raise ValueError("Observation does not contain one full rollout batch")
        for start in range(0, self.batch_size, self.group_size):
            if len(set(seeds[start : start + self.group_size])) != 1:
                raise ValueError("Observation seed order does not match prompt-groups")
        raw = {
            name: component(prompts, completions, **kwargs)
            for name, component in self.components.items()
        }
        records = [
            {
                "group_index": i // self.group_size,
                "slot_index": i % self.group_size,
                "seed": seeds[i],
                "prompt": prompts[i],
                "completion": completion,
                "trajectory_tokens": len(ids[i]),
                "model_tokens": model_token_count(completion, self.tokenizer),
                "correct": raw["env_reward"][i] >= CORRECT_REWARD_THRESHOLD,
                "done": bool(environments[i].done),
                "raw_rewards": {name: values[i] for name, values in raw.items()},
                "composed_reward": total[i],
            }
            for i, completion in enumerate(completions)
        ]
        line = json.dumps(
            {
                "schema_version": 1,
                "optimizer_step": step,
                "global_step_before_update": step - 1,
                "group_size": self.group_size,
                "records": records,
            },
            allow_nan=False,
        )
        with self.path.open("a") as handle:
            handle.write(line + "\n")
        self.recorded.add(step)

    def finish(self) -> None:
        """Require every planned observation before declaring training complete."""
        missing = self.steps - self.recorded
        if missing:
            raise ValueError(
                f"Missing group observations at optimizer steps {sorted(missing)}"
            )

"""Cost records distinguish inference from environment time and survive failure."""

import json

import pytest

from eval import agentic_eval
from eval.metrics import SampleResult, compute_metrics
from training import cost


def test_phase_attempts_preserve_failures_and_allocation(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    clock = iter([100.0, 3700.0, 5000.0, 6800.0])
    monkeypatch.setattr(cost.time, "perf_counter", lambda: next(clock))
    with (
        pytest.raises(RuntimeError, match="failed model"),
        cost.measure_phase(tmp_path, "train"),
    ):
        assert json.loads((tmp_path / "costs.jsonl").read_text())["event"] == "start"
        raise RuntimeError("failed model")
    with cost.measure_phase(tmp_path, "train"):
        pass
    records = [
        json.loads(line) for line in (tmp_path / "costs.jsonl").read_text().splitlines()
    ]
    assert [r["event"] for r in records] == ["start", "end", "start", "end"]
    assert records[0]["attempt_id"] == records[1]["attempt_id"]
    assert records[0]["attempt_id"] != records[2]["attempt_id"]
    assert records[1]["status"] == "failed"
    assert records[1]["error_type"] == "RuntimeError"
    assert records[1]["allocated_gpu_hours"] == 1
    assert records[3]["status"] == "complete"
    assert records[3]["wall_seconds"] == 1800
    assert records[3]["allocated_gpu_hours"] == 0.5


def test_unknown_allocation_does_not_invent_gpu_hours(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with cost.measure_phase(tmp_path, "eval"):
        pass
    end = json.loads((tmp_path / "costs.jsonl").read_text().splitlines()[-1])
    assert end["wall_seconds"] >= 0
    assert end["allocated_gpu_hours"] is None


@pytest.mark.parametrize("multi_turn", [False, True])
def test_episode_cost_excludes_environment_from_inference(multi_turn, monkeypatch):
    now = 0.0

    def spend(seconds):
        nonlocal now
        now += seconds

    monkeypatch.setattr(agentic_eval.time, "perf_counter", lambda: now)

    class Env:
        def reset(self, seed):
            spend(2)
            self.reward, self.done, self.moves = 0.0, False, 0
            return "question"

        def answer(self, answer=""):
            spend(3)
            self.moves += 1
            self.done = not multi_turn or self.moves == 2
            self.reward = float(self.done)
            return "feedback"

    def generate(*args):
        spend(5)
        if multi_turn:
            return {"role": "assistant", "content": ""}, [("answer", {})], 10
        return "answer", 10

    flushed = []

    def on_result(i, result):
        flushed.append(json.loads(agentic_eval._episode_line(i, 100 + i, result)))
        spend(100)  # Serialization/I/O must not inflate the next episode's time.

    if multi_turn:
        results = agentic_eval._run_multiturn_episodes(
            Env(),
            2,
            100,
            generate,
            max_turns=3,
            make_messages=lambda obs: [],
            tool_names={"answer"},
            on_result=on_result,
        )
    else:
        results = agentic_eval._run_episodes(
            Env(), 2, 100, generate, on_result=on_result
        )
    expected_inference, expected_episode = (10, 18) if multi_turn else (5, 10)
    for result, line in zip(results, flushed, strict=True):
        assert result.correct and result.stop_reason == "env_done"
        assert result.inference_wall_seconds == expected_inference
        assert result.episode_wall_seconds == expected_episode
        assert line["inference_wall_seconds"] == expected_inference
        assert line["episode_wall_seconds"] == expected_episode
    report = agentic_eval._metrics_to_dict(compute_metrics(results))
    assert report["cost"] == {
        "n_timed_episodes": 2,
        "episode_wall_seconds": 2 * expected_episode,
        "inference_wall_seconds": 2 * expected_inference,
    }
    old = agentic_eval._metrics_to_dict(compute_metrics([SampleResult(True, 10)]))
    assert old["cost"]["episode_wall_seconds"] is None
    assert old["cost"]["n_timed_episodes"] == 0

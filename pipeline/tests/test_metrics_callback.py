"""Regression test for the per-component reward metrics callback.

The bug: transformers.Trainer.log() appends the log entry to state.log_history
*before* invoking callbacks' on_log (trainer.py: `log_history.append(output)`
then `callback_handler.on_log`). _ComponentMetricsCallback merged the composer's
per-component metrics into the `logs` dict inside on_log - too late, the
persisted entry was already a copy. So reward/<name>/raw_mean never reached
train_log.json (only TRL-native keys did) and the cosine gradient was invisible
in the training curves. The fix updates state.log_history[-1] in place.

These tests replay the exact HF ordering against the real callback with a stub
composer (no torch/trl needed beyond the import guard), so they fail on the old
code and pass on the fix. They skip where the training stack is absent (CPU
.venv-test) and run on the GPU box where transformers/trl/peft exist.
"""
import pytest

pytest.importorskip("transformers")
pytest.importorskip("trl")
pytest.importorskip("peft")

from training.train import _ComponentMetricsCallback  # noqa: E402


class _StubComposer:
    """Stands in for a reward composer: returns fixed metrics, counts drains."""

    def __init__(self, metrics):
        self._metrics = metrics
        self.pops = 0

    def pop_step_metrics(self):
        self.pops += 1
        return dict(self._metrics)


class _FakeState:
    def __init__(self):
        self.log_history = []


def _replay_log(state, logs, callback):
    """Replay transformers.Trainer.log ordering: append a COPY of logs to
    log_history, THEN fire on_log (matching trainer.py: append then on_log)."""
    output = {**logs, "step": 1}
    state.log_history.append(output)
    callback.on_log(args=None, state=state, control=None, logs=logs)


def test_component_metrics_reach_log_history():
    # The persisted entry (what _save_train_log dumps to train_log.json) must
    # carry the per-component keys, not just the live `logs` dict.
    composer = _StubComposer({"reward/EnvReward/raw_mean": 0.7,
                              "reward/CosineLengthReward/raw_mean": -0.2})
    state = _FakeState()
    logs = {"reward": 0.5, "kl": 0.01}
    _replay_log(state, logs, _ComponentMetricsCallback(composer))
    entry = state.log_history[-1]
    assert entry.get("reward/EnvReward/raw_mean") == 0.7
    assert entry.get("reward/CosineLengthReward/raw_mean") == -0.2


def test_live_logs_updated_and_drained_once():
    composer = _StubComposer({"reward/EnvReward/raw_mean": 1.0})
    state = _FakeState()
    logs = {"reward": 0.0}
    _replay_log(state, logs, _ComponentMetricsCallback(composer))
    assert logs["reward/EnvReward/raw_mean"] == 1.0   # live loggers still see it
    assert composer.pops == 1                          # buffer drained exactly once


def test_empty_metrics_is_noop():
    composer = _StubComposer({})
    state = _FakeState()
    logs = {"reward": 0.0}
    _replay_log(state, logs, _ComponentMetricsCallback(composer))
    assert state.log_history[-1] == {"reward": 0.0, "step": 1}

import pytest

from training.env_server import EnvServerProcess


def _srv(**over):
    kwargs = dict(
        env_module="reasoning_gym_env.server.app",
        port=8077,
        repo_envs_path="/workspace/OpenEnv/envs",
        max_concurrent=8,
        python="/venv/bin/python",
    )
    kwargs.update(over)
    return EnvServerProcess(**kwargs)


def test_command_shape():
    cmd = _srv().command()
    assert cmd == ["/venv/bin/python", "-m", "reasoning_gym_env.server.app", "--port", "8077"]


def test_base_url():
    assert _srv(host="127.0.0.1", port=9001).base_url == "http://127.0.0.1:9001"


def test_env_sets_concurrency_and_pythonpath():
    env = _srv(max_concurrent=16)._env()
    assert env["MAX_CONCURRENT_ENVS"] == "16"
    assert env["PYTHONPATH"].startswith("/workspace/OpenEnv/envs")


def test_wait_until_ready_returns_when_ready():
    assert _srv().wait_until_ready(_ready=lambda: True, _sleep=lambda *_: None) is True


def test_wait_until_ready_times_out():
    clock = {"t": 0.0}

    def now():
        clock["t"] += 1.0
        return clock["t"]

    with pytest.raises(TimeoutError):
        _srv().wait_until_ready(
            timeout=3, interval=1, _ready=lambda: False, _sleep=lambda *_: None, _now=now
        )

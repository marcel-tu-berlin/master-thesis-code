import pytest

from domains.env_base import EnvDomain
from training.env_server import EnvServerProcess, build_env_server


class _Dom(EnvDomain):
    server_module = "reasoning_gym_env.server.app"


class _TextDom(EnvDomain):
    server_module = "textarena_env.server.app"

    def server_env(self, env_config=None):
        return {"TEXTARENA_ENV_ID": (env_config or {}).get("env_id", "Wordle-v0")}


def _agentic_cfg(**training):
    t = {"mode": "agentic", "env": "reasoning_gym", "env_config": {"dataset": "chain_sum"}}
    t.update(training)
    return {"training": t}


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


def test_build_env_server_defaults():
    srv = build_env_server(_agentic_cfg(n_rollouts=8, batch_size=1), _Dom(), python="/p")
    assert srv.command()[:3] == ["/p", "-m", "reasoning_gym_env.server.app"]
    assert srv.port == 8077
    assert srv.repo_envs_path == "/workspace/OpenEnv/envs"
    assert srv.max_concurrent == 8  # max(8, 1*8)


def test_build_env_server_sizes_concurrency_to_generation_batch():
    srv = build_env_server(_agentic_cfg(n_rollouts=16, batch_size=2), _Dom())
    assert srv.max_concurrent == 32  # 2*16 > floor of 8


def test_build_env_server_config_overrides():
    cfg = _agentic_cfg(n_rollouts=8, batch_size=1)
    cfg["training"]["env_server"] = {"repo_path": "/custom/envs", "port": 9000}
    srv = build_env_server(cfg, _Dom())
    assert srv.repo_envs_path == "/custom/envs" and srv.port == 9000


def test_env_merges_server_env_vars():
    srv = _srv(server_env={"TEXTARENA_ENV_ID": "Wordle-v0", "TEXTARENA_NUM_PLAYERS": "1"})
    env = srv._env()
    assert env["TEXTARENA_ENV_ID"] == "Wordle-v0"
    assert env["TEXTARENA_NUM_PLAYERS"] == "1"
    # The reasoning_gym defaults still apply alongside the merged vars.
    assert env["MAX_CONCURRENT_ENVS"] == "8"


def test_build_env_server_passes_domain_server_env():
    cfg = _agentic_cfg(n_rollouts=8, batch_size=1)
    cfg["training"]["env_config"] = {"env_id": "Sudoku-v0"}
    srv = build_env_server(cfg, _TextDom())
    assert srv.server_env["TEXTARENA_ENV_ID"] == "Sudoku-v0"


def test_build_env_server_reasoning_gym_server_env_empty():
    srv = build_env_server(_agentic_cfg(n_rollouts=8, batch_size=1), _Dom())
    assert srv.server_env == {}

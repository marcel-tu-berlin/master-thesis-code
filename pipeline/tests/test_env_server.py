import pytest

from domains.env_base import EnvDomain
from training.env_server import EnvServerProcess, build_env_server


class _Dom(EnvDomain):
    server_module = "reasoning_gym_env.server.app"


class _VarDom(EnvDomain):
    """A domain whose server is configured by process env vars, not request args."""

    server_module = "browsergym_env.server.app"

    def server_env(self, env_config=None):
        return {"BROWSERGYM_TASK_NAME": (env_config or {}).get("tasks", ["click-option"])[0]}


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


def test_start_refuses_an_occupied_port():
    # A leftover server from another env keeps the port, the new one dies on bind,
    # and the readiness probe then passes against the wrong server. That silently
    # ran a whole browsergym training against a stale reasoning_gym server.
    srv = _srv()
    srv.is_ready = lambda: True
    with pytest.raises(RuntimeError, match="already in use"):
        srv.start()
    assert srv._proc is None


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
    # 8000 is every OpenEnv server's own default; four of the five domain
    # servers bind it unconditionally and ignore the --port argv, so a default
    # of anything else only ever worked for reasoning_gym.
    assert srv.port == 8000
    assert srv.repo_envs_path == "/workspace/OpenEnv/envs"
    assert srv.max_concurrent == 8  # max(8, 1*8)


def test_build_env_server_sizes_concurrency_to_generation_batch():
    srv = build_env_server(_agentic_cfg(n_rollouts=16, batch_size=2), _Dom())
    assert srv.max_concurrent == 32  # 2*16 > floor of 8


def test_build_env_server_default_geometry_matches_the_trainer():
    # The trainer defaults batch_size to 4 (grpo_runner, via the shared
    # constants in config_schema). A server sized from a stale default of 1
    # capped sessions at 8 while the trainer opened 32 rollout envs, and the
    # 9th slot's first reset() died with SessionCapacityError at step 1.
    srv = build_env_server(_agentic_cfg(), _Dom())
    assert srv.max_concurrent == 32  # DEFAULT_BATCH_SIZE 4 * DEFAULT_N_ROLLOUTS 8


def test_build_env_server_config_overrides():
    cfg = _agentic_cfg(n_rollouts=8, batch_size=1)
    cfg["training"]["env_server"] = {"repo_path": "/custom/envs", "port": 9000}
    srv = build_env_server(cfg, _Dom())
    assert srv.repo_envs_path == "/custom/envs" and srv.port == 9000


def test_env_merges_server_env_vars():
    srv = _srv(server_env={"BROWSERGYM_BENCHMARK": "miniwob", "BROWSERGYM_HEADLESS": "true"})
    env = srv._env()
    assert env["BROWSERGYM_BENCHMARK"] == "miniwob"
    assert env["BROWSERGYM_HEADLESS"] == "true"
    # The reasoning_gym defaults still apply alongside the merged vars.
    assert env["MAX_CONCURRENT_ENVS"] == "8"


def test_build_env_server_passes_domain_server_env():
    cfg = _agentic_cfg(n_rollouts=8, batch_size=1)
    cfg["training"]["env_config"] = {"tasks": ["click-checkboxes"]}
    srv = build_env_server(cfg, _VarDom())
    assert srv.server_env["BROWSERGYM_TASK_NAME"] == "click-checkboxes"


def test_build_env_server_reasoning_gym_server_env_empty():
    srv = build_env_server(_agentic_cfg(n_rollouts=8, batch_size=1), _Dom())
    assert srv.server_env == {}

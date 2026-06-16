import os
import socket
import subprocess
import sys
import time


class EnvServerProcess:
    """Launch an OpenEnv env server as a local subprocess (no Docker) and tear
    it down. One server serves every rollout-slot client, so MAX_CONCURRENT_ENVS
    must be >= the trainer's generation_batch_size.

    Used as a context manager around trainer.train():
        with EnvServerProcess(...) as srv:
            runner.train(..., environment_factory=make_factory(srv.base_url))
    """

    def __init__(self, *, env_module, port, repo_envs_path, max_concurrent,
                 host="127.0.0.1", python=None):
        self.env_module = env_module          # e.g. "reasoning_gym_env.server.app"
        self.port = int(port)
        self.repo_envs_path = repo_envs_path  # dir containing the env package
        self.max_concurrent = int(max_concurrent)
        self.host = host
        self.python = python or sys.executable
        self._proc = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def command(self) -> list[str]:
        return [self.python, "-m", self.env_module, "--port", str(self.port)]

    def _env(self) -> dict:
        env = dict(os.environ)
        env["MAX_CONCURRENT_ENVS"] = str(self.max_concurrent)
        # `python -m <env>.server.app` resolves the env package from the repo
        # envs/ dir; put it on PYTHONPATH so both the launch and the adapter's
        # client import (`from reasoning_gym_env import ...`) find it.
        env["PYTHONPATH"] = self.repo_envs_path + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def start(self) -> "EnvServerProcess":
        self._proc = subprocess.Popen(
            self.command(),
            cwd=self.repo_envs_path,
            env=self._env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return self

    def is_ready(self) -> bool:
        # Readiness = the server's TCP port accepts connections. uvicorn binds
        # and listens only after "Application startup complete", so an accepted
        # connection means the app is serving. A socket probe (vs an HTTP GET)
        # avoids any URL-scheme handling and needs no extra dependency.
        try:
            with socket.create_connection((self.host, self.port), timeout=2):
                return True
        except OSError:
            return False

    def wait_until_ready(self, timeout=60.0, interval=1.0,
                         _ready=None, _sleep=time.sleep, _now=time.monotonic) -> bool:
        ready = _ready or self.is_ready
        deadline = _now() + timeout
        while _now() < deadline:
            if ready():
                return True
            # Surface an early crash instead of waiting out the whole timeout.
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"env server exited early (code {self._proc.returncode}); "
                    f"command: {' '.join(self.command())}"
                )
            _sleep(interval)
        raise TimeoutError(f"env server not ready within {timeout}s at {self.base_url}")

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def __enter__(self) -> "EnvServerProcess":
        self.start()
        self.wait_until_ready()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def build_env_server(config, domain, python=None) -> EnvServerProcess:
    """Construct (unstarted) the env server for an agentic config.

    Sizes MAX_CONCURRENT_ENVS to the trainer's generation_batch_size
    (batch_size * n_rollouts) so every rollout-slot client gets its own session,
    with a floor of the server's own default (8). Repo path and port come from
    training.env_server (defaults target the L4 box clone).
    """
    t = config.get("training", {}) or {}
    es = t.get("env_server", {}) or {}
    n_envs = int(t.get("batch_size", 1)) * int(t.get("n_rollouts", 8))
    return EnvServerProcess(
        env_module=domain.server_module,
        port=int(es.get("port", 8077)),
        repo_envs_path=es.get("repo_path", "/workspace/OpenEnv/envs"),
        max_concurrent=max(8, n_envs),
        python=python,
    )

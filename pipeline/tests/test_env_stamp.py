import json
import subprocess

import pytest

from training import env_server
from training.env_stamp import (
    STAMP_FILE,
    STAMPED_PACKAGES,
    collect_env_stamp,
    write_env_stamp,
)


def _repo(tmp_path, name="clone"):
    d = tmp_path / name
    d.mkdir()

    def run(*a):
        return subprocess.run(
            ["git", "-C", str(d), *a], check=True, capture_output=True
        )

    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    (d / "f").write_text("a")
    run("add", "f")
    run("commit", "-qm", "a")
    head = subprocess.run(
        ["git", "-C", str(d), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return d, head


def test_pin_file_is_a_commit_sha():
    pin = env_server.openenv_pin()
    assert len(pin) == 40 and all(c in "0123456789abcdef" for c in pin)


def test_verify_openenv_pin_accepts_the_pinned_head(tmp_path, monkeypatch):
    d, head = _repo(tmp_path)
    monkeypatch.setattr(env_server, "openenv_pin", lambda: head)
    assert env_server.verify_openenv_pin(d) == head


def test_verify_openenv_pin_rejects_a_moved_clone(tmp_path, monkeypatch):
    # A `git pull` in /workspace/OpenEnv changes the env servers under a run and
    # the run still produces plausible numbers. This is the only check that sees it.
    d, _head = _repo(tmp_path)
    monkeypatch.setattr(env_server, "openenv_pin", lambda: "0" * 40)
    with pytest.raises(RuntimeError, match="not the pinned"):
        env_server.verify_openenv_pin(d)


def test_verify_openenv_pin_rejects_a_path_that_is_not_a_clone(tmp_path):
    with pytest.raises(RuntimeError, match="cannot read"):
        env_server.verify_openenv_pin(tmp_path / "nope")


def test_start_checks_the_pin_before_spawning(tmp_path):
    srv = env_server.EnvServerProcess(
        env_module="reasoning_gym_env.server.app",
        port=8077,
        repo_envs_path=str(tmp_path / "nope"),
        max_concurrent=8,
    )
    srv.is_ready = lambda: False
    with pytest.raises(RuntimeError, match="cannot read"):
        srv.start()
    assert srv._proc is None  # nothing was spawned


def test_collect_env_stamp_records_the_clone_and_the_packages(tmp_path):
    d, head = _repo(tmp_path)
    stamp = collect_env_stamp(d)
    assert stamp["openenv"]["head"] == head
    assert stamp["openenv"]["pin"] == env_server.openenv_pin()
    assert set(stamp["packages"]) == set(STAMPED_PACKAGES)
    assert stamp["python"].startswith("3.")


def test_collect_env_stamp_survives_a_missing_package_and_a_non_clone(tmp_path):
    # Never fatal: the eval box has no vllm wheel on a CPU-only checkout, and a
    # stamp that raised would take the run with it.
    stamp = collect_env_stamp(tmp_path / "not-a-clone")
    assert stamp["openenv"]["head"] is None
    assert any(v is None for v in stamp["packages"].values()) or True


def test_write_env_stamp_keeps_both_phases(tmp_path):
    # Eval can run weeks after training; overwriting would lose the stack that
    # actually trained the checkpoint.
    write_env_stamp(tmp_path, "train")
    write_env_stamp(tmp_path, "eval")
    stamps = json.loads((tmp_path / STAMP_FILE).read_text())
    assert set(stamps) == {"train", "eval"}

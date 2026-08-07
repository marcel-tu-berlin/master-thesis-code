"""Runtime check of the integrated envs under openenv 0.4.2.

Every env's app.py hardcodes port 8000, so instead of running their main() we
import the `app` object and serve it on a distinct port per env. That removes the
port contention that made the shell version silently re-check one server.
"""
import multiprocessing as mp
import os
import sys
import time
import urllib.request
import json

sys.path.insert(0, "/workspace/OpenEnv/envs")

CASES = [
    ("reasoning_gym", "reasoning_gym_env.server.app", 8201,
     {"REASONING_GYM_DATASET": "polynomial_equations"}),
    ("textarena", "textarena_env.server.app", 8202, {"TEXTARENA_ENV_ID": "Wordle-v0"}),
    ("repl", "repl_env.server.app", 8203, {}),
]


def serve(mod, port, envvars):
    os.environ.update(envvars, MAX_CONCURRENT_ENVS="4")
    sys.path.insert(0, "/workspace/OpenEnv/envs")
    import importlib
    import uvicorn
    m = importlib.import_module(mod)
    uvicorn.run(m.app, host="127.0.0.1", port=port, log_level="error")


def post(url, payload, timeout=30):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


for name, mod, port, envvars in CASES:
    p = mp.Process(target=serve, args=(mod, port, envvars), daemon=True)
    p.start()
    up = False
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    up = True
                    break
        except Exception:
            pass
        time.sleep(1)
    if not up:
        print(f"{name:14s} SERVER FAILED TO START", flush=True)
        p.terminate()
        continue
    try:
        d = post(f"http://127.0.0.1:{port}/reset", {})
        obs = d.get("observation", d)
        keys = sorted(obs.keys()) if isinstance(obs, dict) else type(obs).__name__
        preview = json.dumps(obs)[:110] if isinstance(obs, dict) else str(obs)[:110]
        print(f"{name:14s} OK  port={port} obs_keys={keys}\n{'':16s}{preview}", flush=True)
    except Exception as e:
        print(f"{name:14s} RESET FAILED: {type(e).__name__}: {e}", flush=True)
    p.terminate()
    p.join(timeout=10)

"""Base-model difficulty probe for BrowserGym/MiniWoB.

Answers one question: does Qwen3-1.7B land inside the 40-80% band on any MiniWoB
task family, so GRPO gets within-group variance and E1 has headroom to specialize
into. Mirrors the pipeline's agentic eval loop (native tool calling, greedy,
stop-reason bookkeeping) so the numbers transfer.

One model load, one browsergym server per task (BROWSERGYM_TASK_NAME is set at
server start, so the server restarts between tasks).
"""
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, "/workspace/OpenEnv/envs")

TASKS = [
    "click-test",        # trivial: one button
    "click-button",      # trivial: named button among several
    "click-dialog",      # easy: close a dialog
    "click-option",      # easy: radio select then submit
    "enter-text",        # easy: type then submit
    "click-checkboxes",  # medium: multi-select then submit
    "login-user",        # medium: two fields then submit
    "click-tab-2",       # harder: tab nav then click a link
]
N_EPISODES = 20
MAX_TURNS = 10
MAX_NEW = 2048
PORT = 8100
MODEL = "Qwen/Qwen3-1.7B"

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_tool_call(text):
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(m.group(1))
        except (ValueError, TypeError):
            continue
        name = payload.get("name")
        if name is not None:
            return str(name), (payload.get("arguments") or {})
    return None


def start_server(task):
    subprocess.run(["pkill", "-f", "browsergym_env.server.app"],
                   capture_output=True)
    time.sleep(1.5)
    env = {
        **os.environ,
        "MINIWOB_URL": "http://localhost:8080/miniwob/",
        "BROWSERGYM_BENCHMARK": "miniwob",
        "BROWSERGYM_TASK_NAME": task,
        "BROWSERGYM_HEADLESS": "true",
        "BROWSERGYM_PORT": str(PORT),
        "MAX_CONCURRENT_ENVS": "4",
    }
    log = open(f"/tmp/bgym_{task}.log", "w")
    subprocess.Popen(
        ["/workspace/bgym-venv/bin/python", "-m", "browsergym_env.server.app"],
        cwd="/workspace/OpenEnv/envs", env=env, stdout=log, stderr=log,
        start_new_session=True,
    )
    import urllib.request
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from browsergym_env import BrowserGymEnv, BrowserGymAction
    from browsergym_env.harness import build_browsergym_action_str, _BROWSERGYM_TOOLS

    tools = [{"type": "function",
              "function": {"name": t.name, "description": t.description,
                           "parameters": t.input_schema}}
             for t in _BROWSERGYM_TOOLS]

    print(f"loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print("model ready", flush=True)

    all_rows = []
    for task in TASKS:
        if not start_server(task):
            print(f"!! {task}: server failed to start", flush=True)
            continue
        env = BrowserGymEnv(f"http://localhost:{PORT}").sync()
        rows = []
        for ep in range(N_EPISODES):
            try:
                row = run_episode(env, model, tok, tools,
                                  build_browsergym_action_str, BrowserGymAction)
            except Exception as e:
                row = {"reward": 0.0, "terminated": False,
                       "stop_reason": f"error:{type(e).__name__}",
                       "tool_calls": [], "n_tokens": 0}
            rows.append(row)
            all_rows.append({**row, "task": task, "episode": ep})
        summarize(task, rows)
        with open("/tmp/bg_probe_results.json", "w") as f:
            json.dump(all_rows, f)
    print("\n===== SUMMARY =====", flush=True)
    for task in TASKS:
        rows = [r for r in all_rows if r["task"] == task]
        if rows:
            summarize(task, rows)


def run_episode(env, model, tok, tools, build_action, ActionCls):
    import torch
    r = env.reset()
    obs = r.observation
    goal = getattr(obs, "goal", "") or ""
    axtree = getattr(obs, "axtree_txt", "") or getattr(obs, "text", "") or ""
    messages = [{
        "role": "user",
        "content": (
            f"You are controlling a web page. Complete this task:\n{goal}\n\n"
            f"Page accessibility tree (element ids in brackets):\n{axtree}\n\n"
            "Call exactly one tool per turn to act on the page."
        ),
    }]
    calls, total_tok, reward = [], 0, 0.0
    stop_reason = "max_turns"
    for _ in range(MAX_TURNS):
        enc = tok.apply_chat_template(messages, tools=tools,
                                      add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(model.device)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
        comp = out[0][plen:]
        total_tok += int(comp.shape[0])
        text = tok.decode(comp, skip_special_tokens=False)
        parsed = parse_tool_call(text)
        if parsed is None:
            stop_reason = ("hit_generation_cap" if int(comp.shape[0]) >= MAX_NEW
                           else "no_tool_call")
            break
        name, args = parsed
        calls.append(name)
        try:
            action_str = build_action(name, args)
        except KeyError:
            messages += [{"role": "assistant", "content": text},
                         {"role": "tool", "name": name,
                          "content": f"Unknown tool {name}."}]
            continue
        res = env.step(ActionCls(action_str=action_str))
        reward = max(reward, float(res.reward or 0.0))
        o = res.observation
        ax = getattr(o, "axtree_txt", "") or getattr(o, "text", "") or ""
        err = getattr(o, "error", "") or ""
        if res.done:
            stop_reason = "env_done"
            break
        messages += [
            {"role": "assistant", "content": text},
            {"role": "tool", "name": name,
             "content": (f"error: {err}\n" if err else "") + f"Page now:\n{ax}"},
        ]
    return {"reward": reward, "terminated": stop_reason == "env_done",
            "stop_reason": stop_reason, "tool_calls": calls, "n_tokens": total_tok}


def summarize(task, rows):
    n = len(rows)
    succ = sum(1 for r in rows if r["reward"] >= 0.5) / n
    term = sum(1 for r in rows if r["terminated"]) / n
    toks = sum(r["n_tokens"] for r in rows) / n
    steps = sum(len(r["tool_calls"]) for r in rows) / n
    sr = {}
    for r in rows:
        sr[r["stop_reason"]] = sr.get(r["stop_reason"], 0) + 1
    band = "  <-- GOLDILOCKS" if 0.40 <= succ <= 0.80 else ""
    print(f"{task:20s} n={n:3d} success={succ:.2f} term={term:.2f} "
          f"tok={toks:6.0f} steps={steps:.1f} {sr}{band}", flush=True)


if __name__ == "__main__":
    main()

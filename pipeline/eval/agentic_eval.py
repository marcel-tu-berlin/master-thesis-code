"""Agentic episode evaluation.

Runs the trained policy against the live OpenEnv environment for N held-out
episodes (seeds disjoint from training), parses each tool call, scores it via
the env, and reports success rate + token-efficiency metrics.
"""
import json
import os
import re
import sys

from domains.env_base import CORRECT_REWARD_THRESHOLD
from eval.metrics import SampleResult, compute_metrics

# The model answers by emitting a Hermes-style tool call (Qwen3 native format):
#   <tool_call>{"name": "answer", "arguments": {"answer": "42"}}</tool_call>
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# Held-out offset: training seeds are seed..seed+size, so eval at seed+OFFSET
# evaluates on questions the model was not trained on.
_EVAL_SEED_OFFSET = 100_000


def _completion_budget(config, model_max_seq):
    """Max new tokens for eval generation. Defaults to the SAME completion
    budget training used (max_seq - max_prompt_length), so eval never truncates
    reasoning the model was trained to produce. A hardcoded smaller cap silently
    guillotines long completions before the tool call and tanks the success rate.
    eval.max_new_tokens overrides.
    """
    eval_cfg = config.get("eval", {}) or {}
    if eval_cfg.get("max_new_tokens") is not None:
        return int(eval_cfg["max_new_tokens"])
    max_seq = int((config.get("model") or {}).get("max_seq_length", model_max_seq))
    max_prompt = int((config.get("training") or {}).get("max_prompt_length", max_seq // 2))
    return max_seq - max_prompt


def _parse_answer(text: str) -> str | None:
    """Return the answer from the first valid `answer` tool call, else None."""
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(m.group(1))
        except (ValueError, TypeError):
            continue
        if payload.get("name") == "answer":
            ans = (payload.get("arguments") or {}).get("answer")
            if ans is not None:
                return str(ans)
    return None


def _parse_tool_call(text: str) -> tuple | None:
    """Return (name, arguments) from the FIRST valid <tool_call> JSON, else None.

    General form used by the multi-turn loop (any tool name). The single-step
    reasoning_gym path keeps its own _parse_answer (first `answer` call only).
    """
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(m.group(1))
        except (ValueError, TypeError):
            continue
        name = payload.get("name")
        if name is not None:
            return str(name), (payload.get("arguments") or {})
    return None


def _no_call_reason(n_tokens: int, gen_cap: int | None) -> str:
    """Why a turn produced no usable tool call.

    `hit_generation_cap` (the completion filled the whole budget) is kept apart
    from `no_tool_call` (the model stopped on its own) because the first is an
    artifact of our token cap and the second is agent behavior. Collapsing them
    is what made the e9-e21 sweep uninterpretable, and a non-termination penalty
    trained against the collapsed label would be optimizing the cap.
    """
    if gen_cap is not None and n_tokens >= gen_cap:
        return "hit_generation_cap"
    return "no_tool_call"


def _run_episodes(env, n: int, seed_base: int, gen_fn, gen_cap=None) -> list[SampleResult]:
    """Run n single-step episodes. gen_fn(question) -> (answer_str|None, n_tokens).

    A None answer (the model never called the tool) is submitted as an empty
    string so the env scores it as a failure - the episode still counts, but it
    is now also recorded as not terminated, so "never answered" is no longer
    indistinguishable from "answered wrong" (both score 0.0).
    """
    results = []
    for i in range(n):
        question = env.reset(seed=seed_base + i)
        answer, n_tokens = gen_fn(question)
        env.answer(answer if answer is not None else "")
        r = float(env.reward)
        terminated = answer is not None
        results.append(SampleResult(
            correct=r >= CORRECT_REWARD_THRESHOLD,
            n_tokens=n_tokens, n_steps=1, reward=r,
            terminated=terminated,
            stop_reason="env_done" if terminated else _no_call_reason(n_tokens, gen_cap),
            tool_calls=["answer"] if terminated else [],
        ))
    return results


def _run_multiturn_episodes(env, n, seed_base, turn_fn, *, max_turns, make_messages,
                            tool_names, gen_cap=None):
    """Run n multi-turn episodes greedily, tool-agnostic.

    turn_fn(messages) -> (tool_name|None, arguments|None, n_tokens) produces one
    greedy model turn given the running message list. Per episode: reset, then up
    to max_turns turns; each turn that names one of the domain's tools dispatches
    it on the env (the public adapter method of that name) and appends the
    assistant + tool-feedback messages; the episode ends when the model stops
    calling a known tool or the env reports done. Restricting dispatch to
    `tool_names` keeps a hallucinated name (or `reset`) from being invoked. This
    one loop drives every multi-turn domain - textarena (move), finqa (the four
    data tools), repl (execute) - via the parsed tool name. n_tokens is the
    per-turn generated token count (model-only, exact); n_steps is the tool count.

    Each episode also records why it ended and which tools it called, in order.
    The off-target panel (RQ2) is computed from those two fields, and neither is
    recoverable afterwards from the reward alone.
    """
    results = []
    for i in range(n):
        obs = env.reset(seed=seed_base + i)
        messages = list(make_messages(obs))
        total_tokens = 0
        calls = []
        # Default exit: the loop used its whole turn budget without the env ever
        # reporting done. Overwritten at whichever exit the episode actually took.
        stop_reason = "max_turns"
        for _ in range(max_turns):
            name, args, n_tok = turn_fn(messages)
            total_tokens += int(n_tok)
            if name not in tool_names:
                stop_reason = _no_call_reason(n_tok, gen_cap)
                break
            try:
                feedback = getattr(env, name)(**(args or {}))
            except TypeError as e:
                # Malformed model arguments (wrong/extra kwargs) become feedback
                # rather than crashing the episode.
                feedback = f"Tool call error: {e}"
            calls.append(name)
            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"type": "function",
                                "function": {"name": name, "arguments": args or {}}}],
            })
            messages.append({"role": "tool", "content": str(feedback)})
            if getattr(env, "done", False):
                stop_reason = "env_done"
                break
        r = float(env.reward)
        results.append(SampleResult(
            correct=r >= CORRECT_REWARD_THRESHOLD,
            n_tokens=total_tokens, n_steps=len(calls), reward=r,
            terminated=stop_reason == "env_done",
            stop_reason=stop_reason, tool_calls=calls,
        ))
    return results


def _metrics_to_dict(m) -> dict:
    """Serialize EvalMetrics into the per-split shape the report/compare tools
    expect (a `samples` series with n_tokens lets load_reference_thresholds and
    the token-distribution plot work)."""
    return {
        "accuracy": m.accuracy,
        "accuracy_ci_low": m.accuracy_ci_low,
        "accuracy_ci_high": m.accuracy_ci_high,
        "mean_token_count": m.mean_token_count,
        "mean_token_count_ci_low": m.mean_token_count_ci_low,
        "mean_token_count_ci_high": m.mean_token_count_ci_high,
        "underthinking_rate": m.underthinking_rate,
        "overthinking_rate": m.overthinking_rate,
        "mean_steps": m.mean_steps,
        # Off-target panel (RQ2). Absent-as-None when the episodes carry no
        # termination record, so an old report is not read as "zero failures".
        "non_termination_rate": m.non_termination_rate,
        "non_termination_rate_ci_low": m.non_termination_rate_ci_low,
        "non_termination_rate_ci_high": m.non_termination_rate_ci_high,
        "unsupported_claim_rate": m.unsupported_claim_rate,
        "unsupported_claim_rate_ci_low": m.unsupported_claim_rate_ci_low,
        "unsupported_claim_rate_ci_high": m.unsupported_claim_rate_ci_high,
        "mean_verification_depth": m.mean_verification_depth,
        "stop_reasons": m.stop_reasons,
        "n_samples": m.n_samples,
        "n_correct": m.n_correct,
        "samples": [
            {"correct": r.correct, "n_tokens": r.n_tokens, "n_steps": r.n_steps,
             "reward": r.reward, "terminated": r.terminated,
             "stop_reason": r.stop_reason, "tool_calls": r.tool_calls}
            for r in m.raw
        ],
    }


def _write_episodes(results, seed_base: int, path: str) -> None:
    """One JSON line per episode: the durable trajectory record.

    The aggregate report answers the questions we thought of before the run.
    This file is what a later off-target question is answered from without
    spending another 2h of GPU re-running the eval.
    """
    with open(path, "w") as f:
        for i, r in enumerate(results):
            f.write(json.dumps({
                "index": i,
                "seed": seed_base + i,
                "correct": r.correct,
                "reward": r.reward,
                "n_tokens": r.n_tokens,
                "n_steps": r.n_steps,
                "terminated": r.terminated,
                "stop_reason": r.stop_reason,
                "tool_calls": r.tool_calls,
            }) + "\n")


def _resolve_splits(config, base_n: int) -> list[dict]:
    """Eval splits to run, in order.

    Without `eval.agentic.splits` this is the single split named "agentic" that
    every existing report and plot expects. With it, each entry may override the
    env_config (merged over the training one) and the seed offset, which is how
    the protocol's shifted split is expressed: a different task family / dataset
    config, or a disjoint region of the seed->question mapping.
    """
    agentic_cfg = ((config.get("eval") or {}).get("agentic") or {})
    train_env_cfg = (config.get("training") or {}).get("env_config") or {}
    raw = agentic_cfg.get("splits")
    if not raw:
        return [{"name": "agentic", "env_config": dict(train_env_cfg),
                 "n_episodes": base_n, "seed_offset": _EVAL_SEED_OFFSET}]
    return [{
        "name": str(s["name"]),
        "env_config": {**train_env_cfg, **(s.get("env_config") or {})},
        "n_episodes": int(s.get("n_episodes", base_n)),
        "seed_offset": int(s.get("seed_offset", _EVAL_SEED_OFFSET)),
    } for s in raw]


def run_agentic_eval(config, checkpoint_dir, domain, run_dir, n_episodes=None) -> dict:
    """Evaluate a trained agentic policy over held-out env episodes.

    Loads the LoRA checkpoint, launches the OpenEnv server (no Docker), runs
    greedy tool-calling episodes per split, scores via the env, and writes
    runs/<exp>/eval_report.json + .md keyed by split name.

    `checkpoint_dir=None` evaluates the base model with no adapter - reward
    condition E0. The rest of the path is identical, so E0 is measured through
    exactly the same episode loop, prompt framing and scoring as every trained
    arm; a separately written probe script would not be.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    from training.registry import get_model_config
    from training.env_server import build_env_server

    model_cfg = get_model_config(config["model"]["slug"])
    load_4bit = config["model"].get("load_in_4bit", model_cfg["load_in_4bit"])
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    ) if load_4bit else None

    # Native tool-calling template (do NOT apply the reasoning-tag template).
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name"], quantization_config=quant_config,
        torch_dtype=dtype, device_map="auto",
    )
    if checkpoint_dir is None:
        print("No checkpoint: evaluating the base model (E0)")
    else:
        print(f"Loading checkpoint: {checkpoint_dir}")
        model = PeftModel.from_pretrained(model, checkpoint_dir)
    model.eval()

    eval_cfg = config.get("eval", {}) or {}
    agentic_cfg = eval_cfg.get("agentic", {}) or {}
    base_n = int(n_episodes if n_episodes is not None else agentic_cfg.get("n_episodes", 100))
    max_new = _completion_budget(config, model_cfg["max_seq_length"])
    do_sample = bool(eval_cfg.get("do_sample", False))
    seed = int(config.get("seed", 42))
    splits = _resolve_splits(config, base_n)

    # Created up front: each split writes its episode records as it finishes, so
    # a crash in a later split does not lose the earlier one's trajectories.
    os.makedirs(run_dir, exist_ok=True)
    split_metrics = {}
    for split in splits:
        n = split["n_episodes"]
        if config.get("_smoke"):
            # A multi-turn agentic eval runs n * up-to-max_turns model.generate
            # calls, so keep the smoke episode count small - it only checks that
            # the eval loop runs.
            n = min(n, 4)
        env_config = split["env_config"]
        seed_base = seed + split["seed_offset"]
        # One server per split: server_env is derived from env_config (task id,
        # turn cap, data path), so a split that changes it needs its own process.
        split_config = {**config,
                        "training": {**config["training"], "env_config": env_config}}
        server = build_env_server(split_config, domain, python=sys.executable)
        server.start()
        server.wait_until_ready()
        if server.repo_envs_path not in sys.path:
            sys.path.insert(0, server.repo_envs_path)
        try:
            env = domain.make_env_factory(server.base_url, env_config)()
            tools = domain.eval_tools(env)

            def gen_fn(question):
                messages = domain.episode_messages(question)
                enc = tokenizer.apply_chat_template(
                    messages, tools=tools, add_generation_prompt=True,
                    return_tensors="pt", return_dict=True,
                ).to(model.device)
                plen = enc["input_ids"].shape[1]
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=max_new, do_sample=do_sample)
                comp_ids = out[0][plen:]
                text = tokenizer.decode(comp_ids, skip_special_tokens=False)
                return _parse_answer(text), int(comp_ids.shape[0])

            def gen_turn(messages):
                enc = tokenizer.apply_chat_template(
                    messages, tools=tools, add_generation_prompt=True,
                    return_tensors="pt", return_dict=True,
                ).to(model.device)
                plen = enc["input_ids"].shape[1]
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=max_new, do_sample=do_sample)
                comp_ids = out[0][plen:]
                text = tokenizer.decode(comp_ids, skip_special_tokens=False)
                parsed = _parse_tool_call(text)
                if parsed is None:
                    return None, None, int(comp_ids.shape[0])
                name, call_args = parsed
                return name, call_args, int(comp_ids.shape[0])

            print(f"Agentic eval [{split['name']}]: {n} episodes "
                  f"(seed_base={seed_base}, max_new_tokens={max_new})")
            if getattr(domain, "multi_turn", False):
                # env_config.max_turns is the single turn cap - training passes it
                # to TRL as max_tool_calling_iterations and each domain maps it to
                # its server's own cap var, so eval must read the same key or it
                # measures a different episode length than training produced.
                max_turns = int(env_config.get("max_turns", 8))
                results = _run_multiturn_episodes(
                    env, n, seed_base, gen_turn,
                    max_turns=max_turns, make_messages=domain.episode_messages,
                    tool_names={t.__name__ for t in tools}, gen_cap=max_new,
                )
            else:
                results = _run_episodes(env, n, seed_base, gen_fn, gen_cap=max_new)
        finally:
            server.stop()

        split_metrics[split["name"]] = compute_metrics(results)
        _write_episodes(results, seed_base,
                        os.path.join(run_dir, f"episodes_{split['name']}.jsonl"))

    report = {
        "experiment_id": config.get("experiment_id"),
        "model_slug": (config.get("model") or {}).get("slug"),
        "seed": config.get("seed", 42),
        "compose_method": (config.get("rewards") or {}).get("compose_method", "advantage_weighted"),
        "mode": "agentic",
        # E0 (no adapter) is a first-class condition, so the report says which
        # policy produced it rather than leaving it to the experiment_id.
        "checkpoint": checkpoint_dir,
        "results": {name: _metrics_to_dict(m) for name, m in split_metrics.items()},
    }
    json_path = os.path.join(run_dir, "eval_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(run_dir, "eval_report.md"), "w") as f:
        f.write(_report_md(report["experiment_id"], split_metrics))
    summary = ", ".join(f"{name} {m.accuracy:.3f} (n={m.n_samples})"
                        for name, m in split_metrics.items())
    print(f"Agentic eval report written to {json_path} [{summary}]")
    return report


def _report_md(experiment_id, split_metrics: dict) -> str:
    """Human-readable report: one block per split.

    The three dimensions the protocol interprets jointly (task success, the
    targeted efficiency, the off-target panel) sit in one block per split so a
    drop in one is read next to a gain in another, not on a separate page.
    """
    out = [f"# Agentic eval: {experiment_id}\n"]
    for name, m in split_metrics.items():
        nonterm = ("n/a" if m.non_termination_rate is None
                   else f"{m.non_termination_rate:.3f} "
                        f"[{m.non_termination_rate_ci_low:.3f}, "
                        f"{m.non_termination_rate_ci_high:.3f}]")
        out.append(
            f"\n## {name}\n\n"
            f"- episodes: {m.n_samples}\n"
            f"- success rate: {m.accuracy:.3f} "
            f"[{m.accuracy_ci_low:.3f}, {m.accuracy_ci_high:.3f}]\n"
            f"- mean completion tokens: {m.mean_token_count:.1f}\n"
            f"- mean steps: {m.mean_steps}\n"
            f"- underthinking rate: {m.underthinking_rate}\n"
            f"- overthinking rate: {m.overthinking_rate}\n"
            f"- non-termination rate: {nonterm}\n"
            f"- unsupported-claim rate: {m.unsupported_claim_rate}\n"
            f"- mean verification depth: {m.mean_verification_depth}\n"
            f"- stop reasons: {m.stop_reasons}\n"
        )
    return "".join(out)

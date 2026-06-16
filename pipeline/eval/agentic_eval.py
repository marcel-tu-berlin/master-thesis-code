"""Agentic episode evaluation.

Runs the trained policy against the live OpenEnv environment for N held-out
episodes (seeds disjoint from training), parses each tool call, scores it via
the env, and reports success rate + token-efficiency metrics. This is the
agentic counterpart to eval/ood_probes.py (which is dataset-probe oriented).
"""
import json
import os
import re
import sys

from eval.metrics import SampleResult, compute_metrics

# The model answers by emitting a Hermes-style tool call (Qwen3 native format):
#   <tool_call>{"name": "answer", "arguments": {"answer": "42"}}</tool_call>
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# Held-out offset: training seeds are seed..seed+size, so eval at seed+OFFSET
# evaluates on questions the model was not trained on.
_EVAL_SEED_OFFSET = 100_000


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


def _run_episodes(env, n: int, seed_base: int, gen_fn) -> list[SampleResult]:
    """Run n single-step episodes. gen_fn(question) -> (answer_str|None, n_tokens).

    A None answer (the model never called the tool) is submitted as an empty
    string so the env scores it as a failure - the episode still counts.
    """
    results = []
    for i in range(n):
        question = env.reset(seed=seed_base + i)
        answer, n_tokens = gen_fn(question)
        env.answer(answer if answer is not None else "")
        results.append(SampleResult(correct=env.reward > 0, n_tokens=n_tokens, n_steps=1))
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
        "n_samples": m.n_samples,
        "n_correct": m.n_correct,
        "samples": [
            {"correct": r.correct, "n_tokens": r.n_tokens, "n_steps": r.n_steps}
            for r in m.raw
        ],
    }


def run_agentic_eval(config, checkpoint_dir, domain, run_dir, n_episodes=None) -> dict:
    """Evaluate a trained agentic policy over held-out env episodes.

    Loads the LoRA checkpoint, launches the OpenEnv server (no Docker), runs
    greedy tool-calling episodes, scores via the env, and writes
    runs/<exp>/eval_report.json + .md keyed under the "agentic" split.
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
    print(f"Loading checkpoint: {checkpoint_dir}")
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model.eval()

    eval_cfg = config.get("eval", {}) or {}
    agentic_cfg = eval_cfg.get("agentic", {}) or {}
    n = int(n_episodes if n_episodes is not None else agentic_cfg.get("n_episodes", 100))
    if config.get("_smoke"):
        n = min(n, 10)
    max_new = int(eval_cfg.get("max_new_tokens", 512))
    do_sample = bool(eval_cfg.get("do_sample", False))
    env_config = config["training"].get("env_config", {}) or {}
    seed_base = int(config.get("seed", 42)) + _EVAL_SEED_OFFSET

    server = build_env_server(config, domain, python=sys.executable)
    server.start()
    server.wait_until_ready()
    if server.repo_envs_path not in sys.path:
        sys.path.insert(0, server.repo_envs_path)
    try:
        env = domain.make_env_factory(server.base_url, env_config)()
        tools = [env.answer]

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

        print(f"Agentic eval: {n} episodes (seed_base={seed_base}, max_new_tokens={max_new})")
        results = _run_episodes(env, n, seed_base, gen_fn)
    finally:
        server.stop()

    metrics = compute_metrics(results)
    report = {
        "experiment_id": config.get("experiment_id"),
        "model_slug": (config.get("model") or {}).get("slug"),
        "seed": config.get("seed", 42),
        "compose_method": (config.get("rewards") or {}).get("compose_method", "advantage_weighted"),
        "mode": "agentic",
        "results": {"agentic": _metrics_to_dict(metrics)},
    }
    os.makedirs(run_dir, exist_ok=True)
    json_path = os.path.join(run_dir, "eval_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    md = (
        f"# Agentic eval: {report['experiment_id']}\n\n"
        f"- episodes: {metrics.n_samples}\n"
        f"- success rate: {metrics.accuracy:.3f} "
        f"[{metrics.accuracy_ci_low:.3f}, {metrics.accuracy_ci_high:.3f}]\n"
        f"- mean completion tokens: {metrics.mean_token_count:.1f}\n"
        f"- underthinking rate: {metrics.underthinking_rate}\n"
        f"- overthinking rate: {metrics.overthinking_rate}\n"
    )
    with open(os.path.join(run_dir, "eval_report.md"), "w") as f:
        f.write(md)
    print(f"Agentic eval report written to {json_path} "
          f"(success {metrics.accuracy:.3f}, n={metrics.n_samples})")
    return report

"""Agentic episode evaluation.

Runs the trained policy against the live OpenEnv environment for N held-out
episodes (seeds disjoint from training), parses each tool call, scores it via
the env, and reports success rate + token-efficiency metrics.
"""
import inspect
import json
import os
import sys

from domains.env_base import CORRECT_REWARD_THRESHOLD
from eval.metrics import SampleResult, compute_metrics, load_reference_thresholds
from training.config_schema import SEED_BLOCK, resolve_max_turns

# Held-out offset: training takes the bottom of a seed's block, so eval at
# block + OFFSET evaluates on questions the model was not trained on.
_EVAL_SEED_OFFSET = 100_000

# Each seed owns a disjoint block of the seed -> question mapping. Passing the
# raw config seed through as the dataset base made "replicates" overlap almost
# completely: seeds 42/43/44 at size 500 shared 499 of 500 training questions
# and, at a 100-episode eval, 99 of 100 eval questions - and eval is greedy, so
# the shared ones decode identically. Pooling three such runs counts the same
# questions three times and reports a CI far tighter than the data supports.
#
# SEED_BLOCK must exceed every split's seed_offset, or one seed's eval block
# lands inside another seed's; config_schema._split_errors enforces that at
# validation, next to the constant.


def seed_block(seed: int) -> int:
    """Base of the question block belonging to `seed`."""
    return int(seed) * SEED_BLOCK


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


def _tool_calls(msg: dict) -> list[tuple]:
    """Every (name, arguments) tool call in a parsed assistant message, in order.

    All of them, not the first: a turn may request several actions at once, and
    TRL's training loop executes every one of them
    (`grpo_trainer.py`, "Call the tools, and build the new prompt"). An eval that
    dispatched only the first put the policy in a state training never produces.
    Both ways of doing that are wrong and both have run:

    - Appending a stub rebuilt from the one dispatched call kept the transcript
      self-consistent but not what training generates - the model re-requested
      the dropped actions over later turns, inflating steps and tokens.
    - Appending the parsed message while dispatching one call is worse: the
      transcript then shows N tool calls answered by a single tool response, and
      the model reads the unanswered ones as having succeeded. On
      click-checkboxes-transfer, where one turn requests four clicks, it
      announced the task complete after a single click and stopped. That cost 19
      of 100 shifted episodes, every one of them a loss and none a gain.

    The message comes from `trl.chat_template_utils.parse_response`, the same
    function TRL uses to turn a rollout's token ids into a message during
    training. Eval used to hand-roll its own regex over the decoded text, which
    is how eval and training ended up disagreeing about what counts as a call:

    - A Qwen3 completion cut off after the tool-call JSON but before the closing
      `</tool_call>` tag has no tool call. The hand-rolled fallback found the
      bare JSON and scored the episode `env_done`, relabelling a truncation as a
      clean termination - the confound that made e9-e21 uninterpretable.
    - A JSON object quoted inside a `<think>` block is reasoning, not a call.
      parse_response puts it in `reasoning_content` and leaves `tool_calls`
      empty.
    - Llama 3.x emits an untagged object with `parameters` rather than
      `arguments`. parse_response normalises it, so one rule covers both
      lineages instead of a per-lineage branch here.

    Arguments that are not a dict (a model emitting them pre-serialised, say)
    read as no arguments rather than raising.
    """
    out = []
    for call in (msg.get("tool_calls") or []):
        fn = (call or {}).get("function") or {}
        name = fn.get("name")
        if name is None:
            continue
        args = fn.get("arguments")
        out.append((str(name), (args if isinstance(args, dict) else {})))
    return out


def _answer_from(msg: dict) -> str | None:
    """The `answer` argument of the first `answer` tool call, else None."""
    for call in (msg.get("tool_calls") or []):
        fn = (call or {}).get("function") or {}
        if fn.get("name") != "answer":
            continue
        args = fn.get("arguments")
        ans = args.get("answer") if isinstance(args, dict) else None
        if ans is not None:
            return str(ans)
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


def _run_episodes(env, n: int, seed_base: int, gen_fn, gen_cap=None,
                  on_result=None) -> list[SampleResult]:
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
        if on_result is not None:
            on_result(i, results[-1])
    return results


def _run_multiturn_episodes(env, n, seed_base, turn_fn, *, max_turns, make_messages,
                            tool_names, gen_cap=None, count_tokens=None,
                            on_result=None):
    """Run n multi-turn episodes greedily, tool-agnostic.

    turn_fn(messages, budget) -> (message, [(tool_name, arguments), ...],
    n_tokens) produces one greedy model turn given the running message list and
    the tokens left in the episode's generation budget. Per episode: reset, then
    up to max_turns turns; each turn appends the model's own message followed by
    one tool response per call it advertised - every call gets an answer, which
    is what TRL does in training. A call naming one of the domain's tools is
    dispatched to the env; a hallucinated name (or `reset`) is never invoked and
    gets the same {"error": "Tool X not found."} feedback TRL's dispatch
    produces, and the episode continues - training regenerates after error
    feedback, so ending the episode here would score a recoverable
    hallucination as no_tool_call. The episode ends when the model emits no
    call at all, the env reports done, or the budget runs out. This one loop
    drives every multi-turn domain - browsergym (click, noop) today - via the
    parsed tool name rather than a hardcoded one. n_tokens is the
    trajectory total (model-only, exact); n_steps is the count of calls that
    reached the env, so a turn issuing several calls counts several steps.

    `gen_cap` is the budget for the whole trajectory, matching training's
    max_completion_length, not a fresh allowance per turn. `count_tokens(text)`
    charges each tool response against that budget too: TRL counts interleaved
    tool results toward max_completion_length (rolling back results that would
    exceed it), so an eval that charged only model tokens ran materially longer
    trajectories than the cap it claims to mirror. None (unit tests) charges
    nothing.

    Arguments that do not fit the tool's signature (checked by binding them
    before the call) are agent behavior and become error feedback. Any
    exception raised by the call itself - including a TypeError from inside
    the tool body or the env client - propagates and aborts the split. Every
    finished episode is already flushed via on_result, and scoring episodes
    against dead infrastructure (a crashed env server, a reset connection)
    fabricates a policy regression - reward 0.0 and a non-termination spike -
    that reads exactly like the mislabelled-truncation confound this eval was
    rebuilt to eliminate.

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
        # Generation budget for the WHOLE trajectory, not per turn. Training caps
        # the full completion at max_completion_length, so an eval that renewed
        # the budget every turn let an episode generate max_turns times what the
        # policy was trained under - and `hit_generation_cap` could then never
        # fire for the cap that actually bound training.
        budget = gen_cap
        # Default exit: the loop used its whole turn budget without the env ever
        # reporting done. Overwritten at whichever exit the episode actually took.
        stop_reason = "max_turns"
        for _ in range(max_turns):
            if budget is not None and budget <= 0:
                stop_reason = "hit_generation_cap"
                break
            msg, turn_calls, n_tok = turn_fn(messages, budget)
            total_tokens += int(n_tok)
            turn_cap, budget = budget, (None if budget is None else budget - int(n_tok))
            if not turn_calls:
                stop_reason = _no_call_reason(n_tok, turn_cap)
                break
            # The parsed message, not a stub. Rebuilding the turn as
            # {"content": ""} threw away the model's own reasoning, so turn N+1
            # ran on a context training never produced - TRL keeps every prior
            # turn's generated text in the running completion. Qwen3's template
            # re-renders `reasoning_content` on a prior assistant turn, so
            # appending the parsed message reproduces that context.
            messages.append(msg)
            # One tool response per call, because the message just appended
            # advertises every one of them - a call left unanswered reads to the
            # model as having succeeded, and it acts on that.
            for name, args in turn_calls:
                if name not in tool_names:
                    # Never dispatched, but answered: the exact error shape
                    # TRL's training dispatch feeds back for an unknown name.
                    feedback = str({"error": f"Tool {name} not found."})
                else:
                    fn = getattr(env, name)
                    try:
                        inspect.signature(fn).bind(**(args or {}))
                    except TypeError as e:
                        # Malformed model arguments are agent behavior; training
                        # turns the exception into error feedback, so eval does
                        # too. Binding against the signature keeps that
                        # classification to the arguments alone: catching the
                        # call's own TypeError also swallowed TypeErrors raised
                        # inside the tool body (infrastructure, e.g. an openenv
                        # version skew), scoring episodes against broken infra.
                        # Every exception the call raises propagates - see the
                        # docstring.
                        feedback = str({"error": str(e)})
                    else:
                        feedback = fn(**(args or {}))
                        # Only a call that actually reached the env counts as a
                        # step. Counting a failed dispatch inflated n_steps and,
                        # through it, mean_verification_depth in the RQ2 panel.
                        calls.append(name)
                fb = str(feedback)
                # `name` mirrors TRL's tool-message shape. The budget charge
                # mirrors its accounting: the content's token count, leaving
                # only the few per-message template framing tokens uncounted.
                messages.append({"role": "tool", "name": name, "content": fb})
                if budget is not None and count_tokens is not None:
                    budget -= count_tokens(fb)
                if getattr(env, "done", False):
                    stop_reason = "env_done"
                    break
            # A call that ended the episode stops the remaining calls in the same
            # turn from being dispatched into a finished env.
            if stop_reason == "env_done":
                break
        r = float(env.reward)
        results.append(SampleResult(
            correct=r >= CORRECT_REWARD_THRESHOLD,
            n_tokens=total_tokens, n_steps=len(calls), reward=r,
            terminated=stop_reason == "env_done",
            stop_reason=stop_reason, tool_calls=calls,
        ))
        if on_result is not None:
            on_result(i, results[-1])
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
        # The efficiency number. `mean_token_count` pools failures, which run to
        # the generation cap, so it cannot separate "shorter" from "fails more".
        "mean_token_count_correct": m.mean_token_count_correct,
        "mean_token_count_correct_ci_low": m.mean_token_count_correct_ci_low,
        "mean_token_count_correct_ci_high": m.mean_token_count_correct_ci_high,
        "underthinking_rate": m.underthinking_rate,
        "overthinking_rate": m.overthinking_rate,
        # Which yardstick produced those two rates: a pinned reference threshold
        # or this run's own percentile. Without it, two arms' rates look
        # comparable when they may not be.
        "underthinking_threshold": m.underthinking_threshold,
        "overthinking_threshold": m.overthinking_threshold,
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


def _episode_line(index: int, seed: int, r) -> str:
    """One JSON line per episode: the durable trajectory record.

    The aggregate report answers the questions we thought of before the run.
    This file is what a later off-target question is answered from without
    spending another 2h of GPU re-running the eval - so it is written and
    flushed per episode, not once the split finishes. A split that dies on
    episode 95 keeps 94 trajectories instead of none.
    """
    return json.dumps({
        "index": index,
        "seed": seed,
        "correct": r.correct,
        "reward": r.reward,
        "n_tokens": r.n_tokens,
        "n_steps": r.n_steps,
        "terminated": r.terminated,
        "stop_reason": r.stop_reason,
        "tool_calls": r.tool_calls,
    }) + "\n"


def _reference_thresholds(eval_cfg: dict) -> dict:
    """Per-split over/under-thinking thresholds from `eval.reference_report`.

    Without a reference each run uses its own P10/P75, which makes both rates
    invariant to a uniform change in completion length: halve every token count
    and the reported rates do not move, because the threshold moves with them.
    That is useless for the one thing E2 is supposed to show, and it means two
    arms' rates are measured against two different yardsticks.

    Returns `{split: {"underthinking_threshold": x, "overthinking_threshold": y}}`,
    empty when no reference is configured. Splits absent from the reference fall
    back to the per-run percentile, which is why the resolved threshold is
    serialized into every report.
    """
    path = (eval_cfg or {}).get("reference_report")
    if not path:
        return {}
    thresholds = load_reference_thresholds(path)
    print(f"Thinking-rate thresholds pinned to {path}: "
          + ", ".join(f"{s} P10={t['underthinking_threshold']:.0f} "
                      f"P75={t['overthinking_threshold']:.0f}"
                      for s, t in thresholds.items()))
    return thresholds


def _build_report(config, checkpoint_dir, split_metrics: dict) -> dict:
    """The eval_report.json payload.

    Split out of run_agentic_eval so the `smoke` marker is testable without a
    GPU. That marker is the whole point: batch._is_real_report has always looked
    for it and nothing ever wrote it, so a 4-episode --smoke report counted as
    finished work and made the next unattended batch skip the real eval.
    """
    return {
        "experiment_id": config.get("experiment_id"),
        "model_slug": (config.get("model") or {}).get("slug"),
        "seed": config.get("seed", 42),
        "compose_method": (config.get("rewards") or {}).get("compose_method", "advantage_weighted"),
        "mode": "agentic",
        # E0 (no adapter) is a first-class condition, so the report says which
        # policy produced it rather than leaving it to the experiment_id.
        "checkpoint": checkpoint_dir,
        "smoke": bool(config.get("_smoke")),
        "results": {name: _metrics_to_dict(m) for name, m in split_metrics.items()},
    }


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
    from trl.chat_template_utils import add_response_schema, parse_response

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
    # Attaches the response_schema parse_response needs. Same call TRL makes
    # before training, so eval and training parse a completion by one rule.
    add_response_schema(tokenizer)
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
    thresholds = _reference_thresholds(eval_cfg)

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
        seed_base = seed_block(seed) + split["seed_offset"]
        # One server per split: server_env is derived from env_config (task id,
        # turn cap, data path), so a split that changes it needs its own process.
        split_config = {**config,
                        "training": {**config["training"], "env_config": env_config}}
        server = build_env_server(split_config, domain, python=sys.executable)
        server.start()
        server.wait_until_ready()
        if server.repo_envs_path not in sys.path:
            sys.path.insert(0, server.repo_envs_path)

        # Opened before the try so `finally` can always close it: a failure in
        # make_env_factory would otherwise leave the name unbound and raise
        # NameError from the cleanup, masking the real error.
        ep_file = open(os.path.join(run_dir, f"episodes_{split['name']}.jsonl"), "w")

        def on_result(i, r, _f=ep_file, _base=seed_base):
            _f.write(_episode_line(i, _base + i, r))
            _f.flush()

        try:
            env = domain.make_env_factory(server.base_url, env_config)()
            tools = domain.eval_tools(env)

            def _generate(messages, budget):
                enc = tokenizer.apply_chat_template(
                    messages, tools=tools, add_generation_prompt=True,
                    return_tensors="pt", return_dict=True,
                ).to(model.device)
                plen = enc["input_ids"].shape[1]
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=budget, do_sample=do_sample)
                comp_ids = out[0][plen:]
                # parse_response, not a regex over the decoded text: the same
                # function TRL applies to a rollout during training.
                msg = parse_response(tokenizer, comp_ids.tolist())
                return msg, int(comp_ids.shape[0])

            def gen_fn(question):
                messages = domain.episode_messages(question)
                msg, n_tok = _generate(messages, max_new)
                return _answer_from(msg), n_tok

            def gen_turn(messages, budget):
                msg, n_tok = _generate(messages, budget)
                return msg, _tool_calls(msg), n_tok

            print(f"Agentic eval [{split['name']}]: {n} episodes "
                  f"(seed_base={seed_base}, max_new_tokens={max_new})")
            if getattr(domain, "multi_turn", False):
                # env_config.max_turns is the single turn cap - training passes
                # it to TRL as max_tool_calling_iterations via the same
                # resolver, so an unset key means the same episode process on
                # both sides instead of training at 1 iteration while eval runs
                # 8.
                max_turns = resolve_max_turns(env_config)
                # Tool responses are charged against the trajectory budget with
                # the same tokenizer that counts the model's own tokens.
                def count_text_tokens(text):
                    return len(tokenizer(text, add_special_tokens=False)["input_ids"])
                results = _run_multiturn_episodes(
                    env, n, seed_base, gen_turn,
                    max_turns=max_turns, make_messages=domain.episode_messages,
                    tool_names={t.__name__ for t in tools}, gen_cap=max_new,
                    count_tokens=count_text_tokens, on_result=on_result,
                )
            else:
                results = _run_episodes(env, n, seed_base, gen_fn, gen_cap=max_new,
                                        on_result=on_result)
        finally:
            ep_file.close()
            server.stop()

        split_metrics[split["name"]] = compute_metrics(results, **thresholds.get(split["name"], {}))

    report = _build_report(config, checkpoint_dir, split_metrics)
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
        # Correct-only first: it is the efficiency claim. The pooled mean below
        # it moves with the failure rate, so the two must be read together.
        correct_tok = ("n/a" if m.mean_token_count_correct is None
                       else f"{m.mean_token_count_correct:.1f} "
                            f"[{m.mean_token_count_correct_ci_low:.1f}, "
                            f"{m.mean_token_count_correct_ci_high:.1f}]")
        out.append(
            f"\n## {name}\n\n"
            f"- episodes: {m.n_samples}\n"
            f"- success rate: {m.accuracy:.3f} "
            f"[{m.accuracy_ci_low:.3f}, {m.accuracy_ci_high:.3f}]\n"
            f"- mean tokens on CORRECT episodes: {correct_tok}\n"
            f"- mean completion tokens (all episodes): {m.mean_token_count:.1f}\n"
            f"- mean steps: {m.mean_steps}\n"
            f"- underthinking rate: {m.underthinking_rate}\n"
            f"- overthinking rate: {m.overthinking_rate}\n"
            f"- non-termination rate: {nonterm}\n"
            f"- unsupported-claim rate: {m.unsupported_claim_rate}\n"
            f"- mean verification depth: {m.mean_verification_depth}\n"
            f"- stop reasons: {m.stop_reasons}\n"
        )
    return "".join(out)

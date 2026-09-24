"""Episode boundaries must not change the surviving slots' token alignment."""

from types import SimpleNamespace

import pytest

from training.env_termination import environment_tool_call_loop
from training.rewards.non_termination import NonTerminationPenalty


def _call(*names):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": n, "arguments": {}}}
            for n in names
        ],
    }


class _Env:
    def __init__(self, finish_after, reward=1.0):
        self.finish_after = finish_after
        self.terminal_reward = reward
        self.done = False
        self.reward = 0.0
        self.calls = 0

    def act(self):
        self.calls += 1
        self.done = self.calls >= self.finish_after
        self.reward = self.terminal_reward if self.done else 0.0
        return "finished" if self.done else "continue"


def _run(
    envs,
    *,
    turns=8,
    budget=32,
    calls=None,
    sampled=True,
    loop=environment_tool_call_loop,
    generated_ids=None,
):
    generated = []
    generated_ids = [21, 22, 2] if generated_ids is None else generated_ids

    def generate(prompts, images, fields):
        generated.append([p[0] for p in prompts])
        return [list(generated_ids) for _ in prompts], (
            [[-0.25] * len(generated_ids) for _ in prompts] if sampled else None
        )

    trainer = SimpleNamespace(
        environments=envs,
        _sync_tool_dicts=[{"act": e.act} for e in envs],
        _async_tool_dicts=[{} for _ in envs],
        _get_tool_suffix_ids=lambda messages: [90, 91],
        _generate_single_turn=generate,
        max_tool_calling_iterations=turns,
        max_completion_length=budget,
        use_vllm=False,
        _is_vlm=False,
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=100)),
        _tokenizer=None,
    )
    output = loop(
        trainer,
        [[{"role": "user", "content": "task"}] for _ in envs],
        [[100 + i] for i in range(len(envs))],
        [[11, 12, 2] for _ in envs],
        [[c] for c in (calls or [_call("act") for _ in envs])],
        [[-0.5] * 3 for _ in envs] if sampled else None,
        None,
        {},
        parse_response=lambda tokenizer, ids: _call("act"),
    )
    return output, generated


@pytest.mark.parametrize("sampled", [False, True])
def test_mixed_terminal_and_live_slots_keep_ids_masks_and_logps_aligned(sampled):
    envs = [_Env(1), _Env(2), _Env(1, reward=0.0)]
    (mask, completions, ids, logps, count, failures, _), generated = _run(
        envs, sampled=sampled
    )
    assert generated == [[101]]
    assert [e.calls for e in envs] == [1, 2, 1]
    assert count == 4 and failures == 0
    assert ids == [[11, 12, 2], [11, 12, 2, 90, 91, 21, 22, 2], [11, 12, 2]]
    assert mask == [[1, 1, 1], [1, 1, 1, 0, 0, 1, 1, 1], [1, 1, 1]]
    assert all(c[-1]["role"] == "tool" for c in completions)
    if sampled:
        assert logps == [[-0.5] * 3, [-0.5] * 3 + [0.0] * 2 + [-0.25] * 3, [-0.5] * 3]
    else:
        assert logps is None
    assert (
        NonTerminationPenalty(32)(
            [], completions, environments=envs, completion_ids=ids
        )
        == [0.0] * 3
    )


def test_terminal_call_stops_later_calls_in_same_turn():
    env = _Env(1)
    output, generated = _run([env], calls=[_call("missing", "act", "act")])
    assert env.calls == 1 and generated == []
    assert output[4:6] == (2, 1)
    assert [m.get("name") for m in output[1][0][1:]] == ["missing", "act"]


@pytest.mark.parametrize(
    "failure", [ConnectionError, TimeoutError, RuntimeError, TypeError, ValueError]
)
def test_tool_body_failure_aborts_training_instead_of_scoring_infrastructure(failure):
    class BrokenEnv(_Env):
        def act(self):
            raise failure("environment unavailable")

    with pytest.raises(failure, match="environment unavailable"):
        _run([BrokenEnv(1)])


def test_malformed_model_arguments_remain_recoverable_tool_feedback():
    env = _Env(1)
    call = _call("act")
    call["tool_calls"][0]["function"]["arguments"] = {"unexpected": 1}
    output, generated = _run([env], calls=[call], turns=1)
    assert env.calls == 0 and generated == []
    assert output[4:6] == (1, 1)
    assert "unexpected" in output[1][0][-1]["content"]


def test_last_allowed_turn_does_not_generate_an_unacted_assistant_turn():
    env = _Env(100)
    output, generated = _run([env], turns=2)
    assert env.calls == 2 and generated == [[100]]
    assert output[1][0][-1]["role"] == "tool"
    assert NonTerminationPenalty(32)(
        [], output[1], environments=[env], completion_ids=output[2]
    ) == [-1.0]


def test_overlong_feedback_still_rolls_back_only_the_unfinished_slot():
    envs = [_Env(1), _Env(100)]
    output, generated = _run(envs, budget=4)
    assert generated == []
    assert output[2] == [[11, 12, 2], [11, 12, 2]]
    assert output[1][0][-1]["role"] == "tool"
    assert output[1][1][-1]["role"] == "assistant"


def test_native_trainer_boundary_and_unchanged_nonterminal_path():
    pytest.importorskip("trl")
    from types import MethodType

    from transformers import AutoTokenizer
    from trl import GRPOTrainer
    from trl.chat_template_utils import (
        add_response_schema,
        get_training_chat_template,
        is_chat_template_prefix_preserving,
    )

    from training.grpo_runner import _EnvironmentGRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-1.7B",
        revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        local_files_only=True,
    )
    add_response_schema(tokenizer)

    def native_loop(method):
        def call(trainer, *args, parse_response):
            trainer._tokenizer = tokenizer
            trainer.processing_class = tokenizer
            trainer.chat_template = (
                None
                if is_chat_template_prefix_preserving(tokenizer)
                else get_training_chat_template(tokenizer)
            )
            trainer.chat_template_kwargs = {}
            trainer._get_tool_suffix_ids = MethodType(
                GRPOTrainer._get_tool_suffix_ids, trainer
            )
            return method(trainer, *args)

        return call

    continuing_ids = tokenizer.encode(
        "<think>Continue.</think>\n<tool_call>\n"
        '{"name":"act","arguments":{}}\n</tool_call><|im_end|>',
        add_special_tokens=False,
    )
    envs = [_Env(1), _Env(2), _Env(1, reward=0.0), _Env(3)] * 8
    # Distinct state owners, including repeated prompt-like groups.
    envs = [_Env(e.finish_after, e.terminal_reward) for e in envs]
    output, generated = _run(
        envs,
        budget=4096,
        loop=native_loop(_EnvironmentGRPOTrainer._tool_call_loop),
        generated_ids=continuing_ids,
    )
    assert generated == [
        [100 + i for i in range(32) if i % 4 in (1, 3)],
        [100 + i for i in range(32) if i % 4 == 3],
    ]
    assert all(e.calls == e.finish_after for e in envs)
    assert all(
        len(ids) == len(mask) == len(logps)
        for ids, mask, logps in zip(output[2], output[0], output[3], strict=True)
    )
    assert all(c[-1]["role"] == "tool" for c in output[1])

    stop_ids = tokenizer.encode(
        "<think>Stop.</think>\nDone.<|im_end|>", add_special_tokens=False
    )
    old = _run(
        [_Env(100)],
        budget=4096,
        loop=native_loop(GRPOTrainer._tool_call_loop),
        generated_ids=stop_ids,
    )
    new = _run(
        [_Env(100)],
        budget=4096,
        loop=native_loop(_EnvironmentGRPOTrainer._tool_call_loop),
        generated_ids=stop_ids,
    )
    assert old == new

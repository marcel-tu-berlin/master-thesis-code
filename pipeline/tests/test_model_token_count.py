import json

from training.rewards.utils import model_token_count


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return text.split()  # one pseudo-token per whitespace-separated word


def test_string_completion_encodes_whole():
    assert model_token_count("a b c", _Tok()) == 3


def test_message_list_sums_assistant_only_skips_tool():
    c = [
        {"role": "assistant", "content": "a b"},
        {"role": "tool", "content": "x y z w"},   # game feedback -> excluded
        {"role": "assistant", "content": "c"},
    ]
    assert model_token_count(c, _Tok()) == 3  # 2 + 1


def test_counts_serialized_tool_call_arguments():
    args = {"message": "hello world"}
    c = [{
        "role": "assistant", "content": "",
        "tool_calls": [{"type": "function",
                        "function": {"name": "move", "arguments": args}}],
    }]
    assert model_token_count(c, _Tok()) == len(_Tok().encode(json.dumps(args)))


def test_tool_call_arguments_already_string():
    c = [{
        "role": "assistant", "content": "",
        "tool_calls": [{"type": "function",
                        "function": {"name": "move", "arguments": "raw token string"}}],
    }]
    assert model_token_count(c, _Tok()) == 3


def test_counts_reasoning_content_the_chat_parser_splits_out():
    # The real shape TRL hands a reward function, dumped from a live browsergym
    # rollout (/workspace/completion_dump.json): transformers' chat parser puts
    # Qwen3's think block in `reasoning_content` on any assistant message that
    # carries a tool call, leaving `content` empty, and only the final tool-less
    # message keeps `<think>` inline. Counting `content` alone read 18 tokens of
    # a 1024-token completion in e28 and made the cosine reward a near constant.
    c = [
        {"role": "assistant", "content": "",
         "reasoning_content": "t h i n k i n g",
         "tool_calls": [{"type": "function",
                         "function": {"name": "click", "arguments": "bid 24"}}]},
        {"role": "tool", "name": "click", "content": "page page page page"},
        {"role": "assistant", "content": "<think> done </think>"},
    ]
    # 8 reasoning + 2 tool-call args + 3 inline = 13; the tool message stays out.
    assert model_token_count(c, _Tok()) == 13


def test_single_message_dict():
    assert model_token_count({"role": "assistant", "content": "a b c"}, _Tok()) == 3


def test_single_message_dict_counts_reasoning_content():
    m = {"role": "assistant", "content": "a b", "reasoning_content": "c d e"}
    assert model_token_count(m, _Tok()) == 5


def test_empty_and_malformed_are_zero():
    assert model_token_count([], _Tok()) == 0
    assert model_token_count(None, _Tok()) == 0
    assert model_token_count(123, _Tok()) == 0

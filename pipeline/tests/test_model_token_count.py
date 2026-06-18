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


def test_single_message_dict():
    assert model_token_count({"role": "assistant", "content": "a b c"}, _Tok()) == 3


def test_empty_and_malformed_are_zero():
    assert model_token_count([], _Tok()) == 0
    assert model_token_count(None, _Tok()) == 0
    assert model_token_count(123, _Tok()) == 0

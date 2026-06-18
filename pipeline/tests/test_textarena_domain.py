import inspect

from domains.textarena.domain import TextArenaDomain
from domains.textarena.adapter import TextArenaEnvAdapter


class _Obs:
    def __init__(self, prompt="Guess a word."):
        self.prompt = prompt


class _Result:
    def __init__(self):
        self.observation = _Obs()


class _FakeClient:
    def reset(self, **kwargs):
        return _Result()


def test_multi_turn_flag_and_server_module():
    d = TextArenaDomain()
    assert d.multi_turn is True
    assert d.server_module == "textarena_env.server.app"


def test_make_env_factory_is_zero_arg_and_builds_adapter():
    d = TextArenaDomain()
    factory = d.make_env_factory("http://x", {"env_id": "Wordle-v0"}, client_factory=_FakeClient)
    assert callable(factory) and len(inspect.signature(factory).parameters) == 0
    env = factory()
    assert isinstance(env, TextArenaEnvAdapter)
    assert env.reset(seed=1) == "Guess a word."


def test_eval_tools_is_move():
    d = TextArenaDomain()
    env = d.make_env_factory("http://x", {}, client_factory=_FakeClient)()
    assert d.eval_tools(env) == [env.move]


def test_build_seed_dataset_distinct_seeds_and_prompt():
    d = TextArenaDomain()
    ds = d.build_seed_dataset({"env_id": "Wordle-v0"}, n=4, seed_base=10)
    assert len(ds) == 4
    assert [r["seed"] for r in ds] == [10, 11, 12, 13]
    assert all(r["prompt"][0]["role"] == "user" for r in ds)


def test_episode_messages_wraps_observation():
    d = TextArenaDomain()
    msgs = d.episode_messages("BOARD STATE")
    assert msgs[0]["role"] == "user" and "BOARD STATE" in msgs[0]["content"]


def test_server_env_maps_config_to_textarena_vars():
    d = TextArenaDomain()
    env = d.server_env({"env_id": "Wordle-v0", "num_players": 1, "max_turns": 6})
    assert env["TEXTARENA_ENV_ID"] == "Wordle-v0"
    assert env["TEXTARENA_NUM_PLAYERS"] == "1"
    assert env["TEXTARENA_MAX_TURNS"] == "6"


def test_server_env_defaults_and_omits_unset_max_turns():
    d = TextArenaDomain()
    env = d.server_env({})
    assert env["TEXTARENA_ENV_ID"] == "Wordle-v0"
    assert env["TEXTARENA_NUM_PLAYERS"] == "1"
    assert "TEXTARENA_MAX_TURNS" not in env

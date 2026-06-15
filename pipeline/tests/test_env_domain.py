from domains.reasoning_gym.domain import ReasoningGymDomain


class _FakeStep:
    """Stands in for an OpenEnv StepResult."""

    def __init__(self, reward):
        self.reward = reward


class _StubTok:
    pass


def test_episode_reward_reads_step_result():
    assert ReasoningGymDomain().episode_reward(_FakeStep(1.0)) == 1.0


def test_is_correct_from_reward_sign():
    d = ReasoningGymDomain()
    assert d.is_correct(_FakeStep(1.0)) is True
    assert d.is_correct(_FakeStep(0.0)) is False


def test_chat_template_sets_attr():
    tok = _StubTok()
    ReasoningGymDomain().build_chat_template(tok)
    assert "start_working_out" in tok.chat_template
    assert "SOLUTION" in tok.chat_template or "add_generation_prompt" in tok.chat_template

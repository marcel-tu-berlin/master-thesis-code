# The two MiniWoB families the feasibility probe selected (see
# pipeline/runs/browsergym_feasibility_findings.md). click-option supplies
# headroom (0.50 base success, inside the 40-80% band); click-checkboxes supplies
# the property that headroom alone does not: its success (0.35) sits BELOW its
# termination rate (0.50), so task performance and the off-target axis can move
# independently. On click-option every terminated episode is correct, which would
# make a non-termination penalty indistinguishable from a task reward.
_DEFAULT_TASKS = ("click-option", "click-checkboxes")

# Cap the accessibility tree injected back after each action. The probed families
# render ~300 chars, but a MiniWoB page with a long list can be much larger, and
# in a multi-turn episode every result re-enters the context. Qwen3's think block
# already costs ~300 tokens per turn and accumulates, so an unbounded observation
# is what pushes a later turn past the budget.
_MAX_OBS_CHARS = 2000


def _obs_text(observation) -> str:
    """The page as the model should see it: accessibility tree, bids included."""
    text = (getattr(observation, "axtree_txt", "") or
            getattr(observation, "text", "") or "")
    text = str(text).strip()
    if len(text) > _MAX_OBS_CHARS:
        text = text[:_MAX_OBS_CHARS] + "\n...[truncated]"
    return text


class BrowserGymEnvAdapter:
    """TRL environment_factory adapter for the OpenEnv browsergym env (MiniWoB).

    TRL's GRPOTrainer creates one instance per rollout slot, calls reset(**row)
    (whose return becomes the prompt), exposes every *other* public method as a
    tool, and reads the task reward off `self.reward`. The public surface is
    exactly {reset, click, noop}.

    Two tools, not the five browsergym ships. `fill`, `send_keys` and `scroll` do
    nothing on click-option or click-checkboxes - both families are solved by
    clicking radios/checkboxes and then Submit - and a tool that cannot affect the
    task would only add a hallucination target. `noop` is kept deliberately: it is
    a literal do-nothing action, so stalling becomes a tool call we can count
    rather than an absence we have to infer. Add `fill` when a task needs typing.

    The action bridge is upstream: browsergym takes a string DSL
    (`click("13")`), and `harness.build_browsergym_action_str(name, args)`
    translates a tool call into it, so this adapter does not parse or format
    actions itself.

    Unlike finqa, no server patch is needed. reset(seed=N) is deterministic
    (verified: same seed reproduces goal and accessibility tree, and survives an
    intervening reset), and reset(task_name=...) switches family per episode, so
    one server serves the whole mixed task set.

    The client is injectable so the unit tests exercise the adapter logic without
    the OpenEnv install or a running server.
    """

    def __init__(self, base_url, env_config=None, client=None):
        self._env_config = dict(env_config or {})
        # An ABSENT `tasks` takes the default mix; an explicitly EMPTY one is a
        # config error. Collapsing the two would silently train the default mix
        # under a config that says otherwise.
        tasks = self._env_config.get("tasks")
        tasks = _DEFAULT_TASKS if tasks is None else tasks
        self._tasks = tuple(str(t) for t in tasks)
        if not self._tasks:
            raise ValueError("browsergym env_config['tasks'] must name at least one task")
        self._client = client if client is not None else self._connect(base_url)
        self.reward = 0.0
        self.done = False

    @staticmethod
    def _connect(base_url):
        # Lazy: the browsergym client lives in the cloned OpenEnv repo's envs/
        # package (not on PyPI). .sync() wraps it for TRL, which rejects async
        # tools, matching the reasoning_gym/textarena/finqa pattern.
        from browsergym_env import BrowserGymEnv

        return BrowserGymEnv(base_url).sync()

    def _make_action(self, tool, arguments):
        """Tool call -> the browsergym action object.

        The one seam that touches the OpenEnv clone on the action path, imported
        lazily and kept in a single method so the unit tests can replace it and
        run without browsergym installed.
        """
        from browsergym_env import BrowserGymAction
        from browsergym_env.harness import build_browsergym_action_str

        return BrowserGymAction(action_str=build_browsergym_action_str(tool, arguments))

    def _act(self, tool, **arguments) -> str:
        """Send one action and return the resulting page, or the env's error."""
        if self.done:
            # Done-guard: once the episode is over, further tool calls must not
            # overwrite the terminal reward.
            return "The episode is already finished; no further actions are accepted."
        result = self._client.step(self._make_action(tool, arguments))
        # MiniWoB pays out on the terminating step and 0 before it, so last-wins
        # is the terminal reward. The done-guard above stops a later call from
        # overwriting it.
        self.reward = float(getattr(result, "reward", 0.0) or 0.0)
        self.done = bool(getattr(result, "done", False))
        observation = getattr(result, "observation", None)
        error = str(getattr(observation, "error", "") or "")
        page = _obs_text(observation)
        if error:
            return f"Action error: {error}\nPage now:\n{page}"
        return f"Page now:\n{page}"

    def reset(self, *, seed=None, prompt=None, **_row):
        """Load this seed's page and return its goal plus accessibility tree.

        TRL appends the returned string to the prompt's last user message, then
        applies the tool-calling chat template. The task family is a pure function
        of the seed (`tasks[seed % len(tasks)]`) because the eval loop resets with
        a seed and nothing else, and because every rollout slot in a GRPO group
        must land on the same page. `prompt` and other dataset columns arrive as
        kwargs and are ignored.
        """
        s = 0 if seed is None else int(seed)
        task = self._tasks[s % len(self._tasks)]
        result = self._client.reset(seed=s, task_name=task)
        self.reward = 0.0
        self.done = False
        observation = getattr(result, "observation", None)
        goal = str(getattr(observation, "goal", "") or "").strip()
        return f"Task: {goal}\n\nPage:\n{_obs_text(observation)}"

    def click(self, bid: str) -> str:
        """Click an element on the page and see the page that results.

        TRL renders this method into the tool spec the model sees, so this summary
        and the Args entry below are the tool's model-facing documentation.
        transformers' get_json_schema requires a Google-style Args entry for every
        parameter.

        Args:
            bid: The element id in square brackets in the accessibility tree, without
                the brackets (for `[24] radio 'O2F2ioz'`, pass "24").
        """
        return self._act("click", bid=str(bid))

    def noop(self) -> str:
        """Do nothing this turn and look at the page again.

        Use only when no action is appropriate; it does not advance the task.
        """
        return self._act("noop")

import math

from training.config_schema import resolve_interaction_tools, resolve_observation_format

# The two MiniWoB families the feasibility probe selected (see
# pipeline/runs/browsergym_feasibility_findings.md). click-option supplies
# headroom (0.50 base success, inside the 40-80% band); click-checkboxes supplies
# the property that headroom alone does not: its success (0.35) sits BELOW its
# termination rate (0.50), so task performance and the off-target axis can move
# independently. On click-option every terminated episode is correct, which would
# make a non-termination penalty indistinguishable from a task reward.
_DEFAULT_TASKS = ("click-option", "click-checkboxes")


def _obs_text(observation, observation_format="axtree") -> str:
    """Keep the full page, including trailing controls and their action IDs."""
    if observation_format == "visible_html":
        text = getattr(observation, "pruned_html", "")
        if not text:
            raise ValueError("visible_html observation is missing from the env server")
        return str(text).strip()
    text = (
        getattr(observation, "axtree_txt", "") or getattr(observation, "text", "") or ""
    )
    return str(text).strip()


def task_for_seed(tasks, seed) -> str:
    """The MiniWoB family a seed lands on: `tasks[seed % len(tasks)]`.

    Module-level and not just inlined in `reset` because offline analysis needs
    the same mapping to group episodes by family (`eval.paired`), and a second
    copy of it would silently mislabel every per-family number the day the rule
    changes here.
    """
    tasks = tuple(tasks)
    return str(tasks[int(seed) % len(tasks)])


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
    rather than an absence we have to infer. The opt-in adapters below add typing
    and scrolling without changing existing experiments' tool schemas.

    The action bridge is upstream: browsergym takes a string DSL
    (`click("13")`), and `harness.build_browsergym_action_str(name, args)`
    translates the original tools. Scrolling uses the installed BrowserGym
    primitives directly because the pinned harness has an incompatible signature.

    No server patch is needed, which is the bar the deleted finqa env failed:
    reset(seed=N) is deterministic (verified - the same seed reproduces goal and
    accessibility tree, and survives an intervening reset), and
    reset(task_name=...) switches family per episode, so one server serves the
    whole mixed task set.

    The client is injectable so the unit tests exercise the adapter logic without
    the OpenEnv install or a running server.
    """

    def __init__(self, base_url, env_config=None, client=None):
        self._env_config = dict(env_config or {})
        self._observation_format = resolve_observation_format(self._env_config)
        self._interaction_tools = resolve_interaction_tools(self._env_config)
        if (
            any(name.startswith("mouse_") for name in self._interaction_tools)
            and self._observation_format != "visible_html"
        ):
            raise ValueError(
                "Coordinate interaction_tools require visible_html observations"
            )
        # Bind only selected methods: TRL discovers public instance methods, so
        # disabled tools must be absent rather than returning an error on use.
        for name in self._interaction_tools:
            setattr(self, name, getattr(_InteractionTools, name).__get__(self))
        # An ABSENT `tasks` takes the default mix; an explicitly EMPTY one is a
        # config error. Collapsing the two would silently train the default mix
        # under a config that says otherwise.
        tasks = self._env_config.get("tasks")
        tasks = _DEFAULT_TASKS if tasks is None else tasks
        self._tasks = tuple(str(t) for t in tasks)
        if not self._tasks:
            raise ValueError(
                "browsergym env_config['tasks'] must name at least one task"
            )
        self._client = client if client is not None else self._connect(base_url)
        self.reward = 0.0
        self.done = False

    @staticmethod
    def _connect(base_url):
        # Lazy: the browsergym client lives in the cloned OpenEnv repo's envs/
        # package (not on PyPI). .sync() wraps it for TRL, which rejects async
        # tools, matching the reasoning_gym pattern.
        from browsergym_env import BrowserGymEnv

        # Keep active RPC deadlines, but tolerate delayed Pong handling while
        # model computation holds the client's Python interpreter.
        return BrowserGymEnv(base_url, websocket_ping_timeout_s=None).sync()

    def _make_action(self, tool, arguments):
        """Tool call -> the browsergym action object.

        The one seam that touches the OpenEnv clone on the action path, imported
        lazily and kept in a single method so the unit tests can replace it and
        run without browsergym installed.
        """
        from browsergym_env import BrowserGymAction

        if tool in self._interaction_tools:
            if tool == "mouse_drag":
                fx, fy = arguments["from_x"], arguments["from_y"]
                tx, ty = arguments["to_x"], arguments["to_y"]
                # jQuery sortable needs movement after the drag-start event;
                # BrowserGym's single-move drag silently leaves it unchanged.
                actions = [f"mouse_down({fx!r}, {fy!r})"]
                for fraction in (0.25, 0.5, 0.75, 1):
                    actions.append(
                        f"mouse_move({fx + (tx - fx) * fraction!r}, {fy + (ty - fy) * fraction!r})"
                    )
                actions.append(f"mouse_up({tx!r}, {ty!r})")
                return BrowserGymAction(action_str="\n".join(actions))
            native = {
                "select_options": "select_option",
            }.get(tool, tool)
            if tool == "press":
                arguments = {"bid": arguments["bid"], "key_comb": arguments["key"]}
            args = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
            return BrowserGymAction(action_str=f"{native}({args})")
        from browsergym_env.harness import build_browsergym_action_str

        return BrowserGymAction(action_str=build_browsergym_action_str(tool, arguments))

    def _interact(self, tool, **arguments):
        if not self.done:
            for key, value in arguments.items():
                if key == "options":
                    valid = isinstance(value, list) and all(
                        isinstance(v, str) for v in value
                    )
                elif key in {"x", "y", "from_x", "from_y", "to_x", "to_y"}:
                    valid = (
                        type(value) in (int, float)
                        and math.isfinite(value)
                        and value >= 0
                    )
                else:
                    valid = isinstance(value, str) and bool(value.strip())
                if not valid:
                    return f"Action error: invalid {tool} argument {key}"
        return self._act(tool, **arguments)

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
        if not error and getattr(observation, "last_action_error", False):
            # OpenEnv keeps normal BrowserGym action errors in metadata; its
            # top-level error field is reserved for wrapper exceptions.
            metadata = getattr(observation, "metadata", None) or {}
            error = str(
                (metadata.get("browsergym_obs") or {}).get("last_action_error")
                or "Action failed (no error details provided)."
            )
        page = _obs_text(observation, self._observation_format)
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
        task = task_for_seed(self._tasks, s)
        result = self._client.reset(seed=s, task_name=task)
        self.reward = 0.0
        self.done = False
        observation = getattr(result, "observation", None)
        goal = str(getattr(observation, "goal", "") or "").strip()
        return (
            f"Task: {goal}\n\nPage:\n{_obs_text(observation, self._observation_format)}"
        )

    def click(self, bid: str) -> str:
        """Click an element on the page and see the page that results.

        TRL renders this method into the tool spec the model sees, so this summary
        and the Args entry below are the tool's model-facing documentation.
        transformers' get_json_schema requires a Google-style Args entry for every
        parameter.

        Args:
            bid: The element id in square brackets or the HTML bid attribute
                (for `[24] radio` or `bid="24"`, pass "24").
        """
        return self._act("click", bid=str(bid))

    def noop(self) -> str:
        """Do nothing this turn and look at the page again.

        Use only when no action is appropriate; it does not advance the task.
        """
        return self._act("noop")


class _BrowserGymFillEnvAdapter(BrowserGymEnvAdapter):
    """The existing adapter plus fill, selected only by explicit enable_fill."""

    def fill(self, bid: str, text: str) -> str:
        """Replace a text field's value and see the resulting page.

        Args:
            bid: The text field's element id in square brackets or its HTML bid attribute.
            text: The complete value to enter in the field.
        """
        return self._act("fill", bid=str(bid), text=str(text))


class _BrowserGymScrollEnvAdapter(BrowserGymEnvAdapter):
    """Opt-in scrolling, targeted at a visible element in the scroll area."""

    def _make_action(self, tool, arguments):
        if tool != "scroll":
            return super()._make_action(tool, arguments)
        from browsergym_env import BrowserGymAction

        # BrowserGym 0.14.3 accepts hover + wheel in one native action. The pinned
        # OpenEnv harness instead emits scroll('down'), which BrowserGym rejects.
        delta = 120 if arguments["direction"] == "down" else -120
        return BrowserGymAction(
            action_str=f"hover({arguments['bid']!r})\nscroll(0, {delta})"
        )

    def scroll(self, bid: str, direction: str) -> str:
        """Move the pointer to an element, scroll 120 pixels, and see the page.

        Args:
            bid: The HTML bid of a visible scroll area or an element inside it.
            direction: The vertical direction, either "up" or "down".
        """
        if not self.done:
            # Malformed model arguments are recoverable action feedback. Raising
            # here would abort both production loops as an infrastructure error.
            if not isinstance(bid, str) or not bid.strip():
                return "Action error: scroll bid must be a nonempty string"
            if direction not in ("up", "down"):
                return "Action error: scroll direction must be 'up' or 'down'"
        return self._act("scroll", bid=bid, direction=direction)


class _BrowserGymFillScrollEnvAdapter(
    _BrowserGymFillEnvAdapter, _BrowserGymScrollEnvAdapter
):
    """Both opt-ins; the inherited tools retain their original schemas."""


class _InteractionTools(BrowserGymEnvAdapter):
    """Native primitives bound individually to an adapter only when configured."""

    def press(self, bid: str, key: str) -> str:
        """Focus an element and press a key or key combination; see the result.

        Args:
            bid: The element's bid.
            key: A key such as ArrowRight, Home, Enter, or Control+a.
        """
        return self._interact("press", bid=bid, key=key)

    def select_options(self, bid: str, options: list[str]) -> str:
        """Set the selected options of a native select element; see the result.

        Args:
            bid: The select element's bid.
            options: The complete list of option values to select.
        """
        return self._interact("select_options", bid=bid, options=options)

    def hover(self, bid: str) -> str:
        """Move the pointer over an element and see the resulting page.

        Args:
            bid: The element's bid.
        """
        return self._interact("hover", bid=bid)

    def mouse_click(self, x: float, y: float) -> str:
        """Click a page position in pixels and see the result.

        Args:
            x: Horizontal position from the viewport's left edge. HTML box attributes give (left, top, right, bottom) in pixels.
            y: Vertical position from the viewport's top edge.
        """
        return self._interact("mouse_click", x=x, y=y)

    def mouse_move(self, x: float, y: float) -> str:
        """Move the pointer to a page position without clicking; see the result.

        Args:
            x: Horizontal position from the viewport's left edge. HTML box attributes give (left, top, right, bottom) in pixels.
            y: Vertical position from the viewport's top edge.
        """
        return self._interact("mouse_move", x=x, y=y)

    def mouse_drag(self, from_x: float, from_y: float, to_x: float, to_y: float) -> str:
        """Drag between two page positions in pixels and see the result.

        Args:
            from_x: Starting horizontal position. HTML box attributes give (left, top, right, bottom) in pixels.
            from_y: Starting vertical position.
            to_x: Destination horizontal position.
            to_y: Destination vertical position.
        """
        return self._interact(
            "mouse_drag", from_x=from_x, from_y=from_y, to_x=to_x, to_y=to_y
        )

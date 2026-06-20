import os

# Default matches FinQADomain.server_env / the e14 config: the dataset is
# downloaded into the OpenEnv clone. The adapter reads it directly (see reset).
_DEFAULT_DATA_PATH = "/workspace/OpenEnv/envs/finqa_env/data"

# Module-level cache: the question CSV is loaded once per training process and
# shared across every rollout-slot adapter (keyed by csv path).
_QUESTION_CACHE: dict = {}

# Cap each tool result's length. Table dumps can be large, and in a multi-turn
# episode every result is injected back into the context; an unbounded result can
# push the next turn's prompt past max_seq and crash generation. ~1200 chars is
# enough to read table names / a few rows while keeping episodes inside budget.
_MAX_RESULT_CHARS = 1200


def _load_questions(data_path):
    """Load (user_query, company) rows from finqa.csv, in file order.

    The server loads the same CSV the same way (pd.read_csv) and, after our
    seed-patch, selects questions[seed % len]. We mirror that read so the
    question text we show the model matches the row the server scores against -
    both are a deterministic function of the same seed over the same file.
    """
    csv_path = os.path.join(data_path, "benchmark_questions", "finqa.csv")
    cached = _QUESTION_CACHE.get(csv_path)
    if cached is None:
        import pandas as pd

        df = pd.read_csv(csv_path)
        cached = [
            {"user_query": str(r["user_query"]), "company": str(r["company"])}
            for _, r in df.iterrows()
        ]
        _QUESTION_CACHE[csv_path] = cached
    return cached


class FinQAEnvAdapter:
    """TRL environment_factory adapter for the OpenEnv finqa env (multi-turn, MCP).

    TRL's GRPOTrainer creates one instance per rollout slot, calls reset(**row)
    (whose return becomes the prompt), exposes every *other* public method as a
    tool the model may call, and reads the task reward off `self.reward`. The
    public surface is exactly {reset, get_descriptions, get_table_info,
    sql_query, submit_answer}: four tools, since finqa is a tool-use QA env (the
    model explores SEC-filing tables, then submits a numeric answer).

    Three finqa specifics break the single-tool reasoning_gym pattern:

    1. MCP client. The env is served over MCP; `FinQAEnv` is an MCPToolClient.
       Its `call_tool(name, **kw)` returns ONLY the tool's string result and
       discards reward/done. So the non-terminal tools use call_tool, but the
       terminal `submit_answer` MUST go through `step(CallToolAction(...))` to
       read the env reward off the StepResult.

    2. Question delivery. finqa's reset puts the question in Observation.metadata,
       but the OpenEnv serializer drops metadata, so the question never reaches
       the client over the wire. The question is a pure function of the seed over
       finqa.csv, so we read it locally (same file, same seed -> same row the
       server scores) while still calling the server reset to arm its ground
       truth. The company is fixed per episode, so the tool methods inject it.

    3. Multi-turn (up to FINQA_MAX_STEPS tool calls), so the adapter keeps
       `self.done` like textarena. Reward is binary (1 correct / 0 wrong), set
       only after submit_answer.

    The client is injectable so the unit tests exercise the adapter logic without
    the OpenEnv install or a running server.
    """

    def __init__(self, base_url, env_config=None, client=None):
        self._env_config = dict(env_config or {})
        self._data_path = self._env_config.get("data_path", _DEFAULT_DATA_PATH)
        self._client = client if client is not None else self._connect(base_url)
        self._company = ""
        self.reward = 0.0
        self.done = False

    @staticmethod
    def _connect(base_url):
        # Lazy: the finqa client lives in the cloned OpenEnv repo's envs/ package
        # (not on PyPI). FinQAEnv is an MCPToolClient; .sync() wraps it for TRL,
        # matching the reasoning_gym/textarena pattern (the sync client connects
        # on first call - no explicit connect() needed).
        from finqa_env import FinQAEnv

        return FinQAEnv(base_url).sync()

    def _call(self, tool, **arguments):
        # Non-terminal MCP tool call: returns the tool's string result. call_tool
        # discards reward/done, which is correct here (these tools carry neither).
        result = str(self._client.call_tool(tool, **arguments))
        if len(result) > _MAX_RESULT_CHARS:
            result = result[:_MAX_RESULT_CHARS] + "\n...[truncated]"
        return result

    def reset(self, *, seed=None, prompt=None, **_row):
        """Arm the server for this seed and return the question + company text.

        TRL appends the returned string to the prompt's last user message, then
        applies the tool-calling chat template. The server reset arms the
        ground-truth answer for scoring (deterministic after the seed-patch); the
        question text shown to the model is read locally from finqa.csv at the
        same seed (the server's metadata channel is dropped by the serializer).
        `prompt` and other dataset columns arrive as kwargs and are ignored.
        """
        if seed is not None:
            self._client.reset(seed=seed)
        else:
            self._client.reset()
        questions = _load_questions(self._data_path)
        row = questions[(seed or 0) % len(questions)]
        self._company = row["company"]
        self.reward = 0.0
        self.done = False
        return f"Question: {row['user_query']}\nCompany: {self._company}"

    def get_descriptions(self) -> str:
        """List the data tables available for this company's SEC filing.

        Call this first to discover which tables exist. Returns a JSON list of
        table names to pass to get_table_info / sql_query.
        """
        return self._call("get_descriptions", company_name=self._company)

    def get_table_info(self, table_name: str) -> str:
        """Inspect one table's schema before querying it.

        Returns the table's description, column data types, and sample values.

        Args:
            table_name: A table name returned by get_descriptions.
        """
        return self._call("get_table_info", company_name=self._company, table_name=table_name)

    def sql_query(self, table_name: str, query: str) -> str:
        """Run a read-only SQL query against one table and read the rows.

        The query must filter (a WHERE/HAVING/IN/... clause is required; bare
        SELECT * is rejected).

        Args:
            table_name: The table to query (from get_descriptions).
            query: A SQL SELECT statement with a filter clause.
        """
        return self._call("sql_query", company_name=self._company, table_name=table_name, query=query)

    def submit_answer(self, answer: str) -> str:
        """Submit your final numeric answer and end the episode.

        TRL renders this method into the tool spec the model sees, so this
        summary and the Args description below are the tool's model-facing
        documentation. transformers' get_json_schema requires a Google-style Args
        entry for every parameter.

        Args:
            answer: The final answer to the question (a number, e.g. "6.118").
        """
        if self.done:
            # Done-guard: once the episode is over, a model that keeps calling
            # tools cannot corrupt the terminal reward.
            return "The question is already answered; no further tools are accepted."
        # submit_answer carries the reward, which call_tool drops - so go through
        # step(CallToolAction) and read the top-level StepResult.reward.
        from openenv.core.env_server.mcp_types import CallToolAction

        result = self._client.step(
            CallToolAction(tool_name="submit_answer", arguments={"answer": answer})
        )
        self.reward = float(getattr(result, "reward", 0.0) or 0.0)
        self.done = bool(getattr(result, "done", True))
        return f"Recorded answer: {answer}"

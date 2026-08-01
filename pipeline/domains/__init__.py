"""Domain factory: `training.env` -> the EnvDomain instance.

Single source of truth for the dispatch. training.train and eval.runner used to
carry a copy each, and the eval copy went stale - it never learned finqa/repl,
so `python -m eval.runner` raised NotImplementedError on either. Imports are
lazy per branch so importing this package does not pull every env's deps.
"""


def build_domain(config: dict):
    env = (config.get("training") or {}).get("env")
    if env == "reasoning_gym":
        from domains.reasoning_gym import ReasoningGymDomain
        return ReasoningGymDomain()
    if env == "textarena":
        from domains.textarena import TextArenaDomain
        return TextArenaDomain()
    if env == "finqa":
        from domains.finqa import FinQADomain
        return FinQADomain()
    if env == "repl":
        from domains.repl import REPLDomain
        return REPLDomain()
    raise NotImplementedError(
        f"Env: {env!r} (known: reasoning_gym, textarena, finqa, repl)"
    )

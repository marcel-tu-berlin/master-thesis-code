"""Domain factory: `training.env` -> the EnvDomain instance.

Single source of truth for the dispatch. training.train and eval.runner used to
carry a copy each, and the eval copy went stale - it never learned the domains
added after it, so `python -m eval.runner` raised NotImplementedError on them.
Imports are lazy per branch so importing this package does not pull every env's
deps.
"""


def build_domain(config: dict):
    env = (config.get("training") or {}).get("env")
    if env == "reasoning_gym":
        from domains.reasoning_gym import ReasoningGymDomain
        return ReasoningGymDomain()
    if env == "browsergym":
        from domains.browsergym import BrowserGymDomain
        return BrowserGymDomain()
    raise NotImplementedError(
        f"Env: {env!r} (known: reasoning_gym, browsergym)"
    )

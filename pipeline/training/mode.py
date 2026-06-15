def select_mode(config: dict) -> str:
    """Return the training mode: 'dataset' (default) or 'agentic'.

    Kept as a tiny pure helper so train.py's dispatch is unit-testable without
    importing the GPU-heavy runner.
    """
    mode = (config.get("training") or {}).get("mode", "dataset")
    if mode not in ("dataset", "agentic"):
        raise ValueError(f"training.mode must be 'dataset' or 'agentic', got {mode!r}")
    return mode

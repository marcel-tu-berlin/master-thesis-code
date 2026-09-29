"""Pinned OpenEnv app with an opt-in visible DOM observation renderer."""

import importlib
import os


def visible_html(obs):
    from browsergym.utils.obs import flatten_dom_to_str, prune_html

    return prune_html(
        flatten_dom_to_str(
            obs["dom_object"],
            extra_properties=obs["extra_element_properties"],
            filter_visible_only=True,
        )
    )


if os.environ.get("BROWSERGYM_OBSERVATION_FORMAT") == "visible_html":
    environment = importlib.import_module(
        "browsergym_env.server.browsergym_environment"
    )
    # Replace only the pinned serializer's HTML renderer, not the task or scorer.
    vars(environment)["_get_pruned_html"] = visible_html

app = importlib.import_module("browsergym_env.server.app").app

"""Pinned OpenEnv app with an opt-in visible DOM observation renderer."""

import importlib
import os
from functools import partial


def visible_html(obs):
    from browsergym.utils.obs import flatten_dom_to_str, prune_html

    coordinates = os.environ.get("BROWSERGYM_COORDINATES") == "true"
    dom = obs["dom_object"]
    if coordinates:
        from domains.browsergym.observation import with_svg_boxes

        dom = with_svg_boxes(dom, obs["extra_element_properties"])
    return prune_html(
        flatten_dom_to_str(
            dom,
            extra_properties=obs["extra_element_properties"],
            filter_visible_only=True,
            with_bounding_box_coords=coordinates,
        )
    )


if os.environ.get("BROWSERGYM_OBSERVATION_FORMAT") == "visible_html":
    environment = importlib.import_module(
        "browsergym_env.server.browsergym_environment"
    )
    # Replace only the pinned serializer's HTML renderer, not the task or scorer.
    vars(environment)["_get_pruned_html"] = visible_html
    if os.environ.get("BROWSERGYM_COORDINATES") == "true":
        from browsergym.core.action.highlevel import HighLevelActionSet

        # The upstream default accepts bid actions but rejects coordinate
        # actions. Extend only the coordinate opt-in; keep the native executor.
        action_set = HighLevelActionSet(subsets=["bid", "coord"])
        vars(environment)["BrowserGymEnvironment"] = partial(
            environment.BrowserGymEnvironment,
            action_mapping=action_set.to_python_code,
        )

app = importlib.import_module("browsergym_env.server.app").app

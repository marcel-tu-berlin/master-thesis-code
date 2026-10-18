import copy

from domains.browsergym.observation import with_svg_boxes


def test_svg_boxes_use_native_screen_scale_without_mutating_the_snapshot():
    snapshot = {
        "strings": ["DIV", "svg", "circle", "bid", "13", "fill", "black"],
        "documents": [
            {
                "nodes": {
                    "nodeName": [0, 1, 2],
                    "nodeType": [1, 1, 1],
                    "parentIndex": [-1, 0, 1],
                    "attributes": [[], [3, 4], [5, 6]],
                },
                "layout": {
                    "nodeIndex": [0, 1, 2],
                    "bounds": [
                        [0, 0, 160, 210],
                        [2, 52, 160, 160],
                        [63.5, 53.5, 17, 17],
                    ],
                },
            }
        ],
    }
    original = copy.deepcopy(snapshot)
    props = {"13": {"bbox": [3, 78, 240, 240], "visibility": 1}}
    result = with_svg_boxes(snapshot, props)
    attrs = result["documents"][0]["nodes"]["attributes"][2]
    decoded = {
        result["strings"][attrs[i]]: result["strings"][attrs[i + 1]]
        for i in range(0, len(attrs), 2)
    }
    assert decoded == {"fill": "black", "box": "(95.2,80.2,120.8,105.8)"}
    assert snapshot == original
    props["13"]["visibility"] = 0
    assert with_svg_boxes(snapshot, props) == snapshot

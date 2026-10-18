"""Coordinate observations for SVG nodes that BrowserGym does not mark with bids."""

import copy
import math


def with_svg_boxes(dom_snapshot, extra_properties):
    """Add screen-coordinate boxes from native layout bounds, preserving the page."""
    result = copy.deepcopy(dom_snapshot)
    strings = result["strings"]
    for document in result["documents"]:
        nodes = document["nodes"]
        layout = document["layout"]
        bounds = dict(zip(layout["nodeIndex"], layout["bounds"], strict=True))
        attributes = [
            {strings[a[i]]: strings[a[i + 1]] for i in range(0, len(a), 2)}
            for a in nodes["attributes"]
        ]
        for index, box in bounds.items():
            if nodes["nodeType"][index] != 1 or "bid" in attributes[index]:
                continue
            parent = nodes["parentIndex"][index]
            while parent != -1:
                if strings[nodes["nodeName"][parent]].lower() == "svg":
                    props = extra_properties.get(attributes[parent].get("bid"), {})
                    if props.get("bbox") and props.get("visibility", 0) >= 0.5:
                        break
                parent = nodes["parentIndex"][parent]
            if parent == -1 or parent not in bounds:
                continue
            px, py, pw, ph = bounds[parent]
            if pw <= 0 or ph <= 0:
                continue
            sx, sy, sw, sh = props["bbox"]
            x, y, width, height = box
            # CDP reports page coordinates; BrowserGym actions use scaled
            # screenshot coordinates. The marked SVG root supplies that mapping.
            left = sx + (x - px) * sw / pw
            top = sy + (y - py) * sh / ph
            mapped = (left, top, left + width * sw / pw, top + height * sh / ph)
            if not all(math.isfinite(v) for v in mapped):
                raise ValueError("Non-finite SVG layout coordinates")
            value = "(" + ",".join(f"{v:.1f}" for v in mapped) + ")"
            nodes["attributes"][index].extend([len(strings), len(strings) + 1])
            strings.extend(["box", value])
    return result

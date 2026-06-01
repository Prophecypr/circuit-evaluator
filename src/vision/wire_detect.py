"""Wire detection + connection grouping for machine-drawn circuits.

Uses Netlistify's union-find segment merging approach:
- Hough line detection → raw segments
- Union-find merge by endpoint proximity → wire groups
- Match group endpoints to component ports → node graph
"""

import cv2, numpy as np, math
from collections import Counter, defaultdict
from itertools import chain


def detect_wire_segments(gray, components, h, w):
    """Detect wire line segments using Canny + Hough, excluding component interiors.

    Returns list of ((x1,y1),(x2,y2)) segments.
    """
    # Component mask
    comp_mask = np.zeros((h, w), dtype=np.uint8)
    for c in components:
        x1, y1, x2, y2 = c["xyxy"]
        cv2.rectangle(comp_mask, (x1+3, y1+3), (x2-3, y2-3), 255, -1)

    # Edge detection
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges_masked = cv2.bitwise_and(edges, cv2.bitwise_not(comp_mask))

    # Hough lines
    lines = cv2.HoughLinesP(edges_masked, 1, np.pi/180, threshold=30,
                            minLineLength=15, maxLineGap=10)

    segments = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = [int(v) for v in l[0]]
            segments.append(((x1, y1), (x2, y2)))

    return segments


def _distance(p1, p2):
    """Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def deduplicate_segments(segments, threshold=8):
    """Remove near-identical segments."""
    unique = []
    for i, seg in enumerate(segments):
        is_dup = False
        for j in range(i):
            d1 = _distance(segments[i][0], segments[j][0])
            d2 = _distance(segments[i][1], segments[j][1])
            d3 = _distance(segments[i][0], segments[j][1])
            d4 = _distance(segments[i][1], segments[j][0])
            if min(d1+d2, d3+d4) < threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(seg)
    return unique


def group_segments_union_find(segments, threshold=12):
    """Union-find grouping: segments that share endpoints are in the same wire group.

    Returns list of groups, each group = list of segments.
    """
    n = len(segments)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        for j in range(i + 1, n):
            for p1 in segments[i]:
                for p2 in segments[j]:
                    if _distance(p1, p2) <= threshold:
                        union(i, j)
                        break
                else:
                    continue
                break

    # Collect groups
    groups = defaultdict(list)
    for i in range(n):
        root = find(i)
        groups[root].append(segments[i])

    return list(groups.values())


def get_group_endpoints(group):
    """Get unique endpoints of a wire group.

    Points shared by multiple segments within the group are junctions (removed).
    Only points that appear once (leaf endpoints) are kept.
    """
    all_pts = list(chain.from_iterable(group))
    # Round coordinates for matching
    rounded = [(round(p[0]/3)*3, round(p[1]/3)*3) for p in all_pts]
    counts = Counter(rounded)

    unique = []
    for pt, rpt in zip(all_pts, rounded):
        if counts[rpt] == 1:
            unique.append(pt)
    return unique


def match_endpoints_to_ports(components, groups, max_dist=25):
    """Match wire group endpoints to nearest component ports.

    Returns:
        port_groups: {(comp_idx, port_idx): group_idx}
        group_ports: {group_idx: [(comp_idx, port_idx), ...]}
    """
    port_groups = {}
    group_ports = defaultdict(list)

    for gi, group in enumerate(groups):
        endpoints = get_group_endpoints(group)
        for ci, c in enumerate(components):
            for pi, (px, py) in enumerate(c["ports"]):
                best_d = max_dist
                for ex, ey in endpoints:
                    d = math.hypot(px - ex, py - ey)
                    if d < best_d:
                        best_d = d
                if best_d < max_dist:
                    port_groups[(ci, pi)] = gi
                    group_ports[gi].append((ci, pi))

    return port_groups, group_ports


def build_node_graph(components, groups, max_dist=25):
    """Build complete node graph from wire segments and components.

    Key insight: wire segments are cut at component bbox edges, so ports of the
    same component in different wire groups should be connected THROUGH the component.
    E.g., resistor top port ↔ wire group A, resistor bottom port ↔ wire group B
    — this means the resistor connects wire groups A and B.

    Returns:
        port_nodes: {(comp_idx, port_idx): node_id}
        connections: [(ci, pi, cj, pj, distance)]
    """
    port_groups, group_ports = match_endpoints_to_ports(components, groups, max_dist)

    connections = []
    port_nodes = {}
    next_node = 1

    # GND → 0
    for c in components:
        if c["name"] == "GND":
            for pi in range(len(c["ports"])):
                port_nodes[(c["idx"], pi)] = 0

    # 1. Ports in the same wire group share a node
    for gi, ports_in_group in group_ports.items():
        for i in range(len(ports_in_group)):
            for j in range(i + 1, len(ports_in_group)):
                ci, pi = ports_in_group[i]
                cj, pj = ports_in_group[j]
                if ci == cj:
                    continue
                pi_pos = components[ci]["ports"][pi]
                pj_pos = components[cj]["ports"][pj]
                d = math.hypot(pi_pos[0]-pj_pos[0], pi_pos[1]-pj_pos[1])
                connections.append((ci, pi, cj, pj, d))

    # 2. Within a component: its ports are connected through the component itself
    #    Merge wire groups touching the same component's different ports.
    #    Use union-find to handle transitive merges (A-B, B-C, C-D → all merged)
    n_groups = len(group_ports)
    group_ids = list(group_ports.keys())
    parent = {g: g for g in group_ids}

    def find_g(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    def union_g(g1, g2):
        r1, r2 = find_g(g1), find_g(g2)
        if r1 != r2:
            parent[r2] = r1

    for c in components:
        comp_groups = set()
        for pi in range(len(c["ports"])):
            g = port_groups.get((c["idx"], pi))
            if g is not None and g in parent:
                comp_groups.add(g)
        if len(comp_groups) >= 2:
            groups_list = list(comp_groups)
            for i in range(1, len(groups_list)):
                union_g(groups_list[0], groups_list[i])

    # Rebuild group_ports after merging
    merged_ports = defaultdict(list)
    for g in group_ids:
        root = find_g(g)
        merged_ports[root].extend(group_ports.get(g, []))
    # Deduplicate
    for root in merged_ports:
        seen = set()
        unique = []
        for ci, pi in merged_ports[root]:
            key = (ci, pi)
            if key not in seen:
                seen.add(key)
                unique.append((ci, pi))
        merged_ports[root] = unique
    group_ports = merged_ports

    # 3. Rebuild group connections after merging
    connections = []
    for gi, ports_in_group in group_ports.items():
        for i in range(len(ports_in_group)):
            for j in range(i + 1, len(ports_in_group)):
                ci, pi = ports_in_group[i]
                cj, pj = ports_in_group[j]
                if ci == cj:
                    continue
                pi_pos = components[ci]["ports"][pi]
                pj_pos = components[cj]["ports"][pj]
                d = math.hypot(pi_pos[0]-pj_pos[0], pi_pos[1]-pj_pos[1])
                connections.append((ci, pi, cj, pj, d))

    # 4. Proximity fallback: very close ports on different components
    for i, ci in enumerate(components):
        for pi, (px_i, py_i) in enumerate(ci["ports"]):
            for j, cj in enumerate(components):
                if j <= i:
                    continue
                for pj, (px_j, py_j) in enumerate(cj["ports"]):
                    d = math.hypot(px_i - px_j, py_i - py_j)
                    if d < 35:
                        connections.append((ci["idx"], pi, cj["idx"], pj, d))

    # Greedy assignment
    for ci, pi, cj, pj, dist in sorted(connections, key=lambda x: x[4]):
        if ci == cj:
            continue
        if (ci, pi) not in port_nodes and (cj, pj) not in port_nodes:
            nid = next_node; next_node += 1
            port_nodes[(ci, pi)] = nid
            port_nodes[(cj, pj)] = nid
        elif (ci, pi) in port_nodes and (cj, pj) not in port_nodes:
            port_nodes[(cj, pj)] = port_nodes[(ci, pi)]
        elif (cj, pj) in port_nodes and (ci, pi) not in port_nodes:
            port_nodes[(ci, pi)] = port_nodes[(cj, pj)]

    # Anti-short: no two ports of same component share a node
    for c in components:
        seen = set()
        for pi in range(len(c["ports"])):
            key = (c["idx"], pi)
            nid = port_nodes.get(key)
            if nid is not None and nid in seen and len(c["ports"]) <= 2:
                port_nodes[key] = next_node; next_node += 1
            if nid is not None:
                seen.add(nid)

    # Unconnected → solo nodes
    for c in components:
        for pi in range(len(c["ports"])):
            if (c["idx"], pi) not in port_nodes:
                port_nodes[(c["idx"], pi)] = next_node; next_node += 1

    return port_nodes, connections


def draw_wire_overlay(img_path, components, wire_segments, groups, port_nodes,
                      connections, out_path):
    """Draw annotated output with wires, ports, nodes, and connections."""
    img = cv2.imread(str(img_path))

    # Wire segments in light orange (deduplicated)
    for seg in wire_segments:
        cv2.line(img, seg[0], seg[1], (80, 160, 255), 1)

    # Wire groups highlighted (each group = different shade)
    group_colors = [(255,200,100), (200,255,100), (100,255,200),
                    (255,150,150), (150,255,150), (150,150,255)]
    for gi, group in enumerate(groups):
        color = group_colors[gi % len(group_colors)]
        for seg in group:
            cv2.line(img, seg[0], seg[1], color, 2)
        # Draw group endpoints
        for ex, ey in get_group_endpoints(group):
            cv2.circle(img, (ex, ey), 3, color, -1)

    # Components
    colors = {"Resistor": (0,200,0), "Capacitor": (200,200,0), "Inductor": (0,200,200),
              "Diode": (200,0,100), "Zener Diode": (200,0,150),
              "GND": (0,0,200), "V-DC": (200,0,0), "V-AC": (150,0,0),
              "I-DC": (0,100,0), "I-AC": (0,150,100),
              "MOSFET-N": (100,100,0), "MOSFET-P": (100,0,100),
              "BJT-NPN": (150,150,0), "BJT-PNP": (150,0,150),
              "Op-Amp": (0,100,100), "Wire Crossover": (128,128,128)}
    for c in components:
        x1, y1, x2, y2 = c["xyxy"]
        color = colors.get(c["name"], (255,255,255))
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        label = f'{c["name"]}={c["value"]}' if c.get("value") else c["name"]
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        for pi, (px, py) in enumerate(c["ports"]):
            cv2.circle(img, (px, py), 5, (0,0,255), -1)
            cv2.circle(img, (px, py), 6, (255,255,255), 1)
            nid = port_nodes.get((c["idx"], pi), "?")
            cv2.putText(img, f"N{nid}", (px+6, py-6), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)

    # Connections as green lines
    for ci, pi, cj, pj, dist in connections:
        ca, cb = components[ci], components[cj]
        if pi < len(ca["ports"]) and pj < len(cb["ports"]):
            cv2.line(img, ca["ports"][pi], cb["ports"][pj], (0, 255, 0), 2)

    cv2.imwrite(str(out_path), img)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from ultralytics import YOLO

    img_path = sys.argv[1] if len(sys.argv) > 1 else "test2.jpg"
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Component detection
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "runs", "detect")
    comp_model = YOLO(os.path.join(model_dir, "circuit_real", "weights", "best.pt"))
    r = comp_model(img_path)[0]

    # CGHD dataset port positions (normalized 0-1 within bbox)
    CGHD_PORTS = {
        "Resistor":    [(0, 0.5), (1, 0.5)],
        "Capacitor":   [(0, 0.5), (1, 0.5)],
        "Inductor":    [(0, 0.5), (1, 0.5)],
        "Diode":       [(0, 0.5), (1, 0.5)],      # anode left, cathode right
        "Zener Diode": [(0, 0.5), (1, 0.5)],
        "V-DC":        [(0.5, 1), (0.5, 0)],       # negative bottom, positive top
        "V-AC":        [(0.5, 1), (0.5, 0)],
        "I-DC":        [(0.5, 1), (0.5, 0)],
        "I-AC":        [(0.5, 1), (0.5, 0)],
        "BJT-NPN":     [(0, 0.5), (0.7, 1), (0.7, 0)],  # base, collector, emitter
        "BJT-PNP":     [(0, 0.5), (0.7, 1), (0.7, 0)],
        "MOSFET-N":    [(0, 0.5), (0.5, 1), (0.5, 0)],  # gate, source, drain
        "MOSFET-P":    [(0, 0.5), (0.5, 1), (0.5, 0)],
        "Op-Amp":      [(0, 0.5), (0, 0.3), (1, 0.5)],  # -, +, out
        "GND":         [(0.5, 0)],
        "Wire Crossover": [],
    }

    def infer_ports(x1, y1, x2, y2, name):
        w, h = x2-x1, y2-y1
        if name in CGHD_PORTS:
            # CGHD default is horizontal (left-right ports).
            # If component is clearly vertical (h > 1.3*w), swap x↔y.
            vertical = h > w
            ports = []
            for rx, ry in CGHD_PORTS[name]:
                if vertical:
                    # Swap: (rx, ry) → (1-ry, rx)
                    sx, sy = 1.0 - ry, rx
                else:
                    sx, sy = rx, ry
                px = int(x1 + sx * w)
                py = int(y1 + sy * h)
                ports.append((px, py))
            return ports
        cx, cy = (x1+x2)//2, (y1+y2)//2
        return [(x1, cy), (x2, cy)]

    components = []
    for box in (r.boxes or []):
        name = comp_model.names[int(box.cls[0])]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        ports = infer_ports(x1, y1, x2, y2, name)
        components.append(dict(idx=len(components), name=name,
                              xyxy=(x1,y1,x2,y2), cx=(x1+x2)//2, cy=(y1+y2)//2,
                              ports=ports, value=""))

    # Wire detection pipeline
    raw_segments = detect_wire_segments(gray, components, h, w)
    print(f"Raw segments: {len(raw_segments)}")

    unique_segments = deduplicate_segments(raw_segments, threshold=10)
    print(f"After dedup: {len(unique_segments)}")

    # Extend segments by a few pixels at each end to bridge small gaps
    extended = []
    for (sx1, sy1), (sx2, sy2) in unique_segments:
        dx = sx2 - sx1; dy = sy2 - sy1
        length = math.hypot(dx, dy) or 1
        ux, uy = dx / length, dy / length
        ex1 = (int(sx1 - ux * 3), int(sy1 - uy * 3))
        ex2 = (int(sx2 + ux * 3), int(sy2 + uy * 3))
        extended.append((ex1, ex2))

    groups = group_segments_union_find(extended, threshold=18)
    print(f"Wire groups: {len(groups)}")

    port_nodes, connections = build_node_graph(components, groups, max_dist=40)
    print(f"Connections: {len(connections)}")
    print(f"Nodes: {len(set(port_nodes.values()))}")

    for ci, pi, cj, pj, d in connections:
        print(f"  {components[ci]['name']}[{pi}] <-> {components[cj]['name']}[{pj}]  ({d:.0f}px)")

    out_name = os.path.splitext(img_path)[0] + "_wires.jpg"
    draw_wire_overlay(img_path, components, unique_segments, groups,
                     port_nodes, connections, out_name)
    print(f"\nSaved: {out_name}")

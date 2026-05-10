from src.phase2_pipeline.schema import Circuit, SafetyIssue


def check_short_circuits(circuit: Circuit) -> list[SafetyIssue]:
    """Detect power-to-ground shorts: any path from VCC to GND with only wire connections."""
    issues = []
    vcc_nodes = set()
    gnd_nodes = set()

    for comp in circuit.components:
        if comp.type == "voltage_source":
            vcc_nodes.add(comp.nodes[0])
            gnd_nodes.add(comp.nodes[1])
        elif comp.type == "ground":
            gnd_nodes.add(comp.nodes[0])

    # Direct wire connections between VCC and GND
    for conn in circuit.connections:
        if conn.via is None:  # direct wire
            if (conn.from_node in vcc_nodes and conn.to_node in gnd_nodes) or \
               (conn.from_node in gnd_nodes and conn.to_node in vcc_nodes):
                issues.append(SafetyIssue(
                    type="short_circuit",
                    severity="critical",
                    nodes=[conn.from_node, conn.to_node],
                    description=f"电源到地直接短接：{conn.from_node} ↔ {conn.to_node}"
                ))

    return issues


def check_reverse_polarity(circuit: Circuit) -> list[SafetyIssue]:
    """Check polarized components for reverse connection."""
    polarized_types = {"led", "diode", "capacitor"}  # electrolytic
    issues = []

    for comp in circuit.components:
        if comp.type in polarized_types and len(comp.nodes) == 2:
            # Trace: is anode connected toward higher potential?
            anode = comp.nodes[0]
            cathode = comp.nodes[1]

            # Simple check: if cathode connects to VCC or anode to GND via low-impedance path
            for c2 in circuit.components:
                if c2.type == "voltage_source":
                    vcc = c2.nodes[0]
                    gnd = c2.nodes[1]
                    # Direct connection of cathode to VCC
                    if cathode == vcc:
                        issues.append(SafetyIssue(
                            type="reverse_polarity",
                            severity="critical",
                            nodes=[anode, cathode],
                            description=f"{comp.id}({comp.type})极性接反：阴极连接到电源正极"
                        ))
                    # Direct connection of anode to GND
                    if anode == gnd:
                        issues.append(SafetyIssue(
                            type="reverse_polarity",
                            severity="critical",
                            nodes=[anode, cathode],
                            description=f"{comp.id}({comp.type})极性接反：阳极连接到地"
                        ))

    return issues


def check_open_circuits(circuit: Circuit) -> list[SafetyIssue]:
    """Detect floating nodes (nodes connected to only one component)."""
    issues = []
    node_degree = {}

    for comp in circuit.components:
        for node in comp.nodes:
            node_degree[node] = node_degree.get(node, 0) + 1
    for conn in circuit.connections:
        node_degree[conn.from_node] = node_degree.get(conn.from_node, 0) + 1
        node_degree[conn.to_node] = node_degree.get(conn.to_node, 0) + 1

    for node, degree in node_degree.items():
        if degree == 1:
            issues.append(SafetyIssue(
                type="open_circuit",
                severity="critical",
                nodes=[node],
                description=f"悬空节点 {node}：只连接到一个元件，可能断路"
            ))

    return issues


def evaluate_safety(circuit: Circuit) -> list[SafetyIssue]:
    issues = []
    issues.extend(check_short_circuits(circuit))
    issues.extend(check_reverse_polarity(circuit))
    issues.extend(check_open_circuits(circuit))
    return issues

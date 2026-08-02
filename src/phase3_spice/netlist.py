"""Generate SPICE netlist from structured Circuit model."""

from src.phase2_pipeline.schema import Circuit


def generate_netlist(circuit: Circuit, analysis: str = "op") -> str:
    """Convert a Circuit to a SPICE netlist.

    Args:
        circuit: Structured circuit model
        analysis: Type of analysis - "op" (operating point), "dc" (DC sweep), "tran" (transient)
    """
    lines = [f"* {circuit.name}"]
    lines.append(f"* Generated netlist for SPICE simulation")
    lines.append("")

    # Node mapping: assign numbers to node names
    node_map = _build_node_map(circuit)
    lines.append("* Node mapping:")
    for name, num in sorted(node_map.items(), key=lambda x: x[1]):
        if name not in ("0",):
            lines.append(f"*   {name} -> N{num:03d}")
    lines.append("")

    # Generate component lines
    for comp in circuit.components:
        spice_line = _component_to_spice(comp, node_map)
        if spice_line:
            lines.append(spice_line)

    lines.append("")

    # Analysis commands
    if analysis == "op":
        lines.append(".OP")
    elif analysis == "dc":
        # Find voltage sources for DC sweep
        sources = [c for c in circuit.components if c.type == "voltage_source"]
        if sources:
            lines.append(f".DC {sources[0].id} 0 5 0.1")
    elif analysis == "tran":
        lines.append(".TRAN 1u 10m")

    lines.append(".END")
    return "\n".join(lines)


def _build_node_map(circuit: Circuit) -> dict[str, int]:
    """Map named nodes to SPICE node numbers. Node 0 is always GND."""
    node_map = {"0": 0}
    next_node = 1

    for comp in circuit.components:
        for node in comp.nodes:
            if node.upper() in ("GND", "GROUND", "0"):
                node_map[node] = 0
            elif node not in node_map:
                node_map[node] = next_node
                next_node += 1

    return node_map


def _component_to_spice(comp, node_map: dict[str, int]) -> str | None:
    """Convert a single component to its SPICE line."""
    comp_type = comp.type
    nodes = [node_map.get(n, 0) for n in comp.nodes]

    if comp_type == "resistor":
        n1, n2 = nodes[0], nodes[1]
        return f"{comp.id} {n1} {n2} {_parse_value(comp.value)}"

    elif comp_type == "capacitor":
        n1, n2 = nodes[0], nodes[1]
        return f"{comp.id} {n1} {n2} {_parse_value(comp.value)}"

    elif comp_type == "inductor":
        n1, n2 = nodes[0], nodes[1]
        return f"{comp.id} {n1} {n2} {_parse_value(comp.value)}"

    elif comp_type == "voltage_source":
        n_plus, n_minus = nodes[0], nodes[1]
        return f"{comp.id} {n_plus} {n_minus} DC {_parse_value(comp.value)}"

    elif comp_type == "current_source":
        n_plus, n_minus = nodes[0], nodes[1]
        return f"{comp.id} {n_plus} {n_minus} DC {_parse_value(comp.value)}"

    elif comp_type == "diode":
        anode, cathode = nodes[0], nodes[1]
        model_name = f"D_{comp.id}"
        return f"{comp.id} {anode} {cathode} {model_name}\n.MODEL {model_name} D (IS=1e-14 N=1.0)"

    elif comp_type == "led":
        anode, cathode = nodes[0], nodes[1]
        model_name = f"LED_{comp.id}"
        # LED model: use IS to set approximate Vf at 20mA
        # For red LED Vf≈2.0V: IS=1e-20, N=1.8 gives ~2.0V at 20mA
        # For green/blue with higher Vf: adjust N upward
        vf = 2.0
        if comp.value and "vf" in comp.value.lower():
            import re
            match = re.search(r'vf\s*[=:]\s*([\d.]+)', comp.value, re.IGNORECASE)
            if match:
                vf = float(match.group(1))
        # Use a simple approach: model diode with appropriate IS to approximate Vf
        # N=2.0 (ideality factor for LEDs), IS adjusted for target Vf
        is_val = 10 ** (-14 - (vf - 1.5) * 2)  # roughly scales IS for target Vf
        return f"{comp.id} {anode} {cathode} {model_name}\n.MODEL {model_name} D (IS={is_val:.0e} N=2.0 RS=5)"

    elif comp_type in ("transistor_npn",):
        collector, base, emitter = nodes[0], nodes[1], nodes[2]
        return f"{comp.id} {collector} {base} {emitter} QNPN\n.MODEL QNPN NPN (BF=200 IS=1e-14 VAF=100)"

    elif comp_type in ("transistor_pnp",):
        collector, base, emitter = nodes[0], nodes[1], nodes[2]
        return f"{comp.id} {collector} {base} {emitter} QPNP\n.MODEL QPNP PNP (BF=100 IS=1e-14 VAF=80)"

    elif comp_type == "opamp":
        # Simplified opamp model using voltage-controlled voltage source
        n_plus_pwr, n_minus_pwr = nodes[2] if len(nodes) > 2 else 0, nodes[3] if len(nodes) > 3 else 0
        non_inv, inv, out = nodes[0], nodes[1], nodes[2] if len(nodes) <= 3 else nodes[4]
        return (
            f"* {comp.id}: Simplified opamp model (VCVS, gain=1e5)\n"
            f"E_{comp.id} {out} 0 {non_inv} {inv} 1e5\n"
            f"Rin_{comp.id} {non_inv} {inv} 1e6"
        )

    elif comp_type == "ground":
        return None  # Already handled by node 0

    elif comp_type == "wire":
        return None  # Handled by connections

    elif comp_type == "switch":
        n1, n2 = nodes[0], nodes[1]
        return f"S_{comp.id} {n1} {n2} 99 0 SMOD\n.MODEL SMOD SW (RON=0.01 ROFF=1e6 VTH=0.5)"

    else:
        return f"* {comp.id}: {comp_type} not implemented in SPICE converter"


def _parse_value(value_str: str | None) -> str:
    """Parse component value to SPICE-compatible format."""
    if not value_str:
        return "1"
    v = value_str.strip()
    # Handle common suffixes
    v = v.replace("Ω", "").replace("μ", "u")
    # Extract numeric part
    import re
    match = re.match(r'([\d.]+)\s*([kMmunp]?)', v)
    if match:
        num, suffix = match.groups()
        suffix_map = {"k": "k", "M": "Meg", "m": "m", "u": "u", "n": "n", "p": "p"}
        if suffix in suffix_map:
            return num + suffix_map[suffix]
        return num
    return v

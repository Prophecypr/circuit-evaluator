import importlib
from xml.etree import ElementTree

import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.patches import Polygon
from PIL import Image

import src.reference_pack.render as render_module
from src.reference_pack.layouts import LAYOUTS, validate_layout
from src.reference_pack.render import ReferenceCanvas, draw_reference
from src.reference_pack.specs import CIRCUITS


def test_layouts_cover_exactly_the_ten_reference_circuits():
    assert set(LAYOUTS) == {f"C{index:02d}" for index in range(1, 11)}


def test_every_semantic_component_and_net_has_layout_data():
    for cid, circuit in CIRCUITS.items():
        validate_layout(circuit, LAYOUTS[cid])


def test_layouts_keep_symbols_inside_page():
    for layout in LAYOUTS.values():
        for x, y, orientation in layout["components"].values():
            assert 100 <= x <= 900
            assert 100 <= y <= 600
            assert orientation in {"h", "v"}


@pytest.mark.parametrize(("section", "extra_key"), [("components", "X1"), ("hubs", "NX")])
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_validate_layout_rejects_missing_or_extra_entries(section, extra_key, mutation):
    circuit = CIRCUITS["C01"]
    layout = {
        "components": dict(LAYOUTS["C01"]["components"]),
        "hubs": dict(LAYOUTS["C01"]["hubs"]),
    }
    if mutation == "missing":
        layout[section].pop(next(iter(layout[section])))
    else:
        layout[section][extra_key] = (500, 300, "h") if section == "components" else (500, 300)

    with pytest.raises(ValueError, match="C01"):
        validate_layout(circuit, layout)


def test_draw_reference_records_all_semantic_ports_and_orthogonal_routes():
    circuit = CIRCUITS["C10"]
    canvas = draw_reference(circuit, LAYOUTS[circuit.id])
    try:
        expected_ports = {
            f"{component.id}.{port}"
            for component in circuit.components
            for port in component.ports
        }
        assert set(canvas.port_map) == expected_ports
        assert set(canvas.route_records) == set(circuit.nets)

        for net_id, paths in canvas.route_records.items():
            assert len(paths) == len(circuit.nets[net_id])
            for path in paths:
                assert all(
                    start[0] == end[0] or start[1] == end[1]
                    for start, end in zip(path, path[1:])
                )
    finally:
        plt.close(canvas.figure)


def test_draw_reference_marks_only_nets_with_three_or_more_members():
    for circuit in CIRCUITS.values():
        canvas = draw_reference(circuit, LAYOUTS[circuit.id])
        try:
            expected_hubs = {
                net_id: LAYOUTS[circuit.id]["hubs"][net_id]
                for net_id, members in circuit.nets.items()
                if len(members) >= 3
            }
            assert canvas.hub_records == expected_hubs
        finally:
            plt.close(canvas.figure)


def test_draw_reference_uses_exact_visible_title():
    circuit = CIRCUITS["C01"]
    canvas = draw_reference(circuit, LAYOUTS[circuit.id])
    try:
        expected_title = f"{circuit.id}  {circuit.title}"
        assert canvas.title == expected_title
        assert canvas.axes.get_title() == expected_title
    finally:
        plt.close(canvas.figure)


def test_resistor_is_zigzag_and_reference_hides_designator(tmp_path):
    canvas = ReferenceCanvas("test")
    ports = canvas.component("R17", "resistor", "2.2kΩ", (500, 300), "h")
    output = tmp_path / "resistor.svg"
    canvas.save(output)
    svg = output.read_text(encoding="utf-8")

    assert ports == {"1": (420, 300), "2": (580, 300)}
    assert canvas.symbol_records["R17"] == "ansi_zigzag"
    assert canvas.text_records == ["2.2kΩ"]
    assert "2.2kΩ" in svg
    assert "R17" not in svg


@pytest.mark.parametrize(
    ("class_name", "orientation", "expected_ports"),
    [
        pytest.param("resistor", "h", {"1": (420, 300), "2": (580, 300)}, id="resistor-h"),
        pytest.param("resistor", "v", {"1": (500, 220), "2": (500, 380)}, id="resistor-v"),
        pytest.param("voltage.dc", "v", {"+": (500, 230), "-": (500, 370)}, id="voltage-v"),
        pytest.param("gnd", "v", {"GND": (500, 265)}, id="gnd-v"),
        pytest.param(
            "capacitor.unpolarized",
            "h",
            {"1": (430, 300), "2": (570, 300)},
            id="capacitor-h",
        ),
        pytest.param(
            "capacitor.unpolarized",
            "v",
            {"1": (500, 230), "2": (500, 370)},
            id="capacitor-v",
        ),
        pytest.param("inductor", "h", {"1": (420, 300), "2": (580, 300)}, id="inductor-h"),
        pytest.param("inductor", "v", {"1": (500, 220), "2": (500, 380)}, id="inductor-v"),
        pytest.param(
            "diode.light_emitting",
            "h",
            {"+": (430, 300), "-": (570, 300)},
            id="led-h",
        ),
        pytest.param(
            "diode.light_emitting",
            "v",
            {"+": (500, 230), "-": (500, 370)},
            id="led-v",
        ),
        pytest.param(
            "diode.zener",
            "h",
            {"A": (430, 300), "K": (570, 300)},
            id="zener-h",
        ),
        pytest.param(
            "diode.zener",
            "v",
            {"A": (500, 370), "K": (500, 230)},
            id="zener-v",
        ),
        pytest.param(
            "transistor.bjt",
            "v",
            {"B": (430, 300), "C": (545, 235), "E": (545, 365)},
            id="bjt-v",
        ),
    ],
)
def test_component_ports_use_exact_reference_geometry(
    class_name,
    orientation,
    expected_ports,
):
    canvas = ReferenceCanvas("geometry")
    try:
        ports = canvas.component("X1", class_name, None, (500, 300), orientation)

        assert ports == expected_ports
        assert canvas.symbol_records["X1"] == (
            "ansi_zigzag" if class_name == "resistor" else class_name
        )
    finally:
        plt.close(canvas.figure)


def test_values_are_recorded_once_and_component_ids_are_not_visible(tmp_path):
    canvas = ReferenceCanvas("internal title")
    canvas.component("R1", "resistor", "1kΩ", (250, 300), "h")
    canvas.component("V1", "voltage.dc", "5V", (500, 300), "v")
    canvas.component("D1", "diode.light_emitting", None, (750, 300), "h")
    output = tmp_path / "values.svg"
    canvas.save(output)
    svg = output.read_text(encoding="utf-8")

    assert canvas.text_records == ["1kΩ", "5V"]
    assert "1kΩ" in svg
    assert "5V" in svg
    assert "R1" not in svg
    assert "V1" not in svg
    assert "D1" not in svg


def test_vertical_zener_bar_and_leads_match_k_a_polarity():
    canvas = ReferenceCanvas("zener polarity")
    try:
        ports = canvas.component("Z1", "diode.zener", None, (500, 300), "v")
        line_segments = {
            frozenset(zip(line.get_xdata(), line.get_ydata()))
            for line in canvas.axes.lines
        }
        triangle = next(
            patch for patch in canvas.axes.patches if isinstance(patch, Polygon)
        )

        assert ports == {"A": (500, 370), "K": (500, 230)}
        assert frozenset({(466, 278), (534, 278)}) in line_segments
        assert frozenset({(500, 230), (500, 278)}) in line_segments
        assert frozenset({(500, 330), (500, 370)}) in line_segments
        assert set(map(tuple, triangle.get_xy()[:3])) == {
            (470, 330),
            (530, 330),
            (500, 280),
        }
    finally:
        plt.close(canvas.figure)


def test_save_writes_nonempty_png(tmp_path):
    canvas = ReferenceCanvas("png")
    canvas.component("R1", "resistor", "1kΩ", (500, 300), "h")
    output = tmp_path / "smoke.png"

    canvas.save(output)

    assert output.exists()
    assert output.stat().st_size > 8
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_import_and_svg_save_preserve_caller_fonttype(tmp_path):
    original_fonttype = matplotlib.rcParams["svg.fonttype"]
    try:
        matplotlib.rcParams["svg.fonttype"] = "path"
        reloaded = importlib.reload(render_module)

        assert matplotlib.rcParams["svg.fonttype"] == "path"

        canvas = reloaded.ReferenceCanvas("font isolation")
        canvas.component("R1", "resistor", "2.2kΩ", (500, 300), "h")
        output = tmp_path / "isolated.svg"
        canvas.save(output)
        svg = output.read_text(encoding="utf-8")

        assert matplotlib.rcParams["svg.fonttype"] == "path"
        assert "<text" in svg
        assert "2.2kΩ" in svg
    finally:
        matplotlib.rcParams["svg.fonttype"] = original_fonttype


def test_png_and_svg_outputs_keep_fixed_a4_landscape_page(tmp_path):
    variants = [
        ("R1", "resistor", "1kΩ", (180, 150), "h"),
        ("V1", "voltage.dc", "1234567890mV", (800, 550), "v"),
    ]
    png_paths = []
    svg_paths = []

    for index, component_args in enumerate(variants):
        png_path = tmp_path / f"page_{index}.png"
        png_canvas = ReferenceCanvas(f"png {index}")
        png_canvas.component(*component_args)
        png_canvas.save(png_path)
        png_paths.append(png_path)

        svg_path = tmp_path / f"page_{index}.svg"
        svg_canvas = ReferenceCanvas(f"svg {index}")
        svg_canvas.component(*component_args)
        svg_canvas.save(svg_path)
        svg_paths.append(svg_path)

    with Image.open(png_paths[0]) as first_png, Image.open(png_paths[1]) as second_png:
        assert first_png.size == second_png.size == (2338, 1654)

    svg_pages = [ElementTree.parse(path).getroot() for path in svg_paths]
    svg_dimensions = [
        (page.attrib["width"], page.attrib["height"], page.attrib["viewBox"])
        for page in svg_pages
    ]
    assert svg_dimensions[0] == svg_dimensions[1] == (
        "841.68pt",
        "595.44pt",
        "0 0 841.68 595.44",
    )


def test_save_closes_figure_when_savefig_raises(tmp_path, monkeypatch):
    canvas = ReferenceCanvas("save failure")
    figure_number = canvas.figure.number

    def fail_savefig(*args, **kwargs):
        raise RuntimeError("save failed")

    monkeypatch.setattr(canvas.figure, "savefig", fail_savefig)

    with pytest.raises(RuntimeError, match="save failed"):
        canvas.save(tmp_path / "failure.svg")

    assert figure_number not in plt.get_fignums()


@pytest.mark.parametrize(
    ("class_name", "orientation", "message"),
    [
        pytest.param("switch", "h", "unsupported component class", id="class"),
        pytest.param("voltage.dc", "h", "unsupported orientation", id="orientation"),
    ],
)
def test_component_rejects_unsupported_dispatch(class_name, orientation, message):
    canvas = ReferenceCanvas("invalid")
    try:
        with pytest.raises(ValueError, match=message):
            canvas.component("X1", class_name, None, (500, 300), orientation)
    finally:
        plt.close(canvas.figure)

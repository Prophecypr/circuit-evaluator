import hashlib
import importlib
import json
import warnings
from dataclasses import asdict
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


def _copy_layout(circuit_id):
    source = LAYOUTS[circuit_id]
    layout = {
        "components": dict(source["components"]),
        "hubs": dict(source["hubs"]),
    }
    if "waypoints" in source:
        layout["waypoints"] = {
            net_id: dict(member_waypoints)
            for net_id, member_waypoints in source["waypoints"].items()
        }
    return layout


def _route_path(canvas, circuit, net_id, member):
    member_index = circuit.nets[net_id].index(member)
    return canvas.route_records[net_id][member_index]


def _between(value, first, second):
    return min(first, second) <= value <= max(first, second)


def _point_on_segment(point, start, end):
    if start[0] == end[0] == point[0]:
        return _between(point[1], start[1], end[1])
    if start[1] == end[1] == point[1]:
        return _between(point[0], start[0], end[0])
    return False


def _segments_intersect(first_start, first_end, second_start, second_end):
    first_horizontal = first_start[1] == first_end[1]
    second_horizontal = second_start[1] == second_end[1]
    if first_horizontal and second_horizontal:
        return (
            first_start[1] == second_start[1]
            and max(min(first_start[0], first_end[0]), min(second_start[0], second_end[0]))
            <= min(max(first_start[0], first_end[0]), max(second_start[0], second_end[0]))
        )
    if not first_horizontal and not second_horizontal:
        return (
            first_start[0] == second_start[0]
            and max(min(first_start[1], first_end[1]), min(second_start[1], second_end[1]))
            <= min(max(first_start[1], first_end[1]), max(second_start[1], second_end[1]))
        )
    horizontal_start, horizontal_end = (
        (first_start, first_end) if first_horizontal else (second_start, second_end)
    )
    vertical_start, vertical_end = (
        (second_start, second_end) if first_horizontal else (first_start, first_end)
    )
    return _between(vertical_start[0], horizontal_start[0], horizontal_end[0]) and _between(
        horizontal_start[1], vertical_start[1], vertical_end[1]
    )


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


@pytest.mark.parametrize(
    ("waypoints", "message"),
    [
        ({"NX": {"V1.+": ((200, 200),)}}, "waypoint net"),
        ({"N1": {"V1.-": ((200, 200),)}}, "waypoint member"),
        ({"N1": {"V1.+": ((1001, 200),)}}, "waypoint coordinate"),
    ],
)
def test_validate_layout_rejects_invalid_waypoint_data(waypoints, message):
    circuit = CIRCUITS["C01"]
    layout = _copy_layout(circuit.id)
    layout["waypoints"] = waypoints

    with pytest.raises(ValueError, match=rf"C01: {message}"):
        validate_layout(circuit, layout)


def test_draw_reference_rejects_diagonal_waypoint_routes():
    circuit = CIRCUITS["C01"]
    layout = _copy_layout(circuit.id)
    layout["waypoints"] = {"N1": {"V1.+": ((200, 200),)}}

    with pytest.raises(ValueError, match=r"C01: N1/V1\.\+ route is not orthogonal"):
        canvas = draw_reference(circuit, layout)
        plt.close(canvas.figure)


def test_layouts_use_reviewed_hubs_and_member_waypoints():
    assert LAYOUTS["C01"]["hubs"]["N0"] == (650, 500)
    assert LAYOUTS["C02"]["hubs"]["N0"] == (850, 500)
    assert LAYOUTS["C03"]["waypoints"]["N0"]["V1.-"] == (
        (150, 540),
        (720, 540),
    )
    assert LAYOUTS["C08"]["hubs"]["N0"] == (610, 420)
    assert LAYOUTS["C09"]["waypoints"]["N0"] == {
        "V1.-": ((110, 550),),
        "Q1.E": ((720, 485), (720, 550)),
    }
    assert LAYOUTS["C10"]["waypoints"] == {
        "N1": {
            "V1.+": ((100, 100),),
            "R1.1": ((360, 100),),
            "R3.1": ((720, 100),),
        },
        "N0": {
            "V1.-": ((100, 570),),
            "R2.2": ((360, 570),),
            "R4.2": ((720, 570),),
        },
    }


def test_draw_reference_uses_reviewed_waypoint_paths():
    expected_paths = {
        ("C03", "N0", "V1.-"): [(150, 420), (150, 540), (720, 540), (720, 500)],
        ("C09", "N0", "V1.-"): [(110, 420), (110, 550), (620, 550)],
        ("C09", "N0", "Q1.E"): [
            (695, 485),
            (720, 485),
            (720, 550),
            (620, 550),
        ],
        ("C10", "N1", "V1.+"): [(100, 280), (100, 100), (540, 100)],
        ("C10", "N1", "R1.1"): [(360, 120), (360, 100), (540, 100)],
        ("C10", "N1", "R3.1"): [(720, 120), (720, 100), (540, 100)],
        ("C10", "N0", "V1.-"): [(100, 420), (100, 570), (540, 570)],
        ("C10", "N0", "R2.2"): [(360, 560), (360, 570), (540, 570)],
        ("C10", "N0", "R4.2"): [(720, 560), (720, 570), (540, 570)],
    }
    canvases = {}
    try:
        for circuit_id, net_id, member in expected_paths:
            circuit = CIRCUITS[circuit_id]
            if circuit_id not in canvases:
                canvases[circuit_id] = draw_reference(circuit, LAYOUTS[circuit_id])
            canvas = canvases[circuit_id]
            assert _route_path(canvas, circuit, net_id, member) == expected_paths[
                (circuit_id, net_id, member)
            ]
    finally:
        for canvas in canvases.values():
            plt.close(canvas.figure)


def test_reference_routes_do_not_create_cross_net_false_connections():
    failures = []
    for circuit in CIRCUITS.values():
        canvas = draw_reference(circuit, LAYOUTS[circuit.id])
        try:
            segments = [
                (net_id, start, end)
                for net_id, paths in canvas.route_records.items()
                for path in paths
                for start, end in zip(path, path[1:])
            ]
            for index, (first_net, first_start, first_end) in enumerate(segments):
                for second_net, second_start, second_end in segments[index + 1 :]:
                    if first_net == second_net:
                        continue
                    if _segments_intersect(
                        first_start,
                        first_end,
                        second_start,
                        second_end,
                    ):
                        failures.append(
                            f"{circuit.id}: {first_net} {first_start}->{first_end} "
                            f"intersects {second_net} {second_start}->{second_end}"
                        )

            port_nets = {
                member: net_id
                for net_id, members in circuit.nets.items()
                for member in members
            }
            for net_id, start, end in segments:
                for member, point in canvas.port_map.items():
                    if port_nets[member] != net_id and _point_on_segment(point, start, end):
                        failures.append(
                            f"{circuit.id}: {net_id} {start}->{end} crosses {member}"
                        )
        finally:
            plt.close(canvas.figure)

    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(
    ("circuit_id", "net_id", "member"),
    [("C01", "N0", "R1.2"), ("C02", "N0", "R2.2")],
)
def test_resistor_output_routes_do_not_turn_back_into_zigzag(circuit_id, net_id, member):
    circuit = CIRCUITS[circuit_id]
    canvas = draw_reference(circuit, LAYOUTS[circuit.id])
    try:
        centers = {
            component_id: placement[:2]
            for component_id, placement in LAYOUTS[circuit.id]["components"].items()
        }
        path = _route_path(canvas, circuit, net_id, member)
        component_id = member.rsplit(".", 1)[0]
        port, next_point = path[:2]
        center = centers[component_id]
        route_vector = (next_point[0] - port[0], next_point[1] - port[1])
        inward_vector = (center[0] - port[0], center[1] - port[1])

        assert sum(a * b for a, b in zip(route_vector, inward_vector)) <= 0
    finally:
        plt.close(canvas.figure)


def test_all_reference_outputs_use_cjk_titles_and_fixed_page_geometry(tmp_path):
    output_paths = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for circuit in CIRCUITS.values():
            for suffix in ("svg", "png"):
                output = tmp_path / f"{circuit.id}.{suffix}"
                canvas = draw_reference(circuit, LAYOUTS[circuit.id])
                title_font = canvas.axes.title.get_fontproperties().get_name()
                assert title_font in {
                    "Noto Sans SC",
                    "Microsoft YaHei",
                    "SimHei",
                    "DengXian",
                }
                canvas.save(output)
                output_paths.append(output)

                assert output.stat().st_size > 0
                if suffix == "png":
                    with Image.open(output) as image:
                        assert image.size == (2338, 1654)
                        image.verify()
                else:
                    root = ElementTree.parse(output).getroot()
                    assert root.attrib["viewBox"] == "0 0 841.68 595.44"
                    assert "<text" in output.read_text(encoding="utf-8")

    missing_glyph_warnings = [
        str(item.message)
        for item in caught
        if "Glyph" in str(item.message) and "missing" in str(item.message)
    ]
    assert len(output_paths) == 20
    assert missing_glyph_warnings == []
    assert plt.get_fignums() == []


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


def test_generate_pack_writes_complete_versioned_pack_without_overwrite(tmp_path):
    from generate_blind_reference_pack import generate_pack

    output_dir = tmp_path / "v1"

    assert generate_pack(output_dir) == output_dir

    references = output_dir / "references"
    answer_key = output_dir / "answer_key"
    svg_paths = sorted(references.glob("C*_reference.svg"))
    png_paths = sorted(references.glob("C*_reference.png"))
    answer_paths = sorted(answer_key.glob("C*_gt.json"))
    circuit_ids = list(CIRCUITS)

    assert [path.name for path in svg_paths] == [
        f"{circuit_id}_reference.svg" for circuit_id in circuit_ids
    ]
    assert [path.name for path in png_paths] == [
        f"{circuit_id}_reference.png" for circuit_id in circuit_ids
    ]
    assert [path.name for path in answer_paths] == [
        f"{circuit_id}_gt.json" for circuit_id in circuit_ids
    ]

    required_files = {
        "collection_protocol.md",
        "annotation_protocol.md",
        "manifest_template.csv",
        "checksums.sha256",
    }
    assert all((output_dir / name).is_file() for name in required_files)

    for circuit_id, svg_path, png_path, answer_path in zip(
        circuit_ids, svg_paths, png_paths, answer_paths
    ):
        with Image.open(png_path) as image:
            assert image.size == (2338, 1654)
            image.verify()

        svg_root = ElementTree.parse(svg_path).getroot()
        assert svg_root.attrib["viewBox"] == "0 0 841.68 595.44"
        svg_namespace = {"svg": "http://www.w3.org/2000/svg"}
        visible_texts = {
            "".join(node.itertext()).strip()
            for node in svg_root.findall(".//svg:text", svg_namespace)
        }

        spec = CIRCUITS[circuit_id]
        assert all(component.id not in visible_texts for component in spec.components)
        assert all(
            component.value in visible_texts
            for component in spec.components
            if component.value is not None
        )

        answer = json.loads(answer_path.read_text(encoding="utf-8"))
        assert answer == {
            "circuit_id": spec.id,
            "title": spec.title,
            "difficulty": spec.difficulty,
            "components": [
                {**asdict(component), "ports": list(component.ports)}
                for component in spec.components
            ],
            "nets": {
                net_id: list(members) for net_id, members in spec.nets.items()
            },
        }

    manifest = (output_dir / "manifest_template.csv").read_text(encoding="utf-8")
    assert manifest.rstrip("\r\n") == (
        "paper_id,circuit_id,participant_id,difficulty,domain,image_path,gt_path"
    )
    assert "\n" not in manifest.rstrip("\r\n")

    checksum_path = output_dir / "checksums.sha256"
    checksum_lines = checksum_path.read_text(encoding="utf-8").splitlines()
    generated_files = sorted(
        (
            path.relative_to(output_dir).as_posix()
            for path in output_dir.rglob("*")
            if path.is_file() and path != checksum_path
        )
    )
    checksum_entries = []
    for line in checksum_lines:
        digest, relative_path = line.split("  ", 1)
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)
        assert not relative_path.startswith(("/", "\\"))
        assert ":" not in relative_path
        assert (
            hashlib.sha256((output_dir / relative_path).read_bytes()).hexdigest()
            == digest
        )
        checksum_entries.append(relative_path)

    assert checksum_entries == generated_files
    assert checksum_entries == sorted(checksum_entries)
    assert "checksums.sha256" not in checksum_entries

    before_retry = {
        path.relative_to(output_dir).as_posix(): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    with pytest.raises(
        FileExistsError,
        match=r"refusing to overwrite non-empty pack:",
    ):
        generate_pack(output_dir)
    after_retry = {
        path.relative_to(output_dir).as_posix(): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    assert after_retry == before_retry

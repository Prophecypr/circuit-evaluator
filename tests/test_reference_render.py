import matplotlib.pyplot as plt
import pytest
from matplotlib.patches import Polygon

from src.reference_pack.render import ReferenceCanvas


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

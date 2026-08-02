"""Code-native rendering primitives for canonical circuit reference images."""

from functools import cache
from pathlib import Path
from typing import Final

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon

from .layouts import Layout, validate_layout
from .specs import CircuitSpec

Point = tuple[float, float]
Ports = dict[str, Point]

LINE_WIDTH: Final = 2.4
PORT_GEOMETRY: Final = {
    ("voltage.dc", "v"): {"+": (0, -70), "-": (0, 70)},
    ("gnd", "v"): {"GND": (0, -35)},
    ("capacitor.unpolarized", "h"): {"1": (-70, 0), "2": (70, 0)},
    ("capacitor.unpolarized", "v"): {"1": (0, -70), "2": (0, 70)},
    ("inductor", "h"): {"1": (-80, 0), "2": (80, 0)},
    ("inductor", "v"): {"1": (0, -80), "2": (0, 80)},
    ("diode.light_emitting", "h"): {"+": (-70, 0), "-": (70, 0)},
    ("diode.light_emitting", "v"): {"+": (0, -70), "-": (0, 70)},
    ("diode.zener", "h"): {"A": (-70, 0), "K": (70, 0)},
    ("diode.zener", "v"): {"A": (0, 70), "K": (0, -70)},
    ("transistor.bjt", "v"): {"B": (-70, 0), "C": (45, -65), "E": (45, 65)},
}

_ORIENTATIONS: Final = {
    "resistor": {"h", "v"},
    "voltage.dc": {"v"},
    "gnd": {"v"},
    "capacitor.unpolarized": {"h", "v"},
    "inductor": {"h", "v"},
    "diode.light_emitting": {"h", "v"},
    "diode.zener": {"h", "v"},
    "transistor.bjt": {"v"},
}

_CJK_FONT_FAMILIES: Final = (
    "Noto Sans SC",
    "Microsoft YaHei",
    "SimHei",
    "DengXian",
)


@cache
def _cjk_font_properties() -> font_manager.FontProperties:
    """Return the first installed font from the supported CJK fallback list."""

    for family in _CJK_FONT_FAMILIES:
        try:
            path = font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        return font_manager.FontProperties(fname=path)
    raise RuntimeError(
        "no supported Chinese title font found; install one of: "
        + ", ".join(_CJK_FONT_FAMILIES)
    )


class ReferenceCanvas:
    """A 1000 x 700 logical canvas for reusable ANSI-style circuit symbols."""

    def __init__(self, title: str):
        self.title = title
        self.symbol_records: dict[str, str] = {}
        self.text_records: list[str] = []
        self.port_map: dict[str, Point] = {}
        self.route_records: dict[str, list[list[Point]]] = {}
        self.hub_records: dict[str, Point] = {}

        self.figure, self.axes = plt.subplots(figsize=(11.69, 8.27))
        self.figure.patch.set_facecolor("white")
        self.axes.set_facecolor("white")
        self.axes.set_xlim(0, 1000)
        self.axes.set_ylim(700, 0)
        self.axes.set_aspect("equal", adjustable="box")
        self.axes.axis("off")

    @property
    def ax(self):
        """Expose the underlying Matplotlib axes for later layout composition."""

        return self.axes

    def component(
        self,
        component_id: str,
        class_name: str,
        value: str | None,
        center: Point,
        orientation: str,
    ) -> Ports:
        """Draw one supported symbol and return its exact absolute port locations."""

        if class_name not in _ORIENTATIONS:
            raise ValueError(f"unsupported component class: {class_name!r}")
        if orientation not in _ORIENTATIONS[class_name]:
            raise ValueError(
                f"unsupported orientation {orientation!r} for component class {class_name!r}"
            )

        if class_name == "resistor":
            ports = self._resistor(center, orientation, value)
            symbol_style = "ansi_zigzag"
        elif class_name == "voltage.dc":
            ports = self._voltage_dc(center, value)
            symbol_style = class_name
        elif class_name == "gnd":
            ports = self._ground(center, value)
            symbol_style = class_name
        elif class_name == "capacitor.unpolarized":
            ports = self._capacitor(center, orientation, value)
            symbol_style = class_name
        elif class_name == "inductor":
            ports = self._inductor(center, orientation, value)
            symbol_style = class_name
        elif class_name in {"diode.light_emitting", "diode.zener"}:
            ports = self._diode(center, orientation, value, class_name)
            symbol_style = class_name
        else:
            ports = self._bjt(center, value)
            symbol_style = class_name

        self.symbol_records[component_id] = symbol_style
        return ports

    def save(self, path: str | Path) -> None:
        """Save SVG or raster output and release the Matplotlib figure."""

        path = Path(path)
        metadata = {"Date": None} if path.suffix.lower() == ".svg" else None
        try:
            with matplotlib.rc_context(
                {
                    "svg.fonttype": "none",
                    "svg.hashsalt": "blind-reference-pack-v1",
                }
            ):
                self.figure.savefig(path, dpi=200, metadata=metadata)
        finally:
            plt.close(self.figure)

    def _ports(self, class_name: str, orientation: str, center: Point) -> Ports:
        x, y = center
        return {
            label: (x + dx, y + dy)
            for label, (dx, dy) in PORT_GEOMETRY[(class_name, orientation)].items()
        }

    def _line(self, points: list[Point], *, linewidth: float = LINE_WIDTH) -> None:
        self.axes.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color="black",
            linewidth=linewidth,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    def _value(
        self,
        value: str | None,
        position: Point,
        *,
        horizontal_alignment: str = "center",
    ) -> None:
        if not value:
            return
        self.axes.text(
            *position,
            value,
            fontsize=15,
            color="black",
            ha=horizontal_alignment,
            va="center",
        )
        self.text_records.append(value)

    def _resistor(self, center: Point, orientation: str, value: str | None) -> Ports:
        x, y = center
        if orientation == "h":
            points = [
                (x - 80, y),
                (x - 55, y),
                (x - 42, y - 18),
                (x - 24, y + 18),
                (x - 6, y - 18),
                (x + 12, y + 18),
                (x + 30, y - 18),
                (x + 48, y + 18),
                (x + 60, y),
                (x + 80, y),
            ]
            ports = {"1": (x - 80, y), "2": (x + 80, y)}
            self._value(value, (x, y - 42))
        else:
            points = [
                (x, y - 80),
                (x, y - 55),
                (x - 18, y - 42),
                (x + 18, y - 24),
                (x - 18, y - 6),
                (x + 18, y + 12),
                (x - 18, y + 30),
                (x + 18, y + 48),
                (x, y + 60),
                (x, y + 80),
            ]
            ports = {"1": (x, y - 80), "2": (x, y + 80)}
            self._value(value, (x + 38, y), horizontal_alignment="left")
        self._line(points)
        return ports

    def _voltage_dc(self, center: Point, value: str | None) -> Ports:
        x, y = center
        self._line([(x, y - 70), (x, y - 42)])
        self._line([(x, y + 42), (x, y + 70)])
        self.axes.add_patch(
            Circle((x, y), 42, fill=False, edgecolor="black", linewidth=LINE_WIDTH)
        )
        self.axes.text(x, y - 16, "+", fontsize=18, color="black", ha="center", va="center")
        self.axes.text(x, y + 18, "−", fontsize=18, color="black", ha="center", va="center")
        self._value(value, (x + 62, y), horizontal_alignment="left")
        return self._ports("voltage.dc", "v", center)

    def _ground(self, center: Point, value: str | None) -> Ports:
        x, y = center
        self._line([(x, y - 35), (x, y)])
        for offset, half_width in ((0, 34), (11, 23), (22, 11)):
            self._line([(x - half_width, y + offset), (x + half_width, y + offset)])
        self._value(value, (x + 48, y + 11), horizontal_alignment="left")
        return self._ports("gnd", "v", center)

    def _capacitor(self, center: Point, orientation: str, value: str | None) -> Ports:
        x, y = center
        if orientation == "h":
            self._line([(x - 70, y), (x - 12, y)])
            self._line([(x + 12, y), (x + 70, y)])
            self._line([(x - 12, y - 38), (x - 12, y + 38)])
            self._line([(x + 12, y - 38), (x + 12, y + 38)])
            self._value(value, (x, y - 58))
        else:
            self._line([(x, y - 70), (x, y - 12)])
            self._line([(x, y + 12), (x, y + 70)])
            self._line([(x - 38, y - 12), (x + 38, y - 12)])
            self._line([(x - 38, y + 12), (x + 38, y + 12)])
            self._value(value, (x + 56, y), horizontal_alignment="left")
        return self._ports("capacitor.unpolarized", orientation, center)

    def _inductor(self, center: Point, orientation: str, value: str | None) -> Ports:
        x, y = center
        if orientation == "h":
            self._line([(x - 80, y), (x - 48, y)])
            self._line([(x + 48, y), (x + 80, y)])
            for dx in (-36, -12, 12, 36):
                self.axes.add_patch(
                    Arc(
                        (x + dx, y),
                        24,
                        34,
                        theta1=180,
                        theta2=360,
                        color="black",
                        linewidth=LINE_WIDTH,
                    )
                )
            self._value(value, (x, y - 48))
        else:
            self._line([(x, y - 80), (x, y - 48)])
            self._line([(x, y + 48), (x, y + 80)])
            for dy in (-36, -12, 12, 36):
                self.axes.add_patch(
                    Arc(
                        (x, y + dy),
                        34,
                        24,
                        theta1=90,
                        theta2=270,
                        color="black",
                        linewidth=LINE_WIDTH,
                    )
                )
            self._value(value, (x + 52, y), horizontal_alignment="left")
        return self._ports("inductor", orientation, center)

    def _diode(
        self,
        center: Point,
        orientation: str,
        value: str | None,
        class_name: str,
    ) -> Ports:
        x, y = center
        if orientation == "h":
            self._line([(x - 70, y), (x - 30, y)])
            self.axes.add_patch(
                Polygon(
                    [(x - 30, y - 30), (x - 30, y + 30), (x + 20, y)],
                    closed=True,
                    fill=False,
                    edgecolor="black",
                    linewidth=LINE_WIDTH,
                    joinstyle="round",
                )
            )
            self._line([(x + 22, y - 34), (x + 22, y + 34)])
            self._line([(x + 22, y), (x + 70, y)])
            if class_name == "diode.zener":
                self._line([(x + 12, y - 42), (x + 22, y - 34), (x + 32, y - 42)])
                self._line([(x + 12, y + 42), (x + 22, y + 34), (x + 32, y + 42)])
            else:
                self._light_arrows(
                    [
                        ((x + 32, y - 16), (x + 58, y - 42)),
                        ((x + 45, y - 2), (x + 71, y - 28)),
                    ]
                )
            self._value(value, (x, y - 58))
        elif class_name == "diode.zener":
            self._line([(x, y - 70), (x, y - 22)])
            self._line([(x - 34, y - 22), (x + 34, y - 22)])
            self.axes.add_patch(
                Polygon(
                    [(x - 30, y + 30), (x + 30, y + 30), (x, y - 20)],
                    closed=True,
                    fill=False,
                    edgecolor="black",
                    linewidth=LINE_WIDTH,
                    joinstyle="round",
                )
            )
            self._line([(x, y + 30), (x, y + 70)])
            self._line([(x - 42, y - 32), (x - 34, y - 22), (x - 42, y - 12)])
            self._line([(x + 42, y - 32), (x + 34, y - 22), (x + 42, y - 12)])
            self._value(value, (x + 58, y), horizontal_alignment="left")
        else:
            self._line([(x, y - 70), (x, y - 30)])
            self.axes.add_patch(
                Polygon(
                    [(x - 30, y - 30), (x + 30, y - 30), (x, y + 20)],
                    closed=True,
                    fill=False,
                    edgecolor="black",
                    linewidth=LINE_WIDTH,
                    joinstyle="round",
                )
            )
            self._line([(x - 34, y + 22), (x + 34, y + 22)])
            self._line([(x, y + 22), (x, y + 70)])
            self._light_arrows(
                [
                    ((x + 20, y - 10), (x + 48, y - 38)),
                    ((x + 28, y + 7), (x + 56, y - 21)),
                ]
            )
            self._value(value, (x + 58, y), horizontal_alignment="left")
        return self._ports(class_name, orientation, center)

    def _light_arrows(self, arrows: list[tuple[Point, Point]]) -> None:
        for start, end in arrows:
            self.axes.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="->",
                    mutation_scale=12,
                    color="black",
                    linewidth=1.8,
                )
            )

    def _bjt(self, center: Point, value: str | None) -> Ports:
        x, y = center
        self._line([(x - 70, y), (x - 18, y)])
        self._line([(x - 18, y - 45), (x - 18, y + 45)])
        self._line([(x - 18, y - 28), (x + 45, y - 65)])
        self._line([(x - 18, y + 28), (x + 45, y + 65)])
        self.axes.add_patch(
            FancyArrowPatch(
                (x + 18, y + 48),
                (x + 43, y + 64),
                arrowstyle="->",
                mutation_scale=13,
                color="black",
                linewidth=1.8,
            )
        )
        self._value(value, (x, y - 85))
        return self._ports("transistor.bjt", "v", center)


def draw_reference(circuit: CircuitSpec, layout: Layout) -> ReferenceCanvas:
    """Draw one semantic circuit using its deterministic orthogonal layout."""

    validate_layout(circuit, layout)
    title = f"{circuit.id}  {circuit.title}"
    canvas = ReferenceCanvas(title)
    canvas.axes.set_title(
        title,
        fontsize=19,
        pad=12,
        fontproperties=_cjk_font_properties(),
    )

    for component in circuit.components:
        x, y, orientation = layout["components"][component.id]
        ports = canvas.component(
            component.id,
            component.class_name,
            component.value,
            (x, y),
            orientation,
        )
        for label, point in ports.items():
            canvas.port_map[f"{component.id}.{label}"] = point

    for net_id, members in circuit.nets.items():
        hub = layout["hubs"][net_id]
        net_paths: list[list[Point]] = []
        for member in members:
            port = canvas.port_map[member]
            waypoints = layout.get("waypoints", {}).get(net_id, {}).get(member)
            candidate_path = (
                [port, *waypoints, hub]
                if waypoints is not None
                else [port, (hub[0], port[1]), hub]
            )
            path = [
                point
                for index, point in enumerate(candidate_path)
                if index == 0 or point != candidate_path[index - 1]
            ]
            if any(
                start[0] != end[0] and start[1] != end[1]
                for start, end in zip(path, path[1:])
            ):
                plt.close(canvas.figure)
                raise ValueError(
                    f"{circuit.id}: {net_id}/{member} route is not orthogonal"
                )
            canvas._line(path)
            net_paths.append(path)
        canvas.route_records[net_id] = net_paths

        if len(members) >= 3:
            canvas.axes.add_patch(
                Circle(hub, 5, facecolor="black", edgecolor="black", linewidth=0)
            )
            canvas.hub_records[net_id] = hub

    return canvas

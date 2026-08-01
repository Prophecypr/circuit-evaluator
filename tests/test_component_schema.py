import ast
from pathlib import Path

from src.vision import unified_pipeline as pipeline


def _assignment(path, name):
    module = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def _function_assignment(path, function_name, name):
    module = ast.parse(Path(path).read_text(encoding="utf-8"))
    functions = [
        node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    assert len(functions) == 1, f"expected one {function_name} in {path}"
    assignments = [
        node for node in ast.walk(functions[0])
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    assert len(assignments) == 1, f"expected one {name} in {function_name}"
    return ast.literal_eval(assignments[0].value)


def test_bjt_family_has_one_name_and_one_port_order():
    annotation_names = _assignment("gen_annotation_data.py", "CGH_NAME_MAP")
    annotation_positions = _assignment("gen_annotation_data.py", "PORT_POSITIONS")
    annotation_labels = _assignment("gen_annotation_data.py", "PORT_LABELS")

    assert pipeline.CGH_NAME_MAP["transistor.bjt"] == "BJT"
    assert pipeline.CGH_TO_PORT_KEY["transistor.bjt"] == "BJT"
    assert pipeline.PORT_LABELS["BJT"] == ["B", "C", "E"]
    assert pipeline.PORT_POSITIONS["BJT"] == [
        (0.0, 0.5), (0.7, 0.0), (0.7, 1.0),
    ]
    assert annotation_names["transistor.bjt"] == "BJT"
    assert annotation_labels["BJT"] == ["B", "C", "E"]
    assert annotation_positions["BJT"] == [
        (0.0, 0.5), (0.7, 0.0), (0.7, 1.0),
    ]


def test_bjt_runtime_metadata_has_one_family_key():
    colors = _function_assignment(
        "src/vision/unified_pipeline.py", "_draw_result", "COLORS",
    )

    assert pipeline.ANTI_PATTERNS["BJT"] == [r"[ΩFVHzμAH]$"]
    assert pipeline.DESIG["BJT"] == "Q"
    assert pipeline.NM_CH["BJT"] == "BJT三极管"
    assert colors["BJT"] == (150, 150, 0)

    bjt_contracts = (
        pipeline.PORT_LABELS,
        pipeline.PORT_POSITIONS,
        pipeline.ANTI_PATTERNS,
        pipeline.DESIG,
        pipeline.NM_CH,
        colors,
    )
    for contract in bjt_contracts:
        assert "BJT-NPN" not in contract
        assert "BJT-PNP" not in contract


def test_polarity_labels_and_zener_aliases_are_intentional():
    annotation_names = _assignment("gen_annotation_data.py", "CGH_NAME_MAP")
    annotation_labels = _assignment("gen_annotation_data.py", "PORT_LABELS")

    assert pipeline.CGH_NAME_MAP["diode.zener"] == "Zener Diode"
    assert annotation_names["diode.zener"] == "Zener-Diode"
    assert pipeline.PORT_LABELS["Zener Diode"] == ["A", "K"]
    assert pipeline.PORT_LABELS["LED"] == ["+", "-"]
    assert annotation_labels["Zener-Diode"] == ["A", "K"]
    assert annotation_labels["LED"] == ["+", "-"]


def test_annotation_tool_defines_both_editing_entrypoints_consistently():
    annotation_html = Path("benchmark/annotation_tool.html").read_text(encoding="utf-8")

    assert annotation_html.count("'BJT':['B','C','E']") == 2
    assert annotation_html.count("'Zener-Diode':['A','K']") == 2
    assert annotation_html.count("'LED':['+','-']") == 2
    assert annotation_html.count("'BJT':['B','E','C']") == 0
    assert annotation_html.count("'Zener-Diode':['+','-']") == 0

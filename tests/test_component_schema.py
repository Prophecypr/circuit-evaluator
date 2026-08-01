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
    assert annotation_labels["Zener-Diode"] == ["A", "K"]
    annotation_html = Path("benchmark/annotation_tool.html").read_text(encoding="utf-8")
    assert "'BJT':['B','C','E']" in annotation_html
    assert "'Zener-Diode':['A','K']" in annotation_html

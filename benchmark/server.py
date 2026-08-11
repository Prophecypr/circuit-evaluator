"""Circuit annotation server. Images from benchmark/, detections from detections/.
Manual fixes stored in fixed/ and merged on load.
Run: python benchmark/server.py
"""
import copy, os, json
from pathlib import Path
from flask import Flask, send_file, jsonify, request

ROOT = Path(__file__).resolve().parent
DET_DIR = ROOT / "detections"
RESULT_DIR = ROOT / "result"
FIX_DIR = ROOT / "fixed"

app = Flask(__name__, static_folder=str(ROOT))
RESULT_DIR.mkdir(exist_ok=True)
FIX_DIR.mkdir(exist_ok=True)

# ---- Auto-upgrade port definitions ----
# When loading old fixed files, components with fewer ports than the current
# definition get upgraded automatically (missing ports appended).
UPGRADE_PORTS = {
    "Motor":       [(0, 0.5), (1, 0.5), (0.5, 0.0)],
    "Thyristor":   [(0, 0.5), (1, 0.5), (0.5, 0.0)],
    "Switch":      [(0, 0.5), (1, 0.5), (0.5, 0.0)],
    "Microphone":  [(0, 0.5), (1, 0.5)],
    "microphone":  [(0, 0.5), (1, 0.5)],
}
UPGRADE_LABELS = {
    "Motor":       ["M+", "M-", "OUT"],
    "Thyristor":   ["A", "K", "G"],
    "Switch":      ["1", "2", "CTRL"],
    "Microphone":  ["+", "-"],
    "microphone":  ["+", "-"],
}

def _upgrade_component(c):
    """Add missing ports/labels for components upgraded from old definitions."""
    name = c.get("name", "")
    if name in UPGRADE_PORTS:
        expected_ports = UPGRADE_PORTS[name]
        expected_labels = UPGRADE_LABELS[name]
        if len(c.get("ports", [])) < len(expected_ports):
            x1, y1, x2, y2 = c.get("xyxy", [0, 0, 100, 100])
            w, h = x2 - x1, y2 - y1
            while len(c["ports"]) < len(expected_ports):
                pi = len(c["ports"])
                rx, ry = expected_ports[pi]
                c["ports"].append([int(x1 + rx * w), int(y1 + ry * h)])
            c["labels"] = list(expected_labels)
    return c


@app.route("/")
def index():
    return send_file(ROOT / "annotation_tool.html")


@app.route("/api/images")
def api_images():
    imgs = sorted(
        p.name for p in ROOT.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    return jsonify(imgs)


@app.route("/detections/<path:filename>")
def serve_detection(filename):
    # Check fixed/ first for manual overrides, then detections/
    fix = FIX_DIR / filename
    if fix.is_file():
        data = json.loads(fix.read_text(encoding="utf-8"))
        for c in data.get("components", []):
            _upgrade_component(c)
        return jsonify(data)
    det = DET_DIR / filename
    if det.is_file():
        data = json.loads(det.read_text(encoding="utf-8"))
        for c in data.get("components", []):
            _upgrade_component(c)
        return jsonify(data)
    return "Not found", 404


@app.route("/api/save_fix", methods=["POST"])
def save_fix():
    """Save manual corrections to a detection JSON."""
    body = request.get_json()
    fname = body.get("filename", "")
    data = body.get("data", {})
    json_path = FIX_DIR / Path(fname).name
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify({"ok": True, "path": str(json_path)})


@app.route("/api/save", methods=["POST"])
def save_gt():
    body = request.get_json()
    fname = body.get("filename", "unknown")
    lines = body.get("groups", [])
    gt_name = Path(fname).stem + "_gt.txt"
    gt_path = RESULT_DIR / gt_name
    gt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonify({"ok": True, "path": str(gt_path)})


@app.route("/<path:filename>")
def serve_file(filename):
    for d in (FIX_DIR, DET_DIR, ROOT, RESULT_DIR):
        p = d / filename
        if p.is_file():
            return send_file(str(p))
    return "Not found", 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    print(f"\n  http://127.0.0.1:{port}")
    print(f"  Images:    {ROOT}")
    print(f"  Detections:{DET_DIR}")
    print(f"  Fixes:     {FIX_DIR}")
    print()
    app.run(host="127.0.0.1", port=port, debug=False)

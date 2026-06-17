"""Minimal server for the circuit annotation tool.
Run: python benchmark/server.py
"""
import os, json
from pathlib import Path
from flask import Flask, send_file, jsonify, request

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
IMAGE_DIR = PROJECT / "picture"
DET_DIR = ROOT / "detections"
RESULT_DIR = ROOT / "result"

app = Flask(__name__, static_folder=str(ROOT))
RESULT_DIR.mkdir(exist_ok=True)


@app.route("/")
def index():
    return send_file(ROOT / "annotation_tool.html")


@app.route("/api/images")
def api_images():
    imgs = sorted(
        p.name for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    return jsonify(imgs)


@app.route("/detections/<path:filename>")
def serve_detection(filename):
    det = DET_DIR / filename
    if det.is_file():
        return send_file(str(det))
    return "Not found", 404


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
    # Check detections dir first
    det = DET_DIR / filename
    if det.is_file():
        return send_file(str(det))
    # Then images dir
    img = IMAGE_DIR / filename
    if img.is_file():
        return send_file(str(img))
    # Then results dir
    res = RESULT_DIR / filename
    if res.is_file():
        return send_file(str(res))
    # Then root (html, etc.)
    p = ROOT / filename
    if p.is_file():
        return send_file(str(p))
    return "Not found", 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    print(f"Serving at http://localhost:{port}")
    print(f"Images: {IMAGE_DIR}")
    print(f"Detections: {DET_DIR}")
    app.run(host="127.0.0.1", port=port, debug=False)

"""Minimal server for the circuit annotation tool.
Run: python picture/annotation_server.py
"""
import os, json
from pathlib import Path
from flask import Flask, send_file, jsonify

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(ROOT))

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

@app.route("/<path:filename>")
def serve_file(filename):
    p = ROOT / filename
    if p.is_file():
        return send_file(str(p))
    return "Not found", 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    print(f"Serving at http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)

"""Simple HTTP server for the annotation tool.
Usage: python server.py
Then open http://localhost:8765 in browser.
"""
import http.server
import json
import os
from pathlib import Path
from urllib.parse import unquote

BENCHMARK = Path(__file__).parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BENCHMARK), **kwargs)

    def do_GET(self):
        path = unquote(self.path)
        if path == "/api/images":
            manifest = BENCHMARK / "manifest.txt"
            images = []
            if manifest.exists():
                with open(manifest) as f:
                    for line in f:
                        if line.strip():
                            images.append(line.strip().split("\t")[0])
            self.send_json(images)
            return
        if path.startswith("/images/"):
            self.path = "/" + path.split("/", 2)[-1]
            return super().do_GET()
        if path.startswith("/detections/"):
            self.path = "/detections/" + path.split("/detections/", 1)[1]
            return super().do_GET()
        if path == "/" or path == "":
            self.path = "/annotation_tool.html"
            return super().do_GET()
        return super().do_GET()

    def send_json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    os.chdir(BENCHMARK)
    print(f"Annotation tool: http://localhost:8765")
    print(f"Press Ctrl+C to stop")
    http.server.HTTPServer(("0.0.0.0", 8765), Handler).serve_forever()

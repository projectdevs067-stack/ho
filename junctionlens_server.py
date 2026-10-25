"""Traffic Flow Optimiser local vehicle counter and junction estimator.

The image is sent from the browser to this loopback-only server and processed
on this computer with the included Ultralytics YOLO11m model. No frame is
uploaded to a remote service.
"""

from __future__ import annotations

import io
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "yolo11m.pt"
HOST = "127.0.0.1"
PORT = 8787
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck in COCO
_model = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not WEIGHTS.is_file():
                    raise RuntimeError("The included YOLO11m model file is missing.")
                from ultralytics import YOLO

                _model = YOLO(str(WEIGHTS))
    return _model


class JunctionLensHandler(SimpleHTTPRequestHandler):
    server_version = "JunctionLensLocal/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        origin = self.headers.get("Origin")
        if origin in (None, "null", f"http://{HOST}:{PORT}", f"http://localhost:{PORT}"):
            self.send_header("Access-Control-Allow-Origin", origin or "*")
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        if urlsplit(self.path).path == "/api/health":
            self.send_json(200, {"status": "ready", "model": "YOLO11m", "localOnly": True})
            return
        if urlsplit(self.path).path == "/":
            self.path = "/junctionlens-queue-map.html"
        super().do_GET()

    def do_POST(self):
        if urlsplit(self.path).path != "/api/detect":
            self.send_json(404, {"error": "Unknown local detection endpoint."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                raise ValueError("Choose an image or a video frame under 50 MB.")
            frame_bytes = self.rfile.read(length)
            if len(frame_bytes) != length:
                raise ValueError("The image frame did not upload completely.")
            with Image.open(io.BytesIO(frame_bytes)) as raw_image:
                image = ImageOps.exif_transpose(raw_image).convert("RGB")

            model = get_model()
            width, height = image.size
            with _inference_lock:
                result = model.predict(
                    source=image,
                    imgsz=1280,
                    conf=0.08,
                    iou=0.45,
                    max_det=1000,
                    classes=sorted(VEHICLE_CLASSES),
                    agnostic_nms=True,
                    device="cpu",
                    verbose=False,
                )[0]

            counts = {"a": 0, "b": 0, "c": 0, "d": 0}
            for box in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                center_x = ((x1 + x2) / 2) / max(width, 1)
                center_y = ((y1 + y2) / 2) / max(height, 1)
                road = ("a" if center_x < 0.5 else "b") if center_y < 0.5 else ("c" if center_x < 0.5 else "d")
                counts[road] += 1

            active_approaches = [road for road, count in counts.items() if count > 0]
            approach_count = len(active_approaches)
            if approach_count == 4:
                junction_label = "Likely 4-way junction"
            elif approach_count == 3:
                junction_label = "Likely 3-way junction (T/Y)"
            else:
                junction_label = "Junction type unclear"

            self.send_json(200, {
                "total": sum(counts.values()),
                "counts": counts,
                "junction": {
                    "label": junction_label,
                    "approaches": active_approaches,
                    "approachCount": approach_count,
                    "basis": "Detected vehicle locations in the four image quadrants",
                },
            })
        except Exception as exc:
            self.send_json(500, {"error": str(exc) or "Vehicle detection failed."})

    def log_message(self, fmt, *args):
        # Keep the local helper quiet; detection status appears in the page.
        return


def main():
    if not WEIGHTS.is_file():
        raise SystemExit(f"Missing detector model: {WEIGHTS}")
    try:
        server = ThreadingHTTPServer((HOST, PORT), JunctionLensHandler)
    except OSError as exc:
        raise SystemExit(f"Could not start JunctionLens local detector on port {PORT}: {exc}") from exc
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
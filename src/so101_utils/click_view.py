"""Localhost page that turns a click into a pixel. It never moves the arm itself."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread
from urllib.parse import urlsplit
import json

import cv2


PAGE = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>SO-101 click and go</title>
<style>body{font:16px system-ui;background:#101820;color:#e9f1f6;margin:32px}
main{max-width:960px;margin:auto}img{max-width:100%;height:auto;cursor:crosshair;
border:1px solid #55758c}pre{white-space:pre-wrap;background:#1e303f;padding:18px}</style>
<main><h1>Click and go</h1>
<p>Click a point on the table. The arm moves to that spot on the base plane (z = 0),
stopping 2 cm above it.</p>
<img id="rgb" alt="Live RGB"><pre id="result">Waiting for camera...</pre></main>
<script>const image=document.getElementById('rgb'), result=document.getElementById('result');
let busy=false;
async function refresh(){if(busy)return;busy=true;
try{const response=await fetch('/frame',{cache:'no-store'});if(response.ok){
const blob=await response.blob(),url=URL.createObjectURL(blob),old=image.src;
image.src=url;if(old.startsWith('blob:'))URL.revokeObjectURL(old);
}}catch(e){result.textContent='Camera connection unavailable';}finally{busy=false}}
setInterval(refresh,180);refresh();
image.addEventListener('click',async event=>{const rect=image.getBoundingClientRect();
const x=Math.floor((event.clientX-rect.left)*image.naturalWidth/rect.width);
const y=Math.floor((event.clientY-rect.top)*image.naturalHeight/rect.height);
try{const response=await fetch('/click',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({x,y})});result.textContent=(await response.json()).message;
}catch(e){result.textContent='Could not send click';}});</script></html>"""


class ClickView:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self._lock = Lock()
        self._frame = None
        self._pending = []
        self._message = "Waiting for a click"
        self._server = None
        self._thread = None

    def publish(self, image_bgr) -> None:
        success, encoded = cv2.imencode(".jpg", image_bgr)
        if not success:
            raise RuntimeError("Could not encode the RGB frame")
        with self._lock:
            self._frame = encoded.tobytes()

    def consume(self):
        with self._lock:
            selected, self._pending = self._pending, []
        return selected

    def report(self, message: str) -> None:
        with self._lock:
            self._message = message

    def start(self) -> str:
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def _respond(self, code, body, content_type):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                route = urlsplit(self.path).path
                if route == "/":
                    self._respond(200, PAGE.encode(), "text/html; charset=utf-8")
                elif route == "/frame":
                    with view._lock:
                        frame = view._frame
                    if frame is None:
                        self._respond(503, b"Camera warming up", "text/plain")
                    else:
                        self._respond(200, frame, "image/jpeg")
                else:
                    self._respond(404, b"Not found", "text/plain")

            def do_POST(self):
                if urlsplit(self.path).path != "/click":
                    self._respond(404, b"Not found", "text/plain")
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 128:
                        raise ValueError("Invalid click")
                    data = json.loads(self.rfile.read(size))
                    if set(data) != {"x", "y"} or type(data["x"]) is not int or type(data["y"]) is not int:
                        raise ValueError("Pixel coordinates must be integers")
                    x, y = data["x"], data["y"]
                    if not (0 <= x < view.width and 0 <= y < view.height):
                        raise ValueError("Pixel outside the image")
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    self._respond(400, json.dumps({"message": str(exc)}).encode(), "application/json")
                    return
                with view._lock:
                    view._pending.append((x, y))
                    message = view._message
                self._respond(200, json.dumps({"message": f"Queued ({x}, {y}). {message}"}).encode(),
                              "application/json")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://127.0.0.1:{self._server.server_port}/"

    def close(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=2)


@contextmanager
def click_view(width, height):
    viewer = ClickView(width, height)
    address = viewer.start()
    try:
        yield viewer, address
    finally:
        viewer.close()

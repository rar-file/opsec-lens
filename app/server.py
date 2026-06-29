"""OPSEC Lens server — streams the Gemma 4 swarm's progress as NDJSON."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pipeline
from llm import API_KEY, MODEL, reset_usage

PORT = int(os.environ.get("PORT", "8124"))
HERE = os.path.dirname(os.path.abspath(__file__))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _headers(self, code, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                body = open(os.path.join(HERE, "index.html"), "rb").read()
            except FileNotFoundError:
                self._headers(404, "text/plain"); self.wfile.write(b"index.html missing"); return
            self._headers(200, "text/html; charset=utf-8",
                          {"Content-Length": str(len(body)), "Connection": "close"})
            self.wfile.write(body)
        elif self.path.startswith("/sv"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            lat = q.get("lat", ["0"])[0]; lon = q.get("lon", ["0"])[0]; hd = q.get("heading", ["0"])[0]
            w = q.get("w", ["640"])[0]; h = q.get("h", ["420"])[0]
            emb = (f"https://www.google.com/maps?q=&layer=c&cbll={lat},{lon}"
                   f"&cbp=11,{hd},0,0,0&output=svembed")
            html = ('<!doctype html><html><body style="margin:0;background:#000;overflow:hidden">'
                    f'<iframe src="{emb}" width="{w}" height="{h}" style="border:0;display:block">'
                    '</iframe></body></html>')
            body = html.encode()
            self._headers(200, "text/html; charset=utf-8",
                          {"Content-Length": str(len(body)), "Connection": "close"})
            self.wfile.write(body)
        else:
            self._headers(404, "application/json", {"Content-Length": "2", "Connection": "close"})
            self.wfile.write(b"{}")

    def do_POST(self):
        if self.path == "/api/redact":
            return self._do_redact()
        if self.path != "/api/analyze":
            self._headers(404, "application/json", {"Content-Length": "2", "Connection": "close"})
            self.wfile.write(b"{}")
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or "{}")
        except Exception:
            self._headers(400, "application/json", {"Connection": "close"})
            self.wfile.write(b'{"error":"bad json"}')
            return

        image_b64 = req.get("image", "")
        if image_b64.startswith("data:"):
            image_b64 = image_b64.split(",", 1)[1]
        import base64
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            self._headers(400, "application/json", {"Connection": "close"})
            self.wfile.write(b'{"error":"bad image"}')
            return

        mode = req.get("mode", "full")
        cfg = dict(pipeline.LITE_CFG if mode == "lite" else pipeline.DEFAULT_CFG)
        if isinstance(req.get("cfg"), dict):
            cfg = {**cfg, **req["cfg"]}
        cfg["base_url"] = f"http://127.0.0.1:{PORT}"  # for Street View capture

        # stream NDJSON, one event per line, flushed immediately
        self._headers(200, "application/x-ndjson",
                      {"Cache-Control": "no-cache", "Connection": "close"})
        lock = threading.Lock()

        def emit(ev):
            line = (json.dumps(ev, ensure_ascii=False) + "\n").encode("utf-8")
            with lock:
                try:
                    self.wfile.write(line)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    raise

        reset_usage()
        try:
            emit({"stage": "init", "status": "done", "model": MODEL, "mode": mode, "cfg": cfg})
            pipeline.run(image_bytes, emit=emit, cfg=cfg)
        except Exception as e:  # noqa: BLE001
            try:
                emit({"stage": "fatal", "status": "error", "error": str(e)[:400]})
            except Exception:
                pass


    def _do_redact(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or "{}")
        except Exception:
            self._headers(400, "application/json", {"Connection": "close"})
            self.wfile.write(b'{"error":"bad json"}')
            return

        data_url = req.get("image", "")
        if not data_url:
            self._headers(400, "application/json", {"Connection": "close"})
            self.wfile.write(b'{"error":"no image"}')
            return

        import base64
        import io

        import redact
        from llm import load_image

        try:
            raw = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
            img = load_image(base64.b64decode(raw))
            boxes = redact.detect_sensitive(data_url)
            out = redact.redact_image(img, boxes)
            buf = io.BytesIO()
            out.save(buf, format="PNG")
            out_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            payload = json.dumps({"image": out_url, "boxes": boxes}).encode("utf-8")
        except Exception as e:  # noqa: BLE001
            payload = json.dumps({"error": str(e)[:300]}).encode("utf-8")
            self._headers(500, "application/json",
                          {"Content-Length": str(len(payload)), "Connection": "close"})
            self.wfile.write(payload)
            return

        self._headers(200, "application/json",
                      {"Content-Length": str(len(payload)), "Connection": "close"})
        self.wfile.write(payload)


def main():
    if not API_KEY:
        sys.exit("ERROR: export CEREBRAS_API_KEY first")
    print(f"OPSEC Lens on http://0.0.0.0:{PORT}  (model={MODEL})")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

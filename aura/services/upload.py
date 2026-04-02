"""
services.upload -- HTTP file upload → RAG ingestion.

Lightweight upload server on port 8080. Drop files from any browser,
they land in data/input/ and the RAG watchdog auto-ingests them.

No dependencies beyond stdlib. Runs in a daemon thread.
"""

from __future__ import annotations

import cgi
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from core.config import WORKSPACE_ROOT

INPUT_DIR = WORKSPACE_ROOT / "data" / "input"
PORT = int(os.environ.get("AURA_UPLOAD_PORT", "8080"))

_UPLOAD_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aura</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    background: #06091a; color: #dac89b;
    display: flex; justify-content: center; align-items: center;
    min-height: 100vh;
  }
  .drop {
    width: 420px; padding: 48px;
    border: 2px dashed rgba(218,200,155,0.3); border-radius: 24px;
    text-align: center; transition: all 0.2s ease;
    cursor: pointer;
  }
  .drop.over { border-color: #dac89b; background: rgba(218,200,155,0.05); }
  .drop h1 { font-size: 20px; font-weight: 400; letter-spacing: 1px; margin-bottom: 12px; }
  .drop p { font-size: 13px; color: rgba(218,200,155,0.5); }
  .status {
    margin-top: 20px; font-size: 13px; min-height: 18px;
    color: rgba(218,200,155,0.7);
  }
  .ok { color: #50c8a5; }
  .err { color: #c88282; }
  input[type=file] { display: none; }
</style>
</head>
<body>
<div class="drop" id="drop">
  <h1>Drop files for Aura</h1>
  <p>pdf, txt, md, docx, xlsx</p>
  <div class="status" id="status"></div>
</div>
<input type="file" id="file" multiple
       accept=".pdf,.txt,.md,.docx,.xlsx,.xls">
<script>
const drop = document.getElementById('drop');
const file = document.getElementById('file');
const status = document.getElementById('status');

drop.addEventListener('click', () => file.click());
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('over');
  upload(e.dataTransfer.files);
});
file.addEventListener('change', () => upload(file.files));

async function upload(files) {
  for (const f of files) {
    status.textContent = 'Uploading ' + f.name + '...';
    status.className = 'status';
    const form = new FormData();
    form.append('file', f);
    try {
      const r = await fetch('/upload', { method: 'POST', body: form });
      const j = await r.json();
      if (r.ok) {
        status.textContent = f.name + ' — ingested';
        status.className = 'status ok';
      } else {
        status.textContent = j.error || 'Upload failed';
        status.className = 'status err';
      }
    } catch(e) {
      status.textContent = 'Connection error';
      status.className = 'status err';
    }
  }
}
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_UPLOAD_HTML.encode())

    def do_POST(self):
        if self.path != "/upload":
            self._json(404, {"error": "not found"})
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._json(400, {"error": "multipart required"})
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST",
                     "CONTENT_TYPE": content_type},
        )
        item = form["file"]
        if not item.filename:
            self._json(400, {"error": "no file"})
            return

        # Sanitize filename
        fname = os.path.basename(item.filename)
        if not fname:
            self._json(400, {"error": "bad filename"})
            return

        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        dest = INPUT_DIR / fname
        dest.write_bytes(item.file.read())
        size = dest.stat().st_size
        print(f"[upload] {fname} ({size} bytes) → {dest}")
        self._json(200, {"ok": True, "file": fname, "bytes": size})

    def _json(self, code, obj):
        import json
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Suppress per-request logging
        pass


def start():
    """Start upload server in a daemon thread."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = HTTPServer(("0.0.0.0", PORT), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True, name="upload-http")
    t.start()
    print(f"[upload] Listening on port {PORT}")

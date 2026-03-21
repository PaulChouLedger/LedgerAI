"""
services.broadcast_receiver -- Tiny HTTP server inside Aura process.

Listens on port 5050 for broadcast messages from the Farsight Hub
and injects them into the speaker pipeline via the event bus.

Mutes the microphone during broadcast playback to prevent the mic
from picking up the broadcast and triggering a conversation loop.

Started from aura.py after the speaker is initialized.
"""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from core.bus import bus

BROADCAST_PORT = 5050


def _play_broadcast(text: str):
    """Mute mic, speak text, unmute when done."""
    done = threading.Event()

    def _on_finished(**_kw):
        bus.off("tts.finished", _on_finished)
        # Small delay so aplay fully finishes before mic reopens
        import time
        time.sleep(0.5)
        bus.emit("mute.toggled", muted=False)
        print(f"[broadcast] Playback done, mic unmuted", flush=True)
        done.set()

    bus.on("tts.finished", _on_finished)
    bus.emit("mute.toggled", muted=True)
    print(f"[broadcast] Mic muted, speaking: \"{text[:80]}\"", flush=True)
    bus.emit("llm.sentence", text=text)


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/play":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            text = body.get("text", "").strip()
            if text:
                print(f"[broadcast] Received: \"{text[:80]}\"", flush=True)
                _play_broadcast(text)
                self._respond(200, {"ok": True})
            else:
                self._respond(400, {"error": "text required"})
        elif self.path == "/mute":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            muted = bool(body.get("muted", True))
            bus.emit("mute.toggled", muted=muted)
            print(f"[broadcast] Remote mute: {muted}", flush=True)
            self._respond(200, {"ok": True, "muted": muted})
        elif self.path == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not found"})

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not found"})

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt, *args):
        pass  # suppress request logging


def start():
    """Start broadcast receiver on a daemon thread."""
    def _run():
        server = HTTPServer(("0.0.0.0", BROADCAST_PORT), _Handler)
        print(f"[broadcast] Listening on port {BROADCAST_PORT}", flush=True)
        server.serve_forever()

    t = threading.Thread(target=_run, daemon=True, name="broadcast-rx")
    t.start()

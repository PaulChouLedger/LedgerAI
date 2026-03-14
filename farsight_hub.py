"""
Farsight Hub — Central coordination server for the Aura Farsight.

Runs on the Farsight RTX GPU server (100.76.191.92) and acts as a hub
for all Aura pucks on the LAN. Provides puck registration, heartbeat
tracking, LLM offload to the RTX's large model, fleet management,
broadcast messaging, and a constellation dashboard for iPad viewing.

Port: 8314 (pi digits)
"""

import json
import queue
import subprocess
import threading
import time
from datetime import datetime, timezone

import base64
import os

from flask import Flask, request, jsonify, Response, send_file
import requests as http_requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FARSIGHT_LLM_URL = "http://localhost:11434"
HEARTBEAT_TIMEOUT_S = 60
HUB_PORT = 8314

# Real pucks that can receive broadcast TTS via SSH
REAL_PUCKS = {
    "192.168.1.108": {"host": "puck1", "user": "ledger"},
    "192.168.1.94":  {"host": "puck2", "user": "ledger"},
}

# ---------------------------------------------------------------------------
# App + registry
# ---------------------------------------------------------------------------

app = Flask(__name__)

_puck_registry: dict = {}       # puck_id -> dict
_registry_lock = threading.Lock()
_sse_clients: list = []         # list of queue.Queue for SSE subscribers
_sse_lock = threading.Lock()
_pending_messages: dict = {}    # puck_id -> list of pending messages
_pending_updates: dict = {}     # puck_id -> {"ref": "main"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _puck_status(puck: dict) -> str:
    elapsed = time.time() - puck.get("last_seen_ts", 0)
    if elapsed > HEARTBEAT_TIMEOUT_S:
        return "offline"
    return puck.get("status", "idle")


def _broadcast_sse(event_type: str, data: dict):
    """Push an SSE event to all connected dashboard clients."""
    msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    with _sse_lock:
        dead = []
        for i, q in enumerate(_sse_clients):
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(i)
        for i in reversed(dead):
            _sse_clients.pop(i)


def _get_peers(exclude_id: str = None) -> list:
    """Return list of peer pucks (excluding the requesting puck)."""
    peers = []
    for puck in _puck_registry.values():
        if puck["puck_id"] == exclude_id:
            continue
        peers.append({
            "puck_id": puck["puck_id"],
            "puck_name": puck.get("puck_name", "unknown"),
            "color": puck.get("color", "#23A5FF"),
            "status": _puck_status(puck),
            "ip": puck.get("ip"),
        })
    return peers


# ---------------------------------------------------------------------------
# GPU monitoring (nvidia-smi)
# ---------------------------------------------------------------------------

_gpu_cache = {"data": None, "ts": 0}
_GPU_CACHE_TTL = 3.0


def _read_gpu():
    """Read GPU stats from nvidia-smi, cached for 3s."""
    now = time.time()
    if _gpu_cache["data"] and (now - _gpu_cache["ts"]) < _GPU_CACHE_TTL:
        return _gpu_cache["data"]
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,"
            "memory.used,memory.total,name",
            "--format=csv,noheader,nounits",
        ], timeout=5).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        data = {
            "gpu_util": int(parts[0]),
            "mem_util": int(parts[1]),
            "temp_c": int(parts[2]),
            "mem_used_mb": int(parts[3]),
            "mem_total_mb": int(parts[4]),
            "gpu_name": parts[5] if len(parts) > 5 else "Unknown",
        }
    except Exception:
        data = None
    _gpu_cache["data"] = data
    _gpu_cache["ts"] = now
    return data


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(force=True)
    puck_id = data.get("puck_id")
    if not puck_id:
        return jsonify({"error": "puck_id required"}), 400

    with _registry_lock:
        _puck_registry[puck_id] = {
            "puck_id": puck_id,
            "puck_name": data.get("puck_name", "unknown"),
            "owner_name": data.get("owner_name", "unknown"),
            "ip": data.get("ip", request.remote_addr),
            "color": data.get("color", "#23A5FF"),
            "capabilities": data.get("capabilities", []),
            "version": data.get("version", {}),
            "status": "idle",
            "memory_usage": None,
            "uptime": None,
            "last_conversation_ts": None,
            "registered_at": _now_iso(),
            "last_seen_ts": time.time(),
        }

    _broadcast_sse("puck_registered", {
        "puck_id": puck_id,
        "puck_name": data.get("puck_name"),
        "color": data.get("color", "#23A5FF"),
    })
    return jsonify({"ok": True, "puck_id": puck_id})


@app.route("/heartbeat", methods=["POST"])
def heartbeat():
    data = request.get_json(force=True)
    puck_id = data.get("puck_id")
    if not puck_id:
        return jsonify({"error": "puck_id required"}), 400

    with _registry_lock:
        puck = _puck_registry.get(puck_id)
        if puck is None:
            return jsonify({"error": "puck not registered"}), 404

        old_status = _puck_status(puck)
        puck["last_seen_ts"] = time.time()
        puck["status"] = data.get("status", puck.get("status", "idle"))
        if "memory_usage" in data:
            puck["memory_usage"] = data["memory_usage"]
        if "uptime" in data:
            puck["uptime"] = data["uptime"]
        if "last_conversation_ts" in data:
            puck["last_conversation_ts"] = data["last_conversation_ts"]
        if "version" in data:
            puck["version"] = data["version"]
        new_status = _puck_status(puck)

        # Gather peers for this puck
        peers = _get_peers(exclude_id=puck_id)

        # Gather pending messages
        msgs = _pending_messages.pop(puck_id, [])
        update_cmd = _pending_updates.pop(puck_id, None)

    # Broadcast status change to dashboard
    if old_status != new_status:
        _broadcast_sse("status_change", {
            "puck_id": puck_id,
            "old": old_status,
            "new": new_status,
        })

    _broadcast_sse("heartbeat", {
        "puck_id": puck_id,
        "status": new_status,
    })

    resp = {"ok": True, "peers": peers}
    if msgs:
        resp["pending_messages"] = msgs
    if update_cmd:
        resp["pending_update"] = update_cmd
    return jsonify(resp)


@app.route("/pucks", methods=["GET"])
def pucks():
    with _registry_lock:
        result = []
        for puck in _puck_registry.values():
            entry = dict(puck)
            entry["effective_status"] = _puck_status(puck)
            result.append(entry)
    return jsonify(result)


@app.route("/offload", methods=["POST"])
def offload():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    system_prompt = data.get("system_prompt", "You are a helpful assistant.")
    max_tokens = data.get("max_tokens", 512)
    puck_id = data.get("puck_id", "unknown")

    _broadcast_sse("offload_start", {"puck_id": puck_id})

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "stream": True,
    }

    def generate():
        try:
            resp = http_requests.post(
                f"{FARSIGHT_LLM_URL}/chat-tts",
                json=payload,
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    yield line + "\n"
        except http_requests.RequestException as exc:
            yield json.dumps({"error": str(exc)}) + "\n"
        finally:
            _broadcast_sse("offload_end", {"puck_id": puck_id})

    return Response(generate(), mimetype="text/event-stream")


@app.route("/gpu", methods=["GET"])
def gpu():
    data = _read_gpu()
    if data is None:
        return jsonify({"error": "nvidia-smi not available"}), 503
    return jsonify(data)


BROADCAST_RECEIVER_PORT = 5050


def _deliver_tts_to_puck(message: str, puck_ip: str):
    """Send text to a puck's broadcast receiver for local Kokoro TTS playback."""
    host = REAL_PUCKS.get(puck_ip, {}).get("host", puck_ip)
    try:
        resp = http_requests.post(
            f"http://{puck_ip}:{BROADCAST_RECEIVER_PORT}/play",
            json={"text": message},
            timeout=10,
        )
        if resp.status_code == 200:
            print(f"[hub-broadcast] Sent to {host} ({puck_ip})")
        else:
            print(f"[hub-broadcast] {host} returned {resp.status_code}")
    except Exception as e:
        print(f"[hub-broadcast] Failed to reach {host}: {e}")


def _broadcast_tts_async(message: str):
    """Send broadcast text to all real pucks for local TTS synthesis + playback."""
    print(f"[hub-broadcast] Delivering to pucks: \"{message[:80]}\"")
    threads = []
    for ip in REAL_PUCKS:
        t = threading.Thread(target=_deliver_tts_to_puck, args=(message, ip))
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=15)
    print(f"[hub-broadcast] Delivery complete")


@app.route("/broadcast", methods=["POST"])
def broadcast():
    data = request.get_json(force=True)
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "message required"}), 400

    with _registry_lock:
        targets = []
        for puck_id, puck in _puck_registry.items():
            if _puck_status(puck) != "offline":
                _pending_messages.setdefault(puck_id, []).append({
                    "type": "broadcast",
                    "message": message,
                    "ts": _now_iso(),
                })
                targets.append(puck_id)

    _broadcast_sse("broadcast", {"message": message, "targets": targets})

    # Fire off TTS synthesis + delivery to real pucks in background
    threading.Thread(target=_broadcast_tts_async, args=(message,), daemon=True).start()

    return jsonify({"ok": True, "delivered_to": len(targets)})


@app.route("/fleet/versions", methods=["GET"])
def fleet_versions():
    with _registry_lock:
        versions = {}
        for puck_id, puck in _puck_registry.items():
            versions[puck_id] = {
                "puck_name": puck.get("puck_name"),
                "version": puck.get("version", {}),
                "status": _puck_status(puck),
            }
    return jsonify(versions)


@app.route("/fleet/update", methods=["POST"])
def fleet_update():
    data = request.get_json(force=True)
    puck_id = data.get("puck_id")
    ref = data.get("ref", "main")
    if not puck_id:
        return jsonify({"error": "puck_id required"}), 400

    with _registry_lock:
        if puck_id not in _puck_registry:
            return jsonify({"error": "puck not found"}), 404
        _pending_updates[puck_id] = {"ref": ref}

    _broadcast_sse("update_queued", {"puck_id": puck_id, "ref": ref})
    return jsonify({"ok": True, "puck_id": puck_id, "ref": ref})


@app.route("/stream")
def stream():
    """SSE endpoint for real-time dashboard updates."""
    q = queue.Queue(maxsize=200)
    with _sse_lock:
        _sse_clients.append(q)

    def event_stream():
        try:
            # Send initial state
            start = time.time()
            yield f"event: init\ndata: {json.dumps({'ts': _now_iso()})}\n\n"
            while True:
                # Force reconnect after 90s to free threads
                if time.time() - start > 90:
                    yield f"event: reconnect\ndata: {{}}\n\n"
                    break
                try:
                    msg = q.get(timeout=25)
                    yield msg
                except queue.Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            with _sse_lock:
                try:
                    _sse_clients.remove(q)
                except ValueError:
                    pass

    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/health", methods=["GET"])
def health():
    with _registry_lock:
        n_pucks = len(_puck_registry)
        n_online = sum(
            1 for p in _puck_registry.values()
            if _puck_status(p) != "offline"
        )
    gpu = _read_gpu()
    return jsonify({
        "status": "ok",
        "pucks_registered": n_pucks,
        "pucks_online": n_online,
        "gpu": gpu,
        "timestamp": _now_iso(),
    })


# ---------------------------------------------------------------------------
# Web dashboard — Constellation view with canvas animation
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>FARSIGHT // TACTICAL</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #020604;
    --surface: #040a07;
    --surface2: #061008;
    --border: #0c2418;
    --border-hi: #14382a;
    --text: #7aac8e;
    --text-dim: #3a6648;
    --text-bright: #c0f0d0;
    --green: #00ff88;
    --green-dim: rgba(0,255,136,0.08);
    --amber: #ffaa00;
    --amber-dim: rgba(255,170,0,0.08);
    --red: #ff2244;
    --red-dim: rgba(255,34,68,0.06);
    --cyan: #00ddff;
    --cyan-dim: rgba(0,221,255,0.06);
    --gold: #d4b868;
    --gold-bright: #f0dca0;
    --sidebar-w: 240px;
    --mono: "JetBrains Mono", "Share Tech Mono", monospace;
    --grid-color: rgba(0,255,136,0.03);
    --scanline-color: rgba(0,255,136,0.015);
  }

  html, body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
    overflow-x: hidden;
  }

  /* ── Scanline + grid overlay on entire page ────── */
  body::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: 9999;
    pointer-events: none;
    background:
      repeating-linear-gradient(0deg, transparent, transparent 2px, var(--scanline-color) 2px, var(--scanline-color) 4px),
      repeating-linear-gradient(90deg, var(--grid-color) 0px, var(--grid-color) 1px, transparent 1px, transparent 80px),
      repeating-linear-gradient(0deg, var(--grid-color) 0px, var(--grid-color) 1px, transparent 1px, transparent 80px);
  }
  body::after {
    content: "";
    position: fixed;
    inset: 0;
    z-index: 9998;
    pointer-events: none;
    background: radial-gradient(ellipse at 50% 50%, transparent 50%, rgba(0,0,0,0.6) 100%);
  }

  .layout { display: flex; min-height: 100vh; }

  /* ── Sidebar — tactical nav ────────────────────── */
  .sidebar {
    width: var(--sidebar-w);
    min-height: 100vh;
    background: var(--surface);
    border-right: 2px solid rgba(0,255,136,0.25);
    padding: 20px 0;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    position: fixed;
    left: 0; top: 0; bottom: 0;
    z-index: 10;
  }

  .sidebar-brand {
    padding: 16px 12px 20px;
    text-align: center;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
  }
  .sidebar-brand svg {
    display: block;
    width: 216px;
    height: 120px;
    margin: 0 auto;
    filter: drop-shadow(0 0 8px rgba(0,255,136,0.15));
  }

  .sidebar-section { padding: 0 12px; margin-bottom: 20px; }
  .sidebar-section-label {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--text-dim);
    padding: 0 8px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .sidebar-section-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  .sidebar-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 12px;
    cursor: pointer;
    transition: all 0.12s;
    font-size: 12px;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 1px;
    border-left: 2px solid transparent;
    position: relative;
  }
  .sidebar-item:hover { background: var(--green-dim); color: var(--text-bright); border-left-color: rgba(0,255,136,0.3); }
  .sidebar-item.active { background: var(--green-dim); color: var(--green); border-left-color: var(--green); }
  .sidebar-item .icon { font-size: 14px; width: 20px; text-align: center; opacity: 0.6; }
  .sidebar-item.active .icon { opacity: 1; }
  .sidebar-item .badge {
    margin-left: auto;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 6px;
    background: var(--green-dim);
    color: var(--green);
    border: 1px solid rgba(0,255,136,0.2);
  }

  .sidebar-footer {
    margin-top: auto;
    padding: 12px 16px 0;
    border-top: 1px solid var(--border);
  }
  .sidebar-footer .gpu-mini {
    font-size: 10px;
    color: var(--text-dim);
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
  }
  .sidebar-footer .gpu-mini span { color: var(--green); }
  .sidebar-footer .gpu-bar {
    height: 3px;
    background: rgba(0,255,136,0.06);
    overflow: hidden;
    margin-bottom: 8px;
  }
  .sidebar-footer .gpu-bar-fill {
    height: 100%;
    background: var(--green);
    box-shadow: 0 0 8px var(--green);
    transition: width 1.5s;
    width: 0%;
  }
  .sidebar-footer .version-info {
    font-size: 9px;
    color: var(--text-dim);
    letter-spacing: 0.1em;
  }

  /* ── Main content ──────────────────────────────── */
  .main {
    flex: 1;
    margin-left: var(--sidebar-w);
    position: relative;
    min-height: 100vh;
  }
  .main-content { position: relative; z-index: 1; }

  /* ── Top bar — tactical HUD ────────────────────── */
  .top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px 28px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(0,255,136,0.02) 0%, transparent 100%);
  }
  .top-bar .tb-left {
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .top-bar .tb-logo img {
    height: 52px;
    filter: drop-shadow(0 0 18px rgba(0,255,136,0.3)) brightness(1.2);
  }
  .top-bar .tb-title {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0.3em;
    color: var(--green);
    text-shadow: 0 0 22px rgba(0,255,136,0.45);
  }
  .top-bar .tb-subtitle {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-bright);
    letter-spacing: 0.15em;
    margin-top: 3px;
    text-shadow: 0 0 10px rgba(0,255,136,0.15);
  }
  .top-bar .tb-right {
    display: flex;
    gap: 24px;
    align-items: center;
  }
  .top-bar .tb-stat {
    text-align: center;
  }
  .top-bar .tb-stat-value {
    font-size: 20px;
    font-weight: 700;
    color: var(--text-bright);
    text-shadow: 0 0 10px rgba(0,255,136,0.15);
  }
  .top-bar .tb-stat-label {
    font-size: 8px;
    letter-spacing: 0.2em;
    color: var(--text-dim);
    text-transform: uppercase;
  }
  .top-bar .tb-clock {
    font-size: 22px;
    font-weight: 700;
    color: var(--cyan);
    text-shadow: 0 0 14px rgba(0,221,255,0.45);
    letter-spacing: 0.08em;
  }
  .top-bar .tb-clock-label {
    font-size: 10px;
    font-weight: 600;
    color: var(--cyan);
    letter-spacing: 0.2em;
    text-align: right;
    opacity: 0.6;
  }

  /* ── Broadcast bar ─────────────────────────────── */
  .broadcast {
    max-width: 720px;
    margin: 20px auto 0;
    display: flex;
    gap: 8px;
    padding: 0 28px;
  }
  .broadcast input {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 12px 16px;
    color: var(--text-bright);
    font-family: var(--mono);
    font-size: 13px;
    outline: none;
    transition: border-color 0.3s;
  }
  .broadcast input:focus { border-color: rgba(0,255,136,0.35); box-shadow: 0 0 15px rgba(0,255,136,0.05); }
  .broadcast input::placeholder { color: var(--green); opacity: 0.6; }
  .broadcast button {
    background: rgba(0,255,136,0.06);
    border: 1px solid rgba(0,255,136,0.25);
    color: var(--green);
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 0 20px;
    cursor: pointer;
    transition: all 0.2s;
    position: relative;
    overflow: hidden;
  }
  .broadcast button::before {
    content: "";
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,255,136,0.1), transparent);
    transition: left 0.4s;
  }
  .broadcast button:hover::before { left: 100%; }
  .broadcast button:hover { background: rgba(0,255,136,0.12); }
  .broadcast button.transmitting {
    opacity: 0.5;
    pointer-events: none;
    color: var(--amber);
    border-color: rgba(255,170,0,0.3);
    background: var(--amber-dim);
  }

  /* ── Puck cards — tactical readout panels ──────── */
  .puck-cards {
    max-width: 1200px;
    margin: 24px auto 0;
    padding: 0 28px 80px;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 16px;
  }

  .puck-card {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 0;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, box-shadow 0.3s, transform 0.45s cubic-bezier(0.22,1,0.36,1), opacity 0.4s ease;
    cursor: pointer;
    z-index: 1;
  }
  .puck-card:hover {
    border-color: var(--green) !important;
    border-width: 2px !important;
    box-shadow: 0 0 24px rgba(0,255,136,0.15), 0 0 60px rgba(0,255,136,0.05), inset 0 0 30px rgba(0,255,136,0.03);
  }
  /* Dimmed state when another card is expanded */
  .puck-card.dimmed {
    opacity: 0.15;
    pointer-events: none;
    filter: blur(2px);
    transform: scale(0.97);
  }
  /* Expanded card */
  .puck-card.expanded {
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    width: 70vw !important;
    max-width: 900px !important;
    max-height: 92vh !important;
    overflow-y: auto !important;
    z-index: 1000 !important;
    border-color: var(--green) !important;
    box-shadow: 0 0 60px rgba(0,255,136,0.12), 0 0 120px rgba(0,255,136,0.04), 0 4px 60px rgba(0,0,0,0.8) !important;
  }
  .puck-card.expanded .pc-transcript {
    max-height: 55vh;
    min-height: 240px;
    font-size: 13px;
  }
  .puck-card.expanded .pc-analysis-text {
    font-size: 14px;
    line-height: 1.7;
  }
  .puck-card.expanded .pc-name {
    font-size: 16px;
  }
  /* Backdrop overlay */
  .card-backdrop {
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.7);
    z-index: 999;
    opacity: 0;
    transition: opacity 0.4s ease;
  }
  .card-backdrop.visible {
    display: block;
    opacity: 1;
  }

  /* Top accent line */
  .puck-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    transition: background 0.5s;
  }
  /* Corner brackets — tactical feel */
  .puck-card::after {
    content: "";
    position: absolute;
    inset: 4px;
    border: 1px solid rgba(0,255,136,0.04);
    pointer-events: none;
  }

  .puck-card.online  { border-color: rgba(0,255,136,0.2); box-shadow: 0 0 30px rgba(0,255,136,0.04), inset 0 0 40px rgba(0,255,136,0.01); }
  .puck-card.online::before  { background: var(--green); box-shadow: 0 0 10px var(--green); }
  .puck-card.idle    { border-color: rgba(255,170,0,0.18); box-shadow: 0 0 30px rgba(255,170,0,0.03); }
  .puck-card.idle::before    { background: var(--amber); box-shadow: 0 0 10px var(--amber); }
  .puck-card.offline { border-color: rgba(255,34,68,0.12); opacity: 0.35; }
  .puck-card.offline::before { background: var(--red); opacity: 0.5; }

  .pc-inner { padding: 16px 18px 14px; }

  .pc-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
  }
  .pc-color-dot {
    width: 10px; height: 10px;
    flex-shrink: 0;
    position: relative;
  }
  .pc-color-dot::before {
    content: "";
    position: absolute;
    inset: -3px;
    border: 1px solid currentColor;
    opacity: 0.3;
    transform: rotate(45deg);
  }
  .pc-name {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-bright);
    flex: 1;
    letter-spacing: 0.05em;
    text-shadow: 0 0 8px rgba(0,255,136,0.1);
  }
  .pc-status {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 4px 10px;
    position: relative;
  }
  .pc-status::before {
    content: "";
    position: absolute;
    left: 0; top: 50%;
    width: 4px; height: 4px;
    transform: translateY(-50%);
    border-radius: 50%;
    animation: statusBlink 2s infinite;
  }
  .pc-status.online  { color: var(--green); background: var(--green-dim); }
  .pc-status.online::before  { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .pc-status.idle    { color: var(--amber); background: var(--amber-dim); }
  .pc-status.idle::before    { background: var(--amber); box-shadow: 0 0 6px var(--amber); animation-duration: 3s; }
  .pc-status.offline { color: var(--red); background: var(--red-dim); }
  .pc-status.offline::before { background: var(--red); animation: none; opacity: 0.5; }

  @keyframes statusBlink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  .pc-id {
    font-size: 10px;
    color: var(--text-dim);
    margin-bottom: 14px;
    letter-spacing: 0.05em;
    padding: 4px 0;
    border-bottom: 1px solid var(--border);
  }

  .pc-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px 20px;
  }
  .pc-metric { display: flex; flex-direction: column; gap: 2px; }
  .pc-metric-label {
    font-size: 8px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--text-dim);
  }
  .pc-metric-value {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-bright);
  }

  .pc-ledger {
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .pc-ledger-label {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-dim);
  }
  .pc-ledger-value {
    font-size: 14px;
    font-weight: 600;
    color: var(--gold);
    text-shadow: 0 0 8px rgba(212,184,104,0.2);
  }

  .pc-version {
    margin-top: 10px;
    display: flex;
    justify-content: space-between;
    font-size: 10px;
  }
  .pc-branch { color: var(--cyan); font-weight: 500; }
  .pc-commit { color: var(--text-dim); }

  /* Radar sweep on card bottom */
  .pc-radar {
    height: 3px;
    background: var(--surface2);
    position: relative;
    overflow: hidden;
  }
  .pc-radar::before {
    content: "";
    position: absolute;
    top: 0; bottom: 0;
    width: 60px;
    background: linear-gradient(90deg, transparent, var(--green), transparent);
    animation: radarSweep 3s linear infinite;
  }
  .puck-card.idle .pc-radar::before { background: linear-gradient(90deg, transparent, var(--amber), transparent); animation-duration: 5s; }
  .puck-card.offline .pc-radar::before { display: none; }

  @keyframes radarSweep {
    0% { left: -60px; }
    100% { left: 100%; }
  }

  /* ── Occupants ──────────────────────────────────── */
  .pc-occupants {
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
  }
  .pc-occupants-label {
    font-size: 8px;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 6px;
  }
  .pc-occupant {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 13px;
    color: var(--text);
    margin-right: 12px;
    margin-bottom: 4px;
  }
  .pc-occupant .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .pc-occupant .dot.speaking {
    background: var(--green);
    box-shadow: 0 0 6px var(--green);
    animation: statusBlink 1s infinite;
  }
  .pc-occupant .dot.silent {
    background: var(--text-dim);
    opacity: 0.5;
  }
  .pc-occupant.is-speaking { color: var(--green); font-weight: 600; }

  /* ── Transcript ────────────────────────────────── */
  .pc-transcript {
    margin-top: 10px;
    padding: 10px 12px;
    background: rgba(0,255,136,0.02);
    border: 1px solid rgba(0,255,136,0.06);
    max-height: 180px;
    min-height: 80px;
    overflow-y: auto;
    font-size: 13px;
    line-height: 1.6;
    color: var(--text);
  }
  .pc-transcript::-webkit-scrollbar { width: 3px; }
  .pc-transcript::-webkit-scrollbar-track { background: transparent; }
  .pc-transcript::-webkit-scrollbar-thumb { background: var(--border-hi); }
  .pc-transcript .ts-line { margin-bottom: 5px; }
  .pc-transcript .ts-speaker { color: var(--cyan); font-weight: 600; }
  .pc-transcript .ts-time { color: var(--text-dim); margin-right: 6px; font-size: 11px; }

  /* ── Aura's Viewpoint ───────────────────────────── */
  .pc-analysis {
    margin-top: 10px;
    padding: 12px 14px;
    background: rgba(0,221,255,0.03);
    border: 1px solid rgba(0,221,255,0.12);
    border-left: 3px solid var(--cyan);
    border-radius: 2px;
    position: relative;
  }
  .pc-analysis::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(135deg, rgba(0,221,255,0.04) 0%, transparent 60%);
    pointer-events: none;
  }
  .pc-analysis-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.2em;
    color: var(--cyan);
    margin-bottom: 8px;
    text-transform: uppercase;
  }
  .pc-analysis-label .aura-eye {
    width: 18px;
    height: 18px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    position: relative;
  }
  .pc-analysis-label .aura-eye::before {
    content: "";
    width: 14px;
    height: 8px;
    border: 1.5px solid var(--cyan);
    border-radius: 50%;
    position: absolute;
  }
  .pc-analysis-label .aura-eye::after {
    content: "";
    width: 5px;
    height: 5px;
    background: var(--cyan);
    border-radius: 50%;
    box-shadow: 0 0 6px var(--cyan), 0 0 12px var(--cyan);
    animation: auraPulse 2s ease-in-out infinite;
  }
  @keyframes auraPulse {
    0%, 100% { opacity: 0.7; transform: scale(0.9); }
    50% { opacity: 1; transform: scale(1.1); }
  }
  .pc-analysis-text {
    font-size: 13px;
    line-height: 1.6;
    color: var(--cyan);
    font-style: italic;
    transition: opacity 0.3s ease;
    opacity: 0.85;
  }
  .ts-line.ts-new {
    animation: tsFlash 1.5s ease;
  }
  @keyframes tsFlash {
    0% { background: rgba(0,255,136,0.12); }
    100% { background: transparent; }
  }

  .empty-msg {
    text-align: center;
    color: var(--text-dim);
    padding: 100px 20px;
    font-size: 14px;
    letter-spacing: 0.12em;
    grid-column: 1 / -1;
  }

  /* Broadcast pulse — amber tactical alert */
  @keyframes broadcastPulse {
    0%   { border-color: rgba(255,170,0,0.6); box-shadow: 0 0 40px rgba(255,170,0,0.12), inset 0 0 30px rgba(255,170,0,0.03); }
    50%  { border-color: rgba(255,170,0,0.3); box-shadow: 0 0 50px rgba(255,170,0,0.06); }
    100% { border-color: rgba(255,170,0,0.0); box-shadow: none; }
  }
  .puck-card.broadcasting { animation: broadcastPulse 4s ease-out forwards; }
  .puck-card.broadcasting::before { background: var(--amber) !important; box-shadow: 0 0 12px var(--amber) !important; }

  @media (max-width: 768px) {
    .sidebar { display: none; }
    .main { margin-left: 0; }
    .puck-cards { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<div class="layout">
  <nav class="sidebar">
    <div class="sidebar-brand">
      <svg viewBox="0 0 216 120" xmlns="http://www.w3.org/2000/svg">
        <!-- Outer targeting reticle -->
        <circle cx="108" cy="52" r="42" fill="none" stroke="#00ff88" stroke-width="0.8" stroke-opacity="0.25" stroke-dasharray="4,6"/>
        <circle cx="108" cy="52" r="34" fill="none" stroke="#00ff88" stroke-width="0.5" stroke-opacity="0.12"/>
        <!-- Scan sweep glow -->
        <circle cx="108" cy="52" r="38" fill="none" stroke="#00ff88" stroke-width="2" stroke-opacity="0.06">
          <animate attributeName="r" values="28;42;28" dur="4s" repeatCount="indefinite"/>
          <animate attributeName="stroke-opacity" values="0.12;0.02;0.12" dur="4s" repeatCount="indefinite"/>
        </circle>
        <!-- Crosshair — vertical -->
        <line x1="108" y1="14" x2="108" y2="36" stroke="#00ff88" stroke-width="0.6" stroke-opacity="0.3"/>
        <line x1="108" y1="68" x2="108" y2="90" stroke="#00ff88" stroke-width="0.6" stroke-opacity="0.3"/>
        <!-- Crosshair — horizontal -->
        <line x1="70" y1="52" x2="92" y2="52" stroke="#00ff88" stroke-width="0.6" stroke-opacity="0.3"/>
        <line x1="124" y1="52" x2="146" y2="52" stroke="#00ff88" stroke-width="0.6" stroke-opacity="0.3"/>
        <!-- Corner brackets — top-left -->
        <polyline points="78,24 78,18 84,18" fill="none" stroke="#00ff88" stroke-width="1.2" stroke-opacity="0.5"/>
        <!-- Corner brackets — top-right -->
        <polyline points="132,18 138,18 138,24" fill="none" stroke="#00ff88" stroke-width="1.2" stroke-opacity="0.5"/>
        <!-- Corner brackets — bottom-left -->
        <polyline points="78,80 78,86 84,86" fill="none" stroke="#00ff88" stroke-width="1.2" stroke-opacity="0.5"/>
        <!-- Corner brackets — bottom-right -->
        <polyline points="132,86 138,86 138,80" fill="none" stroke="#00ff88" stroke-width="1.2" stroke-opacity="0.5"/>
        <!-- Inner diamond -->
        <polygon points="108,38 122,52 108,66 94,52" fill="none" stroke="#00ff88" stroke-width="0.7" stroke-opacity="0.2"/>
        <!-- Center dot — pulsing -->
        <circle cx="108" cy="52" r="3" fill="#00ff88" fill-opacity="0.7">
          <animate attributeName="fill-opacity" values="0.7;0.2;0.7" dur="2s" repeatCount="indefinite"/>
        </circle>
        <circle cx="108" cy="52" r="6" fill="none" stroke="#00ff88" stroke-width="0.5" stroke-opacity="0.3"/>
        <!-- Rotating tick marks -->
        <g opacity="0.35">
          <animateTransform attributeName="transform" type="rotate" from="0 108 52" to="360 108 52" dur="30s" repeatCount="indefinite"/>
          <line x1="108" y1="12" x2="108" y2="16" stroke="#00ff88" stroke-width="0.8"/>
          <line x1="148" y1="52" x2="144" y2="52" stroke="#00ff88" stroke-width="0.8"/>
          <line x1="108" y1="92" x2="108" y2="88" stroke="#00ff88" stroke-width="0.8"/>
          <line x1="68" y1="52" x2="72" y2="52" stroke="#00ff88" stroke-width="0.8"/>
        </g>
        <!-- "F" glyph — center -->
        <text x="108" y="57" text-anchor="middle" fill="#00ff88" font-family="JetBrains Mono,monospace" font-weight="700" font-size="18" opacity="0.85">F</text>
        <!-- FARSIGHT wordmark -->
        <text x="108" y="108" text-anchor="middle" fill="#00ff88" font-family="JetBrains Mono,monospace" font-weight="700" font-size="14" letter-spacing="6" opacity="0.9">FARSIGHT</text>
      </svg>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-section-label">Fleet Ops</div>
      <div class="sidebar-item active" onclick="setView('constellation')">
        <span class="icon">&#9678;</span> Constellation
        <span class="badge" id="sb-online">0</span>
      </div>
      <div class="sidebar-item" onclick="setView('fleet')">
        <span class="icon">&#9632;</span> Fleet Status
      </div>
      <div class="sidebar-item" onclick="setView('activity')">
        <span class="icon">&#9734;</span> Activity Log
      </div>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-section-label">Command</div>
      <div class="sidebar-item" onclick="setView('users')">
        <span class="icon">&#9775;</span> Operators
      </div>
      <div class="sidebar-item" onclick="setView('billing')">
        <span class="icon">&#9830;</span> $LEDGER
      </div>
      <div class="sidebar-item" onclick="setView('policies')">
        <span class="icon">&#9881;</span> Directives
      </div>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-section-label">Systems</div>
      <div class="sidebar-item" onclick="setView('updates')">
        <span class="icon">&#8635;</span> OTA Deploy
      </div>
      <div class="sidebar-item" onclick="setView('models')">
        <span class="icon">&#9830;</span> Models
      </div>
      <div class="sidebar-item" onclick="setView('settings')">
        <span class="icon">&#9881;</span> Config
      </div>
    </div>

    <div class="sidebar-footer">
      <div class="gpu-mini" id="gpu-mini"><span>RTX</span> ...</div>
      <div class="gpu-bar"><div class="gpu-bar-fill" id="gpu-bar-fill"></div></div>
      <div class="version-info">FARSIGHT v2.0 // TACTICAL</div>
    </div>
  </nav>

  <div class="main">
    <div class="main-content">
      <div class="top-bar">
        <div class="tb-left">
          <div class="tb-logo"><img src="/logo.png" alt="AURA"></div>
          <div>
            <div class="tb-title">FARSIGHT COMMAND</div>
            <div class="tb-subtitle">AURA NETWORK COORDINATION</div>
          </div>
        </div>
        <div class="tb-right">
          <div class="tb-stat">
            <div class="tb-stat-value" id="stat-total">0</div>
            <div class="tb-stat-label">UNITS</div>
          </div>
          <div class="tb-stat">
            <div class="tb-stat-value" id="stat-online" style="color:var(--green)">0</div>
            <div class="tb-stat-label">ONLINE</div>
          </div>
          <div class="tb-stat">
            <div class="tb-stat-value" id="stat-offline" style="color:var(--red)">0</div>
            <div class="tb-stat-label">OFFLINE</div>
          </div>
          <div style="border-left:1px solid var(--border);height:36px;margin:0 4px;"></div>
          <div>
            <div class="tb-clock" id="tb-clock">00:00:00</div>
            <div class="tb-clock-label">UTC ZULU</div>
          </div>
        </div>
      </div>

      <div class="broadcast">
        <input type="text" id="bc-input" placeholder="// BROADCAST TO ALL UNITS..." maxlength="200">
        <button id="bc-btn" onclick="sendBroadcast()">TRANSMIT</button>
      </div>

      <div class="card-backdrop" id="card-backdrop"></div>
      <div class="puck-cards" id="cards"></div>
    </div>
  </div>
</div>

<script>
var pucks = [];
var gpuData = null;
var currentView = "constellation";
var ledgerAccum = {};
var ledgerStart = Date.now();

/* ── UTC Clock ────────────────────────────────── */
function tickClock(){
  var d = new Date();
  var h = String(d.getUTCHours()).padStart(2,"0");
  var m = String(d.getUTCMinutes()).padStart(2,"0");
  var s = String(d.getUTCSeconds()).padStart(2,"0");
  var el = document.getElementById("tb-clock");
  if(el) el.textContent = h+":"+m+":"+s;
}
tickClock();
setInterval(tickClock, 1000);

function setView(v) {
  currentView = v;
  document.querySelectorAll(".sidebar-item").forEach(function(el){ el.classList.remove("active"); });
  event.currentTarget.classList.add("active");
}

function fmtUp(s) {
  if (s == null) return "\u2014";
  var d = Math.floor(s/86400), h = Math.floor((s%86400)/3600), m = Math.floor((s%3600)/60);
  return d > 0 ? d+"d "+h+"h" : h > 0 ? h+"h "+m+"m" : m+"m";
}
function fmtMem(mb) {
  if (mb == null) return "\u2014";
  return mb >= 1024 ? (mb/1024).toFixed(1)+" GB" : Math.round(mb)+" MB";
}
function sc(s) { return s==="offline"?"offline":s==="idle"?"idle":"online"; }
function sl(s) { return s.toUpperCase(); }

function sendBroadcast() {
  var inp = document.getElementById("bc-input"), btn = document.getElementById("bc-btn");
  var msg = inp.value.trim();
  if (!msg || btn.classList.contains("transmitting")) return;
  btn.classList.add("transmitting");
  btn.textContent = "TRANSMITTING...";
  inp.disabled = true;
  fetch("/broadcast",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({message:msg})}).then(function(r){return r.json();}).then(function(){
    inp.value="";
    var words = msg.split(/\s+/).length;
    var waitMs = Math.max(8000, words * 150 + 8000);
    setTimeout(function(){
      btn.classList.remove("transmitting");
      btn.textContent = "TRANSMIT";
      inp.disabled = false;
    }, waitMs);
  }).catch(function(){
    btn.classList.remove("transmitting");
    btn.textContent = "TRANSMIT";
    inp.disabled = false;
  });
}
document.getElementById("bc-input").addEventListener("keydown",function(e){if(e.key==="Enter")sendBroadcast();});

function refresh() {
  fetch("/pucks").then(function(r){return r.json();}).then(function(data){
    pucks = data;
    pucks.sort(function(a,b){
      var ao = a.effective_status==="offline"?1:0, bo = b.effective_status==="offline"?1:0;
      if(ao!==bo) return ao-bo;
      return (a.puck_name||"").localeCompare(b.puck_name||"");
    });
    renderCards();
    var on = pucks.filter(function(p){return p.effective_status!=="offline";}).length;
    var off = pucks.length - on;
    document.getElementById("stat-total").textContent = pucks.length;
    document.getElementById("stat-online").textContent = on;
    document.getElementById("stat-offline").textContent = off;
    document.getElementById("sb-online").textContent = on;
  }).catch(function(){});

  fetch("/gpu").then(function(r){return r.json();}).then(function(d){
    if(d && !d.error){gpuData=d; renderGpu();}
  }).catch(function(){});
}

function renderGpu() {
  if(!gpuData) return;
  var p = gpuData.gpu_util;
  var el = document.getElementById("gpu-mini");
  el.innerHTML = "<span>"+esc(gpuData.gpu_name||"RTX")+"</span>";
  el.innerHTML += " "+p+"% &middot; "+gpuData.temp_c+"&deg;C";
  document.getElementById("gpu-bar-fill").style.width = p+"%";
  var bar = document.getElementById("gpu-bar-fill");
  if(p>90){ bar.style.background="var(--red)"; bar.style.boxShadow="0 0 8px var(--red)"; }
  else if(p>70){ bar.style.background="var(--amber)"; bar.style.boxShadow="0 0 8px var(--amber)"; }
  else { bar.style.background="var(--green)"; bar.style.boxShadow="0 0 8px var(--green)"; }
}

function renderCards() {
  var el = document.getElementById("cards");
  if(!pucks.length){el.innerHTML='<div class="empty-msg">// AWAITING UNIT REGISTRATION...</div>';return;}
  initLiveState();
  var h = "";
  for(var i=0;i<pucks.length;i++){
    var p = pucks[i], s = sc(p.effective_status), col = p.color||"#00ff88";
    var pid = p.puck_id||"";
    if(!ledgerAccum[pid]) ledgerAccum[pid] = (Math.abs(hashCode(pid)) % 2000 + 500) / 100;
    if(p.effective_status!=="offline") ledgerAccum[pid] += 0.01 + Math.random()*0.04;
    var ledger = ledgerAccum[pid];
    var occ = p.occupants||[];
    // Use live state instead of server snapshot
    var trans = liveTranscripts[pid] || p.transcript || [];
    var analysis = liveViewpoints[pid] || p.analysis || "Monitoring...";
    var cid = "card-"+i;
    h += '<div class="puck-card '+s+'" id="'+cid+'">'
      +'<div class="pc-inner">'
      +'<div class="pc-header">'
      +'  <div class="pc-color-dot" style="color:'+col+';background:'+col+'"></div>'
      +'  <div class="pc-name">'+esc(p.puck_name||"UNNAMED")+'</div>'
      +'  <div class="pc-status '+s+'">'+sl(p.effective_status)+'</div>'
      +'</div>';
    // Occupants with speaking indicators — randomize on each render
    if(occ.length){
      h+='<div class="pc-occupants">';
      for(var j=0;j<occ.length;j++){
        var o=occ[j], spk = Math.random() > 0.45;
        h+='<span class="pc-occupant'+(spk?" is-speaking":"")+'">'
          +'<span class="dot '+(spk?"speaking":"silent")+'"></span>'
          +esc(o.name)+'</span>';
      }
      h+='</div>';
    }
    // Live transcript
    h+='<div class="pc-transcript" id="ts-'+i+'">';
    for(var j=0;j<trans.length;j++){
      var t=trans[j];
      h+='<div class="ts-line"><span class="ts-time">'+esc(t.time)+'</span>'
        +'<span class="ts-speaker">'+esc(t.speaker)+':</span> '
        +esc(t.text)+'</div>';
    }
    h+='</div>';
    // Aura's Viewpoint
    h+='<div class="pc-analysis">'
      +'<div class="pc-analysis-label"><span class="aura-eye"></span>Aura\'s Viewpoint</div>'
      +'<div class="pc-analysis-text" id="av-'+i+'">'+esc(analysis)+'</div>'
      +'</div>';
    // Ledger
    h+='<div class="pc-ledger"><span class="pc-ledger-label">$LEDGER</span><span class="pc-ledger-value">$'+ledger.toFixed(2)+'</span></div>';
    h+='</div>';  // pc-inner
    h+='<div class="pc-radar"></div>';
    h+='</div>';
  }
  el.innerHTML = h;
  // Auto-scroll transcript divs to bottom
  for(var i=0;i<pucks.length;i++){
    var tsd = document.getElementById("ts-"+i);
    if(tsd) tsd.scrollTop = tsd.scrollHeight;
  }
  // Re-apply expanded/dimmed state after rebuild
  if (expandedIdx >= 0) {
    var cards = document.querySelectorAll(".puck-card");
    cards.forEach(function(c, i) {
      if (i === expandedIdx) { c.classList.add("expanded"); }
      else { c.classList.add("dimmed"); }
    });
  }
}

// ── Card expand / collapse ──────────────────────
var expandedIdx = -1;

function expandCard(idx) {
  if (expandedIdx === idx) return;
  expandedIdx = idx;
  var cards = document.querySelectorAll(".puck-card");
  var backdrop = document.getElementById("card-backdrop");
  cards.forEach(function(c, i) {
    if (i === idx) {
      c.classList.remove("dimmed");
      c.classList.add("expanded");
    } else {
      c.classList.remove("expanded");
      c.classList.add("dimmed");
    }
  });
  backdrop.style.display = "block";
  requestAnimationFrame(function(){ backdrop.classList.add("visible"); });
}

function collapseCards() {
  if (expandedIdx < 0) return;
  expandedIdx = -1;
  var cards = document.querySelectorAll(".puck-card");
  var backdrop = document.getElementById("card-backdrop");
  cards.forEach(function(c) {
    c.classList.remove("expanded","dimmed");
  });
  backdrop.classList.remove("visible");
  setTimeout(function(){ backdrop.style.display = "none"; }, 400);
}

// Backdrop click → collapse
document.getElementById("card-backdrop").addEventListener("click", function(e) {
  e.stopPropagation();
  collapseCards();
});

// Delegate clicks on cards
document.getElementById("cards").addEventListener("click", function(e) {
  var card = e.target.closest(".puck-card");
  if (!card) return;
  if (card.classList.contains("expanded")) return; // don't re-expand
  var idx = Array.prototype.indexOf.call(document.querySelectorAll(".puck-card"), card);
  if (idx >= 0) expandCard(idx);
});

// ESC key to collapse
document.addEventListener("keydown", function(e) {
  if (e.key === "Escape") collapseCards();
});

function esc(s){var d=document.createElement("div");d.textContent=s;return d.innerHTML;}
function hashCode(s){var h=0;for(var i=0;i<s.length;i++){h=((h<<5)-h)+s.charCodeAt(i);h|=0;}return h;}

// ── Live transcript & viewpoint generation ──────────────
var liveTranscripts = {};  // puck_id -> [{time, speaker, text}, ...]
var liveViewpoints = {};   // puck_id -> string
var _tsInited = false;

var _convPools = {
  "Exec Boardroom": {
    speakers: ["Paul","David","Dr. Rafael"],
    lines: [
      ["Paul","We need to finalize the partnership terms before Friday."],
      ["David","The legal team flagged two clauses that need revision."],
      ["Paul","Which clauses? Send me the redlines after this."],
      ["Dr. Rafael","The IP assignment clause is the blocker. I reviewed it with outside counsel."],
      ["David","Revenue share is at 60-40 but they're pushing for 55-45."],
      ["Paul","Hold at 60-40. We have the leverage here."],
      ["Dr. Rafael","Agreed. Our tech stack is the differentiator."],
      ["David","I'll push back on their legal team this afternoon."],
      ["Paul","What about the deployment timeline? Are we still on track?"],
      ["Dr. Rafael","Hardware arrives next week. Software is ready."],
      ["Paul","Good. Let's not slip on this."],
      ["David","There's a facilities request for the new server room buildout."],
      ["Paul","How much?"],
      ["David","About 340K including cooling and redundancy."],
      ["Dr. Rafael","That's reasonable for what we're getting."],
      ["Paul","Approve it. Move fast before the price goes up."],
      ["David","I'll sign the PO today."],
      ["Paul","What's the status on the Series B conversations?"],
      ["David","Two firms are in. Term sheet expected next week."],
      ["Dr. Rafael","Valuation looks strong given the Q2 numbers."],
    ],
    viewpoints: [
      "Paul is running this meeting at pace. David's legal hesitation is valid but he needs to stop hedging and commit to a timeline. Dr. Rafael just dropped the key insight about IP assignment \u2014 that's the real issue, not revenue share. Paul should listen to him more.",
      "The 60-40 hold is the right call. David will fold under pressure from their legal team unless Paul backs him explicitly. I'd get that in writing. The hardware timeline is tight \u2014 'next week' means 'maybe next week' in my experience.",
      "Series B conversations are heating up. Two firms is good but three is leverage. Paul knows this. The 340K server room approval was instant \u2014 he's spending with conviction. Good sign for the board.",
      "Dr. Rafael finally engaged and he's the smartest person in the room. His silence earlier was concerning. The IP clause insight just saved them weeks of back-and-forth. David needs to lead with that in the afternoon call.",
      "This meeting is productive but running long. There's a visitor waiting in the lobby for Paul. Someone needs to flag that \u2014 keeping partners waiting is exactly the kind of thing that costs deals.",
      "David just said 'I'll sign the PO today' \u2014 he's said that exact phrase three times this month and signed zero POs. Paul trusts him but I don't. Track the actual signature, not the promise. Dr. Rafael noticed too. Watch his face.",
      "Everyone in this room is smart enough to know the Series B valuation is inflated by 20%, but nobody's saying it out loud. That's a problem. Honest conversations now save painful board meetings later. Paul knows this. He's choosing optimism over accuracy.",
    ]
  },
  "Engineering Lab": {
    speakers: ["Mason","Lucas"],
    lines: [
      ["Mason","I refactored the attention layer. Memory dropped 40%."],
      ["Lucas","What's the throughput impact?"],
      ["Mason","Actually improved. 2200 tokens per second now."],
      ["Lucas","That's insane. Show me the profiler output."],
      ["Mason","Pull it up. Kernel fusion did most of the heavy lifting."],
      ["Lucas","We should upstream this to the main branch."],
      ["Mason","Already opened the PR. Needs your review."],
      ["Lucas","I'll look at it after lunch. Any regressions in the test suite?"],
      ["Mason","All green. 847 tests passing."],
      ["Lucas","What about the edge case with long context windows?"],
      ["Mason","Fixed. The sliding window approach handles it now."],
      ["Lucas","Nice. I've been working on the streaming decoder."],
      ["Mason","How's latency looking?"],
      ["Lucas","First token at 45ms. Way under our 100ms target."],
      ["Mason","Paul's going to love that number."],
      ["Lucas","Should we present at the all-hands?"],
      ["Mason","Definitely. Let's put together the benchmarks."],
      ["Lucas","I'll grab the A/B test results too."],
    ],
    viewpoints: [
      "Mason just casually dropped a 40% memory reduction like it's nothing. This is the kind of work that should be in the boardroom conversation upstairs, not buried in a PR review. Lucas is a good foil \u2014 he asks the right questions.",
      "2200 tokens/sec with 40% less memory. These two are operating at a level the rest of the org doesn't fully appreciate. The 45ms first-token latency is world-class. They should present this to Paul directly, not wait for an all-hands.",
      "All 847 tests green is reassuring but I'd want to see the long-context edge case tested under load, not just unit tested. Mason says 'fixed' but sliding window approaches have failure modes at the boundaries. Trust but verify.",
      "The streaming decoder work is the real product differentiator. If Lucas can maintain 45ms first-token consistently at scale, that's a competitive moat. Patent this before publishing any benchmarks.",
      "Brutal truth: Mason is carrying this team. Lucas asks good questions but hasn't shipped original work in two weeks. He's becoming a reviewer, not a builder. Someone needs to give him a hard problem of his own before he gets comfortable.",
      "They keep benchmarking instead of shipping. This is classic engineer procrastination disguised as rigor. The numbers were good three days ago. Push to production, measure in the wild, and stop polishing in the lab.",
    ]
  },
  "Kitchen & Lounge": {
    speakers: ["Jorge","Bob","Sarah","Kim"],
    lines: [
      ["Jorge","Did you see the email about the rooftop social tonight?"],
      ["Bob","Yeah, 5pm. Weather looks perfect."],
      ["Sarah","I'm bringing that dip from last time. People kept asking about it."],
      ["Kim","The one with the roasted garlic? Yes please."],
      ["Jorge","Who's handling the drinks order?"],
      ["Bob","Facilities said they'd set up a bar station."],
      ["Sarah","We should get some non-alcoholic options too."],
      ["Kim","Good call. There's a great NA beer brand."],
      ["Jorge","Did anyone try the new lunch spot on 5th?"],
      ["Bob","The ramen place? Incredible."],
      ["Sarah","I went yesterday. The tonkotsu is amazing."],
      ["Kim","We should do a team lunch there."],
      ["Jorge","Speaking of food, someone keeps stealing my yogurt."],
      ["Bob","That's a war crime. Label it better."],
      ["Sarah","I'll send a passive-aggressive email to all-staff."],
      ["Kim","Just put a fake label on it. 'Contains medicine.'"],
      ["Jorge","Genius. Kim, you're promoted."],
    ],
    viewpoints: [
      "Four people in the kitchen again. Jorge is the social glue of this office \u2014 every casual conversation traces back to him. This isn't wasted time; this is how culture gets built. The yogurt theft is genuinely annoying though. Somebody check the fridge cam.",
      "The rooftop social planning is happening organically and that's better than any HR-organized event. Sarah's dip is apparently legendary. If the weather holds, tonight's going to be great for morale. Let them have their 15 minutes.",
      "Kitchen conversations reveal more about team health than any engagement survey. Everyone's relaxed, joking about yogurt theft \u2014 that's a team that trusts each other. Kim's 'contains medicine' suggestion is actually brilliant social engineering.",
      "Bob keeps gravitating to the kitchen. Third time this week. Either he's avoiding his desk or he genuinely needs the social recharge. Given his output has been strong, I'll assume the latter. Not flagging this.",
      "Jorge has spent 47 minutes in the kitchen today. At some point 'culture building' becomes 'not working.' His Q3 deliverables are behind. I like the guy but the numbers don't lie. Someone should casually mention his sprint velocity in standup tomorrow.",
    ]
  },
  "Medical Suite": {
    speakers: ["Dr. Chen","Nurse Adams"],
    lines: [
      ["Dr. Chen","Next patient is the 3pm follow-up. Knee replacement recovery."],
      ["Nurse Adams","Chart's pulled. Physical therapy notes look good."],
      ["Dr. Chen","ROM improved to 110 degrees. That's ahead of schedule."],
      ["Nurse Adams","Patient's been doing the home exercises consistently."],
      ["Dr. Chen","That's what makes the difference. Compliance is everything."],
      ["Nurse Adams","Lab results for the Walker case just came in."],
      ["Dr. Chen","And?"],
      ["Nurse Adams","All within normal range. A1C is down to 6.1."],
      ["Dr. Chen","Excellent. That's a significant improvement from 7.8."],
      ["Nurse Adams","Should I schedule the medication review?"],
      ["Dr. Chen","Yes. We might be able to reduce the dosage."],
      ["Nurse Adams","The pharmacy integration is working well by the way."],
      ["Dr. Chen","It's saving us about 20 minutes per patient."],
      ["Nurse Adams","The patients notice too. Less waiting."],
    ],
    viewpoints: [
      "This room is a well-oiled machine. Dr. Chen runs through patients with surgical precision \u2014 pun intended. The A1C drop from 7.8 to 6.1 is a genuine clinical win. Nurse Adams is the unsung hero here; she pre-stages everything.",
      "The pharmacy integration saving 20 minutes per patient is huge at scale. If they see 30 patients a day, that's 10 hours recovered per week. Dr. Chen should present this efficiency gain to the board.",
      "ROM at 110 degrees post-knee replacement is excellent progress. The patient compliance note is important \u2014 Dr. Chen's bedside manner clearly motivates follow-through. This is evidence-based care done right.",
      "Smooth operations. No bottlenecks, no delays. If every department ran like this suite, the whole building would be a different place. Nurse Adams deserves a raise and I'm not even being hyperbolic.",
      "Dr. Chen is the fastest clinician I've observed but speed has a shadow side. He spent 3 minutes and 40 seconds on a patient discharge. That's efficient or it's rushed \u2014 depends on whether the patient felt heard. The metrics say yes. My instinct says barely.",
    ]
  },
  "Server Room": {
    speakers: ["System","Alert"],
    lines: [
      ["System","Rack 3 temperature: 18.4\u00b0C. Within tolerance."],
      ["System","UPS battery test completed. All cells nominal."],
      ["Alert","Humidity sensor B2-7 reading 43%. Above 40% threshold."],
      ["System","Network switch port 24 link speed renegotiated to 10Gbps."],
      ["System","Scheduled backup completed. 2.4TB replicated to offsite."],
      ["Alert","Rack 5 fan module RPM fluctuation detected. Monitoring."],
      ["System","Power consumption: 14.2 kW. Below 18 kW budget."],
      ["System","Cooling unit A cycling normally. Compressor runtime: 340h."],
      ["Alert","DNS query latency spike: 45ms. Resolved. ISP upstream issue."],
      ["System","Storage array health: all disks green. 847TB available."],
    ],
    viewpoints: [
      "That humidity reading is creeping up again. 43% and the threshold is 40%. Last time this happened it was a failed seal on the CRAC unit. Facilities 'fixed' it but I'm skeptical. If it hits 45% I'm sending an automated alert to maintenance.",
      "Rack 5 fan module has been 'fluctuating' for a week now. This is how cascading failures start. Replace the module proactively \u2014 a $200 fan is cheaper than a $50K server going thermal. Flagging this as priority.",
      "The 2.4TB offsite backup completed clean. Good. But nobody's tested a restore in 3 months. Backups without restore tests are just expensive hope. Adding this to the ops review agenda.",
      "Power at 14.2 kW against an 18 kW budget gives us headroom for the new racks. The cooling is keeping up. This room is healthy but needs preventive attention before the summer heat load arrives.",
    ]
  },
  "Reception Lobby": {
    speakers: ["Front Desk","Visitor","Courier"],
    lines: [
      ["Front Desk","Good morning, welcome to Aura. How can I help you?"],
      ["Visitor","I'm here for the 11:30 with the engineering team."],
      ["Front Desk","Of course. I'll let them know you're here. Can I get you a coffee?"],
      ["Visitor","That would be great, thank you."],
      ["Front Desk","The Wi-Fi password is on the card in the lounge area."],
      ["Courier","Package delivery for David Chen. Signature required."],
      ["Front Desk","I can sign for that. Thank you."],
      ["Courier","Have a good day."],
      ["Front Desk","You too. David, you have a package at reception."],
      ["Visitor","Beautiful office by the way. Love the design."],
      ["Front Desk","Thank you! The renovation was just completed last month."],
    ],
    viewpoints: [
      "Visitor arrived for the 11:30 engineering meeting. Front desk is handling it well \u2014 offered coffee, Wi-Fi, the whole experience. First impressions matter and this lobby is earning its keep. The visitor complimented the space, which is always a good sign for client conversion.",
      "David's package arrived and Front Desk signed for it. That's three packages this week for David. Either it's equipment for the lab or he's running a side hustle from the office. Joking. Mostly.",
      "The lobby is the first thing visitors see and right now it's performing perfectly. No wait times, warm reception, ambient music. This is the kind of invisible excellence that closes deals. Keep it up.",
      "Two visitors in the last hour. Traffic is picking up which tracks with the partnership discussions happening upstairs. If this pace continues, we might need a second front desk person during peak hours.",
    ]
  },
  "Rooftop Terrace": {
    speakers: ["Wind Sensor","Ambient"],
    lines: [
      ["Ambient","Temperature: 22.4\u00b0C. Humidity: 35%. UV index: 6."],
      ["Wind Sensor","Wind speed: 8 km/h NNW. Gusts to 12 km/h."],
      ["Ambient","Noise level: 42 dB. Well within comfort range."],
      ["Wind Sensor","Barometric pressure: 1013 hPa. Stable."],
      ["Ambient","Solar panel output: 4.2 kW. Above daily average."],
      ["Wind Sensor","Wind direction shifting to NW. Speed holding steady."],
      ["Ambient","Air quality index: 28. Excellent."],
    ],
    viewpoints: [
      "Perfect conditions up here. 22\u00b0C, clear skies, gentle breeze. The 5pm social tonight is going to be excellent. Solar panels are overperforming which is nice. This rooftop is genuinely the best asset in the building and it's criminally underused during work hours.",
      "UV index at 6 means sunscreen is advisable for the social tonight. Someone should mention that in the event announcement. The air quality is excellent \u2014 better than most offices' indoor air. People should be working up here.",
      "Wind is calm, pressure is stable, no weather changes incoming. Tonight's event has zero weather risk. The solar output at 4.2 kW means the terrace lighting can run entirely off the panels. Sustainable entertaining.",
      "Empty again during business hours. The ROI on this terrace renovation needs people actually using it. I'm going to recommend a 'rooftop Fridays' policy. Fresh air and sunlight improve cognitive performance by 15-20% according to the research.",
    ]
  }
};

// Initialize live transcript state from server data
function initLiveState() {
  if (_tsInited || !pucks.length) return;
  _tsInited = true;
  for (var i=0; i<pucks.length; i++) {
    var p = pucks[i], pid = p.puck_id;
    if (!liveTranscripts[pid]) {
      liveTranscripts[pid] = (p.transcript || []).slice();
    }
    if (!liveViewpoints[pid]) {
      liveViewpoints[pid] = p.analysis || "Monitoring...";
    }
  }
}

// Generate a time string offset from now
function fakeTime(offsetMin) {
  var d = new Date(Date.now() + offsetMin*60000);
  return ("0"+d.getHours()).slice(-2)+":"+("0"+d.getMinutes()).slice(-2);
}

var _lineCounters = {};
var _vpCounters = {};

function tickTranscripts() {
  if (!pucks.length) return;
  // Add a new transcript line to 1-3 random rooms
  var active = pucks.filter(function(p){ return p.effective_status !== "offline"; });
  if (!active.length) return;
  var count = 1 + Math.floor(Math.random() * 2);
  for (var c=0; c<count; c++) {
    var p = active[Math.floor(Math.random() * active.length)];
    var pid = p.puck_id;
    var pool = _convPools[p.puck_name];
    if (!pool) continue;
    if (!_lineCounters[pid]) _lineCounters[pid] = Math.floor(Math.random() * pool.lines.length);
    var idx = _lineCounters[pid] % pool.lines.length;
    _lineCounters[pid]++;
    var line = pool.lines[idx];
    var entry = {time: fakeTime(0), speaker: line[0], text: line[1]};
    if (!liveTranscripts[pid]) liveTranscripts[pid] = [];
    liveTranscripts[pid].push(entry);
    // Keep max 20 lines
    if (liveTranscripts[pid].length > 20) liveTranscripts[pid] = liveTranscripts[pid].slice(-20);
    // Update DOM directly
    var el = document.getElementById("ts-" + pucks.indexOf(p));
    if (el) {
      var div = document.createElement("div");
      div.className = "ts-line ts-new";
      div.innerHTML = '<span class="ts-time">'+esc(entry.time)+'</span>'
        +'<span class="ts-speaker">'+esc(entry.speaker)+':</span> '+esc(entry.text);
      el.appendChild(div);
      el.scrollTop = el.scrollHeight;
    }
  }
}

function tickViewpoints() {
  if (!pucks.length) return;
  // Update 1-2 random room viewpoints
  var active = pucks.filter(function(p){ return p.effective_status !== "offline"; });
  if (!active.length) return;
  var count = 1 + Math.floor(Math.random() * 1);
  for (var c=0; c<count; c++) {
    var p = active[Math.floor(Math.random() * active.length)];
    var pid = p.puck_id;
    var pool = _convPools[p.puck_name];
    if (!pool || !pool.viewpoints || !pool.viewpoints.length) continue;
    if (!_vpCounters[pid]) _vpCounters[pid] = Math.floor(Math.random() * pool.viewpoints.length);
    var idx = _vpCounters[pid] % pool.viewpoints.length;
    _vpCounters[pid]++;
    liveViewpoints[pid] = pool.viewpoints[idx];
    // Update DOM directly
    var el = document.getElementById("av-" + pucks.indexOf(p));
    if (el) {
      el.style.opacity = "0";
      setTimeout((function(e, txt) {
        return function() { e.textContent = txt; e.style.opacity = "1"; };
      })(el, pool.viewpoints[idx]), 300);
    }
  }
}

// Run transcript ticks every 4-7 seconds
function scheduleTranscriptTick() {
  var delay = 4000 + Math.random() * 3000;
  setTimeout(function(){ tickTranscripts(); scheduleTranscriptTick(); }, delay);
}
// Run viewpoint ticks every 12-18 seconds
function scheduleViewpointTick() {
  var delay = 12000 + Math.random() * 6000;
  setTimeout(function(){ tickViewpoints(); scheduleViewpointTick(); }, delay);
}
scheduleTranscriptTick();
scheduleViewpointTick();

// SSE with auto-reconnect
function connectSSE(){
  var es = new EventSource("/stream");
  es.addEventListener("puck_registered",function(){refresh();});
  es.addEventListener("status_change",function(){refresh();});
  es.addEventListener("reconnect",function(){es.close(); setTimeout(connectSSE, 500);});
  es.addEventListener("broadcast",function(e){
    var inp=document.getElementById("bc-input");
    inp.style.borderColor="rgba(255,170,0,0.6)";
    inp.style.boxShadow="0 0 20px rgba(255,170,0,0.1)";
    setTimeout(function(){inp.style.borderColor="";inp.style.boxShadow="";},2000);
    var cards=document.querySelectorAll(".puck-card");
    cards.forEach(function(c){
      c.classList.remove("broadcasting");
      void c.offsetWidth;
      c.classList.add("broadcasting");
    });
    setTimeout(function(){
      cards.forEach(function(c){c.classList.remove("broadcasting");});
    },4500);
  });
  es.onerror = function(){ es.close(); setTimeout(connectSSE, 2000); };
}
connectSSE();

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


@app.route("/logo.png", methods=["GET"])
def logo():
    """Serve the AURA logo."""
    logo_paths = [
        "/tmp/AuraLogo.png",
        os.path.expanduser("~/LedgerAI/aura-control/voices/mp3/AuraLogo.png"),
    ]
    for p in logo_paths:
        if os.path.isfile(p):
            return send_file(p, mimetype="image/png")
    return "", 404


@app.route("/", methods=["GET"])
def dashboard():
    return _DASHBOARD_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _seed_demo_pucks():
    """Inject demo pucks so the dashboard always has something to show."""
    demos = [
        {"puck_id": "aura-16113256-035751-314159", "puck_name": "Exec Boardroom",
         "owner_name": "Floor 12", "color": "#981225", "ip": "192.168.1.108",
         "status": "listening", "uptime": 86400*2+3600*7, "memory_usage": 3840,
         "occupants": [
             {"name": "Paul", "speaking": True},
             {"name": "David", "speaking": False},
             {"name": "Dr. Rafael", "speaking": False},
         ],
         "transcript": [
             {"time": "10:42", "speaker": "Paul", "text": "The Q3 projections need to account for the new partnership revenue."},
             {"time": "10:42", "speaker": "David", "text": "Agreed. I'll have the revised model by Thursday."},
             {"time": "10:43", "speaker": "Paul", "text": "Let's also factor in the hardware costs for the new deployments."},
         ],
         "analysis": "Paul is driving this meeting well but David keeps hedging on the timeline. The revised model should have been done last week. The hardware CapEx discussion is overdue \u2014 if they don\u2019t lock numbers today, Q3 will slip. Dr. Rafael hasn\u2019t spoken in 8 minutes. He\u2019s either deep in thought or checked out. Somebody should pull him in."},
        {"puck_id": "aura-d4v1d002-035751-314159", "puck_name": "Engineering Lab",
         "owner_name": "Floor 3", "color": "#23A5FF", "ip": "192.168.1.55",
         "status": "listening", "uptime": 86400+1800, "memory_usage": 2910,
         "occupants": [
             {"name": "Mason", "speaking": True},
             {"name": "Lucas", "speaking": True},
         ],
         "transcript": [
             {"time": "10:40", "speaker": "Mason", "text": "The latency on the new inference pipeline dropped to 180ms."},
             {"time": "10:41", "speaker": "Lucas", "text": "That's with quantization? What about accuracy?"},
             {"time": "10:41", "speaker": "Mason", "text": "Negligible loss. Point-two percent on the benchmark."},
         ],
         "analysis": "These two are genuinely brilliant but they\u2019re going to rabbit-hole on benchmarks for another hour if nobody stops them. The 180ms result is production-ready \u2014 ship it. \u201cNegligible loss\u201d means they\u2019re overthinking it. Mason should present this to Paul\u2019s boardroom meeting upstairs before David finalizes the model without these numbers."},
        {"puck_id": "aura-b0b00003-a4c820-314159", "puck_name": "Kitchen & Lounge",
         "owner_name": "Floor 1", "color": "#8B5CF6", "ip": "192.168.1.78",
         "status": "listening", "uptime": 3600*14, "memory_usage": 4200,
         "occupants": [
             {"name": "Jorge", "speaking": True},
             {"name": "Bob", "speaking": False},
             {"name": "Sarah", "speaking": False},
             {"name": "Kim", "speaking": False},
         ],
         "transcript": [
             {"time": "10:38", "speaker": "Jorge", "text": "Has anyone tried the new espresso machine?"},
             {"time": "10:39", "speaker": "Bob", "text": "It's incredible. Way better than the old one."},
             {"time": "10:39", "speaker": "Jorge", "text": "I heard facilities is planning to add one on every floor."},
         ],
         "analysis": "Four people in the kitchen at 10:40 on a Tuesday. Jorge is the instigator \u2014 he\u2019s been here 12 minutes. Bob followed him down. This is a morale thing, not a productivity problem. Let them have their coffee. The espresso machine investment is paying off in retention whether finance tracks it or not."},
        {"puck_id": "aura-j0rg3004-f17e22-314159", "puck_name": "Medical Suite",
         "owner_name": "Floor 2", "color": "#F59E0B", "ip": "192.168.1.103",
         "status": "listening", "uptime": 86400*5, "memory_usage": 3100,
         "occupants": [
             {"name": "Dr. Chen", "speaking": True},
             {"name": "Nurse Adams", "speaking": False},
         ],
         "transcript": [
             {"time": "10:35", "speaker": "Dr. Chen", "text": "Patient vitals are stable. BP is 120 over 78."},
             {"time": "10:36", "speaker": "Nurse Adams", "text": "Labs came back normal. Discharge paperwork is ready."},
             {"time": "10:37", "speaker": "Dr. Chen", "text": "Good. Schedule the follow-up for two weeks out."},
         ],
         "analysis": "Clean discharge. Dr. Chen is efficient \u2014 this whole consult took under 4 minutes. Nurse Adams had the paperwork pre-staged, which is exactly how this should work. The two-week follow-up window is conservative but appropriate given the labs. No concerns here. This room runs like clockwork."},
        {"puck_id": "aura-mas0n005-38d1a5-314159", "puck_name": "Server Room",
         "owner_name": "Basement B2", "color": "#10B981", "ip": "192.168.1.91",
         "status": "idle", "uptime": 7200, "memory_usage": 5020,
         "occupants": [],
         "transcript": [],
         "analysis": "Nobody\u2019s been down here since yesterday. Temperature is holding at 18.2\u00b0C which is fine, but humidity crept up to 42% \u2014 should be under 40. If facilities doesn\u2019t check the HVAC by end of day I\u2019m going to flag it. Rack 3 had a fan warning last week that was \u201cresolved\u201d but I\u2019m not convinced."},
        {"puck_id": "aura-1ucas006-cc71b9-314159", "puck_name": "Reception Lobby",
         "owner_name": "Ground Floor", "color": "#EC4899", "ip": "192.168.1.67",
         "status": "listening", "uptime": 86400*3+10800, "memory_usage": 2680,
         "occupants": [
             {"name": "Front Desk", "speaking": False},
         ],
         "transcript": [
             {"time": "10:30", "speaker": "Visitor", "text": "I have a 10:45 with Paul Chou."},
             {"time": "10:30", "speaker": "Front Desk", "text": "Of course. Let me notify his office. Please have a seat."},
         ],
         "analysis": "Visitor just arrived for Paul\u2019s 10:45. Front desk handled it perfectly. But Paul is still mid-sentence in the boardroom and that meeting is running long. Someone should text him. Making a guest wait more than 5 minutes in the lobby is a bad look, especially if this is the partnership contact David keeps mentioning."},
        {"puck_id": "aura-raf3l007-dd82c0-314159", "puck_name": "Rooftop Terrace",
         "owner_name": "Floor 14", "color": "#4ecdc4", "ip": "192.168.1.94",
         "status": "idle", "uptime": 86400*1+7200, "memory_usage": 3400,
         "occupants": [],
         "transcript": [],
         "analysis": "Empty and gorgeous up here. 22\u00b0C, not a cloud in sight. The team social at 5pm is going to be perfect weather. Honestly, half the engineering floor should be working up here instead of under fluorescents. Whoever booked this space only for after-hours is wasting the best real estate in the building."},
    ]
    with _registry_lock:
        for d in demos:
            _puck_registry[d["puck_id"]] = {
                **d,
                "capabilities": {},
                "version": {"branch": "main", "commit": "88be7ae", "dirty": False},
                "last_conversation_ts": None,
                "registered_at": _now_iso(),
                "last_seen_ts": time.time(),
            }


def _demo_heartbeat_loop():
    """Keep demo pucks alive by refreshing their last_seen_ts."""
    while True:
        time.sleep(30)
        with _registry_lock:
            for puck in _puck_registry.values():
                if puck["puck_id"].startswith("aura-"):
                    puck["last_seen_ts"] = time.time()


if __name__ == "__main__":
    _seed_demo_pucks()
    threading.Thread(target=_demo_heartbeat_loop, daemon=True).start()
    print(f"Farsight Hub starting on port {HUB_PORT}")
    print(f"LLM offload target: {FARSIGHT_LLM_URL}")
    print(f"Dashboard: http://0.0.0.0:{HUB_PORT}/")
    app.run(host="0.0.0.0", port=HUB_PORT, threaded=True)

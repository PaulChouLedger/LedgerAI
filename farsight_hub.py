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
    border-right: 1px solid var(--border);
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
    font-size: 13px;
    font-weight: 600;
    color: var(--cyan);
    text-shadow: 0 0 8px rgba(0,221,255,0.3);
    letter-spacing: 0.05em;
  }
  .top-bar .tb-clock-label {
    font-size: 8px;
    color: var(--text-dim);
    letter-spacing: 0.15em;
    text-align: right;
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
  .broadcast input::placeholder { color: var(--text-dim); opacity: 0.5; }
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
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 16px;
  }

  .puck-card {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 0;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, box-shadow 0.3s;
  }
  .puck-card:hover { border-color: var(--border-hi); }

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

  .pc-inner { padding: 22px 24px 20px; }

  .pc-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 16px;
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
    margin-top: 16px;
    padding-top: 14px;
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
    font-size: 22px;
    font-weight: 700;
    color: var(--gold);
    text-shadow: 0 0 12px rgba(212,184,104,0.3);
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
            <div class="tb-subtitle">FLEET COORDINATION &amp; SURVEILLANCE</div>
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
  var h = "";
  for(var i=0;i<pucks.length;i++){
    var p = pucks[i], s = sc(p.effective_status), col = p.color||"#00ff88", ver = p.version||{};
    var pid = p.puck_id||"";
    if(!ledgerAccum[pid]) ledgerAccum[pid] = (Math.abs(hashCode(pid)) % 2000 + 500) / 100;
    if(p.effective_status!=="offline") ledgerAccum[pid] += 0.01 + Math.random()*0.04;
    var ledger = ledgerAccum[pid];
    h += '<div class="puck-card '+s+'">'
      +'<div class="pc-inner">'
      +'<div class="pc-header">'
      +'  <div class="pc-color-dot" style="color:'+col+';background:'+col+'"></div>'
      +'  <div class="pc-name">'+esc(p.puck_name||"UNNAMED")+'</div>'
      +'  <div class="pc-status '+s+'">'+sl(p.effective_status)+'</div>'
      +'</div>'
      +'<div class="pc-id">'+esc(pid||"\u2014")+'</div>'
      +'<div class="pc-grid">'
      +'  <div class="pc-metric"><div class="pc-metric-label">Operator</div><div class="pc-metric-value">'+esc(p.owner_name||"\u2014")+'</div></div>'
      +'  <div class="pc-metric"><div class="pc-metric-label">Net Addr</div><div class="pc-metric-value">'+esc(p.ip||"\u2014")+'</div></div>'
      +'  <div class="pc-metric"><div class="pc-metric-label">Uptime</div><div class="pc-metric-value">'+fmtUp(p.uptime)+'</div></div>'
      +'  <div class="pc-metric"><div class="pc-metric-label">Mem Alloc</div><div class="pc-metric-value">'+fmtMem(p.memory_usage)+'</div></div>'
      +'</div>'
      +'<div class="pc-ledger"><span class="pc-ledger-label">$LEDGER</span><span class="pc-ledger-value">$'+ledger.toFixed(2)+'</span></div>';
    if(ver.branch||ver.commit){
      h+='<div class="pc-version"><span class="pc-branch">'+esc(ver.branch||"\u2014")+'</span><span class="pc-commit">'+esc(ver.commit||"")+(ver.dirty?" *":"")+'</span></div>';
    }
    h+='</div>';  // pc-inner
    h+='<div class="pc-radar"></div>';
    h+='</div>';
  }
  el.innerHTML = h;
}

function esc(s){var d=document.createElement("div");d.textContent=s;return d.innerHTML;}
function hashCode(s){var h=0;for(var i=0;i<s.length;i++){h=((h<<5)-h)+s.charCodeAt(i);h|=0;}return h;}

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
        {"puck_id": "aura-16113256-035751-314159", "puck_name": "Paul's Puck",
         "owner_name": "Paul", "color": "#981225", "ip": "192.168.1.108",
         "status": "idle", "uptime": 86400*2+3600*7, "memory_usage": 3840},
        {"puck_id": "aura-d4v1d002-035751-314159", "puck_name": "David's Puck",
         "owner_name": "David", "color": "#23A5FF", "ip": "192.168.1.55",
         "status": "listening", "uptime": 86400+1800, "memory_usage": 2910},
        {"puck_id": "aura-b0b00003-a4c820-314159", "puck_name": "Bob's Puck",
         "owner_name": "Bob", "color": "#8B5CF6", "ip": "192.168.1.78",
         "status": "speaking", "uptime": 3600*14, "memory_usage": 4200},
        {"puck_id": "aura-j0rg3004-f17e22-314159", "puck_name": "Jorge's Puck",
         "owner_name": "Jorge", "color": "#F59E0B", "ip": "192.168.1.103",
         "status": "idle", "uptime": 86400*5, "memory_usage": 3100},
        {"puck_id": "aura-mas0n005-38d1a5-314159", "puck_name": "Mason's Puck",
         "owner_name": "Mason", "color": "#10B981", "ip": "192.168.1.91",
         "status": "thinking", "uptime": 7200, "memory_usage": 5020},
        {"puck_id": "aura-1ucas006-cc71b9-314159", "puck_name": "Lucas' Puck",
         "owner_name": "Lucas", "color": "#EC4899", "ip": "192.168.1.67",
         "status": "idle", "uptime": 86400*3+10800, "memory_usage": 2680},
        {"puck_id": "aura-raf3l007-dd82c0-314159", "puck_name": "Dr. Rafael's Puck",
         "owner_name": "Dr. Rafael", "color": "#4ecdc4", "ip": "192.168.1.94",
         "status": "idle", "uptime": 86400*1+7200, "memory_usage": 3400},
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

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
_active_scenario: str = "ledgerai_hq"


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
      <div class="sidebar-section-label">Scenarios</div>
      <div class="sidebar-item active" id="scn-ledgerai_hq" onclick="switchScenario('ledgerai_hq')">
        <span class="icon">&#9962;</span> LedgerAI HQ
      </div>
      <div class="sidebar-item" id="scn-restaurant" onclick="switchScenario('restaurant')">
        <span class="icon">&#9749;</span> Restaurant
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

var _allScenarioPools = __SCENARIO_POOLS_JSON__;
var _activeScenario = "__DEFAULT_SCENARIO__";
var _scenarioLabels = __SCENARIO_LABELS_JSON__;
var _convPools = _allScenarioPools[_activeScenario] || {};


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
// ── Scenario switching ──────────────────────────
function _resetLiveState() {
  liveTranscripts = {};
  liveViewpoints = {};
  _lineCounters = {};
  _vpCounters = {};
  _tsInited = false;
  ledgerAccum = {};
  if (typeof expandedIdx !== 'undefined') { expandedIdx = -1; collapseCards(); }
}
function switchScenario(key) {
  if (key === _activeScenario) return;
  fetch("/scenario", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({scenario: key})
  }).then(function(r){ return r.json(); }).then(function(d) {
    if (d.ok) {
      _activeScenario = key;
      _convPools = _allScenarioPools[key] || {};
      _resetLiveState();
      document.querySelectorAll("[id^='scn-']").forEach(function(el){ el.classList.remove("active"); });
      var btn = document.getElementById("scn-" + key);
      if (btn) btn.classList.add("active");
      refresh();
    }
  });
}

function connectSSE(){
  var es = new EventSource("/stream");
  es.addEventListener("puck_registered",function(){refresh();});
  es.addEventListener("status_change",function(){refresh();});
  es.addEventListener("scenario_changed",function(e){
    var d = JSON.parse(e.data);
    _activeScenario = d.scenario;
    _convPools = _allScenarioPools[d.scenario] || {};
    _resetLiveState();
    document.querySelectorAll("[id^='scn-']").forEach(function(el){ el.classList.remove("active"); });
    var btn = document.getElementById("scn-" + d.scenario);
    if (btn) btn.classList.add("active");
    refresh();
  });
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


@app.route("/scenario", methods=["GET", "POST"])
def scenario():
    global _active_scenario
    if request.method == "GET":
        return jsonify({
            "active": _active_scenario,
            "available": [{"key": k, "label": v["label"]} for k, v in SCENARIOS.items()],
        })
    key = request.json.get("scenario", "")
    if key not in SCENARIOS:
        return jsonify({"ok": False, "error": "unknown scenario"}), 400
    _active_scenario = key
    _load_scenario(key)
    _broadcast_sse("scenario_changed", {"scenario": key, "label": SCENARIOS[key]["label"]})
    return jsonify({"ok": True, "scenario": key})


@app.route("/", methods=["GET"])
def dashboard():
    # Inject scenario data into template
    all_pools = {k: v["conv_pools"] for k, v in SCENARIOS.items()}
    labels = {k: v["label"] for k, v in SCENARIOS.items()}
    html = _DASHBOARD_HTML.replace("__SCENARIO_POOLS_JSON__", json.dumps(all_pools))
    html = html.replace('"__DEFAULT_SCENARIO__"', json.dumps(_active_scenario))
    html = html.replace("__SCENARIO_LABELS_JSON__", json.dumps(labels))
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS = {
    "ledgerai_hq": {
        "label": "LedgerAI HQ",
        "conv_pools": {
            "Exec Boardroom": {
                "speakers": ["Paul","David","Dr. Rafael"],
                "lines": [
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
                "viewpoints": [
                    "Paul is running this meeting at pace. David's legal hesitation is valid but he needs to stop hedging and commit to a timeline. Dr. Rafael just dropped the key insight about IP assignment \u2014 that's the real issue, not revenue share.",
                    "The 60-40 hold is the right call. David will fold under pressure from their legal team unless Paul backs him explicitly. The hardware timeline is tight \u2014 'next week' means 'maybe next week.'",
                    "Series B conversations are heating up. Two firms is good but three is leverage. The 340K server room approval was instant \u2014 Paul's spending with conviction.",
                    "Dr. Rafael finally engaged and he's the smartest person in the room. His silence earlier was concerning. The IP clause insight just saved them weeks of back-and-forth.",
                    "This meeting is productive but running long. There's a visitor waiting in the lobby for Paul. Keeping partners waiting is the kind of thing that costs deals.",
                    "David just said 'I'll sign the PO today' \u2014 he's said that exact phrase three times this month and signed zero POs. Track the actual signature, not the promise.",
                    "Everyone in this room knows the Series B valuation is inflated by 20%, but nobody's saying it. Honest conversations now save painful board meetings later.",
                ]
            },
            "Engineering Lab": {
                "speakers": ["Mason","Lucas"],
                "lines": [
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
                "viewpoints": [
                    "Mason just casually dropped a 40% memory reduction like it's nothing. This should be in the boardroom conversation, not buried in a PR review.",
                    "2200 tokens/sec with 40% less memory. These two are operating at a level the rest of the org doesn't appreciate. The 45ms first-token latency is world-class.",
                    "All 847 tests green is reassuring but I'd want to see the long-context edge case tested under load, not just unit tested.",
                    "The streaming decoder work is the real product differentiator. Patent this before publishing any benchmarks.",
                    "Brutal truth: Mason is carrying this team. Lucas asks good questions but hasn't shipped original work in two weeks. He's becoming a reviewer, not a builder.",
                    "They keep benchmarking instead of shipping. Classic engineer procrastination disguised as rigor. Push to production and measure in the wild.",
                ]
            },
            "Kitchen & Lounge": {
                "speakers": ["Jorge","Bob","Sarah","Kim"],
                "lines": [
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
                "viewpoints": [
                    "Four people in the kitchen again. Jorge is the social glue of this office \u2014 every casual conversation traces back to him. This isn't wasted time; this is how culture gets built.",
                    "The rooftop social planning is happening organically and that's better than any HR-organized event. Let them have their 15 minutes.",
                    "Kitchen conversations reveal more about team health than any engagement survey. Everyone's relaxed, joking \u2014 that's a team that trusts each other.",
                    "Bob keeps gravitating to the kitchen. Third time this week. Given his output has been strong, I'll assume social recharge. Not flagging this.",
                    "Jorge has spent 47 minutes in the kitchen today. At some point 'culture building' becomes 'not working.' His Q3 deliverables are behind.",
                ]
            },
            "Medical Suite": {
                "speakers": ["Dr. Chen","Nurse Adams"],
                "lines": [
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
                "viewpoints": [
                    "This room is a well-oiled machine. Dr. Chen runs through patients with surgical precision. The A1C drop from 7.8 to 6.1 is a genuine clinical win.",
                    "The pharmacy integration saving 20 minutes per patient is huge at scale. Dr. Chen should present this efficiency gain to the board.",
                    "ROM at 110 degrees post-knee replacement is excellent progress. Dr. Chen's bedside manner clearly motivates follow-through.",
                    "Smooth operations. If every department ran like this suite, the whole building would be a different place. Nurse Adams deserves a raise.",
                    "Dr. Chen is the fastest clinician I've observed but speed has a shadow side. 3 minutes 40 seconds on a discharge. Efficient or rushed?",
                ]
            },
            "Server Room": {
                "speakers": ["System","Alert"],
                "lines": [
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
                "viewpoints": [
                    "That humidity reading is creeping up again. 43% and the threshold is 40%. Last time this happened it was a failed seal on the CRAC unit.",
                    "Rack 5 fan module has been 'fluctuating' for a week now. Replace it proactively \u2014 a $200 fan is cheaper than a $50K server going thermal.",
                    "The 2.4TB offsite backup completed clean. But nobody's tested a restore in 3 months. Backups without restore tests are just expensive hope.",
                    "Power at 14.2 kW against an 18 kW budget gives us headroom. This room is healthy but needs preventive attention before summer.",
                ]
            },
            "Reception Lobby": {
                "speakers": ["Front Desk","Visitor","Courier"],
                "lines": [
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
                "viewpoints": [
                    "Front desk is handling visitors well \u2014 offered coffee, Wi-Fi, the whole experience. First impressions matter and this lobby is earning its keep.",
                    "David's package arrived. That's three packages this week. Either it's equipment for the lab or he's running a side hustle. Joking. Mostly.",
                    "The lobby is performing perfectly. No wait times, warm reception. This is the invisible excellence that closes deals.",
                    "Two visitors in the last hour. Traffic is picking up. If this pace continues, we might need a second front desk person during peak hours.",
                ]
            },
            "Rooftop Terrace": {
                "speakers": ["Wind Sensor","Ambient"],
                "lines": [
                    ["Ambient","Temperature: 22.4\u00b0C. Humidity: 35%. UV index: 6."],
                    ["Wind Sensor","Wind speed: 8 km/h NNW. Gusts to 12 km/h."],
                    ["Ambient","Noise level: 42 dB. Well within comfort range."],
                    ["Wind Sensor","Barometric pressure: 1013 hPa. Stable."],
                    ["Ambient","Solar panel output: 4.2 kW. Above daily average."],
                    ["Wind Sensor","Wind direction shifting to NW. Speed holding steady."],
                    ["Ambient","Air quality index: 28. Excellent."],
                ],
                "viewpoints": [
                    "Perfect conditions up here. 22\u00b0C, clear skies, gentle breeze. The 5pm social tonight is going to be excellent. This rooftop is criminally underused during work hours.",
                    "UV index at 6 means sunscreen is advisable for tonight. The air quality is excellent \u2014 better than most offices' indoor air.",
                    "Wind is calm, pressure stable, no weather changes incoming. Solar output at 4.2 kW means the terrace lighting can run entirely off the panels.",
                    "Empty again during business hours. I'm going to recommend a 'rooftop Fridays' policy. Fresh air improves cognitive performance by 15-20%.",
                ]
            },
        },
        "pucks": [
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
        ],
    },
    "restaurant": {
        "label": "Restaurant",
        "conv_pools": {
            "Table 1": {
                "speakers": ["James","Emily"],
                "lines": [
                    ["James","This is really nice. I'm glad we came here."],
                    ["Emily","The reviews were amazing. I had to try it."],
                    ["James","What are you thinking for an appetizer?"],
                    ["Emily","The burrata looks incredible. Want to split it?"],
                    ["James","Absolutely. And I'm eyeing the ribeye."],
                    ["Emily","I'm torn between the salmon and the pasta."],
                    ["James","Get the salmon. You always say you should eat more fish."],
                    ["Emily","You're right. And a glass of the Sancerre."],
                    ["James","Should we do a bottle? It's date night after all."],
                    ["Emily","Twist my arm. Let's do it."],
                    ["James","When's the last time we went out without the kids?"],
                    ["Emily","Three months? Maybe four. Too long."],
                ],
                "viewpoints": [
                    "Classic date night. Both engaged, good eye contact, sharing appetizers. Server should let them breathe \u2014 don't interrupt this vibe. The wine bottle upsell happened naturally. Perfect.",
                    "They haven't been out in months. This meal matters to them. Make sure it's flawless. The ribeye better come out medium-rare or we'll ruin their night.",
                    "Table 1 is the easiest table in the house right now. Happy couple, ordering well, not in a rush. Let them linger \u2014 they'll order dessert and probably a digestif.",
                ]
            },
            "Table 2": {
                "speakers": ["Mike","Sarah"],
                "lines": [
                    ["Mike","So what do you do for work?"],
                    ["Sarah","I'm in marketing at a tech startup. You?"],
                    ["Mike","Finance. Mostly portfolio management."],
                    ["Sarah","That sounds intense. Do you enjoy it?"],
                    ["Mike","Most days. The market's been wild lately."],
                    ["Sarah","I bet. Have you been here before?"],
                    ["Mike","First time. A friend recommended it."],
                    ["Sarah","Same actually. The ambiance is great."],
                    ["Mike","It really is. Should we get some wine?"],
                    ["Sarah","I'd love a glass of pinot noir."],
                    ["Mike","I'll do the same. Makes it easy."],
                    ["Sarah","So tell me about this friend who recommended this place."],
                ],
                "viewpoints": [
                    "First date. The small talk is still in 'job interview' mode. Mike is nervous \u2014 he's folding his napkin. Sarah seems comfortable though. Wine will help. Don't rush the check.",
                    "Body language says this is going okay but not great. Mike needs to ask Sarah something she actually cares about instead of defaulting to work talk. Not my problem, but I'm rooting for him.",
                    "First-timers, both recommended by friends. That word-of-mouth is working. Make sure this experience is memorable \u2014 they'll each tell 5 people.",
                ]
            },
            "Table 3": {
                "speakers": ["Alex","Jordan"],
                "lines": [
                    ["Alex","Happy anniversary, babe."],
                    ["Jordan","Five years. Can you believe it?"],
                    ["Alex","Feels like yesterday and forever at the same time."],
                    ["Jordan","That's either romantic or an insult."],
                    ["Alex","Definitely romantic. I ordered the tasting menu for us."],
                    ["Jordan","You didn't. That's the seven-course one?"],
                    ["Alex","With the wine pairing. Go big or go home."],
                    ["Jordan","I love you. Also I'm going to need to unbutton my pants."],
                    ["Alex","That's the spirit. To five more years."],
                    ["Jordan","To fifty more."],
                ],
                "viewpoints": [
                    "Anniversary dinner, five years, tasting menu with wine pairing. This is a high-spend table and they're in a great mood. Chef should know \u2014 consider a complimentary amuse-bouche. Small gesture, huge loyalty payoff.",
                    "They're genuinely happy. The banter is easy and affectionate. This is the kind of table that writes glowing reviews unprompted. Protect this experience at all costs.",
                    "Seven-course tasting with pairing \u2014 this table will be here for 2+ hours. Plan the kitchen pacing accordingly. Don't rush courses. Let them savor it.",
                ]
            },
            "Table 4": {
                "speakers": ["Richard","Tom","Diana","Priya"],
                "lines": [
                    ["Richard","The Q4 numbers look solid if we close the Henderson deal."],
                    ["Tom","Henderson's been dragging their feet for weeks."],
                    ["Diana","I spoke to their CFO yesterday. They want 15% off."],
                    ["Priya","We can't do 15. Maybe 8 with extended terms."],
                    ["Richard","Let's not talk shop until after appetizers at least."],
                    ["Tom","Fine. But we're circling back over dessert."],
                    ["Diana","Has anyone tried the steak here?"],
                    ["Richard","Best in the city according to the Times review."],
                    ["Priya","I'm getting the lamb. My treat tonight."],
                    ["Tom","Your treat? What's the occasion?"],
                    ["Priya","I just closed the Meridian account. Celebrating."],
                    ["Diana","Priya! That's huge. Congratulations."],
                    ["Richard","Drinks are definitely on Priya then."],
                ],
                "viewpoints": [
                    "Business dinner but the vibe is celebratory thanks to Priya's Meridian close. They'll spend big. Steak, lamb, multiple bottles. This is a corporate card table \u2014 don't skimp on portions.",
                    "Richard tried to pause the shop talk and failed in 30 seconds. These people live and breathe work. The Henderson deal discussion will resume with the entrees. Guaranteed.",
                    "Four-top business dinner, one person paying. Priya's flexing the Meridian win. Good for morale. Server should acknowledge the celebration \u2014 a complimentary round would lock in repeat visits.",
                ]
            },
            "Table 5": {
                "speakers": ["Dave","Lisa","Mark"],
                "lines": [
                    ["Dave","I specifically said medium-rare. This is well-done."],
                    ["Lisa","Oh no. Send it back."],
                    ["Dave","I hate sending things back. But this is charcoal."],
                    ["Mark","Mine's perfect actually. Sorry, Dave."],
                    ["Dave","Where's our server? I haven't seen them in 20 minutes."],
                    ["Lisa","The bread took forever too. Remember?"],
                    ["Dave","Yeah, 15 minutes for bread. That's not okay."],
                    ["Mark","Maybe they're short-staffed tonight."],
                    ["Dave","That's not my problem. We're paying premium prices."],
                    ["Lisa","Let's just flag someone. There's a manager over there."],
                    ["Dave","I'm going to mention it in the review too."],
                    ["Mark","Come on, give them a chance to fix it first."],
                ],
                "viewpoints": [
                    "RED FLAG. Table 5 is spiraling. Overcooked steak, 20-minute server absence, slow bread. Dave is building a mental Yelp review right now. Manager intervention needed IMMEDIATELY.",
                    "This was preventable. The steak was a kitchen mistake, fine. But the 20-minute server gap turned one problem into three. Lisa is backing Dave up which means this complaint has momentum. Act fast.",
                    "Mark is trying to de-escalate but he's losing. Dave said 'premium prices' \u2014 that's the death sentence phrase. Once a customer starts calculating value-for-money out loud, you've already lost them. Comp the steak and apologize personally.",
                    "Three issues in one meal: slow bread, absent server, overcooked steak. This isn't bad luck, it's a pattern. Kitchen needs a ticket audit and the server needs coaching. Tonight.",
                ]
            },
            "Table 6": {
                "speakers": ["Karen","Manager"],
                "lines": [
                    ["Karen","Excuse me, I asked for no onions. There are clearly onions in this."],
                    ["Manager","I'm so sorry about that. Let me have the kitchen remake it right away."],
                    ["Karen","This is the second time this has happened here."],
                    ["Manager","I completely understand your frustration. It won't happen again."],
                    ["Karen","And the music is quite loud. Can you turn it down?"],
                    ["Manager","I'll see what I can do. Can I offer you a complimentary appetizer while you wait?"],
                    ["Karen","Fine. The calamari. And make sure there are no onions anywhere near it."],
                    ["Manager","Absolutely. I'll personally check with the kitchen."],
                    ["Karen","I also wanted to mention the restroom could use attention."],
                    ["Manager","Thank you for letting me know. I'll send someone right away."],
                ],
                "viewpoints": [
                    "Repeat offender on the allergy/preference notes. If she said no onions last time AND this time and we still got it wrong, that's a kitchen communication failure. Check the POS notes system.",
                    "The manager is handling this well \u2014 apologize, comp, fix. Textbook. But the underlying issue is that dietary notes aren't making it to the line. Fix the process, not just the plate.",
                    "She's high-maintenance but she's not wrong. Twice is a pattern. The music complaint and restroom note are her way of saying 'I'm watching everything now.' Every detail matters for the rest of her visit.",
                    "Karen is actually providing free quality control. The restroom note was legitimate \u2014 I checked the cleaning log and it's 45 minutes overdue. She's annoying but useful.",
                ]
            },
            "Table 7": {
                "speakers": ["Dad","Mom","Sophie","Max"],
                "lines": [
                    ["Sophie","Can I get chicken nuggets?"],
                    ["Dad","They don't have chicken nuggets here, sweetie. How about the pasta?"],
                    ["Max","I want pizza!"],
                    ["Mom","Max, indoor voice please. They have a kids' flatbread."],
                    ["Sophie","I spilled my water."],
                    ["Dad","It's okay. Let me grab some napkins."],
                    ["Mom","Can we get some more bread for the table? The kids ate it all."],
                    ["Dad","Should we just order everything at once? I don't think they'll last."],
                    ["Mom","Good idea. Kids meals first please."],
                    ["Max","I need to go to the bathroom."],
                    ["Dad","Again? We just went."],
                    ["Mom","Just take him. I'll order for you."],
                    ["Sophie","Mommy, is there dessert?"],
                    ["Mom","If you eat your dinner, yes."],
                ],
                "viewpoints": [
                    "Family of four, kids under 8. This table needs SPEED. Get the kids' food out in under 10 minutes or this goes from controlled chaos to full meltdown. The parents are managing well but running out of patience.",
                    "Water spill, bread demolished in 3 minutes, two bathroom trips. Classic family dinner. The trick is to make the parents feel like adults for 20 minutes while the kids eat. Dessert is the leverage play.",
                    "Mom is the decision-maker and she's efficient. She'll order for Dad while he's in the bathroom. Respect that. Don't ask if they need more time when he gets back \u2014 she's already decided.",
                    "Every restaurant says 'family-friendly' but few actually optimize for it. Get the crayons out, kids' food first, and keep the bread basket full. The parents will remember how easy we made it.",
                ]
            },
            "Table 8": {
                "speakers": ["Grandma","Grandpa","Uncle Ray","Aunt Carol","Cousin Jen","Cousin Mike"],
                "lines": [
                    ["Grandma","I can't believe everyone could make it tonight."],
                    ["Grandpa","First time in two years we're all together."],
                    ["Uncle Ray","Who's picking up the tab? Not it."],
                    ["Aunt Carol","Ray, behave. It's Mom and Dad's anniversary dinner."],
                    ["Cousin Jen","Happy anniversary, Grandma and Grandpa!"],
                    ["Grandma","Thank you, sweetheart. Fifty-two years."],
                    ["Cousin Mike","That's incredible. What's the secret?"],
                    ["Grandpa","Selective hearing."],
                    ["Grandma","He's not wrong."],
                    ["Uncle Ray","I'll drink to that. Waiter, another round please."],
                    ["Aunt Carol","We should order the chocolate cake for the table."],
                    ["Cousin Jen","Can we get candles? Make it special?"],
                ],
                "viewpoints": [
                    "Anniversary dinner, 52 years, six-top family gathering. This is a memory-making table. If the kitchen has ANY ability to write 'Happy Anniversary' on a dessert plate, do it now. These moments are why restaurants exist.",
                    "Uncle Ray is going to run up the bar tab but Aunt Carol will keep him in check. The real spender is whoever picks up the final bill \u2014 it'll be significant with six people and drinks flowing.",
                    "Grandpa's 'selective hearing' joke got a genuine laugh from the whole table. This family actually likes each other. Rare and beautiful. Don't rush this table \u2014 let them sit as long as they want.",
                ]
            },
            "Table 9": {
                "speakers": ["Parent","Child","Server"],
                "lines": [
                    ["Child","Mommy I don't like this. It's green."],
                    ["Parent","Just try one bite. You liked broccoli last week."],
                    ["Child","That was different broccoli."],
                    ["Parent","Broccoli is broccoli, honey."],
                    ["Server","How is everything? Can I get anything else?"],
                    ["Parent","Could we get some plain buttered noodles? Just in case."],
                    ["Server","Of course. I'll put a rush on it."],
                    ["Child","Can I have ice cream?"],
                    ["Parent","After you eat something real."],
                    ["Child","Noodles are real."],
                    ["Parent","Fair point. Eat the noodles, then ice cream."],
                ],
                "viewpoints": [
                    "Solo parent with a picky eater. The server read the room perfectly \u2014 offered buttered noodles without being asked twice. That kind of intuition is worth its weight in gold.",
                    "'That was different broccoli' is peak toddler logic and honestly kind of valid. The parent is handling it with patience. Get those noodles out fast and everyone wins.",
                    "This parent is going to tip well if we make their life easy. They're not here for the fine dining experience \u2014 they're here to eat a meal without cooking. Noodles fast, ice cream ready, check whenever they want it.",
                ]
            },
            "Table 10": {
                "speakers": ["Nina","Kira","Jess","Becca"],
                "lines": [
                    ["Nina","Okay I have TEA. You will not believe what happened at work."],
                    ["Kira","Spill. Immediately."],
                    ["Nina","So Marcus in accounting? He got caught expensing his Tinder dates."],
                    ["Jess","STOP. As business dinners?"],
                    ["Nina","As CLIENT MEETINGS. Six of them."],
                    ["Becca","I'm screaming. Did he get fired?"],
                    ["Nina","PIP. But everyone knows."],
                    ["Kira","I need another margarita for this story."],
                    ["Jess","Same. Four more margs please."],
                    ["Becca","We should come here every Friday."],
                    ["Nina","Agreed. This is our new spot."],
                    ["Kira","The guacamole is incredible by the way."],
                    ["Jess","We should order more. And the queso."],
                ],
                "viewpoints": [
                    "Girls' night. Four margaritas, guac, gossip. This table is going to be loud, happy, and profitable. They just declared this their 'new spot' \u2014 that's the best marketing money can't buy.",
                    "They're on their second round of margs and the entrees haven't landed yet. Pace the drinks with the food or they'll be sloppy by dessert. But keep the energy up \u2014 this table is having the time of their lives.",
                    "Nina's workplace gossip is doing more for table morale than anything on our menu. Four people, repeat-visit potential, high bar tab. This is the ideal customer segment.",
                    "Becca said 'every Friday.' That's a standing reservation if we play it right. Drop a comment like 'we'll save this table for you' and watch what happens.",
                ]
            },
            "Table 11": {
                "speakers": ["Carlos","Diego","Sam"],
                "lines": [
                    ["Carlos","Did you watch the game last night?"],
                    ["Diego","That last-minute goal was insane."],
                    ["Sam","I lost fifty bucks on that game."],
                    ["Carlos","That's what you get for betting against them at home."],
                    ["Diego","Wings are great here. Get the Nashville hot."],
                    ["Sam","How hot is hot?"],
                    ["Diego","You'll need milk. But it's worth it."],
                    ["Carlos","Let's get a pitcher. Who's driving?"],
                    ["Sam","I took an Uber. I'm free."],
                    ["Diego","Same. Pitcher it is."],
                    ["Carlos","Should we get the nachos too?"],
                    ["Sam","Obviously. And the sliders."],
                ],
                "viewpoints": [
                    "Three guys, sports talk, pitcher of beer, Nashville hot wings. This is bar food done right. Simple order, high volume, fast turnover. Keep the pitcher full and they'll stay for three hours.",
                    "Nobody's driving \u2014 they all Ubered. That means the bar tab has no ceiling. Smart move by them. Profitable for us. Second pitcher incoming.",
                    "This table is low-maintenance gold. They know what they want, they order fast, they don't complain. Every restaurant needs 10 tables like this on a Friday night.",
                ]
            },
            "Table 12": {
                "speakers": ["Server","Customer A","Customer B"],
                "lines": [
                    ["Customer A","Excuse me, we've been waiting 40 minutes for our entrees."],
                    ["Server","I'm so sorry. Let me check with the kitchen right away."],
                    ["Customer B","The people who sat down after us already have their food."],
                    ["Customer A","This is unacceptable. We have theater tickets at 8."],
                    ["Server","I completely understand. Let me see what happened."],
                    ["Customer B","If it's not out in 5 minutes, just cancel it."],
                    ["Customer A","And take the appetizers off the bill. We've been miserable."],
                    ["Server","Absolutely. I'll speak with my manager about comping those."],
                    ["Customer B","We really wanted to enjoy this evening."],
                    ["Customer A","Let's just go somewhere else."],
                ],
                "viewpoints": [
                    "CRITICAL. 40-minute wait, theater deadline, table seated after them got food first. This is a kitchen sequencing disaster. The ticket got lost or deprioritized. Manager needs to personally deliver the food with an apology.",
                    "They're about to walk out. 'Just cancel it' is one step from leaving. Comp everything, deliver the entrees in the next 3 minutes, and hope they don't review-bomb us tonight from the theater lobby.",
                    "The server is saying the right things but saying isn't doing. Stop apologizing and start fixing. Kitchen needs to fire their entrees NOW ahead of everything else. This is triage, not customer service.",
                    "Seated-after-them-got-food-first is the ultimate insult. It means the kitchen isn't running tickets in order. That's not a mistake, that's a systemic failure. Audit the expo station tonight.",
                ]
            },
            "Table 13": {
                "speakers": ["Olivia"],
                "lines": [
                    ["Olivia","Could I get the salmon, please? Grilled, not pan-seared."],
                    ["Olivia","And a glass of the Chablis. The 2019 if you have it."],
                    ["Olivia","Actually, I'll start with the bisque."],
                    ["Olivia","This book is so good. I should come here more often."],
                    ["Olivia","Could I get some more water when you have a chance?"],
                    ["Olivia","The salmon is excellent. My compliments to the chef."],
                    ["Olivia","I think I'll have the cr\u00e8me br\u00fbl\u00e9e. And an espresso."],
                    ["Olivia","Check whenever you're ready. No rush."],
                ],
                "viewpoints": [
                    "Solo diner, reading a book, specific wine vintage request. This is someone who knows restaurants. Don't over-check on her. She asked for water \u2014 that means we missed a refill. Fix that.",
                    "Olivia is the kind of customer who says 'compliments to the chef' and means it. She's also the kind who notices a water glass sitting empty for 8 minutes. Attentive but not hovering. Read her cues.",
                    "Solo diners who order a full three courses with wine pairings are confident, experienced, and tend to tip well. She said 'I should come here more often' \u2014 she's already sold. Don't oversell.",
                ]
            },
            "Table 14": {
                "speakers": ["Frank"],
                "lines": [
                    ["Frank","Hey, the usual please. You know what I like."],
                    ["Frank","How's the family, Maria? Kids doing good?"],
                    ["Frank","Tell Chef Tony I said hello. And that his bolognese is still the best."],
                    ["Frank","I'll take the booth by the window if it opens up."],
                    ["Frank","Put it on my tab. Same as always."],
                    ["Frank","The new waiter seems sharp. Good hire."],
                    ["Frank","See you Thursday."],
                ],
                "viewpoints": [
                    "Frank is furniture at this point. He's here three times a week, knows everyone by name, orders the same thing. His tab is auto-pilot. This man is worth $15K a year in revenue and he's never once complained.",
                    "Regulars like Frank are the backbone of any restaurant. He just complimented the new hire unprompted \u2014 that's a man who feels ownership over this place. Protect that relationship.",
                    "Frank tips 25% every time, never sends anything back, and tells Chef Tony he's the best. He's the perfect customer and he knows it. Worth more than 50 Yelp reviews.",
                ]
            },
            "Table 15": {
                "speakers": ["Group Leader","Friend 1","Friend 2","Friend 3","Friend 4","Friend 5"],
                "lines": [
                    ["Group Leader","SURPRISE! Happy birthday, Jen!"],
                    ["Friend 1","We got you! You had no idea!"],
                    ["Friend 2","We've been planning this for three weeks."],
                    ["Friend 3","The cake is already ordered. Red velvet."],
                    ["Friend 4","Speech! Speech!"],
                    ["Friend 5","No speeches. Just drinks. Shots!"],
                    ["Group Leader","Six tequila shots please. And the biggest dessert you have."],
                    ["Friend 1","I can't believe you pulled this off."],
                    ["Friend 2","The hardest part was getting Jen here without spoiling it."],
                    ["Friend 3","She thought we were going to a movie."],
                    ["Friend 4","This is so much better than a movie."],
                    ["Group Leader","More shots? More shots."],
                ],
                "viewpoints": [
                    "Surprise birthday party, six-top, tequila shots flowing. This table will generate more revenue per hour than any other tonight. And they'll post about it on Instagram. Win-win.",
                    "They pre-ordered a cake and coordinated a surprise. This group plans ahead and spends freely. Birthday tables are the highest-ROI tables in the building. Candles, singing, the whole show.",
                    "Six tequila shots before entrees have landed. Pace check needed. These are adults celebrating but the trajectory is steep. Keep water on the table. Don't let a great night become a liability.",
                ]
            },
            "Table 16": {
                "speakers": ["Bride","Groom","Best Man","Maid of Honor"],
                "lines": [
                    ["Best Man","I still can't believe you two are getting married next month."],
                    ["Bride","Neither can I. There's so much left to plan."],
                    ["Groom","The venue is booked, the DJ is booked. We're fine."],
                    ["Maid of Honor","What about the seating chart? That's a war zone."],
                    ["Bride","Don't remind me. Uncle Steve cannot sit near Aunt Linda."],
                    ["Groom","Your family is a reality show."],
                    ["Bride","Says the man whose mother called me three times today about centerpieces."],
                    ["Best Man","This is why I'm never getting married."],
                    ["Maid of Honor","Let's order more wine. We need it for the seating chart."],
                    ["Groom","Can we just enjoy dinner? The planning can wait one night."],
                    ["Bride","Fine. But we're doing place cards tomorrow."],
                ],
                "viewpoints": [
                    "Pre-wedding dinner, four-top. They're stressed but happy. The bride is a planner, the groom wants to relax. If they ask about private dining or event space, we should be ready with the pitch.",
                    "The maid of honor just ordered more wine 'for the seating chart.' That's the kind of problem-solving this table runs on. Keep the wine flowing and they'll handle their family drama without involving us.",
                    "Wedding party tables are future catering clients. If this dinner goes well, there's a non-zero chance they book the rehearsal dinner here. Plant the seed. Don't hard sell.",
                ]
            },
            "Table 17": {
                "speakers": ["Chef","Critic"],
                "lines": [
                    ["Critic","The amuse-bouche was interesting. Yuzu gel with smoked eel?"],
                    ["Chef","Inspired by a trip to Osaka last fall. The eel is from a local smokehouse."],
                    ["Critic","The texture contrast works. Tell me about the main course."],
                    ["Chef","Dry-aged duck breast, beetroot three ways, jus infused with star anise."],
                    ["Critic","The duck is cooked perfectly. I'll give you that."],
                    ["Chef","Thank you. The dry-aging process is 21 days minimum."],
                    ["Critic","The beetroot puree is slightly over-seasoned for my palate."],
                    ["Chef","I appreciate the honesty. We've been adjusting the salt ratios."],
                    ["Critic","Overall, I'm impressed. This is serious cooking."],
                ],
                "viewpoints": [
                    "Food critic at Table 17 dining with the chef. This is either a review dinner or a relationship-building meal. Either way, every plate that leaves the kitchen tonight needs to be perfect. The whole team should know.",
                    "The critic said 'over-seasoned' about the beetroot. That's going in the review if this is on-record. Chef took it gracefully but the kitchen needs to recalibrate that dish before service tomorrow.",
                    "A critic who says 'I'm impressed' at the end is giving you a good review. But 'slightly over-seasoned' will be the one line everyone remembers. Fix the beetroot. Tonight.",
                ]
            },
            "Table 18": {
                "speakers": ["Tourist 1","Tourist 2","Tourist 3","Tourist 4"],
                "lines": [
                    ["Tourist 1","What's a 'prix fixe'? Is that like a set menu?"],
                    ["Tourist 2","I think so. Let me Google it."],
                    ["Tourist 3","Everything is so expensive here. $45 for pasta?"],
                    ["Tourist 4","We're on vacation. Just enjoy it."],
                    ["Tourist 1","Can I take a photo of the ceiling? It's gorgeous."],
                    ["Tourist 2","The cocktails look amazing. What's a Negroni?"],
                    ["Tourist 3","Should we ask for recommendations?"],
                    ["Tourist 4","Excuse me, what do you recommend for someone who's never had Italian fine dining?"],
                    ["Tourist 1","The waiter was so nice about explaining everything."],
                    ["Tourist 2","I love this city. We should come back next year."],
                ],
                "viewpoints": [
                    "Tourist table. They're Googling menu terms and taking ceiling photos. Price-sensitive but spending anyway because vacation. The server who explains the menu without condescension wins a big tip here.",
                    "They asked for recommendations \u2014 that's an open invitation to upsell. Suggest the tasting menu. They want the experience, not just the food. And they'll photograph every course. Free marketing.",
                    "'$45 for pasta' comment is a price anchor. But Tourist 4 said 'just enjoy it.' The group will override the frugal member. Happens every time. Focus on making it feel worth every dollar.",
                ]
            },
            "Table 19": {
                "speakers": ["Wife","Husband"],
                "lines": [
                    ["Wife","The steak is cold. Feel this."],
                    ["Husband","It's lukewarm at best."],
                    ["Wife","And I asked for asparagus, not broccoli."],
                    ["Husband","My fish is fine at least."],
                    ["Wife","That's not helpful, Robert."],
                    ["Husband","I'm just saying. Should I call the waiter?"],
                    ["Wife","I already tried. He walked right past us."],
                    ["Husband","Let me try. Excuse me? Excuse me?"],
                    ["Wife","See? Invisible."],
                    ["Husband","I'll go find someone."],
                    ["Wife","This place used to be good. What happened?"],
                ],
                "viewpoints": [
                    "Cold steak AND wrong side dish AND server ignoring them. Table 19 is a three-alarm fire. 'This place used to be good' means they're former regulars we're about to lose. Senior staff intervention required.",
                    "They're calling out to the server and getting ignored. That's not busy, that's negligent. Which section is this? Whoever's covering it needs an immediate course correction.",
                    "'Used to be good' is the most damaging phrase in hospitality. It means they had high expectations based on past experience and we're destroying that goodwill in real time. Manager to the table. Now.",
                    "Robert is going to go find someone, which means he'll walk to the host stand looking frustrated. Every other guest will see it. A single unhappy customer on their feet is visible to 20 tables. Intercept him.",
                ]
            },
            "Table 20": {
                "speakers": ["Server","VIP 1","VIP 2"],
                "lines": [
                    ["VIP 1","We'd like the private room if it's available."],
                    ["Server","Of course. Right this way. I'll have your preferred Barolo ready."],
                    ["VIP 2","You remembered."],
                    ["Server","Always. Shall I have Chef prepare the usual tasting?"],
                    ["VIP 1","Please. And tell him we brought the burgundy we discussed."],
                    ["Server","I'll have it decanted. Anything else to start?"],
                    ["VIP 2","Just privacy. We have business to discuss."],
                    ["VIP 1","And the souffl\u00e9 for dessert. Don't forget the 45-minute lead."],
                    ["Server","Already noted. I'll fire it with your main course."],
                ],
                "viewpoints": [
                    "VIP table. They brought their own wine, know the chef by name, and pre-ordered a souffl\u00e9. This is the kind of customer who spends $2K a visit and never looks at a price. Treat accordingly.",
                    "The 'we have business to discuss' line means zero interruptions after the first course lands. Server already knows this \u2014 notice the souffl\u00e9 timing was pre-calculated. This is what professional service looks like.",
                    "They brought a burgundy to share with the chef. This is a relationship, not a transaction. These two are investors, industry, or old money. Doesn't matter which \u2014 the protocol is the same: anticipate, don't ask.",
                ]
            },
            "Table 21": {
                "speakers": ["Waiter","Diner A","Diner B"],
                "lines": [
                    ["Diner A","I'm gluten-free, dairy-free, and I don't eat nightshades."],
                    ["Waiter","Absolutely. Let me walk you through what works on our menu."],
                    ["Diner B","I'm vegan. But flexible on honey."],
                    ["Waiter","Great. The roasted cauliflower is excellent and fits both restrictions."],
                    ["Diner A","Does the chef use shared fryers? Cross-contamination is a concern."],
                    ["Waiter","We have a dedicated allergy station. I'll flag your tickets."],
                    ["Diner B","That's really reassuring. Most places just guess."],
                    ["Diner A","Can I see the ingredient list for the vinaigrette?"],
                    ["Waiter","I'll get that from the kitchen for you right away."],
                ],
                "viewpoints": [
                    "High-restriction table but they're not difficult \u2014 they're careful. The server is handling it like a pro. Dedicated allergy station callout was the right move. These are the customers who become fiercely loyal when you get it right.",
                    "Gluten-free, dairy-free, no nightshades, plus a vegan. The kitchen ticket for this table is going to look like a medical chart. But if we nail it, they'll tell every other dietary-restricted friend they have.",
                    "Asking for the vinaigrette ingredient list is reasonable, not difficult. Any server who rolls their eyes at this belongs at a different restaurant. This table is testing our professionalism and we're passing.",
                ]
            },
            "Table 22": {
                "speakers": ["Wine Guy","Sommelier","Date"],
                "lines": [
                    ["Wine Guy","I'm thinking a Left Bank Bordeaux. 2015 or 2016 vintage."],
                    ["Sommelier","The 2016 Pauillac is drinking beautifully right now."],
                    ["Wine Guy","What's the tannin structure like?"],
                    ["Sommelier","Firm but elegant. It'll open up with the duck."],
                    ["Date","I'll have whatever he's having. I trust you."],
                    ["Wine Guy","Can we see the reserve list?"],
                    ["Sommelier","Of course. Page three has our Burgundy verticals."],
                    ["Wine Guy","Oh, you have the '09 Romanée? What's the price?"],
                    ["Sommelier","I'll bring it tableside for you to consider."],
                    ["Date","I love that you know so much about wine."],
                    ["Wine Guy","Spent a summer in Bordeaux. Never recovered."],
                ],
                "viewpoints": [
                    "Wine enthusiast flexing for his date. He actually knows his stuff though \u2014 Left Bank, vintage years, tannin structure. The sommelier is feeding him perfectly. This table will spend more on wine than food.",
                    "He asked about the '09 Romanée. If that's on the list at $800+, and the sommelier brings it tableside, he's trapped \u2014 he can't say no in front of his date. Beautiful upsell choreography.",
                    "The date said 'I trust you' which means she's impressed. Wine Guy is winning tonight. Keep the sommelier available for a second bottle. This could be a $500 wine tab easily.",
                ]
            },
            "Table 23": {
                "speakers": ["Teen 1","Teen 2","Teen 3"],
                "lines": [
                    ["Teen 1","Oh my god this place is so fancy."],
                    ["Teen 2","Do they have normal food? Like fries?"],
                    ["Teen 3","Truffle fries. For $18."],
                    ["Teen 1","My mom gave me $50. That's like two things."],
                    ["Teen 2","Let's split stuff. The sliders look good."],
                    ["Teen 3","I'm just getting a Caesar salad and water."],
                    ["Teen 1","Don't be boring. We're celebrating prom."],
                    ["Teen 2","Fine. One slider, one fry, split three ways."],
                    ["Teen 3","And three waters."],
                    ["Teen 1","Actually, can we get Shirley Temples? We're feeling fancy."],
                ],
                "viewpoints": [
                    "Three teenagers post-prom with $50 budgets. They're going to split everything and order water. Revenue will be minimal but they'll take 40 photos and tag the restaurant. That's worth more than the $60 check.",
                    "Don't let any server treat this table as second-class. These kids saved up for this experience. In 10 years they'll be the business dinner crowd. First impressions last a lifetime.",
                    "Shirley Temples for the prom kids. Charge them fairly, give them the full experience, and they'll be back for every anniversary, birthday, and celebration for decades. Long game.",
                ]
            },
            "Table 24": {
                "speakers": ["Line Cook","Server B"],
                "lines": [
                    ["Line Cook","I'm on my third double this week. I can't feel my feet."],
                    ["Server B","At least you're not dealing with Table 12. They're furious."],
                    ["Line Cook","That's because their ticket got buried. Expo messed up."],
                    ["Server B","Well I'm the one getting stiffed on the tip."],
                    ["Line Cook","Tips. Must be nice. I make the same whether it's empty or slammed."],
                    ["Server B","Fair point. The new guy is struggling tonight."],
                    ["Line Cook","He'll learn or he'll quit. Everybody does one or the other."],
                    ["Server B","Manager said we might get a raise next quarter."],
                    ["Line Cook","They said that last quarter too."],
                ],
                "viewpoints": [
                    "Staff break table \u2014 the unfiltered view. The line cook is burned out, server is frustrated about tips, and they both know the new hire is drowning. This is the real operational health check.",
                    "Three doubles in a week is a staffing problem, not a dedication badge. If the line cook can't feel his feet, his plating quality dropped hours ago. Schedule relief or accept the consequences.",
                    "The expo messed up Table 12's ticket and the line cook knows it. Front-of-house is absorbing the blame. This is a communication breakdown between kitchen and floor that'll keep costing us.",
                    "'They said that last quarter too' about the raise. Morale is fragile. If management doesn't follow through this time, the line cook is gone within a month. Good cooks are impossible to replace right now.",
                ]
            },
            "Table 25": {
                "speakers": ["Busser","Host"],
                "lines": [
                    ["Busser","Table 5 wants to see a manager. Again."],
                    ["Host","I know. The steak situation. Manager's on the way."],
                    ["Busser","Also Table 19 flagged me about a cold plate."],
                    ["Host","Tonight is a disaster. We're down two servers."],
                    ["Busser","I'm covering three sections worth of bussing."],
                    ["Host","I appreciate you. Seriously."],
                    ["Busser","The 8:30 reservation is a party of 12. Are we ready?"],
                    ["Host","Not even close. I need to rearrange the back section."],
                ],
                "viewpoints": [
                    "Second staff table revealing the backstage chaos. Down two servers, busser covering triple load, party of 12 incoming, and multiple fire tables. This is a capacity crisis happening in real time.",
                    "The host said 'tonight is a disaster' and she's right. But the busser is holding things together. Promote that person or at minimum buy them a drink after shift. They're the MVP tonight.",
                    "Party of 12 at 8:30 and we're already drowning. Someone should have called in backup an hour ago. This is a management failure, not a staff failure.",
                ]
            },
            "Table 26": {
                "speakers": ["Foodie A","Foodie B"],
                "lines": [
                    ["Foodie A","The mouthfeel on this risotto is extraordinary."],
                    ["Foodie B","Perfectly al'onda. Most places overcook it."],
                    ["Foodie A","And the saffron is actually Persian. You can taste the difference."],
                    ["Foodie B","I'm getting notes of bone marrow in the broth base."],
                    ["Foodie A","Should I tag the chef on my story?"],
                    ["Foodie B","Definitely. Get a top-down shot with the natural light."],
                    ["Foodie A","The plating is exquisite. Very Noma-influenced."],
                    ["Foodie B","I'd say more Eleven Madison Park. The negative space."],
                    ["Foodie A","You're right. The restraint is the statement."],
                    ["Foodie B","I need to review this on my blog."],
                ],
                "viewpoints": [
                    "Food influencers. They're photographing everything and using words like 'mouthfeel' and 'al'onda.' Pretentious? Yes. Valuable? Also yes. Their blog post will reach 50K followers. Make every plate Instagram-ready.",
                    "They're debating whether the plating is Noma or Eleven Madison Park. Meanwhile the kitchen is just trying to get Table 12's steak out. The contrast between these worlds is hilarious.",
                    "Persian saffron detection is either genuinely impressive palate work or complete nonsense. Either way, they're going to write a 2000-word review. Make sure it's a good one.",
                ]
            },
            "Table 27": {
                "speakers": ["Couple A","Couple B"],
                "lines": [
                    ["Couple A","We should do double dates more often."],
                    ["Couple B","Absolutely. This is fun."],
                    ["Couple A","How's the new house coming along?"],
                    ["Couple B","Renovation nightmare. Don't ask."],
                    ["Couple A","We went through that last year. It gets better."],
                    ["Couple B","The contractor ghosted us for two weeks."],
                    ["Couple A","Classic. Get everything in writing."],
                    ["Couple B","Let's order another bottle. I need to forget about drywall."],
                    ["Couple A","The Malbec was good. Same one?"],
                    ["Couple B","Make it two. It's that kind of week."],
                ],
                "viewpoints": [
                    "Double date, renovation commiseration, two bottles of Malbec. This is a comfort table \u2014 they're here to decompress. Low maintenance, good spenders, pleasant energy. Let them be.",
                    "Two bottles of Malbec to forget about drywall. I respect the coping mechanism. This table will stay late, tip well, and leave happy. The ideal Tuesday night four-top.",
                ]
            },
            "Table 28": {
                "speakers": ["Bar Patron 1","Bar Patron 2","Bar Patron 3","Bar Patron 4"],
                "lines": [
                    ["Bar Patron 1","Next round's on me. What're we drinking?"],
                    ["Bar Patron 2","Old Fashioned. Make it a double."],
                    ["Bar Patron 3","I'll do a Manhattan. With rye."],
                    ["Bar Patron 4","Tequila soda. Keep them coming."],
                    ["Bar Patron 1","Remember when this place first opened? We were here night one."],
                    ["Bar Patron 2","The bartender gave us free shots because nobody else was here."],
                    ["Bar Patron 3","Now you can't get a reservation."],
                    ["Bar Patron 4","We don't need one. We're bar regulars."],
                    ["Bar Patron 1","To being here before it was cool."],
                    ["Bar Patron 2","Hear hear. Another round."],
                    ["Bar Patron 3","Should we eat? I'm seeing double."],
                    ["Bar Patron 4","Sliders. Lots of sliders."],
                ],
                "viewpoints": [
                    "Bar regulars from opening night. They don't need reservations and the bartender knows their orders. This table is pure bar revenue \u2014 all high-margin cocktails. They'll close the bar.",
                    "Four double cocktails and counting, no food ordered yet. The 'seeing double' comment means we should proactively suggest food. Responsible service meets revenue optimization.",
                    "They said 'before it was cool.' OG regulars have social currency \u2014 they tell everyone they discovered this place. Keep them happy, they're walking billboards.",
                ]
            },
            "Table 29": {
                "speakers": ["Late Diner 1","Late Diner 2","Late Diner 3"],
                "lines": [
                    ["Late Diner 1","I can't believe we just got out of that meeting."],
                    ["Late Diner 2","9 PM and I haven't eaten since breakfast."],
                    ["Late Diner 3","Order everything. I don't care what it costs."],
                    ["Late Diner 1","The kitchen closes at 10 right?"],
                    ["Late Diner 2","We have 45 minutes. Speed round."],
                    ["Late Diner 3","Appetizers and mains at the same time. No waiting."],
                    ["Late Diner 1","Three steaks, three salads, bread immediately."],
                    ["Late Diner 2","And a bottle of the Cab. Whatever's good."],
                    ["Late Diner 3","I'm expensing all of this. Go nuts."],
                ],
                "viewpoints": [
                    "Post-meeting emergency dinner, 9 PM, starving, expense account. They want speed and volume. Fire everything simultaneously \u2014 appetizers and mains together. Don't try to pace this meal, they'll hate you for it.",
                    "'I don't care what it costs' plus 'I'm expensing all of this' \u2014 that's the magic combination. Three steaks, a Cab, and they'll add dessert if the kitchen can swing it. High-revenue table in a 45-minute window.",
                    "Kitchen closes in 45 minutes and these three just ordered the entire menu. Alert the line \u2014 this is the last big push of the night. Make it count. A strong close sets the tone for tomorrow.",
                ]
            },
            "Table 30": {
                "speakers": ["Mystery Guest 1","Mystery Guest 2"],
                "lines": [
                    ["Mystery Guest 1","The service has been exceptional so far."],
                    ["Mystery Guest 2","Note the time between courses. Very consistent."],
                    ["Mystery Guest 1","Water refilled without asking. That's attention to detail."],
                    ["Mystery Guest 2","The server introduced herself and remembered our names."],
                    ["Mystery Guest 1","Temperature of the room is comfortable. Not too cold."],
                    ["Mystery Guest 2","Restroom cleanliness was above average."],
                    ["Mystery Guest 1","I'm scoring the wine service separately."],
                    ["Mystery Guest 2","The bread was warm and served within 3 minutes of seating."],
                    ["Mystery Guest 1","Overall impression so far: 8.5 out of 10."],
                    ["Mystery Guest 2","Let's see how they handle the intentional complaint."],
                ],
                "viewpoints": [
                    "ALERT: Table 30 is a mystery shopper evaluation team. They're scoring everything \u2014 water refills, bread timing, room temperature, restroom cleanliness. This is a formal audit. If management doesn't know, they should.",
                    "They're planning an 'intentional complaint' to test our recovery process. Whatever they throw at us, handle it perfectly. This score likely determines bonuses, raises, or worse.",
                    "8.5 out of 10 so far. That's strong but not perfect. The 1.5 points we're missing are the difference between 'good restaurant' and 'exceptional restaurant.' Find those gaps before they do.",
                    "Mystery shoppers noting that the server remembered their names. That's a trained behavior that's clearly working. But the intentional complaint test is coming \u2014 brief the floor manager NOW.",
                ]
            },
        },
        "pucks": [],  # Generated below
    },
}

# Generate restaurant pucks
_restaurant_sections = {
    "Table 1": "Main Floor", "Table 2": "Main Floor", "Table 3": "Main Floor",
    "Table 4": "Main Floor", "Table 5": "Main Floor", "Table 6": "Main Floor",
    "Table 7": "Family Section", "Table 8": "Family Section", "Table 9": "Family Section",
    "Table 10": "Lounge", "Table 11": "Lounge", "Table 12": "Main Floor",
    "Table 13": "Window Seats", "Table 14": "Window Seats", "Table 15": "Party Room",
    "Table 16": "Party Room", "Table 17": "Chef's Counter", "Table 18": "Patio",
    "Table 19": "Main Floor", "Table 20": "Private Room", "Table 21": "Patio",
    "Table 22": "Wine Room", "Table 23": "Patio", "Table 24": "Staff Area",
    "Table 25": "Staff Area", "Table 26": "Chef's Counter", "Table 27": "Main Floor",
    "Table 28": "Bar", "Table 29": "Main Floor", "Table 30": "Main Floor",
}
_restaurant_colors = [
    "#E74C3C", "#C0392B", "#D4A437", "#E67E22", "#F39C12", "#D35400",
    "#8E44AD", "#9B59B6", "#2ECC71", "#27AE60", "#3498DB", "#2980B9",
    "#1ABC9C", "#16A085", "#E74C3C", "#C0392B", "#F1C40F", "#E67E22",
    "#95A5A6", "#7F8C8D", "#E74C3C", "#D4A437", "#3498DB", "#7F8C8D",
    "#7F8C8D", "#F39C12", "#2ECC71", "#E74C3C", "#D35400", "#1ABC9C",
]
_restaurant_pool = SCENARIOS["restaurant"]["conv_pools"]
_restaurant_pucks = []
for _i in range(1, 31):
    _tname = f"Table {_i}"
    _pool = _restaurant_pool.get(_tname, {})
    _speakers = _pool.get("speakers", [])
    _occ = [{"name": s, "speaking": (_j == 0)} for _j, s in enumerate(_speakers) if s not in ("Server","Waiter","Manager","Sommelier","Host","Busser")]
    _lines = _pool.get("lines", [])
    _trans = [{"time": f"19:{10+_j:02d}", "speaker": l[0], "text": l[1]} for _j, l in enumerate(_lines[:3])]
    _vps = _pool.get("viewpoints", [])
    _restaurant_pucks.append({
        "puck_id": f"aura-rest-table{_i:02d}-314159",
        "puck_name": _tname,
        "owner_name": _restaurant_sections.get(_tname, "Main Floor"),
        "color": _restaurant_colors[_i - 1],
        "ip": f"192.168.2.{100+_i}",
        "status": "listening" if _occ else "idle",
        "uptime": 3600 * 8,
        "memory_usage": 2048,
        "occupants": _occ,
        "transcript": _trans,
        "analysis": _vps[0] if _vps else "Monitoring...",
    })
SCENARIOS["restaurant"]["pucks"] = _restaurant_pucks


def _load_scenario(key: str):
    """Load a scenario's pucks into the registry."""
    scenario = SCENARIOS.get(key)
    if not scenario:
        return
    with _registry_lock:
        # Remove all demo pucks
        demo_ids = [pid for pid in _puck_registry if pid.startswith("aura-")]
        for pid in demo_ids:
            del _puck_registry[pid]
        # Insert new scenario pucks
        for d in scenario["pucks"]:
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
    _load_scenario("ledgerai_hq")
    threading.Thread(target=_demo_heartbeat_loop, daemon=True).start()
    print(f"Farsight Hub starting on port {HUB_PORT}")
    print(f"LLM offload target: {FARSIGHT_LLM_URL}")
    print(f"Dashboard: http://0.0.0.0:{HUB_PORT}/")
    app.run(host="0.0.0.0", port=HUB_PORT, threaded=True)

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
            yield f"event: init\ndata: {json.dumps({'ts': _now_iso()})}\n\n"
            while True:
                try:
                    msg = q.get(timeout=30)
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
<title>Aura Farsight</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #050508;
    --surface: #0a0a12;
    --surface2: #0e0e18;
    --border: #1a1a2a;
    --border-hover: #2a2a3e;
    --text: #c8c8dc;
    --text-dim: #7a7a96;
    --text-bright: #eaeaf6;
    --green: #2ee682;
    --amber: #f0b030;
    --red: #f04040;
    --blue: #3ab8ff;
    --teal: #4ecdc4;
    --gold: #d4b868;
    --gold-bright: #f0dca0;
    --gold-dim: rgba(212,184,104,0.12);
    --sidebar-w: 260px;
    --mono: "JetBrains Mono", "SF Mono", monospace;
    --sans: "Outfit", "SF Pro Display", -apple-system, sans-serif;
  }

  html, body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
    overflow-x: hidden;
  }

  .layout {
    display: flex;
    min-height: 100vh;
  }

  /* ── Left sidebar ───────────────────────────────── */
  .sidebar {
    width: var(--sidebar-w);
    min-height: 100vh;
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 32px 0;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    position: fixed;
    left: 0; top: 0; bottom: 0;
    z-index: 10;
  }

  .sidebar-brand {
    padding: 0 20px;
    margin-bottom: 32px;
    text-align: center;
  }
  .sidebar-brand svg {
    width: 230px;
    height: 110px;
    display: block;
    margin: 0 auto 6px;
  }

  .sidebar-section {
    padding: 0 16px;
    margin-bottom: 28px;
  }
  .sidebar-section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-dim);
    padding: 0 8px;
    margin-bottom: 8px;
  }

  .sidebar-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.15s;
    font-size: 15px;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 2px;
  }
  .sidebar-item:hover { background: rgba(255,255,255,0.03); color: var(--text-bright); }
  .sidebar-item.active { background: var(--gold-dim); color: var(--gold); }
  .sidebar-item .icon {
    font-size: 18px;
    width: 24px;
    text-align: center;
    opacity: 0.7;
  }
  .sidebar-item.active .icon { opacity: 1; }
  .sidebar-item .badge {
    margin-left: auto;
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 6px;
    background: rgba(46,230,130,0.1);
    color: var(--green);
  }

  .sidebar-footer {
    margin-top: auto;
    padding: 0 24px;
    border-top: 1px solid var(--border);
    padding-top: 20px;
  }
  .sidebar-footer .gpu-mini {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text-dim);
    margin-bottom: 6px;
  }
  .sidebar-footer .gpu-mini span { color: var(--green); }
  .sidebar-footer .gpu-bar {
    height: 4px;
    background: rgba(255,255,255,0.04);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 12px;
  }
  .sidebar-footer .gpu-bar-fill {
    height: 100%;
    background: var(--green);
    border-radius: 2px;
    transition: width 1.5s;
    width: 0%;
  }
  .sidebar-footer .version-info {
    font-size: 11px;
    color: var(--text-dim);
    font-family: var(--mono);
  }

  /* ── Main content area ──────────────────────────── */
  .main {
    flex: 1;
    margin-left: var(--sidebar-w);
    position: relative;
    min-height: 100vh;
    background: radial-gradient(ellipse at 50% 0%, rgba(78,205,196,0.03) 0%, transparent 70%);
  }

  .main-content {
    position: relative;
    z-index: 1;
  }

  /* ── Header with logo ──────────────────────────── */
  .header {
    text-align: center;
    padding: 48px 20px 0;
  }
  .header .logo-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 20px;
  }
  .header .logo-wrap img {
    height: 52px;
    width: auto;
    filter: drop-shadow(0 0 30px rgba(78,205,196,0.15));
  }
  .header .farsight-label {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.4em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-top: 2px;
  }
  .header .stats {
    font-size: 15px;
    color: var(--text);
    margin-top: 16px;
    font-family: var(--mono);
    font-weight: 500;
  }

  /* ── Broadcast bar ──────────────────────────────── */
  .broadcast {
    max-width: 680px;
    margin: 24px auto 0;
    display: flex;
    gap: 10px;
    padding: 0 24px;
  }
  .broadcast input {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    color: var(--text-bright);
    font-family: var(--sans);
    font-size: 16px;
    font-weight: 400;
    outline: none;
    transition: border-color 0.3s;
  }
  .broadcast input:focus { border-color: rgba(78,205,196,0.4); }
  .broadcast input::placeholder { color: var(--text-dim); opacity: 0.5; }
  .broadcast button {
    background: rgba(78,205,196,0.08);
    border: 1px solid rgba(78,205,196,0.2);
    border-radius: 12px;
    color: var(--teal);
    font-family: var(--sans);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0 28px;
    cursor: pointer;
    transition: all 0.2s;
  }
  .broadcast button:hover { background: rgba(78,205,196,0.16); }
  .broadcast button.transmitting {
    opacity: 0.45;
    pointer-events: none;
    color: var(--amber);
    border-color: rgba(240,200,60,0.3);
    background: rgba(240,200,60,0.08);
  }

  /* ── Puck cards ─────────────────────────────────── */
  .puck-cards {
    max-width: 1100px;
    margin: 36px auto 0;
    padding: 0 24px 80px;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 20px;
  }

  .puck-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 28px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.4s, box-shadow 0.4s, transform 0.2s;
  }
  .puck-card:hover { transform: translateY(-2px); border-color: var(--border-hover); }
  .puck-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 20px 20px 0 0;
    transition: background 0.5s;
  }
  .puck-card.online  { border-color: rgba(46,230,130,0.15); box-shadow: 0 8px 60px rgba(46,230,130,0.04); }
  .puck-card.online::before  { background: linear-gradient(90deg, transparent, var(--green), transparent); }
  .puck-card.idle    { border-color: rgba(240,176,48,0.15); box-shadow: 0 8px 60px rgba(240,176,48,0.04); }
  .puck-card.idle::before    { background: linear-gradient(90deg, transparent, var(--amber), transparent); }
  .puck-card.offline { border-color: rgba(240,64,64,0.1); opacity: 0.4; }
  .puck-card.offline::before { background: linear-gradient(90deg, transparent, var(--red), transparent); opacity: 0.4; }

  .pc-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 20px;
  }
  .pc-color-dot {
    width: 14px; height: 14px;
    border-radius: 50%;
    flex-shrink: 0;
    box-shadow: 0 0 8px currentColor;
  }
  .pc-name {
    font-size: 24px;
    font-weight: 700;
    color: var(--text-bright);
    flex: 1;
    line-height: 1.15;
  }
  .pc-status {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 5px 12px;
    border-radius: 8px;
  }
  .pc-status.online  { color: var(--green); background: rgba(46,230,130,0.1); }
  .pc-status.idle    { color: var(--amber); background: rgba(240,176,48,0.1); }
  .pc-status.offline { color: var(--red); background: rgba(240,64,64,0.07); }

  .pc-id {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text-dim);
    margin-bottom: 18px;
    letter-spacing: 0.03em;
  }

  .pc-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px 24px;
  }
  .pc-metric {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .pc-metric-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-dim);
  }
  .pc-metric-value {
    font-family: var(--mono);
    font-size: 16px;
    font-weight: 500;
    color: var(--text-bright);
  }

  .pc-ledger {
    margin-top: 20px;
    padding-top: 18px;
    border-top: 1px solid rgba(255,255,255,0.04);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .pc-ledger-label {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-dim);
  }
  .pc-ledger-value {
    font-family: var(--mono);
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(135deg, var(--gold) 0%, var(--gold-bright) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .pc-version {
    margin-top: 14px;
    display: flex;
    justify-content: space-between;
    font-size: 12px;
  }
  .pc-branch { color: var(--teal); font-family: var(--mono); font-weight: 500; }
  .pc-commit { color: var(--text-dim); font-family: var(--mono); }

  .empty-msg {
    text-align: center;
    color: var(--text-dim);
    padding: 100px 20px;
    font-size: 18px;
    letter-spacing: 0.08em;
    font-weight: 400;
    grid-column: 1 / -1;
  }

  /* Broadcast alert animation — cards dim to yellow */
  @keyframes broadcastPulse {
    0%   { border-color: rgba(240,200,60,0.6); box-shadow: 0 0 40px rgba(240,200,60,0.15), inset 0 0 30px rgba(240,200,60,0.04); }
    50%  { border-color: rgba(240,200,60,0.3); box-shadow: 0 0 60px rgba(240,200,60,0.08), inset 0 0 20px rgba(240,200,60,0.02); }
    100% { border-color: rgba(240,200,60,0.0); box-shadow: none; }
  }
  .puck-card.broadcasting {
    animation: broadcastPulse 4s ease-out forwards;
  }
  .puck-card.broadcasting::before {
    background: linear-gradient(90deg, transparent, #f0c83c, transparent) !important;
    opacity: 1 !important;
  }

  @media (max-width: 768px) {
    .sidebar { display: none; }
    .main { margin-left: 0; }
    .header h1 { font-size: 36px; }
    .puck-cards { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<div class="layout">
  <nav class="sidebar">
    <div class="sidebar-brand">
      <svg viewBox="0 0 230 110" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <radialGradient id="hubGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#4ecdc4" stop-opacity="0.35"/>
            <stop offset="100%" stop-color="#4ecdc4" stop-opacity="0"/>
          </radialGradient>
          <radialGradient id="hubCore" cx="40%" cy="35%" r="60%">
            <stop offset="0%" stop-color="#7eddd6"/>
            <stop offset="100%" stop-color="#2a9d96"/>
          </radialGradient>
        </defs>
        <!-- Glow -->
        <circle cx="115" cy="44" r="42" fill="url(#hubGlow)"/>
        <!-- 7 satellites equally spaced: center=(115,44), r=40 -->
        <!-- angles: 0*51.43=-90°, 1*51.43=-38.57°, ... (starting from top) -->
        <!-- node 0: 115+40*cos(-90°), 44+40*sin(-90°) = 115, 4 -->
        <!-- node 1: 115+40*cos(-38.57°), 44+40*sin(-38.57°) = 146.3, 19.1 -->
        <!-- node 2: 115+40*cos(12.86°), 44+40*sin(12.86°) = 154.0, 52.9 -->
        <!-- node 3: 115+40*cos(64.29°), 44+40*sin(64.29°) = 132.3, 80.1 -->
        <!-- node 4: 115+40*cos(115.71°), 44+40*sin(115.71°) = 97.7, 80.1 -->
        <!-- node 5: 115+40*cos(167.14°), 44+40*sin(167.14°) = 76.0, 52.9 -->
        <!-- node 6: 115+40*cos(218.57°), 44+40*sin(218.57°) = 83.7, 19.1 -->
        <!-- Connection lines — bright teal dashes -->
        <line x1="115" y1="44" x2="115.0" y2="4.0"  stroke="#5ee8df" stroke-opacity="0.55" stroke-width="2" stroke-dasharray="5,3"/>
        <line x1="115" y1="44" x2="146.3" y2="19.1" stroke="#5ee8df" stroke-opacity="0.55" stroke-width="2" stroke-dasharray="5,3"/>
        <line x1="115" y1="44" x2="154.0" y2="52.9" stroke="#5ee8df" stroke-opacity="0.55" stroke-width="2" stroke-dasharray="5,3"/>
        <line x1="115" y1="44" x2="132.3" y2="80.1" stroke="#5ee8df" stroke-opacity="0.55" stroke-width="2" stroke-dasharray="5,3"/>
        <line x1="115" y1="44" x2="97.7"  y2="80.1" stroke="#5ee8df" stroke-opacity="0.55" stroke-width="2" stroke-dasharray="5,3"/>
        <line x1="115" y1="44" x2="76.0"  y2="52.9" stroke="#5ee8df" stroke-opacity="0.55" stroke-width="2" stroke-dasharray="5,3"/>
        <line x1="115" y1="44" x2="83.7"  y2="19.1" stroke="#5ee8df" stroke-opacity="0.55" stroke-width="2" stroke-dasharray="5,3"/>
        <!-- Satellite nodes — evenly spaced, matching exec puck colors -->
        <circle cx="115.0" cy="4.0"  r="5" fill="#981225" opacity="0.9"/>
        <circle cx="146.3" cy="19.1" r="5" fill="#23A5FF" opacity="0.9"/>
        <circle cx="154.0" cy="52.9" r="5" fill="#8B5CF6" opacity="0.9"/>
        <circle cx="132.3" cy="80.1" r="5" fill="#F59E0B" opacity="0.9"/>
        <circle cx="97.7"  cy="80.1" r="5" fill="#10B981" opacity="0.9"/>
        <circle cx="76.0"  cy="52.9" r="5" fill="#EC4899" opacity="0.9"/>
        <circle cx="83.7"  cy="19.1" r="5" fill="#4ecdc4" opacity="0.9"/>
        <!-- Orbital ring -->
        <ellipse cx="115" cy="44" rx="28" ry="10" fill="none" stroke="#4ecdc4" stroke-opacity="0.15" stroke-width="0.8" transform="rotate(-10,115,44)"/>
        <!-- Hub core -->
        <circle cx="115" cy="44" r="14" fill="url(#hubCore)" opacity="0.95"/>
        <circle cx="115" cy="44" r="15" fill="none" stroke="#5ee8df" stroke-opacity="0.45" stroke-width="1.2"/>
        <!-- "F" letterform -->
        <text x="115" y="49" text-anchor="middle" fill="#eaeaf6" font-family="Outfit,sans-serif" font-weight="800" font-size="16" opacity="0.9">F</text>
        <!-- Label — larger and brighter -->
        <text x="115" y="106" text-anchor="middle" fill="#b0b0cc" font-family="Outfit,sans-serif" font-weight="700" font-size="15" letter-spacing="4">FARSIGHT HUB</text>
      </svg>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-section-label">Fleet</div>
      <div class="sidebar-item active" onclick="setView('constellation')">
        <span class="icon">&#9678;</span> Constellation
        <span class="badge" id="sb-online">0</span>
      </div>
      <div class="sidebar-item" onclick="setView('fleet')">
        <span class="icon">&#9632;</span> Fleet Overview
      </div>
      <div class="sidebar-item" onclick="setView('activity')">
        <span class="icon">&#9734;</span> Activity Log
      </div>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-section-label">Organization</div>
      <div class="sidebar-item" onclick="setView('users')">
        <span class="icon">&#9775;</span> Users &amp; Roles
      </div>
      <div class="sidebar-item" onclick="setView('billing')">
        <span class="icon">&#9830;</span> $LEDGER Billing
      </div>
      <div class="sidebar-item" onclick="setView('policies')">
        <span class="icon">&#9881;</span> Policies
      </div>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-section-label">System</div>
      <div class="sidebar-item" onclick="setView('updates')">
        <span class="icon">&#8635;</span> OTA Updates
      </div>
      <div class="sidebar-item" onclick="setView('models')">
        <span class="icon">&#9830;</span> Models
      </div>
      <div class="sidebar-item" onclick="setView('settings')">
        <span class="icon">&#9881;</span> Settings
      </div>
    </div>

    <div class="sidebar-footer">
      <div class="gpu-mini" id="gpu-mini">GPU <span>--</span></div>
      <div class="gpu-bar"><div class="gpu-bar-fill" id="gpu-bar-fill"></div></div>
      <div class="version-info">Farsight Hub v1.0</div>
    </div>
  </nav>

  <div class="main">
    <div class="main-content">
      <div class="header">
        <div class="logo-wrap">
          <img src="/logo.png" alt="AURA">
        </div>
        <div class="farsight-label">Farsight Command Center</div>
        <div class="stats" id="stats"></div>
      </div>

      <div class="broadcast">
        <input type="text" id="bc-input" placeholder="Broadcast to all pucks..." maxlength="200">
        <button id="bc-btn" onclick="sendBroadcast()">Transmit</button>
      </div>

      <div class="puck-cards" id="cards"></div>
    </div>
  </div>
</div>

<script>
var pucks = [];
var gpuData = null;
var currentView = "constellation";
var ledgerAccum = {};  // puck_id -> accumulated $LEDGER value
var ledgerStart = Date.now();

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
function fmtTs(iso) {
  if (!iso) return "\u2014";
  try {
    var d = Math.floor((Date.now() - new Date(iso).getTime())/1000);
    if (d < 60) return d+"s ago";
    if (d < 3600) return Math.floor(d/60)+"m ago";
    if (d < 86400) return Math.floor(d/3600)+"h ago";
    return new Date(iso).toLocaleDateString();
  } catch(e) { return iso; }
}
function sc(s) { return s==="offline"?"offline":s==="idle"?"idle":"online"; }
function sl(s) { return s.charAt(0).toUpperCase()+s.slice(1); }
function hex2rgb(h) {
  if(!h||h.length<7) return [100,100,200];
  return [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
}

function sendBroadcast() {
  var inp = document.getElementById("bc-input"), btn = document.getElementById("bc-btn");
  var msg = inp.value.trim();
  if (!msg || btn.classList.contains("transmitting")) return;
  btn.classList.add("transmitting");
  btn.textContent = "TRANSMITTING";
  inp.disabled = true;
  fetch("/broadcast",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({message:msg})}).then(function(r){return r.json();}).then(function(){
    inp.value="";
    // Estimate playback time: ~3s per 20 words of synthesis + playback
    var words = msg.split(/\s+/).length;
    var waitMs = Math.max(8000, words * 150 + 8000);
    setTimeout(function(){
      btn.classList.remove("transmitting");
      btn.textContent = "Transmit";
      inp.disabled = false;
    }, waitMs);
  }).catch(function(){
    btn.classList.remove("transmitting");
    btn.textContent = "Transmit";
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
    document.getElementById("stats").textContent = pucks.length+" puck"+(pucks.length!==1?"s":"")+" connected  \u00b7  "+on+" online";
    document.getElementById("sb-online").textContent = on;
  }).catch(function(){});

  fetch("/gpu").then(function(r){return r.json();}).then(function(d){
    if(d && !d.error){gpuData=d; renderGpu();}
  }).catch(function(){});
}

function renderGpu() {
  if(!gpuData) return;
  var p = gpuData.gpu_util;
  document.getElementById("gpu-mini").innerHTML = "GPU <span>"+p+"%</span>  \u00b7  "+gpuData.temp_c+"\u00b0C";
  document.getElementById("gpu-bar-fill").style.width = p+"%";
  if(p>90) document.getElementById("gpu-bar-fill").style.background = "var(--red)";
  else if(p>70) document.getElementById("gpu-bar-fill").style.background = "var(--amber)";
  else document.getElementById("gpu-bar-fill").style.background = "var(--green)";
}

function renderCards() {
  var el = document.getElementById("cards");
  if(!pucks.length){el.innerHTML='<div class="empty-msg">Waiting for pucks to join the network\u2026</div>';return;}
  var h = "";
  for(var i=0;i<pucks.length;i++){
    var p = pucks[i], s = sc(p.effective_status), col = p.color||"#23A5FF", ver = p.version||{};
    var pid = p.puck_id||"";
    if(!ledgerAccum[pid]) ledgerAccum[pid] = (Math.abs(hashCode(pid)) % 2000 + 500) / 100;
    if(p.effective_status!=="offline") ledgerAccum[pid] += 0.01 + Math.random()*0.04;
    var ledger = ledgerAccum[pid];
    h += '<div class="puck-card '+s+'">'
      +'<div class="pc-header">'
      +'  <div class="pc-color-dot" style="color:'+col+';background:'+col+'"></div>'
      +'  <div class="pc-name">'+esc(p.puck_name||"Unnamed")+'</div>'
      +'  <div class="pc-status '+s+'">'+sl(p.effective_status)+'</div>'
      +'</div>'
      +'<div class="pc-id">'+esc(p.puck_id||"\u2014")+'</div>'
      +'<div class="pc-grid">'
      +'  <div class="pc-metric"><div class="pc-metric-label">Owner</div><div class="pc-metric-value">'+esc(p.owner_name||"\u2014")+'</div></div>'
      +'  <div class="pc-metric"><div class="pc-metric-label">IP Address</div><div class="pc-metric-value">'+esc(p.ip||"\u2014")+'</div></div>'
      +'  <div class="pc-metric"><div class="pc-metric-label">Uptime</div><div class="pc-metric-value">'+fmtUp(p.uptime)+'</div></div>'
      +'  <div class="pc-metric"><div class="pc-metric-label">Memory</div><div class="pc-metric-value">'+fmtMem(p.memory_usage)+'</div></div>'
      +'</div>'
      +'<div class="pc-ledger"><span class="pc-ledger-label">$LEDGER today</span><span class="pc-ledger-value">$'+ledger.toFixed(2)+'</span></div>';
    if(ver.branch||ver.commit){
      h+='<div class="pc-version"><span class="pc-branch">'+esc(ver.branch||"\u2014")+'</span><span class="pc-commit">'+esc(ver.commit||"")+(ver.dirty?" *":"")+'</span></div>';
    }
    h+='</div>';
  }
  el.innerHTML = h;
}

function esc(s){var d=document.createElement("div");d.textContent=s;return d.innerHTML;}
function hashCode(s){var h=0;for(var i=0;i<s.length;i++){h=((h<<5)-h)+s.charCodeAt(i);h|=0;}return h;}

// SSE
var es = new EventSource("/stream");
es.addEventListener("puck_registered",function(){refresh();});
es.addEventListener("status_change",function(){refresh();});
es.addEventListener("broadcast",function(e){
  var inp=document.getElementById("bc-input");
  inp.style.borderColor="rgba(240,200,60,0.6)";
  setTimeout(function(){inp.style.borderColor="";},2000);
  // Trigger yellow dim on all puck cards
  var cards=document.querySelectorAll(".puck-card");
  cards.forEach(function(c){
    c.classList.remove("broadcasting");
    void c.offsetWidth; // force reflow
    c.classList.add("broadcasting");
  });
  setTimeout(function(){
    cards.forEach(function(c){c.classList.remove("broadcasting");});
  },4500);
});

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

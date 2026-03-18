"""
services.health -- Container health checks + auto-start.

Pings Whisper, LLM, and Memory containers on startup.
Ensures Docker Compose services are running before polling health.
"""

from __future__ import annotations

import os
import subprocess
import time
import urllib.request

from core.config import WHISPER_URL, LLM_URL, MEMORY_URL, WORKSPACE_ROOT

_SERVICE_URLS = {
    "whisper": WHISPER_URL,
    "llm":     LLM_URL,
    "memory":  MEMORY_URL,
}


def _ping(url: str, timeout: float = 2.0) -> bool:
    try:
        r = urllib.request.urlopen(f"{url}/health", timeout=timeout)
        return r.status == 200
    except Exception:
        return False


def check_service(name: str) -> bool:
    """Non-blocking health check for a single service by name.

    Returns True if the service responds to /health, False otherwise.
    """
    url = _SERVICE_URLS.get(name)
    if url is None:
        return False
    return _ping(url)


def _stop_stale_chatterbox() -> None:
    """Stop the chatterbox-tts container if running (speaker.py uses in-process TTS).

    This reclaims ~2.9 GB of RAM that would otherwise be wasted.
    """
    compose_dir = WORKSPACE_ROOT / "setup"
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--status", "running", "--format", "{{.Name}}"],
            cwd=str(compose_dir),
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.strip().splitlines():
            if "chatterbox" in line.lower():
                print(f"[health] Stopping stale chatterbox container ({line}) — speaker uses in-process TTS")
                subprocess.run(
                    ["docker", "compose", "stop", "chatterbox-tts"],
                    cwd=str(compose_dir),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=30,
                )
                print("[health] Chatterbox container stopped (~2.9 GB reclaimed)")
                return
    except Exception:
        pass


def _kill_stale_port(port: int) -> None:
    """Kill any process holding *port* that isn't responding to health checks."""
    try:
        out = subprocess.check_output(
            ["fuser", f"{port}/tcp"], stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()
        for pid in out.split():
            pid = pid.strip()
            if pid.isdigit():
                os.kill(int(pid), 9)
                print(f"[health] Killed stale process {pid} on port {port}")
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass  # nothing on the port — fine


def _kill_competing_llm_processes() -> None:
    """Kill any duplicate container_rest.py processes competing for GPU memory.

    On Jetson with 16GB unified RAM, stale Docker or native LLM processes that
    didn't shut down cleanly will hold GPU memory and starve the active LLM.
    This keeps only the newest container_rest.py process alive.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-af", "container_rest.py"],
            capture_output=True, text=True, timeout=5,
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if len(lines) <= 1:
            return  # 0 or 1 process — nothing to clean up

        # Parse PIDs, keep the newest (highest PID), kill the rest
        pids = []
        for line in lines:
            parts = line.split(None, 1)
            if parts and parts[0].isdigit():
                pids.append(int(parts[0]))
        if len(pids) <= 1:
            return

        pids.sort()
        stale = pids[:-1]  # all except newest
        for pid in stale:
            try:
                os.kill(pid, 9)
                print(f"[health] Killed competing LLM process PID {pid}")
            except OSError:
                pass
        if stale:
            print(f"[health] Cleaned up {len(stale)} stale LLM process(es)")
    except Exception:
        pass


def _ensure_native_llm() -> None:
    """Start the native LLM server if not already running.

    The LLM runs as a native Flask process (not Docker) for direct GPU access.
    Checks if port 11434 is already responding before launching.
    Kills stale processes holding the port if the health check fails.
    """
    # Always clean up competing LLM processes first
    _kill_competing_llm_processes()

    if _ping(LLM_URL):
        print("[health] Native LLM already running")
        return

    # Kill any stale process squatting on the port
    _kill_stale_port(11434)

    # Also stop any Docker LLM container that might conflict
    for name in ("setup-llm-generic-1", "setup-llm-medical-1"):
        try:
            subprocess.run(
                ["docker", "stop", name],
                capture_output=True, timeout=10,
            )
        except Exception:
            pass

    script = WORKSPACE_ROOT / "run_llm_native.sh"
    if not script.exists():
        print(f"[health] WARNING: {script} not found, LLM will not start")
        return

    print("[health] Starting native LLM server...")
    try:
        subprocess.Popen(
            ["bash", str(script)],
            stdout=open("/tmp/aura-llm.log", "a"),
            stderr=subprocess.STDOUT,
            cwd=str(WORKSPACE_ROOT),
            start_new_session=True,
        )
        print("[health] Native LLM server launched (log: /tmp/aura-llm.log)")
    except Exception as e:
        print(f"[health] Failed to start native LLM: {e}")


def ensure_containers() -> None:
    """Start Docker Compose services and native LLM if not already running.

    Runs `docker compose up -d whisper memory` from the setup/ directory.
    LLM runs natively (not in Docker) for better GPU performance on Jetson.
    Also stops the chatterbox-tts container if it was left running
    (speaker.py loads ChatterboxTTS in-process — the container is dead weight).
    Non-fatal: logs errors but does not raise.
    """
    compose_dir = WORKSPACE_ROOT / "setup"

    # Stop duplicate chatterbox container if lingering from previous runs
    _stop_stale_chatterbox()

    # Start native LLM (no Docker)
    _ensure_native_llm()

    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--status", "running", "-q"],
            cwd=str(compose_dir),
            capture_output=True, text=True, timeout=10,
        )
        running = [l for l in result.stdout.strip().splitlines() if l.strip()]
        if len(running) >= 2:
            print(f"[health] Containers already running ({len(running)} up)")
            return
    except Exception:
        pass

    print("[health] Starting containers (docker compose up -d whisper memory)...")
    try:
        subprocess.run(
            ["docker", "compose", "up", "-d", "whisper", "memory"],
            cwd=str(compose_dir),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=60,
        )
        print("[health] Containers started (whisper + memory)")
    except Exception as e:
        print(f"[health] Failed to start containers: {e}")


def wait_for_containers(timeout: float = 30.0) -> dict:
    """Wait up to *timeout* seconds for containers to respond.

    Returns dict of {name: bool} indicating which responded.
    Non-blocking: returns immediately with whatever is available.
    """
    services = dict(_SERVICE_URLS)
    results = {name: False for name in services}
    deadline = time.time() + timeout

    while time.time() < deadline:
        all_up = True
        for name, url in services.items():
            if not results[name]:
                results[name] = _ping(url)
                if not results[name]:
                    all_up = False
        if all_up:
            break
        time.sleep(1.0)

    for name, ok in results.items():
        status = "UP" if ok else "DOWN"
        print(f"[health] {name}: {status}")
    return results

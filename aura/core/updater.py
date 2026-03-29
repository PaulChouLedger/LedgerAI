"""
core.updater -- OTA update checker.

Background thread that periodically runs ``git fetch`` and compares the
local HEAD with the remote.  When new commits are available it emits
``updates.available`` on the bus so the Settings complication can flash
an alert.  The user can then review commits and choose to apply the
update (git pull + automatic restart via systemd).

Usage:
    from core.updater import updater
    updater.start()                # call once at boot
    updater.commits                # list of pending commits
    updater.available              # bool
    updater.apply_update()         # pull + restart
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional

from core.bus import bus
from core.config import WORKSPACE_ROOT


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECK_INTERVAL_S = float(os.environ.get("AURA_UPDATE_INTERVAL", "60"))
GIT_REMOTE       = os.environ.get("AURA_UPDATE_REMOTE", "origin")
GIT_BRANCH       = os.environ.get("AURA_UPDATE_BRANCH", "rafael_2.0")
GIT_TIMEOUT_S    = 30


# ---------------------------------------------------------------------------
# Updater
# ---------------------------------------------------------------------------

class _Updater:
    """Singleton OTA update checker."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._commits: List[Dict[str, str]] = []
        self._available = False
        self._applying = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._available

    @property
    def commits(self) -> List[Dict[str, str]]:
        """List of dicts: {hash, subject, author, date}."""
        return list(self._commits)

    @property
    def applying(self) -> bool:
        return self._applying

    def start(self) -> None:
        """Start background polling thread."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="updater", daemon=True,
        )
        self._thread.start()
        print(f"[updater] started — checking {GIT_REMOTE}/{GIT_BRANCH}"
              f" every {int(CHECK_INTERVAL_S)}s", flush=True)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def check_now(self) -> None:
        """Force an immediate check (runs in caller's thread)."""
        self._check_once()

    def apply_update(self) -> bool:
        """Pull latest and restart the service.

        Returns True if the pull succeeded.  The process will exit
        shortly after (systemd restarts it with new code).
        """
        if self._applying:
            return False
        self._applying = True
        bus.emit("updates.applying")

        try:
            result = subprocess.run(
                ["git", "pull", "--ff-only", GIT_REMOTE, GIT_BRANCH],
                cwd=str(WORKSPACE_ROOT),
                capture_output=True, text=True,
                timeout=GIT_TIMEOUT_S,
            )
            if result.returncode != 0:
                print(f"[updater] pull failed: {result.stderr.strip()}")
                bus.emit("updates.failed", error=result.stderr.strip())
                self._applying = False
                return False

            print(f"[updater] pull succeeded: {result.stdout.strip()}")

            # Sync systemd service file (copy from repo → /etc/systemd/system)
            # Symlinks break after reboot ("Link has been severed"), so we copy.
            svc_src = str(WORKSPACE_ROOT / "aura" / "services" / "aura-v2.service")
            svc_dst = "/etc/systemd/system/aura4.service"
            try:
                subprocess.run(
                    ["sudo", "cp", svc_src, svc_dst],
                    timeout=5, capture_output=True,
                )
                subprocess.run(
                    ["sudo", "systemctl", "daemon-reload"],
                    timeout=5, capture_output=True,
                )
                print("[updater] service file synced + daemon-reload")
            except Exception as e:
                print(f"[updater] service file sync failed (non-fatal): {e}")

            bus.emit("updates.applied")

            # Give the bus event a moment to propagate, then restart.
            # systemd Restart=always: use os._exit(42) to trigger restart.
            # sys.exit(0) would be a clean exit that systemd won't restart.
            def _delayed_exit():
                time.sleep(1.5)
                print("[updater] restarting service...", flush=True)
                os._exit(42)  # non-zero → systemd restarts us

            threading.Thread(target=_delayed_exit, daemon=True).start()
            return True

        except subprocess.TimeoutExpired:
            print("[updater] pull timed out")
            bus.emit("updates.failed", error="git pull timed out")
            self._applying = False
            return False
        except Exception as exc:
            print(f"[updater] pull error: {exc}")
            bus.emit("updates.failed", error=str(exc))
            self._applying = False
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        # Initial delay — let the system boot first
        print("[updater] waiting 30s before first check", flush=True)
        self._stop.wait(30)
        while not self._stop.is_set():
            try:
                print("[updater] checking for updates...", flush=True)
                self._check_once()
            except Exception as exc:
                print(f"[updater] check error: {exc}", flush=True)
            self._stop.wait(CHECK_INTERVAL_S)

    def _check_once(self) -> None:
        cwd = str(WORKSPACE_ROOT)

        # Fetch latest from remote
        try:
            result = subprocess.run(
                ["git", "fetch", GIT_REMOTE],
                cwd=cwd, capture_output=True, text=True,
                timeout=GIT_TIMEOUT_S,
            )
            if result.returncode != 0:
                print(f"[updater] fetch failed: {result.stderr.strip()}", flush=True)
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            print(f"[updater] fetch exception: {exc}", flush=True)
            return

        # Compare local HEAD vs remote
        try:
            local = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd, capture_output=True, text=True, timeout=5,
            ).stdout.strip()

            remote = subprocess.run(
                ["git", "rev-parse", f"{GIT_REMOTE}/{GIT_BRANCH}"],
                cwd=cwd, capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return

        if local == remote:
            if self._available:
                self._available = False
                self._commits = []
                bus.emit("updates.none")
            return

        # New commits available — get the log
        try:
            log = subprocess.run(
                ["git", "log", f"HEAD..{GIT_REMOTE}/{GIT_BRANCH}",
                 "--format=%H|%s|%an|%ai", "--no-merges"],
                cwd=cwd, capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return

        commits = []
        for line in log.splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                commits.append({
                    "hash": parts[0][:8],
                    "subject": parts[1],
                    "author": parts[2],
                    "date": parts[3][:10],
                })

        if commits:
            self._commits = commits
            self._available = True
            bus.emit("updates.available",
                     count=len(commits), commits=commits)
            print(f"[updater] {len(commits)} update(s) available", flush=True)


# Module-level singleton
updater = _Updater()

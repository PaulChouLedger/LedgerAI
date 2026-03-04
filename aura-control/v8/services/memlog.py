"""
services.memlog -- Lightweight memory profiler for Aura.

Logs process RSS, GPU memory, and system-wide RAM at key milestones.
Designed for Jetson (unified memory) — reads from /proc and tegrastats.

Usage:
    from services.memlog import memlog
    memlog("boot start")          # logs a checkpoint
    memlog("tts loaded")          # another checkpoint
    memlog.summary()              # prints a table of all checkpoints
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Optional


def _read_proc_status(pid: Optional[int] = None) -> dict:
    """Read /proc/<pid>/status for memory fields."""
    pid = pid or os.getpid()
    result = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmSize:", "VmSwap:", "VmHWM:")):
                    parts = line.split()
                    # Convert kB to MB
                    result[parts[0].rstrip(":")] = int(parts[1]) / 1024
    except Exception:
        pass
    return result


def _read_meminfo() -> dict:
    """Read /proc/meminfo for system-wide memory."""
    result = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                for key in ("MemTotal", "MemAvailable", "MemFree", "SwapTotal",
                            "SwapFree", "Buffers", "Cached"):
                    if line.startswith(key + ":"):
                        parts = line.split()
                        result[key] = int(parts[1]) / 1024  # kB → MB
    except Exception:
        pass
    return result


def _docker_mem() -> dict:
    """Get memory usage of running Docker containers."""
    result = {}
    try:
        out = subprocess.check_output(
            ["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.MemUsage}}"],
            text=True, timeout=5, stderr=subprocess.DEVNULL,
        )
        for line in out.strip().splitlines():
            parts = line.split("\t")
            if len(parts) == 2:
                name = parts[0].strip()
                usage = parts[1].split("/")[0].strip()
                # Parse "420MiB" or "2.86GiB"
                if "GiB" in usage:
                    mb = float(usage.replace("GiB", "").strip()) * 1024
                elif "MiB" in usage:
                    mb = float(usage.replace("MiB", "").strip())
                elif "KiB" in usage:
                    mb = float(usage.replace("KiB", "").strip()) / 1024
                else:
                    mb = 0.0
                result[name] = mb
    except Exception:
        pass
    return result


class MemLog:
    """Lightweight memory checkpoint logger."""

    def __init__(self):
        self._checkpoints: list[dict] = []
        self._t0 = time.monotonic()

    def __call__(self, label: str, include_docker: bool = False) -> None:
        """Log a memory checkpoint with the given label."""
        elapsed = time.monotonic() - self._t0
        proc = _read_proc_status()
        sysinfo = _read_meminfo()

        rss = proc.get("VmRSS", 0)
        hwm = proc.get("VmHWM", 0)
        swap = proc.get("VmSwap", 0)
        sys_total = sysinfo.get("MemTotal", 0)
        sys_avail = sysinfo.get("MemAvailable", 0)
        sys_used = sys_total - sys_avail if sys_total else 0

        cp = {
            "label": label,
            "elapsed": elapsed,
            "rss_mb": rss,
            "hwm_mb": hwm,
            "swap_mb": swap,
            "sys_used_mb": sys_used,
            "sys_avail_mb": sys_avail,
            "sys_total_mb": sys_total,
        }

        if include_docker:
            cp["docker"] = _docker_mem()

        self._checkpoints.append(cp)

        # Compact one-line log
        docker_str = ""
        if include_docker and cp.get("docker"):
            parts = [f"{n}={m:.0f}M" for n, m in cp["docker"].items()]
            docker_str = f"  docker=[{', '.join(parts)}]"

        print(
            f"[memlog] {label:30s}  "
            f"rss={rss:7.0f}M  hwm={hwm:7.0f}M  swap={swap:5.0f}M  "
            f"sys={sys_used:.0f}/{sys_total:.0f}M (avail={sys_avail:.0f}M)"
            f"{docker_str}"
            f"  +{elapsed:.1f}s"
        )

    def delta(self, label: str) -> None:
        """Log a checkpoint and print delta from previous."""
        prev_rss = self._checkpoints[-1]["rss_mb"] if self._checkpoints else 0
        self(label)
        curr = self._checkpoints[-1]
        diff = curr["rss_mb"] - prev_rss
        if abs(diff) > 1:
            sign = "+" if diff > 0 else ""
            print(f"[memlog]   ^ delta rss: {sign}{diff:.0f}M from previous checkpoint")

    def summary(self) -> None:
        """Print a summary table of all checkpoints."""
        if not self._checkpoints:
            print("[memlog] No checkpoints recorded")
            return

        print("\n[memlog] ═══ Memory Profile Summary ═══")
        print(f"{'#':>3}  {'Elapsed':>7}  {'Label':30s}  {'RSS':>7}  {'HWM':>7}  {'Swap':>6}  {'SysUsed':>8}  {'SysAvail':>8}")
        print("─" * 105)

        for i, cp in enumerate(self._checkpoints):
            print(
                f"{i+1:3d}  {cp['elapsed']:6.1f}s  {cp['label']:30s}  "
                f"{cp['rss_mb']:6.0f}M  {cp['hwm_mb']:6.0f}M  {cp['swap_mb']:5.0f}M  "
                f"{cp['sys_used_mb']:7.0f}M  {cp['sys_avail_mb']:7.0f}M"
            )

        # Totals
        first = self._checkpoints[0]
        last = self._checkpoints[-1]
        rss_growth = last["rss_mb"] - first["rss_mb"]
        sys_growth = last["sys_used_mb"] - first["sys_used_mb"]
        print("─" * 105)
        print(f"     Process RSS growth: {rss_growth:+.0f} MB over {last['elapsed']:.1f}s")
        print(f"     System RAM growth:  {sys_growth:+.0f} MB")
        print(f"     Peak RSS (HWM):     {last['hwm_mb']:.0f} MB")
        print()


# Singleton
memlog = MemLog()

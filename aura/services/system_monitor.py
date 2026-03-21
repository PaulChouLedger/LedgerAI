"""
services.system_monitor -- Background system metrics daemon.

Polls Jetson hardware sensors every 10 seconds and emits bus events
for live GUI updates (alerts, gauges, service health).

All hardware reads are wrapped in try/except so this never crashes,
even on non-Jetson dev machines.

Health checks use short timeouts and run on a staggered schedule
to avoid blocking the Jetson's limited CPU.
"""

from __future__ import annotations

import glob
import threading
import time
import urllib.request

from core.bus import bus

# ── Severity constants (match alerts.py) ─────────────────────
_SEV_INFO = 0
_SEV_WARN = 1
_SEV_CRIT = 2

# ── INA3221 sensor paths ─────────────────────────────────────
_HWMON = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1"
_VDD_V = f"{_HWMON}/in1_input"   # mV
_VDD_I = f"{_HWMON}/curr1_input"  # mA

# ── Service health endpoints ─────────────────────────────────
_SERVICES = {
    "whisper": "http://localhost:5000/health",
    "llm":     "http://localhost:11434/health",
    "memory":  "http://localhost:11438/health",
}

_FARSIGHT_URL = "http://100.76.191.92:11435/health"

# ── Intervals ────────────────────────────────────────────────
_SENSOR_INTERVAL = 10.0    # sensor reads every 10s (lightweight)
_HEALTH_INTERVAL = 30.0    # HTTP health checks every 30s (expensive)


class SystemMonitor:
    """Daemon thread that polls hardware sensors and emits bus events."""

    def __init__(self):
        self._stop = threading.Event()
        self._thread = None
        self._thermal_paths = {}  # cached: {"GPU-therm": "/sys/.../temp", ...}
        # Cached health state (updated on slower cadence)
        self._services = {"whisper": False, "llm": False, "memory": False}
        self._farsight_ok = False
        self._last_health_check = 0.0

    def start(self):
        self._discover_thermal_zones()
        self._thread = threading.Thread(target=self._poll_loop,
                                        name="system-monitor", daemon=True)
        self._thread.start()
        print("[system_monitor] started", flush=True)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=8)

    # ── Thermal zone discovery ───────────────────────────────
    def _discover_thermal_zones(self):
        """Find GPU-therm and CPU-therm sysfs paths once at startup."""
        try:
            for zone in glob.glob("/sys/devices/virtual/thermal/thermal_zone*"):
                try:
                    name = open(f"{zone}/type").read().strip()
                    if name in ("GPU-therm", "CPU-therm"):
                        self._thermal_paths[name] = f"{zone}/temp"
                except Exception:
                    pass
        except Exception:
            pass

    # ── Sensor reads (all file-based, fast) ──────────────────
    def _read_ram(self):
        """Return (used_gb, total_gb, pct 0-100)."""
        try:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split()
                    if parts[0] in ("MemTotal:", "MemAvailable:"):
                        info[parts[0]] = int(parts[1])  # kB
            total = info.get("MemTotal:", 0)
            avail = info.get("MemAvailable:", 0)
            used = total - avail
            total_gb = total / 1048576.0
            used_gb = used / 1048576.0
            pct = (used / total * 100.0) if total else 0
            return used_gb, total_gb, pct
        except Exception:
            return 0, 0, 0

    def _read_temp(self, zone_name):
        """Return temperature in °C for a named thermal zone."""
        path = self._thermal_paths.get(zone_name)
        if not path:
            return 0.0
        try:
            return int(open(path).read().strip()) / 1000.0
        except Exception:
            return 0.0

    def _read_gpu_load(self):
        """Return GPU usage 0-100."""
        try:
            val = int(open("/sys/devices/gpu.0/load").read().strip())
            return val / 10.0  # sysfs reports 0-1000
        except Exception:
            return 0.0

    def _read_power(self):
        """Return power in watts from INA3221."""
        try:
            v = int(open(_VDD_V).read().strip()) / 1000.0  # mV→V
            i = int(open(_VDD_I).read().strip()) / 1000.0  # mA→A
            return v * i
        except Exception:
            return 0.0

    def _read_uptime(self):
        """Return human-readable uptime string."""
        try:
            secs = float(open("/proc/uptime").read().split()[0])
            days = int(secs // 86400)
            hours = int((secs % 86400) // 3600)
            mins = int((secs % 3600) // 60)
            if days > 0:
                return f"{days}d {hours}h"
            elif hours > 0:
                return f"{hours}h {mins}m"
            else:
                return f"{mins}m"
        except Exception:
            return "?"

    def _check_service(self, url, timeout=1.0):
        """HTTP GET with short timeout, return True if 200."""
        try:
            r = urllib.request.urlopen(url, timeout=timeout)
            return r.status == 200
        except Exception:
            return False

    # ── Health checks (expensive, run less frequently) ───────
    def _maybe_check_health(self):
        """Run HTTP health checks only every _HEALTH_INTERVAL seconds."""
        now = time.time()
        if (now - self._last_health_check) < _HEALTH_INTERVAL:
            return  # use cached values

        self._last_health_check = now

        # Local services — 1s timeout each, sequential but fast for localhost
        for name, url in _SERVICES.items():
            if self._stop.is_set():
                return
            self._services[name] = self._check_service(url, timeout=1.0)

        # Farsight — remote, use 1.5s timeout, skip if stop requested
        if not self._stop.is_set():
            self._farsight_ok = self._check_service(_FARSIGHT_URL, timeout=1.5)

    # ── Main loop ────────────────────────────────────────────
    def _poll_loop(self):
        # Initial delay — let boot complete before hammering sensors
        self._stop.wait(15.0)

        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception as exc:
                print(f"[system_monitor] poll error: {exc}", flush=True)
            self._stop.wait(_SENSOR_INTERVAL)

    def _poll_once(self):
        # Fast sensor reads (file-based, microseconds)
        ram_used, ram_total, ram_pct = self._read_ram()
        gpu_temp = self._read_temp("GPU-therm")
        cpu_temp = self._read_temp("CPU-therm")
        gpu_pct = self._read_gpu_load()
        power_w = self._read_power()
        uptime_str = self._read_uptime()

        # Slow health checks (HTTP, only every 30s)
        self._maybe_check_health()

        # Emit metrics (uses cached health state)
        bus.emit("system.metrics",
                 gpu_pct=gpu_pct, gpu_temp=gpu_temp, cpu_temp=cpu_temp,
                 ram_pct=ram_pct, ram_used_gb=ram_used, ram_total_gb=ram_total,
                 power_w=power_w, uptime_str=uptime_str,
                 services=dict(self._services), farsight_ok=self._farsight_ok)

        # Build alerts
        alerts = []

        if gpu_temp > 85:
            alerts.append({"msg": f"GPU thermal warning — {gpu_temp:.0f}°C",
                           "sev": _SEV_CRIT, "ago": "now"})
        elif gpu_temp > 75:
            alerts.append({"msg": f"GPU temp elevated — {gpu_temp:.0f}°C",
                           "sev": _SEV_WARN, "ago": "now"})

        if ram_pct > 90:
            alerts.append({"msg": f"Memory pressure — {ram_pct:.0f}% used",
                           "sev": _SEV_WARN, "ago": "now"})

        for name, ok in self._services.items():
            if not ok:
                alerts.append({"msg": f"{name.capitalize()} service unreachable",
                               "sev": _SEV_WARN, "ago": "now"})

        if self._farsight_ok:
            alerts.append({"msg": "Farsight uplink active",
                           "sev": _SEV_INFO, "ago": "now"})
        else:
            alerts.append({"msg": "Farsight uplink offline",
                           "sev": _SEV_INFO, "ago": "now"})

        # Nominal status
        if not any(a["sev"] >= _SEV_WARN for a in alerts):
            alerts.append({"msg": f"All systems nominal — uptime {uptime_str}",
                           "sev": _SEV_INFO, "ago": "now"})

        bus.emit("alerts.update", alerts=alerts[:5])

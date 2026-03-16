#!/usr/bin/env python3
"""
Adaptive Power Manager — auto-detect wall vs battery power and switch
nvpmodel modes to prevent brownout crashes.

Strategy:
  1. Boot in 25W mode (safe for any power source)
  2. Run a brief GPU stress probe (~3 seconds)
  3. If VDD_IN voltage stays stable under load → wall power → switch to MAXN
  4. If voltage sags or power can't sustain → battery → stay at 25W
  5. Continue monitoring; if voltage drops during operation, downshift

The Orin NX 16GB INA3221 sensor provides:
  - VDD_IN voltage/current (module's 5V rail, post-regulator)
  - We compute instantaneous power from V*I
  - Battery (36W cap) will show instability above ~28W module draw
  - Wall (60W cap) sustains 40W+ without sag

Runs as a systemd service or background thread within Aura.

Usage:
    python3 power_manager.py              # run standalone daemon
    python3 power_manager.py --probe-only # detect and print, don't loop
"""

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("power_mgr")

# ── Sensor paths ─────────────────────────────────────────────
HWMON = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1"
VDD_IN_V = f"{HWMON}/in1_input"      # mV
VDD_IN_I = f"{HWMON}/curr1_input"    # mA

# ── Thresholds ───────────────────────────────────────────────
# Voltage sag thresholds (module's 5V rail, normally ~5.04V)
VOLTAGE_NOMINAL = 5.04    # V — healthy supply
VOLTAGE_SAG_WARN = 4.85   # V — supply is straining
VOLTAGE_SAG_CRIT = 4.70   # V — about to brownout, downshift immediately

# Power thresholds
POWER_BATTERY_CEILING = 28.0   # W — if we hit this, we're near 36W system limit
POWER_WALL_PROOF = 32.0        # W — if supply sustains this, it's wall power

# Probe stress duration — long enough for power to stabilize
PROBE_DURATION_S = 8.0
PROBE_SAMPLES = 30  # sample every ~270ms during probe

# Monitor interval
MONITOR_INTERVAL_S = 2.0

# Hysteresis: must see stable readings for N cycles before upshifting
UPSHIFT_STABLE_CYCLES = 30   # ~60 seconds of stable voltage at low load
DOWNSHIFT_IMMEDIATE = True    # downshift immediately on sag


@dataclass
class PowerReading:
    voltage: float  # V
    current: float  # A
    power: float    # W
    timestamp: float


# ── Sensor reads ─────────────────────────────────────────────
def read_power() -> Optional[PowerReading]:
    """Read VDD_IN voltage and current from INA3221."""
    try:
        v = int(open(VDD_IN_V).read()) / 1000.0  # mV → V
        i = int(open(VDD_IN_I).read()) / 1000.0  # mA → A
        return PowerReading(
            voltage=v, current=i, power=v * i, timestamp=time.time()
        )
    except (FileNotFoundError, ValueError, PermissionError) as e:
        log.warning(f"Cannot read power sensor: {e}")
        return None


def get_current_mode() -> str:
    """Get current nvpmodel mode name."""
    try:
        result = subprocess.run(
            ["sudo", "nvpmodel", "-q"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "NV Power Mode" in line:
                return line.split(":")[-1].strip()
    except Exception:
        pass
    return "UNKNOWN"


def set_power_mode(mode_id: int, mode_name: str) -> bool:
    """Switch nvpmodel power mode."""
    current = get_current_mode()
    if current == mode_name:
        log.debug(f"Already in {mode_name} mode")
        return True

    log.info(f"Switching power mode: {current} → {mode_name} (id={mode_id})")
    try:
        result = subprocess.run(
            ["sudo", "nvpmodel", "-m", str(mode_id)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            log.info(f"Power mode set to {mode_name}")
            # Also apply jetson_clocks for MAXN to get full performance
            if mode_id == 0:
                subprocess.run(
                    ["sudo", "jetson_clocks"],
                    capture_output=True, timeout=10
                )
            return True
        else:
            log.error(f"nvpmodel failed: {result.stderr}")
            return False
    except Exception as e:
        log.error(f"Failed to set power mode: {e}")
        return False


# ── GPU stress for probing ───────────────────────────────────
def _gpu_stress_probe(duration: float) -> list[PowerReading]:
    """Run a heavy GPU workload and collect power readings.

    Uses large matrix multiplies (4096x4096 fp16) to push GPU power to
    near the 25W mode cap. A 36W battery will show voltage sag at this
    draw level; a 60W wall supply will not.
    """
    readings = []
    interval = duration / PROBE_SAMPLES

    # Use the aura venv python which has torch+CUDA
    VENV_PYTHON = "/home/ledger/aura-env/bin/python3"
    _python = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

    # Heavy GPU stress — large matmuls that actually push power draw
    gpu_proc = subprocess.Popen(
        [_python, "-c", f"""
import time
try:
    import torch
    if torch.cuda.is_available():
        d = torch.device('cuda')
        # Pre-allocate large tensors to spike memory + compute
        a = torch.randn(4096, 4096, device=d, dtype=torch.float16)
        start = time.time()
        while time.time() - start < {duration + 3}:
            b = torch.matmul(a, a)
            torch.cuda.synchronize()
except Exception:
    time.sleep({duration + 3})
"""],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # Simultaneous CPU stress (all cores) to simulate real workload
    cpu_proc = subprocess.Popen(
        [_python, "-c", f"""
import time, os, multiprocessing
os.environ['OPENBLAS_NUM_THREADS'] = '8'
def burn(secs):
    import numpy as np
    end = time.time() + secs
    while time.time() < end:
        a = np.random.randn(1500, 1500).astype('float32')
        np.matmul(a, a)
procs = [multiprocessing.Process(target=burn, args=({duration + 3},))
         for _ in range(4)]
for p in procs: p.start()
for p in procs: p.join()
"""],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    time.sleep(1.5)  # let stress ramp up fully

    for _ in range(PROBE_SAMPLES):
        r = read_power()
        if r:
            readings.append(r)
        time.sleep(interval)

    # Clean up
    for proc in [gpu_proc, cpu_proc]:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    return readings


# ── Power source detection ───────────────────────────────────
def detect_power_source() -> str:
    """
    Detect whether we're on wall power or battery.

    Returns: "wall", "battery", or "unknown"
    """
    log.info("Probing power source...")

    # Take baseline reading
    baseline = read_power()
    if not baseline:
        log.warning("Cannot read sensors — assuming battery (safe default)")
        return "battery"

    log.info(f"Baseline: {baseline.voltage:.3f}V, {baseline.power:.2f}W")

    # Run stress probe
    readings = _gpu_stress_probe(PROBE_DURATION_S)
    if not readings:
        log.warning("No probe readings — assuming battery")
        return "battery"

    # Analyze readings
    voltages = [r.voltage for r in readings]
    powers = [r.power for r in readings]
    min_v = min(voltages)
    max_p = max(powers)
    avg_v = sum(voltages) / len(voltages)

    log.info(f"Probe results: V_min={min_v:.3f}V, V_avg={avg_v:.3f}V, "
             f"P_max={max_p:.2f}W ({len(readings)} samples)")

    # Decision logic
    if min_v < VOLTAGE_SAG_CRIT:
        log.info(f"CRITICAL voltage sag detected ({min_v:.3f}V) → BATTERY")
        return "battery"

    if min_v < VOLTAGE_SAG_WARN:
        log.info(f"Voltage sag under load ({min_v:.3f}V) → BATTERY")
        return "battery"

    # Voltage stability is the primary signal. In 25W mode the GPU is
    # capped at 408MHz so power draw only reaches ~16W even on wall.
    # But wall power keeps voltage rock-steady (<0.03V range) while
    # battery shows more variation under sustained load.
    voltage_range = max(voltages) - min_v
    log.info(f"Voltage range: {voltage_range:.4f}V (max-min)")

    # High power without sag = definitely wall
    if max_p > POWER_WALL_PROOF:
        log.info(f"Sustained high power ({max_p:.1f}W) without sag → WALL POWER")
        return "wall"

    # Voltage rock-solid + near nominal = wall power
    # (even if power draw is modest due to 25W mode GPU cap)
    if voltage_range < 0.03 and avg_v > (VOLTAGE_NOMINAL - 0.15) and max_p > 10:
        log.info(f"Voltage rock-solid ({voltage_range:.4f}V range), "
                 f"near nominal ({avg_v:.3f}V), power {max_p:.1f}W → WALL POWER")
        return "wall"

    # Voltage instability under load = battery
    if voltage_range > 0.08:
        log.info(f"Voltage instability ({voltage_range:.3f}V range) → BATTERY")
        return "battery"

    # Very low power even under stress = supply is current-limited (battery)
    # In 25W mode, wall power reaches 14-17W; battery may be lower
    if max_p < 10.0:
        log.info(f"Very low peak power under stress ({max_p:.1f}W) — "
                 f"supply appears severely current-limited → BATTERY")
        return "battery"

    # Moderate voltage range (0.03-0.08V) with decent power = likely wall
    if voltage_range < 0.05 and avg_v > (VOLTAGE_NOMINAL - 0.15) and max_p > 13:
        log.info(f"Voltage stable ({voltage_range:.4f}V range), "
                 f"adequate power ({max_p:.1f}W) → WALL POWER")
        return "wall"

    # Ambiguous — be conservative
    log.info(f"Ambiguous (V_range={voltage_range:.3f}, P_max={max_p:.1f}W, "
             f"V_avg={avg_v:.3f}) — defaulting to BATTERY (safe)")
    return "battery"


# ── Main loop ────────────────────────────────────────────────
def run_daemon():
    """Main adaptive power management loop."""
    log.info("=== Aura Adaptive Power Manager ===")

    # Step 1: Start in 25W mode (safe for any source)
    current_mode = get_current_mode()
    log.info(f"Current mode: {current_mode}")

    if current_mode == "MAXN":
        log.info("Starting in MAXN — switching to 25W for safe probe")
        set_power_mode(3, "25W")
        time.sleep(2)

    # Step 2: Detect power source
    source = detect_power_source()
    log.info(f"Detected power source: {source}")

    # Step 3: Set appropriate mode
    if source == "wall":
        set_power_mode(0, "MAXN")
        target_mode = "MAXN"
    else:
        set_power_mode(3, "25W")
        target_mode = "25W"

    # Step 4: Continuous monitoring
    log.info(f"Monitoring power (mode={target_mode})...")
    stable_cycles = 0
    sag_events = 0

    while True:
        try:
            reading = read_power()
            if not reading:
                time.sleep(MONITOR_INTERVAL_S)
                continue

            v = reading.voltage
            p = reading.power

            # Critical sag — immediate downshift
            if v < VOLTAGE_SAG_CRIT:
                sag_events += 1
                if target_mode != "25W":
                    log.warning(f"CRITICAL SAG: {v:.3f}V — emergency downshift!")
                    set_power_mode(3, "25W")
                    target_mode = "25W"
                    stable_cycles = 0

                # If sag persists even at 25W, go to 15W
                if sag_events > 5 and target_mode == "25W":
                    log.warning(f"Persistent sag at 25W — dropping to 15W")
                    set_power_mode(2, "15W")
                    target_mode = "15W"
                    sag_events = 0

            elif v < VOLTAGE_SAG_WARN:
                stable_cycles = 0
                sag_events += 1
                if target_mode == "MAXN":
                    log.warning(f"Voltage sag: {v:.3f}V — downshifting to 25W")
                    set_power_mode(3, "25W")
                    target_mode = "25W"

            else:
                # Voltage is healthy
                sag_events = max(0, sag_events - 1)
                if target_mode != "MAXN":
                    stable_cycles += 1
                else:
                    stable_cycles = 0

                # Consider upshifting after sustained stability
                if stable_cycles >= UPSHIFT_STABLE_CYCLES and target_mode != "MAXN":
                    log.info(f"Voltage stable for {stable_cycles} cycles — "
                             f"re-probing for upshift")
                    source = detect_power_source()
                    if source == "wall":
                        log.info("Wall power confirmed — upshifting to MAXN")
                        set_power_mode(0, "MAXN")
                        target_mode = "MAXN"
                    stable_cycles = 0

            # Periodic status log (every 30 seconds)
            if int(time.time()) % 30 < MONITOR_INTERVAL_S:
                log.debug(f"[{target_mode}] {v:.3f}V {p:.1f}W "
                          f"sag={sag_events} stable={stable_cycles}")

            time.sleep(MONITOR_INTERVAL_S)

        except KeyboardInterrupt:
            log.info("Power manager stopped")
            break
        except Exception as e:
            log.error(f"Monitor error: {e}")
            time.sleep(5)


# ── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if "--probe-only" in sys.argv:
        source = detect_power_source()
        reading = read_power()
        mode = get_current_mode()
        print(f"\nPower source: {source}")
        print(f"Current mode: {mode}")
        if reading:
            print(f"VDD_IN: {reading.voltage:.3f}V  {reading.current:.3f}A  "
                  f"{reading.power:.2f}W")
        if source == "wall":
            print("Recommendation: MAXN (mode 0)")
        else:
            print("Recommendation: 25W (mode 3)")
    else:
        run_daemon()

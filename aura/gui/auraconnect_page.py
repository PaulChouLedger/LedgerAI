"""
gui.auraconnect_page -- AuraConnect BLE sub-page (Settings overlay).

Patek Philippe watchface treatment matching ``_draw_updates_page``:
beveled bezel, 60-tick chapter ring, engine-turned guilloché, arc headers,
scheme-aware palette, Helvetica Neue. Reads as a feature advertisement
for the AuraConnect Mac companion — not a clinical toggle.
"""

from __future__ import annotations

import math
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QLinearGradient, QPen,
    QRadialGradient, QPainterPath,
)

from gui.renderer import clamp

# Must match LOCAL_NAME in ble_server.py — shown on the device cartouche.
LOCAL_NAME = "Aura Puck"

# Rotating advertising taglines (5s each + crossfade).
_TAGLINES = (
    "PAIR FROM MAC  ·  DROP FILES",
    "WHAT YOU SEND, SHE REMEMBERS",
    "ON-DEVICE  ·  PRIVATE  ·  YOURS",
)
_TAGLINE_DWELL = 5.0      # seconds per line
_TAGLINE_FADE  = 0.6      # crossfade duration


# ---------------------------------------------------------------------------
# BLE GATT server — delegates to standalone ble_server.py process
# (avoids GIL starvation from aura.py's CUDA/GUI threads). Backend logic
# unchanged — only the drawing has been rebuilt.
# ---------------------------------------------------------------------------

import threading

_BLE_SERVER_SCRIPT = Path(__file__).resolve().parents[1] / "ble_server.py"
_BLE_PID_FILE = Path("/tmp/aura_ble.pid")

_ble_running = False
_ble_error: Optional[str] = None
_ble_connected = False
# Off-thread worker for launches/kills so the GUI never blocks.
_ble_worker_lock = threading.Lock()
# Last-toggle timestamp for tap debounce (prevents ON→OFF→ON races).
_last_toggle_t: float = 0.0


def _get_ble_pid() -> Optional[int]:
    try:
        if _BLE_PID_FILE.exists():
            pid = int(_BLE_PID_FILE.read_text().strip())
            os.kill(pid, 0)
            return pid
    except (ValueError, OSError):
        pass
    return None


def _spawn_ble_server() -> None:
    """Run on a worker thread — launch ble_server.py, wait for PID file."""
    global _ble_running, _ble_error, _ble_connected
    with _ble_worker_lock:
        if _get_ble_pid():
            _ble_running = True
            _ble_error = None
            return
        _ble_error = None
        _ble_connected = False
        if not _BLE_SERVER_SCRIPT.exists():
            _ble_error = f"ble_server.py not found at {_BLE_SERVER_SCRIPT}"
            return
        try:
            subprocess.Popen(
                ["python3", "-u", str(_BLE_SERVER_SCRIPT), "--daemon"],
                stdout=open("/tmp/ble.log", "a"),
                stderr=subprocess.STDOUT,
                cwd=str(_BLE_SERVER_SCRIPT.parent),
            )
            # Poll up to 6s — bluez can be slow to release GATT after a stop,
            # but on a clean start the PID file lands in <500ms.
            for _ in range(60):
                time.sleep(0.1)
                if _get_ble_pid():
                    _ble_running = True
                    print("[auraconnect] Standalone BLE server launched")
                    return
            _ble_error = "BLE server failed to start — check /tmp/ble.log"
            print(f"[auraconnect] {_ble_error}")
        except Exception as e:
            _ble_error = str(e)
            print(f"[auraconnect] Failed to launch BLE server: {e}")


def _stop_ble_server() -> None:
    """Run on a worker thread — kill the BLE server cleanly."""
    global _ble_running, _ble_connected
    with _ble_worker_lock:
        pid = _get_ble_pid()
        if pid:
            try:
                os.kill(pid, 15)
            except OSError:
                pass
            # Wait briefly for the server's own SIGTERM handler to clean up
            # bluez registrations — without this, an immediate restart fails.
            for _ in range(30):
                time.sleep(0.1)
                if not _get_ble_pid():
                    break
            else:
                # SIGTERM didn't take — escalate
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass
        _ble_running = False
        _ble_connected = False
        _BLE_PID_FILE.unlink(missing_ok=True)
        print("[auraconnect] BLE server stopped")


def start_ble() -> None:
    """Non-blocking — spawn a worker thread to launch the BLE server.

    GUI must NOT freeze waiting for bluez. The worker holds a lock so
    rapid taps cannot stack two launches.
    """
    threading.Thread(target=_spawn_ble_server, daemon=True,
                     name="ble-spawn").start()


def stop_ble() -> None:
    """Non-blocking — spawn a worker thread to kill the BLE server."""
    threading.Thread(target=_stop_ble_server, daemon=True,
                     name="ble-stop").start()


def is_ble_running() -> bool:
    global _ble_running
    _ble_running = _get_ble_pid() is not None
    return _ble_running


def is_ble_connected() -> bool:
    return _ble_connected


def get_ble_error() -> Optional[str]:
    return _ble_error


# ---------------------------------------------------------------------------
# Drawing — Patek Philippe watchface (matches _draw_updates_page)
# ---------------------------------------------------------------------------

# Match the settings overlay R — sub-page slots into the same dial size
# the user sees on the previous screen.
_R_FRAC = 0.30


def _palette(trans: float) -> dict:
    """Scheme-aware color set, mirroring _draw_updates_page conventions."""
    from core.config import COLOR_SCHEMES
    from core.state import state

    scheme = COLOR_SCHEMES.get(state.color_scheme, COLOR_SCHEMES.get("rafael", {}))
    is_red = scheme.get("ring_palette", "blue") == "red"

    A  = int(240 * trans)
    A2 = int(175 * trans)
    A3 = int(120 * trans)

    if is_red:
        accent_strong = QColor(220, 180, 120, A)
        accent_mid    = QColor(200, 165, 105, A2)
        accent_faint  = QColor(180, 150, 95, A3)
        glow_tint     = QColor(255, 200, 160)
        base_bg       = QColor(22, 6, 8, int(220 * trans))
        mid_bg        = QColor(35, 10, 14, int(210 * trans))
        jewel_rgb     = (220, 170, 90)
        accent_rgb    = (210, 175, 115)
    else:
        accent_strong = QColor(145, 175, 215, A)
        accent_mid    = QColor(145, 175, 215, A2)
        accent_faint  = QColor(145, 175, 215, A3)
        glow_tint     = QColor(255, 255, 255)
        base_bg       = QColor(8, 9, 12, int(210 * trans))
        mid_bg        = QColor(14, 15, 20, int(200 * trans))
        jewel_rgb     = (100, 170, 255)
        accent_rgb    = (145, 175, 215)

    return {
        "trans":         trans,
        "is_red":        is_red,
        "accent_strong": accent_strong,
        "accent_mid":    accent_mid,
        "accent_faint":  accent_faint,
        "glow_tint":     glow_tint,
        "base_bg":       base_bg,
        "mid_bg":        mid_bg,
        "jewel_rgb":     jewel_rgb,
        "accent_rgb":    accent_rgb,
        "tick":          (lambda a, c=accent_rgb: QColor(c[0], c[1], c[2], a)),
        "ok_rgb":        (110, 210, 165),     # CONNECTED accent
        "warn_rgb":      (220, 150, 130),     # ERROR accent
    }


def _arc_text(p, cx, cy, text, radius, center_deg, font, color,
              spacing_deg=6.5, flip=False):
    """Curve `text` along an arc centered at (cx, cy)."""
    chars = list(text)
    if flip:
        chars = chars[::-1]
    n = len(chars)
    span = (n - 1) * spacing_deg if n > 1 else 0.0
    start = center_deg - span * 0.5
    p.save()
    p.setFont(font)
    fm = p.fontMetrics()
    ch_h = fm.height()
    extra_rot = 180.0 if flip else 0.0
    for i, ch in enumerate(chars):
        a = math.radians(start + i * spacing_deg)
        x = cx + radius * math.cos(a)
        y = cy + radius * math.sin(a)
        ch_w = max(fm.horizontalAdvance(ch), ch_h)
        p.save()
        p.translate(x, y)
        p.rotate(math.degrees(a) + 90.0 + extra_rot)
        rect = QRectF(-ch_w * 0.7, -ch_h * 0.6, ch_w * 1.4, ch_h * 1.2)
        p.setPen(color)
        p.drawText(rect, Qt.AlignCenter, ch)
        p.restore()
    p.restore()


def _draw_bt_glyph(p, cx, cy, r, color):
    """Refined Bluetooth rune. `color` is QColor (alpha already applied)."""
    pen = QPen(color)
    pen.setWidthF(max(1.6, r * 0.13))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)

    # Soft drop shadow
    shd = QPen(QColor(0, 0, 0, 90))
    shd.setWidthF(pen.widthF() * 1.15)
    shd.setCapStyle(Qt.RoundCap)
    shd.setJoinStyle(Qt.RoundJoin)
    off = max(0.4, r * 0.025)

    h = r * 0.85
    w = r * 0.42

    def _strokes(ox, oy, q):
        q.drawLine(QPointF(cx + ox, cy - h + oy), QPointF(cx + ox, cy + h + oy))
        q.drawLine(QPointF(cx + ox, cy - h + oy),
                   QPointF(cx + w + ox, cy - h * 0.35 + oy))
        q.drawLine(QPointF(cx + w + ox, cy - h * 0.35 + oy),
                   QPointF(cx - w + ox, cy + h * 0.35 + oy))
        q.drawLine(QPointF(cx + ox, cy + h + oy),
                   QPointF(cx + w + ox, cy + h * 0.35 + oy))
        q.drawLine(QPointF(cx + w + ox, cy + h * 0.35 + oy),
                   QPointF(cx - w + ox, cy - h * 0.35 + oy))

    p.setPen(shd); p.setBrush(Qt.NoBrush); _strokes(off, off, p)
    p.setPen(pen); _strokes(0, 0, p)


def _tagline_state(t: float) -> tuple[str, float]:
    """Pick the active tagline + crossfade alpha (0..1) for time `t`."""
    cycle = _TAGLINE_DWELL
    n = len(_TAGLINES)
    phase = (t % (cycle * n))
    idx = int(phase // cycle)
    in_idx = phase - idx * cycle
    # Fade in over _TAGLINE_FADE, hold, fade out over _TAGLINE_FADE
    if in_idx < _TAGLINE_FADE:
        a = in_idx / _TAGLINE_FADE
    elif in_idx > cycle - _TAGLINE_FADE:
        a = (cycle - in_idx) / _TAGLINE_FADE
    else:
        a = 1.0
    return _TAGLINES[idx % n], clamp(a, 0.0, 1.0)


def draw_auraconnect_page(p, cx, cy, mind, t, trans, state=None):
    """Draw the AuraConnect BLE status / advertising page."""
    a = clamp(float(trans), 0.0, 1.0)
    if a <= 0.002:
        return

    pal = _palette(a)
    R = mind * _R_FRAC

    r_bezel_outer = R * 0.98
    r_bezel_inner = R * 0.90
    r_chapter_out = R * 0.84
    r_chapter_in  = R * 0.74
    rg = R * 0.64

    A  = int(240 * a)
    A2 = int(175 * a)
    A3 = int(120 * a)

    accent_strong = pal["accent_strong"]
    accent_mid    = pal["accent_mid"]
    accent_faint  = pal["accent_faint"]
    glow_tint     = pal["glow_tint"]
    base_bg       = pal["base_bg"]
    mid_bg        = pal["mid_bg"]
    jewel_rgb     = pal["jewel_rgb"]
    tick          = pal["tick"]

    # Status (drives glyph color, jewel color, status word)
    running   = is_ble_running()
    connected = is_ble_connected()
    error     = get_ble_error()

    if running and connected:
        status_text = "CONNECTED"
        status_rgb  = pal["ok_rgb"]
    elif running:
        status_text = "ACTIVATED"
        status_rgb  = jewel_rgb
    elif error:
        status_text = "ERROR"
        status_rgb  = pal["warn_rgb"]
    else:
        status_text = "OFF"
        status_rgb  = (140, 140, 155)

    # Header text-exclusion zones (12 o'clock arc, 6 o'clock arc)
    def tick_in_text_zone(i):
        ang = (i / 60.0) * 360.0
        if ang > 335 or ang < 25:        # AURACONNECT at 12
            return True
        if 155 < ang < 205:              # ◀ BACK at 6
            return True
        return False

    p.save()
    p.setRenderHint(p.Antialiasing, True)

    # ── Clip to circle so nothing escapes the page ───────────────
    clip = QPainterPath()
    clip.addEllipse(QPointF(cx, cy), R, R)
    p.setClipPath(clip)

    # =========================================================
    # Base plate (deep enamel) + crystal gloss
    # =========================================================
    p.setPen(Qt.NoPen)
    p.setBrush(base_bg)
    p.drawEllipse(QPointF(cx, cy), R, R)

    grad = QRadialGradient(QPointF(cx, cy), R * 0.98)
    grad.setColorAt(0.0, QColor(glow_tint.red(), glow_tint.green(), glow_tint.blue(), int(A * 0.12)))
    grad.setColorAt(0.55, QColor(glow_tint.red(), glow_tint.green(), glow_tint.blue(), int(A * 0.06)))
    grad.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.setBrush(QBrush(grad))
    p.drawEllipse(QPointF(cx, cy), R * 0.98, R * 0.98)

    # Outer bevel highlight
    g1 = QRadialGradient(QPointF(cx, cy), r_bezel_outer)
    g1.setColorAt(0.70, QColor(accent_mid.red(), accent_mid.green(), accent_mid.blue(), int(34 * a)))
    g1.setColorAt(0.86, QColor(accent_mid.red(), accent_mid.green(), accent_mid.blue(), int(62 * a)))
    g1.setColorAt(1.00, QColor(0, 0, 0, 0))
    p.setBrush(QBrush(g1))
    p.drawEllipse(QPointF(cx, cy), r_bezel_outer, r_bezel_outer)

    # Inner chamfer shadow
    g2 = QRadialGradient(QPointF(cx, cy), r_bezel_inner)
    g2.setColorAt(0.60, QColor(0, 0, 0, 0))
    g2.setColorAt(0.92, QColor(0, 0, 0, int(80 * a)))
    g2.setColorAt(1.00, QColor(0, 0, 0, int(110 * a)))
    p.setBrush(QBrush(g2))
    p.drawEllipse(QPointF(cx, cy), r_bezel_inner, r_bezel_inner)

    # Bezel rings (outer polished + inner faint)
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(accent_mid, max(2.0, mind * 0.0042)))
    p.drawEllipse(QPointF(cx, cy), r_bezel_outer, r_bezel_outer)
    p.setPen(QPen(accent_faint, max(1.2, mind * 0.0026)))
    p.drawEllipse(QPointF(cx, cy), r_bezel_inner, r_bezel_inner)

    # =========================================================
    # Chapter ring (skip ticks behind arc text)
    # =========================================================
    for i in range(60):
        if tick_in_text_zone(i):
            continue
        ang = (i / 60.0) * 2.0 * math.pi - math.pi / 2.0
        is_major = (i % 5 == 0)
        tick_len = (R * 0.080) if is_major else (R * 0.045)
        tick_w   = (mind * 0.0038) if is_major else (mind * 0.0022)
        rr_out = r_chapter_out
        rr_in  = rr_out - tick_len
        col = tick(int((170 if is_major else 115) * a))
        p.setPen(QPen(col, tick_w, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(
            QPointF(cx + rr_in * math.cos(ang), cy + rr_in * math.sin(ang)),
            QPointF(cx + rr_out * math.cos(ang), cy + rr_out * math.sin(ang)),
        )
    p.setPen(QPen(accent_faint, max(1.0, mind * 0.0018)))
    p.drawEllipse(QPointF(cx, cy), r_chapter_in, r_chapter_in)

    # =========================================================
    # Engine-turned guilloché (concentric ellipses with wobble)
    # =========================================================
    p.setPen(Qt.NoPen)
    p.setBrush(mid_bg)
    p.drawEllipse(QPointF(cx, cy), rg, rg)

    gu_alpha = int(22 * a) if not pal["is_red"] else int(16 * a)
    gu_pen = QPen(QColor(glow_tint.red(), glow_tint.green(), glow_tint.blue(), gu_alpha))
    gu_pen.setWidthF(max(1.0, mind * 0.0016))
    p.setPen(gu_pen)
    p.setBrush(Qt.NoBrush)
    steps = 30
    for k in range(steps):
        frac = k / (steps - 1)
        rad = rg * (0.18 + 0.82 * frac)
        wob = 1.0 + 0.016 * math.sin(t * 0.9 + k * 0.55)
        p.drawEllipse(QPointF(cx, cy), rad * wob, rad)

    # =========================================================
    # Header arcs: AURACONNECT (12), ◀ BACK (6), BLUETOOTH (6 inner)
    # =========================================================
    header_font = QFont("Helvetica Neue", max(9, int(mind * 0.016)))
    header_font.setWeight(QFont.DemiBold)
    header_font.setLetterSpacing(QFont.PercentageSpacing, 145)

    _arc_text(p, cx, cy, "AURACONNECT", R * 0.78, -90.0,
              header_font, accent_strong, spacing_deg=6.0)

    sub_font = QFont("Helvetica Neue", max(6, int(mind * 0.0095)))
    sub_font.setWeight(QFont.Medium)
    sub_font.setLetterSpacing(QFont.PercentageSpacing, 140)
    _arc_text(p, cx, cy, "BLUETOOTH", R * 0.68, -90.0,
              sub_font, accent_faint, spacing_deg=7.5)

    # =========================================================
    # Bluetooth glyph (the visual hub) with status-tinted halo
    # =========================================================
    glyph_cy = cy - R * 0.05
    glyph_r  = R * 0.20

    # Status pulse: ADVERTISING pulses; CONNECTED breathes; OFF static
    if running and not connected:
        pulse = 0.55 + 0.45 * math.sin(t * 2.8)         # active advertising
    elif running and connected:
        pulse = 0.65 + 0.20 * math.sin(t * 1.1)         # gentle connected breath
    else:
        pulse = 0.0                                     # off / error

    # Halo
    if pulse > 0:
        halo = QRadialGradient(QPointF(cx, glyph_cy), glyph_r * 2.4)
        halo.setColorAt(0.0, QColor(status_rgb[0], status_rgb[1], status_rgb[2],
                                     int((55 + 60 * pulse) * a)))
        halo.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(halo))
        p.drawEllipse(QPointF(cx, glyph_cy), glyph_r * 2.4, glyph_r * 2.4)

    glyph_alpha = int((220 if running else 140) * a)
    glyph_col = QColor(status_rgb[0], status_rgb[1], status_rgb[2], glyph_alpha)
    _draw_bt_glyph(p, cx, glyph_cy, glyph_r, glyph_col)

    # =========================================================
    # Status word + device cartouche, just below the glyph
    # =========================================================
    status_font = QFont("Helvetica Neue", max(8, int(mind * 0.014)))
    status_font.setWeight(QFont.DemiBold)
    status_font.setLetterSpacing(QFont.PercentageSpacing, 165)
    p.setFont(status_font)
    sc = QColor(status_rgb[0], status_rgb[1], status_rgb[2], int(225 * a))
    p.setPen(QColor(0, 0, 0, int(120 * a)))
    p.drawText(QRectF(cx - R, cy + R * 0.18 + 1, 2 * R, R * 0.13),
               Qt.AlignCenter, status_text)
    p.setPen(sc)
    p.drawText(QRectF(cx - R, cy + R * 0.18, 2 * R, R * 0.13),
               Qt.AlignCenter, status_text)

    name_font = QFont("Helvetica Neue", max(7, int(mind * 0.0115)))
    name_font.setWeight(QFont.Light)
    name_font.setLetterSpacing(QFont.PercentageSpacing, 130)
    p.setFont(name_font)
    p.setPen(accent_faint)
    p.drawText(QRectF(cx - R, cy + R * 0.32, 2 * R, R * 0.10),
               Qt.AlignCenter, f"“{LOCAL_NAME}”")

    # Hairline separator under cartouche
    sep_y = cy + R * 0.46
    sep_pen = QPen(QColor(accent_faint.red(), accent_faint.green(), accent_faint.blue(),
                          int(70 * a)), max(0.6, mind * 0.0008))
    p.setPen(sep_pen)
    p.drawLine(QPointF(cx - R * 0.32, sep_y), QPointF(cx + R * 0.32, sep_y))

    # =========================================================
    # Rotating tagline (the "advertisement" line)
    # =========================================================
    tag_text, tag_a = _tagline_state(t)
    if error and not running:
        # Error supersedes the ad — show the actual error
        tag_text = error[:34] + ("…" if len(error) > 34 else "")
        tag_a = 1.0
        tag_col = QColor(pal["warn_rgb"][0], pal["warn_rgb"][1], pal["warn_rgb"][2],
                         int(200 * a))
    else:
        tag_col = QColor(accent_strong.red(), accent_strong.green(), accent_strong.blue(),
                         int(200 * a * tag_a))

    tag_font = QFont("Helvetica Neue", max(6, int(mind * 0.0098)))
    tag_font.setWeight(QFont.Medium)
    tag_font.setLetterSpacing(QFont.PercentageSpacing, 155)
    p.setFont(tag_font)
    p.setPen(tag_col)
    p.drawText(QRectF(cx - R, cy + R * 0.50, 2 * R, R * 0.10),
               Qt.AlignCenter, tag_text)

    # =========================================================
    # Jewel pusher (START / STOP) — tucked under the 12 o'clock header
    # =========================================================
    btn_cx = cx
    btn_cy = cy - R * 0.55
    btn_r  = R * 0.10
    pulse_btn = 0.5 + 0.5 * math.sin(t * 2.0)

    # Pick jewel color: scheme accent if OFF, ok-green if RUNNING
    if running:
        b_rgb = pal["ok_rgb"] if connected else jewel_rgb
    else:
        b_rgb = jewel_rgb

    # Halo
    p.setPen(Qt.NoPen)
    btn_halo = QRadialGradient(QPointF(btn_cx, btn_cy), btn_r * 3.4)
    btn_halo.setColorAt(0.0, QColor(b_rgb[0], b_rgb[1], b_rgb[2],
                                     int((40 + 35 * pulse_btn) * a)))
    btn_halo.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.setBrush(QBrush(btn_halo))
    p.drawEllipse(QPointF(btn_cx, btn_cy), btn_r * 3.4, btn_r * 3.4)

    # Bezel ring around the jewel
    p.setPen(QPen(QColor(b_rgb[0], b_rgb[1], b_rgb[2],
                          int((150 + 60 * pulse_btn) * a)),
                  max(1.1, mind * 0.0022)))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(btn_cx, btn_cy), btn_r * 1.3, btn_r * 1.3)

    # Jewel body
    jg = QRadialGradient(QPointF(btn_cx - btn_r * 0.2, btn_cy - btn_r * 0.3),
                         btn_r * 1.2)
    jg.setColorAt(0.0, QColor(min(255, b_rgb[0] + 60),
                              min(255, b_rgb[1] + 50),
                              min(255, b_rgb[2] + 40),
                              int((200 + 55 * pulse_btn) * a)))
    jg.setColorAt(0.6, QColor(b_rgb[0], b_rgb[1], b_rgb[2], int(180 * a)))
    jg.setColorAt(1.0, QColor(max(0, b_rgb[0] - 40),
                              max(0, b_rgb[1] - 40),
                              max(0, b_rgb[2] - 30),
                              int(160 * a)))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(jg))
    p.drawEllipse(QPointF(btn_cx, btn_cy), btn_r, btn_r)

    # Highlight catchlight
    p.setBrush(QColor(255, 255, 255, int(85 * a)))
    p.drawEllipse(QPointF(btn_cx - btn_r * 0.25, btn_cy - btn_r * 0.28),
                  btn_r * 0.28, btn_r * 0.20)

    # Arc text label above the jewel: START / STOP
    label = "STOP" if running else "START"
    lbl_font = QFont("Helvetica Neue", max(5, int(mind * 0.0090)))
    lbl_font.setWeight(QFont.DemiBold)
    lbl_font.setLetterSpacing(QFont.PercentageSpacing, 140)
    p.setFont(lbl_font)
    lbl_r = btn_r * 2.4
    lbl_spacing = 8.0
    lbl_span = (len(label) - 1) * lbl_spacing
    lbl_start = -90.0 - lbl_span * 0.5
    lbl_col = QColor(b_rgb[0], b_rgb[1], b_rgb[2],
                     int((170 + 60 * pulse_btn) * a))
    for ci, ch in enumerate(label):
        ang = math.radians(lbl_start + ci * lbl_spacing)
        ax = btn_cx + lbl_r * math.cos(ang)
        ay = btn_cy + lbl_r * math.sin(ang)
        p.save()
        p.translate(ax, ay)
        p.rotate(math.degrees(ang) + 90.0)
        p.setPen(lbl_col)
        p.drawText(QRectF(-30, -10, 60, 20), Qt.AlignCenter, ch)
        p.restore()

    # =========================================================
    # 6 o'clock arc: ◀ BACK
    # =========================================================
    if a > 0.5:
        back_font = QFont("Helvetica Neue", max(6, int(mind * 0.0105)))
        back_font.setWeight(QFont.Normal)
        back_font.setLetterSpacing(QFont.PercentageSpacing, 125)
        ba = int(180 * (a - 0.5) / 0.5)
        ba = max(0, min(180, ba))
        back_col = QColor(accent_faint.red(), accent_faint.green(),
                          accent_faint.blue(), ba)
        _arc_text(p, cx, cy, "◀  BACK", R * 0.78, 90.0,
                  back_font, back_col, spacing_deg=5.0, flip=True)

    p.restore()


# ---------------------------------------------------------------------------
# Touch handling
# ---------------------------------------------------------------------------

def handle_auraconnect_tap(x, y, cx, cy, mind):
    """Returns 'back' or None.

    Touch model: the bottom ~20% of the dial is the BACK strip; everything
    else inside the dial — jewel pusher, BT glyph, status word, cartouche —
    is one big BLE toggle. Tapping anywhere "in the middle" turns
    advertising on/off. Generous targets so fingertips on the puck's
    small touchscreen don't miss.
    """
    R = mind * _R_FRAC

    dy = y - cy

    # The parent settings overlay already filtered out taps with
    # dist > R * 1.05 (returns to main settings). Anything reaching
    # here is inside-or-touching the dial, so just split top/bottom.
    if dy > R * 0.45:
        return "back"

    # Toggle debounce — bluez state transitions are slow, and a quick
    # double-tap would otherwise flip ON→OFF→ON in a way that races the
    # background spawn/kill workers. Ignore taps within 1.2s of the last.
    global _last_toggle_t
    now = time.time()
    if now - _last_toggle_t < 1.2:
        print("[auraconnect] tap ignored (debounce)")
        return None
    _last_toggle_t = now

    if is_ble_running():
        print("[auraconnect] Stopping BLE...")
        stop_ble()
    else:
        print("[auraconnect] Starting BLE...")
        start_ble()
    return None

"""
gui.wifi_page -- WiFi configuration for the Aura round display.

Patek Philippe Grand Complications aesthetic: engine-turned guilloché,
polished bezel with chapter-ring ticks, champagne gold accents, deep
lacquer backgrounds.  Uses nmcli for scanning and connecting.
"""

from __future__ import annotations

import math
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QLinearGradient, QPen,
    QRadialGradient, QPainterPath,
)

from gui.renderer import clamp

# ── Palette ──────────────────────────────────────────────────────────
_CHAMPAGNE   = lambda a=255: QColor(218, 200, 155, a)
_IVORY       = lambda a=255: QColor(240, 234, 218, a)
_DEEP_NAVY   = lambda a=255: QColor(6, 9, 22, a)
_SLATE       = lambda a=255: QColor(42, 48, 68, a)
_ACCENT_TEAL = lambda a=255: QColor(80, 200, 165, a)
_ROSE        = lambda a=255: QColor(200, 130, 130, a)
_DIM_GOLD    = lambda a=255: QColor(165, 152, 118, a)


@dataclass
class WifiNetwork:
    ssid: str
    signal: int        # 0-100
    security: str      # "WPA2", "WPA3", "WEP", "" (open)
    connected: bool = False


# ── WiFi backend (nmcli) ────────────────────────────────────────────

def scan_networks() -> List[WifiNetwork]:
    """Scan for available WiFi networks via nmcli."""
    try:
        # Force a fresh scan first
        subprocess.run(
            ["nmcli", "device", "wifi", "rescan"],
            capture_output=True, timeout=8,
        )
        time.sleep(1.5)  # give radio time to populate results
        result = subprocess.run(
            ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY,ACTIVE",
             "device", "wifi", "list"],
            capture_output=True, text=True, timeout=10,
        )
        networks = []
        seen = set()
        for line in result.stdout.strip().splitlines():
            parts = line.split(":")
            if len(parts) < 4:
                continue
            ssid = parts[0].strip()
            if not ssid or ssid in seen:
                continue
            seen.add(ssid)
            try:
                signal = int(parts[1])
            except ValueError:
                signal = 0
            security = parts[2].strip()
            active = parts[3].strip().lower() == "yes"
            networks.append(WifiNetwork(ssid=ssid, signal=signal,
                                        security=security, connected=active))
        networks.sort(key=lambda n: (-n.connected, -n.signal))
        return networks
    except Exception as e:
        print(f"[wifi] Scan failed: {e}")
        return []


def connect_network(ssid: str, password: str = "") -> tuple[bool, str]:
    """Connect to a WiFi network. Returns (success, message)."""
    try:
        cmd = ["nmcli", "device", "wifi", "connect", ssid]
        if password:
            cmd += ["password", password]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return True, f"Connected to {ssid}"
        else:
            err = result.stderr.strip() or result.stdout.strip()
            return False, err[:80]
    except Exception as e:
        return False, str(e)[:80]


def get_current_ip() -> str:
    """Get current WiFi IP address."""
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "IP4.ADDRESS", "device", "show", "wlan0"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if "IP4.ADDRESS" in line:
                addr = line.split(":", 1)[1].strip()
                return addr.split("/")[0]
    except Exception:
        pass
    return ""


# ── Keyboard layout ─────────────────────────────────────────────────

_ROWS_LOWER = [
    list("1234567890"),
    list("qwertyuiop"),
    list("asdfghjkl"),
    list("zxcvbnm"),
]

_ROWS_UPPER = [
    list("1234567890"),
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
]

_ROWS_SYMBOL = [
    list("!@#$%^&*()"),
    list("-_=+[]{}\\|"),
    list(";:'\",.<>/?"),
    list("`~"),
]


# ── WiFi page state ─────────────────────────────────────────────────

class WifiPageState:
    def __init__(self):
        self.networks: List[WifiNetwork] = []
        self.scroll_offset: int = 0
        self.selected_ssid: Optional[str] = None
        self.password: str = ""
        self.shift: bool = False
        self.symbols: bool = False
        self.show_keyboard: bool = False
        self.status_msg: str = ""
        self.status_time: float = 0.0
        self.connecting: bool = False
        self.scanning: bool = False
        self._last_scan: float = 0.0
        self._cached_ip: str = ""
        self._ip_fetch_time: float = 0.0
        # Tap feedback: (row_idx, col_idx, timestamp) or ("special", label, timestamp)
        self._tap_flash: Optional[tuple] = None

    def trigger_scan(self):
        if self.scanning or (time.time() - self._last_scan < 5.0):
            return
        self.scanning = True

        def _scan():
            nets = scan_networks()
            self.networks = nets
            self.scanning = False
            self._last_scan = time.time()

        threading.Thread(target=_scan, daemon=True).start()

    def trigger_connect(self):
        if self.connecting or not self.selected_ssid:
            return
        self.connecting = True
        self.status_msg = f"Connecting to {self.selected_ssid}..."
        self.status_time = time.time()

        ssid = self.selected_ssid
        pwd = self.password

        def _connect():
            ok, msg = connect_network(ssid, pwd)
            self.status_msg = msg
            self.status_time = time.time()
            self.connecting = False
            if ok:
                self.show_keyboard = False
                self.password = ""
                self.trigger_scan()

        threading.Thread(target=_connect, daemon=True).start()


# ── Radius constant (shared between draw + touch) ───────────────────
_WIFI_R_FRAC = 0.33   # fraction of min(W,H) — larger for keyboard usability


# ── Drawing ──────────────────────────────────────────────────────────

def draw_wifi_page(p, cx, cy, mind, t, trans, state: WifiPageState):
    """Render the WiFi configuration page with Patek Philippe aesthetic."""
    a = clamp(float(trans), 0.0, 1.0)
    if a <= 0.002:
        return

    R = mind * _WIFI_R_FRAC

    # Auto-scan on first render
    if not state.networks and not state.scanning:
        state.trigger_scan()

    p.save()
    p.setRenderHint(p.Antialiasing, True)

    # ── Clip to circle ───────────────────────────────────────────
    clip = QPainterPath()
    clip.addEllipse(QPointF(cx, cy), R, R)
    p.setClipPath(clip)

    # ── Deep lacquer background with vignette ────────────────────
    bg = QRadialGradient(cx, cy, R)
    bg.setColorAt(0.00, QColor(14, 18, 38, int(252 * a)))
    bg.setColorAt(0.55, QColor(8, 12, 28, int(248 * a)))
    bg.setColorAt(0.85, QColor(4, 6, 16, int(245 * a)))
    bg.setColorAt(1.00, QColor(2, 3, 8, int(240 * a)))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(bg))
    p.drawEllipse(QPointF(cx, cy), R, R)

    # ── Engine-turned guilloché ──────────────────────────────────
    # Radial lines — fine engraving
    gu_pen = QPen(_CHAMPAGNE(int(14 * a)), max(0.4, R * 0.0015))
    p.setPen(gu_pen)
    p.setBrush(Qt.NoBrush)
    for i in range(90):
        angle = (2 * math.pi * i) / 90
        wave = 1.0 + 0.008 * math.sin(i * 7 + t * 0.25)
        x1 = cx + R * 0.08 * math.cos(angle)
        y1 = cy + R * 0.08 * math.sin(angle)
        x2 = cx + R * 0.88 * wave * math.cos(angle)
        y2 = cy + R * 0.88 * wave * math.sin(angle)
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    # Concentric rings (basket-weave depth)
    ring_pen = QPen(_CHAMPAGNE(int(10 * a)), max(0.3, R * 0.001))
    p.setPen(ring_pen)
    for frac in (0.18, 0.32, 0.46, 0.60, 0.74, 0.86):
        rr = R * frac
        wob = 1.0 + 0.006 * math.sin(t * 0.4 + frac * 12)
        p.drawEllipse(QPointF(cx, cy), rr * wob, rr)

    # ── Triple bezel (polished + brushed + polished) ─────────────
    p.setBrush(Qt.NoBrush)

    # Outer polished bezel
    outer_pen = QPen(_CHAMPAGNE(int(160 * a)), max(3.0, mind * 0.005))
    p.setPen(outer_pen)
    p.drawEllipse(QPointF(cx, cy), R * 0.97, R * 0.97)

    # Brushed middle ring (subtle gradient feel)
    mid_pen = QPen(_DIM_GOLD(int(55 * a)), max(1.5, mind * 0.0025))
    p.setPen(mid_pen)
    p.drawEllipse(QPointF(cx, cy), R * 0.94, R * 0.94)

    # Inner polished bezel
    inner_pen = QPen(_CHAMPAGNE(int(90 * a)), max(1.2, mind * 0.002))
    p.setPen(inner_pen)
    p.drawEllipse(QPointF(cx, cy), R * 0.91, R * 0.91)

    # ── Chapter ring (60 ticks, like a watch dial) ───────────────
    for i in range(60):
        ang = (i / 60.0) * 2.0 * math.pi - math.pi / 2.0
        is_major = (i % 5 == 0)
        tick_out = R * 0.965
        tick_in = tick_out - (R * 0.055 if is_major else R * 0.030)
        tw = (mind * 0.003 if is_major else mind * 0.0015)
        col = _CHAMPAGNE(int((150 if is_major else 75) * a))
        p.setPen(QPen(col, tw, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(
            QPointF(cx + tick_in * math.cos(ang), cy + tick_in * math.sin(ang)),
            QPointF(cx + tick_out * math.cos(ang), cy + tick_out * math.sin(ang)),
        )

    # ── Beveled depth shadow inside inner bezel ──────────────────
    depth = QRadialGradient(cx, cy, R * 0.91)
    depth.setColorAt(0.70, QColor(0, 0, 0, 0))
    depth.setColorAt(0.92, QColor(0, 0, 0, int(60 * a)))
    depth.setColorAt(1.00, QColor(0, 0, 0, int(100 * a)))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(depth))
    p.drawEllipse(QPointF(cx, cy), R * 0.91, R * 0.91)

    # ── Crystal reflection highlight ─────────────────────────────
    highlight = QRadialGradient(cx - R * 0.15, cy - R * 0.25, R * 0.7)
    highlight.setColorAt(0.0, QColor(255, 255, 255, int(8 * a)))
    highlight.setColorAt(0.5, QColor(255, 255, 255, int(3 * a)))
    highlight.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.setBrush(QBrush(highlight))
    p.drawEllipse(QPointF(cx, cy), R * 0.90, R * 0.90)

    # ── Header: "WI-FI" ─────────────────────────────────────────
    hdr_font = QFont("DejaVu Serif", max(11, int(mind * 0.020)))
    hdr_font.setLetterSpacing(QFont.AbsoluteSpacing, mind * 0.012)
    hdr_font.setBold(True)
    p.setFont(hdr_font)
    # Shadow
    p.setPen(QColor(0, 0, 0, int(120 * a)))
    p.drawText(QRectF(cx - R, cy - R * 0.85 + 1, 2 * R, R * 0.10),
               Qt.AlignCenter, "WI-FI")
    p.setPen(_CHAMPAGNE(int(240 * a)))
    p.drawText(QRectF(cx - R, cy - R * 0.85, 2 * R, R * 0.10),
               Qt.AlignCenter, "WI-FI")

    # Thin gold separator below header
    sep_y = cy - R * 0.74
    p.setPen(QPen(_CHAMPAGNE(int(45 * a)), max(0.6, mind * 0.001)))
    p.drawLine(QPointF(cx - R * 0.50, sep_y), QPointF(cx + R * 0.50, sep_y))

    # Back chevron (top-left)
    back_font = QFont("DejaVu Sans", max(14, int(mind * 0.028)))
    p.setFont(back_font)
    p.setPen(_CHAMPAGNE(int(160 * a)))
    p.drawText(QRectF(cx - R * 0.88, cy - R * 0.88, R * 0.25, R * 0.14),
               Qt.AlignCenter, "\u2039")

    # IP address (top-right, subtle)
    now = time.time()
    if now - state._ip_fetch_time > 10.0:
        state._cached_ip = get_current_ip()
        state._ip_fetch_time = now
    ip = state._cached_ip
    if ip:
        ip_font = QFont("DejaVu Sans Mono", max(7, int(mind * 0.011)))
        p.setFont(ip_font)
        p.setPen(_DIM_GOLD(int(110 * a)))
        p.drawText(QRectF(cx, cy - R * 0.86, R * 0.82, R * 0.07),
                   Qt.AlignRight | Qt.AlignVCenter, ip)

    if state.show_keyboard:
        _draw_keyboard(p, cx, cy, R, mind, a, state)
    else:
        _draw_network_list(p, cx, cy, R, mind, t, a, state)

    # ── Status message ───────────────────────────────────────────
    if state.status_msg and (time.time() - state.status_time < 5.0):
        stat_font = QFont("DejaVu Sans", max(8, int(mind * 0.014)))
        p.setFont(stat_font)
        fade = max(0.0, 1.0 - (time.time() - state.status_time) / 5.0)
        p.setPen(_IVORY(int(170 * fade * a)))
        p.drawText(QRectF(cx - R * 0.65, cy + R * 0.76, R * 1.30, R * 0.09),
                   Qt.AlignCenter, state.status_msg)

    p.restore()


def _draw_network_list(p, cx, cy, R, mind, t, a, state: WifiPageState):
    """Draw the scrollable network list."""
    if state.scanning:
        sf = QFont("DejaVu Serif", max(10, int(mind * 0.016)))
        sf.setLetterSpacing(QFont.AbsoluteSpacing, mind * 0.005)
        p.setFont(sf)
        dots = "." * (1 + int(t * 2) % 3)
        p.setPen(_CHAMPAGNE(int(140 * a)))
        p.drawText(QRectF(cx - R, cy - R * 0.2, 2 * R, R * 0.1),
                   Qt.AlignCenter, f"SCANNING{dots}")
        return

    if not state.networks:
        sf = QFont("DejaVu Sans", max(9, int(mind * 0.015)))
        p.setFont(sf)
        p.setPen(_DIM_GOLD(int(120 * a)))
        p.drawText(QRectF(cx - R, cy - R * 0.15, 2 * R, R * 0.1),
                   Qt.AlignCenter, "No networks found")
        return

    # List area
    list_top = cy - R * 0.65
    row_h = mind * 0.048
    max_visible = 7
    offset = state.scroll_offset

    visible = state.networks[offset:offset + max_visible]

    ssid_font = QFont("DejaVu Sans", max(9, int(mind * 0.016)))

    for i, net in enumerate(visible):
        row_y = list_top + i * row_h
        row_rect = QRectF(cx - R * 0.72, row_y, R * 1.44, row_h - 2)

        # Selected highlight — warm champagne glow
        if net.ssid == state.selected_ssid:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(180, 165, 120, int(28 * a)))
            p.drawRoundedRect(row_rect, 6, 6)

        # Connected dot
        if net.connected:
            p.setPen(Qt.NoPen)
            p.setBrush(_ACCENT_TEAL(int(200 * a)))
            dot_r = mind * 0.004
            p.drawEllipse(QPointF(cx - R * 0.65, row_y + row_h * 0.45),
                          dot_r, dot_r)

        # SSID
        p.setFont(ssid_font)
        if net.connected:
            p.setPen(_ACCENT_TEAL(int(230 * a)))
        else:
            p.setPen(_IVORY(int(195 * a)))
        p.drawText(QRectF(cx - R * 0.57, row_y, R * 0.85, row_h),
                   Qt.AlignVCenter | Qt.AlignLeft, net.ssid)

        # Signal bars (refined)
        _draw_signal_bars(p, cx + R * 0.48, row_y + row_h * 0.28,
                          mind * 0.020, net.signal, a)

        # Lock icon for secured networks
        if net.security:
            lock_font = QFont("DejaVu Sans", max(6, int(mind * 0.010)))
            p.setFont(lock_font)
            p.setPen(_DIM_GOLD(int(100 * a)))
            p.drawText(QRectF(cx + R * 0.36, row_y, R * 0.10, row_h),
                       Qt.AlignVCenter | Qt.AlignCenter, "\u2022")

        # Subtle separator
        if i < len(visible) - 1:
            sep_y2 = row_y + row_h - 1
            p.setPen(QPen(_CHAMPAGNE(int(16 * a)), 0.5))
            p.drawLine(QPointF(cx - R * 0.45, sep_y2),
                       QPointF(cx + R * 0.45, sep_y2))

    # Scroll indicators (elegant chevrons)
    chev_font = QFont("DejaVu Sans", max(11, int(mind * 0.018)))
    if offset > 0:
        p.setPen(_CHAMPAGNE(int(90 * a)))
        p.setFont(chev_font)
        p.drawText(QRectF(cx - R * 0.08, list_top - row_h * 0.45,
                           R * 0.16, row_h * 0.35),
                   Qt.AlignCenter, "\u2303")
    if offset + max_visible < len(state.networks):
        p.setPen(_CHAMPAGNE(int(90 * a)))
        p.setFont(chev_font)
        p.drawText(QRectF(cx - R * 0.08, list_top + max_visible * row_h,
                           R * 0.16, row_h * 0.35),
                   Qt.AlignCenter, "\u2304")

    # SCAN button — elegant pill shape
    btn_w = R * 0.38
    btn_h = R * 0.09
    btn_rect = QRectF(cx - btn_w / 2, cy + R * 0.62, btn_w, btn_h)

    # Button gradient
    btn_grad = QLinearGradient(btn_rect.topLeft(), btn_rect.bottomLeft())
    btn_grad.setColorAt(0.0, QColor(30, 35, 55, int(200 * a)))
    btn_grad.setColorAt(1.0, QColor(18, 22, 38, int(200 * a)))
    p.setPen(QPen(_CHAMPAGNE(int(100 * a)), max(1.0, mind * 0.0018)))
    p.setBrush(QBrush(btn_grad))
    p.drawRoundedRect(btn_rect, btn_h / 2, btn_h / 2)

    btn_font = QFont("DejaVu Serif", max(7, int(mind * 0.012)))
    btn_font.setLetterSpacing(QFont.AbsoluteSpacing, mind * 0.008)
    btn_font.setBold(True)
    p.setFont(btn_font)
    p.setPen(_CHAMPAGNE(int(190 * a)))
    p.drawText(btn_rect, Qt.AlignCenter, "SCAN")


def _draw_signal_bars(p, x, y, size, signal, a):
    """Refined signal strength bars — polished metal look."""
    bars = 4
    level = signal // 25
    for i in range(bars):
        bh = size * (0.30 + 0.22 * i)
        bw = size * 0.15
        bx = x + i * (bw + size * 0.08)
        by = y + size - bh
        if i < level:
            p.setPen(Qt.NoPen)
            p.setBrush(_ACCENT_TEAL(int(200 * a)))
        else:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(45, 50, 65, int(70 * a)))
        p.drawRoundedRect(QRectF(bx, by, bw, bh), 1.2, 1.2)


def _draw_keyboard(p, cx, cy, R, mind, a, state: WifiPageState):
    """Luxury on-screen keyboard for password entry."""
    if state.symbols:
        rows = _ROWS_SYMBOL
    elif state.shift:
        rows = _ROWS_UPPER
    else:
        rows = _ROWS_LOWER

    now = time.time()

    # ── Keyboard geometry — compute first so we can place everything ──
    kb_width = R * 1.54
    key_gap = max(2.0, mind * 0.005)
    key_h = max(mind * 0.054, 26)
    row_gap = max(2.0, mind * 0.004)
    n_char_rows = len(rows)
    # Total keyboard height: char rows + utility row + confirm row
    total_kb_h = (n_char_rows + 1) * (key_h + row_gap) + row_gap + key_h * 1.2
    # Center the entire block vertically, shifted down slightly
    kb_top = cy - total_kb_h * 0.38

    # ── SSID + CANCEL header ──────────────────────────────────────
    hdr_y = kb_top - R * 0.22
    ssid_font = QFont("DejaVu Sans", max(8, int(mind * 0.013)))
    ssid_font.setLetterSpacing(QFont.AbsoluteSpacing, mind * 0.003)
    p.setFont(ssid_font)
    p.setPen(_CHAMPAGNE(int(170 * a)))
    p.drawText(QRectF(cx - R * 0.50, hdr_y, R * 1.00, R * 0.06),
               Qt.AlignCenter, state.selected_ssid or "")

    cancel_font = QFont("DejaVu Sans", max(7, int(mind * 0.012)))
    p.setFont(cancel_font)
    p.setPen(_ROSE(int(160 * a)))
    cancel_rect = QRectF(cx + R * 0.42, hdr_y, R * 0.28, R * 0.06)
    p.drawText(cancel_rect, Qt.AlignCenter, "CANCEL")

    # ── Password field — compact, centered ─────────────────────────
    pw_w = R * 0.95  # narrower — nobody needs 30 chars visible
    pw_h = R * 0.08
    pw_y = kb_top - R * 0.12
    pw_rect = QRectF(cx - pw_w / 2, pw_y, pw_w, pw_h)

    # Inset shadow
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(3, 5, 12, int(220 * a)))
    p.drawRoundedRect(pw_rect, pw_h / 2, pw_h / 2)
    # Gold border
    p.setPen(QPen(_CHAMPAGNE(int(80 * a)), max(1.0, mind * 0.0018)))
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(pw_rect, pw_h / 2, pw_h / 2)

    pw_font = QFont("DejaVu Sans Mono", max(10, int(mind * 0.018)))
    p.setFont(pw_font)
    p.setPen(_IVORY(int(215 * a)))
    display = state.password if len(state.password) <= 16 else "..." + state.password[-13:]
    cursor = "\u2502" if int(now * 2) % 2 == 0 else ""
    p.drawText(pw_rect.adjusted(10, 0, -10, 0),
               Qt.AlignVCenter | Qt.AlignLeft, display + cursor)

    # ── Tap flash helpers ──────────────────────────────────────────
    def _is_flashing(row_i, col_i):
        f = state._tap_flash
        if f and f[0] == row_i and f[1] == col_i:
            return max(0.0, 1.0 - (now - f[2]) * 5.0)
        return 0.0

    def _is_special_flashing(label):
        f = state._tap_flash
        if f and f[0] == "special" and f[1] == label:
            return max(0.0, 1.0 - (now - f[2]) * 5.0)
        return 0.0

    # ── Character rows ─────────────────────────────────────────────
    key_font = QFont("DejaVu Sans", max(12, int(mind * 0.024)))
    key_font.setBold(False)

    for row_idx, row in enumerate(rows):
        row_y = kb_top + row_idx * (key_h + row_gap)
        n = len(row)
        key_w = (kb_width - (n - 1) * key_gap) / n
        row_x = cx - kb_width / 2

        for col_idx, key_char in enumerate(row):
            kx = row_x + col_idx * (key_w + key_gap)
            key_rect = QRectF(kx, row_y, key_w, key_h)
            flash = _is_flashing(row_idx, col_idx)

            key_bg = QLinearGradient(key_rect.topLeft(), key_rect.bottomLeft())
            br, bg_, bb = 48, 54, 74
            if flash > 0:
                fr = int(br + (180 - br) * flash)
                fg = int(bg_ + (165 - bg_) * flash)
                fb = int(bb + (120 - bb) * flash)
                key_bg.setColorAt(0.0, QColor(fr, fg, fb, int(190 * a)))
                key_bg.setColorAt(1.0, QColor(fr - 10, fg - 10, fb - 10, int(180 * a)))
            else:
                key_bg.setColorAt(0.0, QColor(br + 8, bg_ + 8, bb + 8, int(170 * a)))
                key_bg.setColorAt(1.0, QColor(br - 6, bg_ - 6, bb - 6, int(160 * a)))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(key_bg))
            p.drawRoundedRect(key_rect, 3, 3)

            border_a = int((55 + 100 * flash) * a)
            p.setPen(QPen(_CHAMPAGNE(border_a), max(0.7, mind * 0.0012)))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(key_rect, 3, 3)

            p.setFont(key_font)
            p.setPen(_IVORY(int((220 + 35 * flash) * a)))
            p.drawText(key_rect, Qt.AlignCenter, key_char)

    # ── Utility row: ⇧ | #+= | SPACE | ⌫  ────────────────────────
    bottom_y = kb_top + n_char_rows * (key_h + row_gap)
    sym_label = "ABC" if state.symbols else "#+="
    util_keys = [
        ("\u21E7",   0.12),
        (sym_label,  0.12),
        ("SPACE",    0.58),
        ("\u232B",   0.18),
    ]
    util_x = cx - kb_width / 2

    sp_font = QFont("DejaVu Sans", max(10, int(mind * 0.018)))
    sp_font.setBold(False)

    for label, frac in util_keys:
        w = kb_width * frac - key_gap
        key_rect = QRectF(util_x, bottom_y, w, key_h)
        flash = _is_special_flashing(label)
        is_active = ((label == "\u21E7" and state.shift) or
                     (label in ("#+=", "ABC") and state.symbols))

        key_bg = QLinearGradient(key_rect.topLeft(), key_rect.bottomLeft())
        if is_active:
            key_bg.setColorAt(0.0, QColor(120, 115, 80, int(160 * a)))
            key_bg.setColorAt(1.0, QColor(100, 95, 65, int(150 * a)))
        elif flash > 0:
            key_bg.setColorAt(0.0, QColor(int(48 + 130 * flash), int(54 + 110 * flash),
                                           int(74 + 50 * flash), int(180 * a)))
            key_bg.setColorAt(1.0, QColor(int(38 + 130 * flash), int(44 + 110 * flash),
                                           int(64 + 50 * flash), int(170 * a)))
        else:
            key_bg.setColorAt(0.0, QColor(38, 44, 62, int(160 * a)))
            key_bg.setColorAt(1.0, QColor(28, 34, 52, int(150 * a)))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(key_bg))
        p.drawRoundedRect(key_rect, 4, 4)

        border_a = int((45 + 80 * flash) * a)
        p.setPen(QPen(_CHAMPAGNE(border_a), max(0.7, mind * 0.0012)))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(key_rect, 4, 4)

        p.setFont(sp_font)
        p.setPen(_IVORY(int((190 + 50 * flash) * a)))
        p.drawText(key_rect, Qt.AlignCenter, label)
        util_x += w + key_gap

    # ── CONFIRM — Patek Philippe applied index style ───────────────
    # Not a flat green pill. A recessed channel with polished chamfers,
    # brushed steel face, and engraved serif lettering.
    confirm_y = bottom_y + key_h + row_gap * 2
    confirm_h = key_h * 1.2
    confirm_w = kb_width * 0.65
    confirm_rect = QRectF(cx - confirm_w / 2, confirm_y, confirm_w, confirm_h)
    cf = _is_special_flashing("CONFIRM")
    corner_r = max(4.0, mind * 0.008)

    # Layer 1: Outer shadow (depth)
    shadow_rect = confirm_rect.adjusted(-1.5, -1.5, 1.5, 2.5)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(0, 0, 0, int(100 * a)))
    p.drawRoundedRect(shadow_rect, corner_r + 1, corner_r + 1)

    # Layer 2: Outer chamfer — polished gold bevel
    chamfer_grad = QLinearGradient(confirm_rect.topLeft(), confirm_rect.bottomLeft())
    chamfer_grad.setColorAt(0.0, QColor(220, 200, 160, int((140 + 80 * cf) * a)))
    chamfer_grad.setColorAt(0.35, QColor(180, 160, 120, int((100 + 60 * cf) * a)))
    chamfer_grad.setColorAt(0.65, QColor(140, 120, 85, int((80 + 50 * cf) * a)))
    chamfer_grad.setColorAt(1.0, QColor(200, 185, 145, int((120 + 70 * cf) * a)))
    p.setBrush(QBrush(chamfer_grad))
    p.drawRoundedRect(confirm_rect, corner_r, corner_r)

    # Layer 3: Recessed face — dark brushed steel
    face_rect = confirm_rect.adjusted(2.5, 2.5, -2.5, -2.5)
    face_grad = QLinearGradient(face_rect.topLeft(), face_rect.bottomLeft())
    face_grad.setColorAt(0.0, QColor(int(28 + 25 * cf), int(32 + 20 * cf),
                                      int(42 + 15 * cf), int(230 * a)))
    face_grad.setColorAt(0.3, QColor(int(22 + 20 * cf), int(26 + 16 * cf),
                                      int(36 + 12 * cf), int(225 * a)))
    face_grad.setColorAt(0.7, QColor(int(18 + 18 * cf), int(22 + 14 * cf),
                                      int(32 + 10 * cf), int(225 * a)))
    face_grad.setColorAt(1.0, QColor(int(24 + 22 * cf), int(28 + 18 * cf),
                                      int(38 + 13 * cf), int(230 * a)))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(face_grad))
    p.drawRoundedRect(face_rect, corner_r - 1, corner_r - 1)

    # Layer 4: Fine horizontal brushing lines (brushed steel texture)
    brush_pen = QPen(QColor(180, 165, 130, int((8 + 12 * cf) * a)), 0.5)
    p.setPen(brush_pen)
    fy_start = int(face_rect.top()) + 2
    fy_end = int(face_rect.bottom()) - 2
    for by in range(fy_start, fy_end, 2):
        p.drawLine(QPointF(face_rect.left() + 4, by),
                   QPointF(face_rect.right() - 4, by))

    # Layer 5: Inner highlight (top edge catch light)
    p.setPen(QPen(QColor(240, 225, 190, int((25 + 40 * cf) * a)),
                  max(0.5, mind * 0.001)))
    p.setBrush(Qt.NoBrush)
    highlight_rect = face_rect.adjusted(1, 1, -1, -face_rect.height() * 0.6)
    p.drawRoundedRect(highlight_rect, corner_r - 2, corner_r - 2)

    # Layer 6: "CONFIRM" — engraved serif, inlaid gold
    cf_font = QFont("DejaVu Serif", max(9, int(mind * 0.018)))
    cf_font.setBold(False)
    cf_font.setLetterSpacing(QFont.AbsoluteSpacing, max(4.0, mind * 0.014))
    p.setFont(cf_font)
    # Engraved shadow (recessed into the steel)
    p.setPen(QColor(0, 0, 0, int(140 * a)))
    p.drawText(confirm_rect.adjusted(0, 1.5, 0, 1.5), Qt.AlignCenter, "CONFIRM")
    # Gold fill (inlaid lettering)
    gold_a = int((200 + 55 * cf) * a)
    p.setPen(QColor(218, 200, 155, gold_a))
    p.drawText(confirm_rect, Qt.AlignCenter, "CONFIRM")
    # Top highlight on letters (polished facet catch)
    p.setPen(QColor(255, 248, 230, int((40 + 60 * cf) * a)))
    p.drawText(confirm_rect.adjusted(0, -0.8, 0, -0.8), Qt.AlignCenter, "CONFIRM")


# ── Touch handling ───────────────────────────────────────────────────

def handle_wifi_tap(x: float, y: float, cx: float, cy: float,
                    mind: float, state: WifiPageState) -> Optional[str]:
    """Handle a tap on the WiFi page. Returns action string or None."""
    R = mind * _WIFI_R_FRAC

    # Back chevron (top-left)
    if (cx - R * 0.88 < x < cx - R * 0.63 and
            cy - R * 0.88 < y < cy - R * 0.74):
        if state.show_keyboard:
            state.show_keyboard = False
            state.password = ""
            return None
        return "back"

    if state.show_keyboard:
        return _handle_keyboard_tap(x, y, cx, cy, mind, R, state)
    else:
        return _handle_list_tap(x, y, cx, cy, mind, R, state)


def _handle_list_tap(x, y, cx, cy, mind, R, state: WifiPageState) -> Optional[str]:
    """Handle taps on the network list view."""
    list_top = cy - R * 0.65
    row_h = mind * 0.048
    max_visible = 7

    # Scan button
    btn_w = R * 0.38
    btn_h = R * 0.09
    btn_rect = QRectF(cx - btn_w / 2, cy + R * 0.62, btn_w, btn_h)
    if btn_rect.contains(QPointF(x, y)):
        state.trigger_scan()
        return "scan"

    # Scroll up
    if (cy - R * 0.65 - row_h * 0.5 < y < cy - R * 0.65 and
            cx - R * 0.1 < x < cx + R * 0.1 and state.scroll_offset > 0):
        state.scroll_offset -= 1
        return "scroll_up"

    # Scroll down
    scroll_bottom_y = list_top + max_visible * row_h
    if (scroll_bottom_y < y < scroll_bottom_y + row_h * 0.5 and
            cx - R * 0.1 < x < cx + R * 0.1 and
            state.scroll_offset + max_visible < len(state.networks)):
        state.scroll_offset += 1
        return "scroll_down"

    # Network row tap
    for i in range(max_visible):
        idx = state.scroll_offset + i
        if idx >= len(state.networks):
            break
        row_y = list_top + i * row_h
        row_rect = QRectF(cx - R * 0.72, row_y, R * 1.44, row_h)
        if row_rect.contains(QPointF(x, y)):
            net = state.networks[idx]
            state.selected_ssid = net.ssid
            if net.connected:
                state.status_msg = f"Already connected to {net.ssid}"
                state.status_time = time.time()
            elif net.security:
                state.show_keyboard = True
                state.password = ""
            else:
                state.trigger_connect()
            return None

    return None


def _handle_keyboard_tap(x, y, cx, cy, mind, R, state: WifiPageState) -> Optional[str]:
    """Handle taps on the keyboard view."""
    if state.symbols:
        rows = _ROWS_SYMBOL
    elif state.shift:
        rows = _ROWS_UPPER
    else:
        rows = _ROWS_LOWER

    kb_width = R * 1.54
    key_gap = max(2.0, mind * 0.005)
    key_h = max(mind * 0.054, 26)
    row_gap = max(2.0, mind * 0.004)
    n_char_rows = len(rows)
    total_kb_h = (n_char_rows + 1) * (key_h + row_gap) + row_gap + key_h * 1.2
    kb_top = cy - total_kb_h * 0.38  # must match _draw_keyboard
    now = time.time()

    # Cancel button (positioned relative to kb_top, matching draw)
    hdr_y = kb_top - R * 0.22
    cancel_rect = QRectF(cx + R * 0.42, hdr_y, R * 0.28, R * 0.06)
    if cancel_rect.contains(QPointF(x, y)):
        state.show_keyboard = False
        state.password = ""
        state.selected_ssid = None
        return None

    # Character rows
    for row_idx, row in enumerate(rows):
        row_y = kb_top + row_idx * (key_h + row_gap)
        n = len(row)
        key_w = (kb_width - (n - 1) * key_gap) / n
        row_x = cx - kb_width / 2

        if row_y <= y <= row_y + key_h:
            for col_idx, key_char in enumerate(row):
                kx = row_x + col_idx * (key_w + key_gap)
                if kx <= x <= kx + key_w:
                    state.password += key_char
                    state._tap_flash = (row_idx, col_idx, now)
                    return None

    # Utility row (shift, symbols, space, backspace)
    bottom_y = kb_top + len(rows) * (key_h + row_gap)
    if bottom_y <= y <= bottom_y + key_h:
        sym_label = "ABC" if state.symbols else "#+="
        util_keys = [
            ("\u21E7",   0.12),
            (sym_label,  0.12),
            ("SPACE",    0.58),
            ("\u232B",   0.18),
        ]
        ux = cx - kb_width / 2
        for label, frac in util_keys:
            w = kb_width * frac - key_gap
            if ux <= x <= ux + w:
                state._tap_flash = ("special", label, now)
                if label == "\u21E7":
                    state.shift = not state.shift
                    state.symbols = False
                elif label in ("#+=", "ABC"):
                    state.symbols = not state.symbols
                    state.shift = False
                elif label == "SPACE":
                    state.password += " "
                elif label == "\u232B":
                    state.password = state.password[:-1]
                return None
            ux += w + key_gap

    # CONFIRM button row (must match _draw_keyboard geometry)
    confirm_y = bottom_y + key_h + row_gap * 2
    confirm_h = key_h * 1.2
    confirm_w = kb_width * 0.65
    confirm_rect = QRectF(cx - confirm_w / 2, confirm_y, confirm_w, confirm_h)
    if confirm_rect.contains(QPointF(x, y)):
        state._tap_flash = ("special", "CONFIRM", now)
        state.trigger_connect()
        return None

    return None

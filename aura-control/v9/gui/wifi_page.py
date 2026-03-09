"""
gui.wifi_page -- On-screen WiFi configuration for the round display.

Renders inside the Settings overlay when settings_page == "wifi".
Uses nmcli for scanning and connecting.
"""

from __future__ import annotations

import math
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QBrush, QColor, QFont, QPen, QRadialGradient

from gui.renderer import clamp

# Horology steel-blue tone (matches settings.py)
GOLD = lambda a=255: QColor(145, 175, 215, a)  # noqa: E731


@dataclass
class WifiNetwork:
    ssid: str
    signal: int        # 0-100
    security: str      # "WPA2", "WPA3", "WEP", "" (open)
    connected: bool = False


# ---------------------------------------------------------------------------
# WiFi backend (nmcli)
# ---------------------------------------------------------------------------

def scan_networks() -> List[WifiNetwork]:
    """Scan for available WiFi networks via nmcli."""
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY,ACTIVE", "device", "wifi", "list"],
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
        # Sort: connected first, then by signal strength
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
                return addr.split("/")[0]  # strip CIDR
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Keyboard layout
# ---------------------------------------------------------------------------

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

_SPECIAL_KEYS = {"⇧": "shift", "␣": "space", "⌫": "backspace", "GO": "connect"}


# ---------------------------------------------------------------------------
# WiFi page state
# ---------------------------------------------------------------------------

class WifiPageState:
    """Holds the state for the WiFi configuration page."""

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

    def trigger_scan(self):
        """Start a background WiFi scan."""
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
        """Start a background connect attempt."""
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
                self.trigger_scan()  # refresh to show connected state

        threading.Thread(target=_connect, daemon=True).start()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_wifi_page(p, cx, cy, mind, t, trans, state: WifiPageState):
    """Render the WiFi configuration page."""
    a = clamp(float(trans), 0.0, 1.0)
    if a <= 0.002:
        return

    R = mind * 0.40

    # Auto-scan on first render
    if not state.networks and not state.scanning:
        state.trigger_scan()

    p.save()
    p.setRenderHint(p.Antialiasing, True)

    # --- Background ---
    bg = QRadialGradient(cx, cy, R)
    bg.setColorAt(0.00, QColor(6, 8, 14, int(245 * a)))
    bg.setColorAt(0.85, QColor(4, 6, 12, int(230 * a)))
    bg.setColorAt(1.00, QColor(0, 0, 0, 0))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(bg))
    p.drawEllipse(QPointF(cx, cy), R, R)

    # --- Bezel ---
    p.setPen(QPen(GOLD(int(160 * a)), max(2.0, mind * 0.003)))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(cx, cy), R * 0.97, R * 0.97)

    # --- Header ---
    hdr_font = QFont("DejaVu Sans", max(14, int(mind * 0.028)))
    hdr_font.setBold(True)
    p.setFont(hdr_font)
    p.setPen(GOLD(int(230 * a)))
    p.drawText(QRectF(cx - R, cy - R * 0.88, 2 * R, R * 0.15),
               Qt.AlignCenter, "WI-FI")

    # Back arrow (top-left)
    back_font = QFont("DejaVu Sans", max(16, int(mind * 0.032)))
    p.setFont(back_font)
    p.setPen(GOLD(int(180 * a)))
    p.drawText(QRectF(cx - R * 0.85, cy - R * 0.88, R * 0.3, R * 0.15),
               Qt.AlignCenter, "←")

    # IP address + SSH info (top-right) — cached to avoid subprocess spam
    now = time.time()
    if now - state._ip_fetch_time > 10.0:
        state._cached_ip = get_current_ip()
        state._ip_fetch_time = now
    ip = state._cached_ip
    if ip:
        ip_font = QFont("DejaVu Sans", max(8, int(mind * 0.014)))
        p.setFont(ip_font)
        p.setPen(QColor(120, 180, 255, int(180 * a)))
        p.drawText(QRectF(cx - R * 0.1, cy - R * 0.88, R * 0.95, R * 0.08),
                   Qt.AlignRight | Qt.AlignVCenter, ip)
        # SSH hint below IP
        ssh_font = QFont("DejaVu Sans", max(7, int(mind * 0.011)))
        p.setFont(ssh_font)
        p.setPen(QColor(100, 160, 220, int(120 * a)))
        p.drawText(QRectF(cx - R * 0.1, cy - R * 0.80, R * 0.95, R * 0.07),
                   Qt.AlignRight | Qt.AlignVCenter, f"SSH: ledger@{ip}")

    if state.show_keyboard:
        _draw_keyboard(p, cx, cy, R, mind, a, state)
    else:
        _draw_network_list(p, cx, cy, R, mind, t, a, state)

    # --- Status message ---
    if state.status_msg and (time.time() - state.status_time < 5.0):
        stat_font = QFont("DejaVu Sans", max(10, int(mind * 0.018)))
        p.setFont(stat_font)
        fade = max(0.0, 1.0 - (time.time() - state.status_time) / 5.0)
        p.setPen(QColor(180, 230, 255, int(200 * fade * a)))
        p.drawText(QRectF(cx - R * 0.8, cy + R * 0.78, R * 1.6, R * 0.12),
                   Qt.AlignCenter, state.status_msg)

    p.restore()


def _draw_network_list(p, cx, cy, R, mind, t, a, state: WifiPageState):
    """Draw the scrollable network list."""
    # Scanning indicator
    if state.scanning:
        sf = QFont("DejaVu Sans", max(10, int(mind * 0.018)))
        p.setFont(sf)
        p.setPen(QColor(150, 200, 255, int(150 * a)))
        p.drawText(QRectF(cx - R, cy - R * 0.68, 2 * R, R * 0.1),
                   Qt.AlignCenter, "Scanning...")
        return

    if not state.networks:
        sf = QFont("DejaVu Sans", max(10, int(mind * 0.018)))
        p.setFont(sf)
        p.setPen(QColor(150, 150, 150, int(150 * a)))
        p.drawText(QRectF(cx - R, cy - R * 0.2, 2 * R, R * 0.1),
                   Qt.AlignCenter, "No networks found")
        return

    # List area
    list_top = cy - R * 0.65
    row_h = mind * 0.055
    max_visible = 7
    offset = state.scroll_offset

    visible = state.networks[offset:offset + max_visible]

    ssid_font = QFont("DejaVu Sans", max(11, int(mind * 0.020)))
    detail_font = QFont("DejaVu Sans", max(8, int(mind * 0.014)))

    for i, net in enumerate(visible):
        row_y = list_top + i * row_h
        row_rect = QRectF(cx - R * 0.80, row_y, R * 1.60, row_h - 2)

        # Highlight selected
        if net.ssid == state.selected_ssid:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(80, 140, 220, int(50 * a)))
            p.drawRoundedRect(row_rect, 6, 6)

        # Connected indicator
        if net.connected:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(60, 200, 120, int(180 * a)))
            p.drawEllipse(QPointF(cx - R * 0.72, row_y + row_h * 0.4),
                          mind * 0.006, mind * 0.006)

        # SSID
        p.setFont(ssid_font)
        col = QColor(230, 240, 255, int(220 * a)) if not net.connected \
            else QColor(100, 230, 160, int(240 * a))
        p.setPen(col)
        p.drawText(QRectF(cx - R * 0.65, row_y, R * 1.0, row_h),
                   Qt.AlignVCenter | Qt.AlignLeft, net.ssid)

        # Signal bars
        _draw_signal_bars(p, cx + R * 0.55, row_y + row_h * 0.35,
                          mind * 0.025, net.signal, a)

        # Security icon
        if net.security:
            p.setFont(detail_font)
            p.setPen(QColor(180, 180, 200, int(140 * a)))
            p.drawText(QRectF(cx + R * 0.35, row_y, R * 0.2, row_h),
                       Qt.AlignVCenter | Qt.AlignRight, "🔒")

    # Scroll indicators
    if offset > 0:
        p.setPen(GOLD(int(120 * a)))
        p.setFont(QFont("DejaVu Sans", max(14, int(mind * 0.024))))
        p.drawText(QRectF(cx - R * 0.1, list_top - row_h * 0.6, R * 0.2, row_h * 0.5),
                   Qt.AlignCenter, "▲")
    if offset + max_visible < len(state.networks):
        p.setPen(GOLD(int(120 * a)))
        p.drawText(QRectF(cx - R * 0.1, list_top + max_visible * row_h, R * 0.2, row_h * 0.5),
                   Qt.AlignCenter, "▼")

    # Refresh button at bottom
    btn_font = QFont("DejaVu Sans", max(10, int(mind * 0.018)))
    btn_font.setBold(True)
    p.setFont(btn_font)
    p.setPen(GOLD(int(170 * a)))
    btn_rect = QRectF(cx - R * 0.25, cy + R * 0.60, R * 0.50, R * 0.12)
    p.drawRoundedRect(btn_rect, 8, 8)
    p.drawText(btn_rect, Qt.AlignCenter, "SCAN")


def _draw_signal_bars(p, x, y, size, signal, a):
    """Draw WiFi signal strength bars."""
    bars = 4
    level = signal // 25  # 0-4
    for i in range(bars):
        bh = size * (0.3 + 0.2 * i)
        bw = size * 0.18
        bx = x + i * (bw + size * 0.06)
        by = y + size - bh
        if i < level:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(100, 200, 140, int(200 * a)))
        else:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(80, 80, 100, int(100 * a)))
        p.drawRoundedRect(QRectF(bx, by, bw, bh), 2, 2)


def _draw_keyboard(p, cx, cy, R, mind, a, state: WifiPageState):
    """Draw the on-screen keyboard for password entry."""
    if state.symbols:
        rows = _ROWS_SYMBOL
    elif state.shift:
        rows = _ROWS_UPPER
    else:
        rows = _ROWS_LOWER

    # Password field
    pw_font = QFont("DejaVu Sans Mono", max(12, int(mind * 0.022)))
    p.setFont(pw_font)
    pw_rect = QRectF(cx - R * 0.75, cy - R * 0.62, R * 1.50, R * 0.10)
    p.setPen(QPen(GOLD(int(140 * a)), 1.5))
    p.setBrush(QColor(15, 15, 20, int(200 * a)))
    p.drawRoundedRect(pw_rect, 6, 6)

    # Show selected SSID above + cancel button
    ssid_font = QFont("DejaVu Sans", max(9, int(mind * 0.016)))
    p.setFont(ssid_font)
    p.setPen(QColor(150, 200, 255, int(180 * a)))
    p.drawText(QRectF(cx - R * 0.45, cy - R * 0.74, R * 0.90, R * 0.10),
               Qt.AlignCenter, state.selected_ssid or "")

    # Cancel button (top-right of keyboard area)
    cancel_font = QFont("DejaVu Sans", max(10, int(mind * 0.017)))
    p.setFont(cancel_font)
    p.setPen(QColor(220, 140, 140, int(200 * a)))
    cancel_rect = QRectF(cx + R * 0.45, cy - R * 0.74, R * 0.30, R * 0.10)
    p.drawText(cancel_rect, Qt.AlignCenter, "CANCEL")

    # Password text (masked)
    p.setFont(pw_font)
    p.setPen(QColor(230, 240, 255, int(230 * a)))
    display = state.password if len(state.password) <= 24 else "..." + state.password[-21:]
    cursor = "│" if int(time.time() * 2) % 2 == 0 else ""
    p.drawText(pw_rect.adjusted(8, 0, -8, 0),
               Qt.AlignVCenter | Qt.AlignLeft, display + cursor)

    # Keyboard rows
    key_h = mind * 0.048
    key_gap = mind * 0.004
    kb_top = cy - R * 0.46

    key_font = QFont("DejaVu Sans", max(11, int(mind * 0.020)))
    key_font.setBold(True)

    for row_idx, row in enumerate(rows):
        row_y = kb_top + row_idx * (key_h + key_gap)
        total_keys = len(row)
        key_w = min(mind * 0.042, (R * 1.50) / max(total_keys, 1))
        row_w = total_keys * (key_w + key_gap) - key_gap
        row_x = cx - row_w / 2

        for col_idx, key_char in enumerate(row):
            kx = row_x + col_idx * (key_w + key_gap)
            key_rect = QRectF(kx, row_y, key_w, key_h)

            # Key background
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(30, 35, 45, int(200 * a)))
            p.drawRoundedRect(key_rect, 4, 4)

            # Key border
            p.setPen(QPen(QColor(80, 100, 140, int(120 * a)), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(key_rect, 4, 4)

            # Key label
            p.setFont(key_font)
            p.setPen(QColor(210, 225, 245, int(220 * a)))
            p.drawText(key_rect, Qt.AlignCenter, key_char)

    # Bottom row: SHIFT, SYMBOLS, SPACE, BACKSPACE, CONNECT
    bottom_y = kb_top + len(rows) * (key_h + key_gap)
    sym_label = "ABC" if state.symbols else "#+="
    special_keys = [("⇧", R * 0.18), (sym_label, R * 0.18), ("SPACE", R * 0.42), ("⌫", R * 0.18), ("GO", R * 0.22)]
    total_w = sum(w for _, w in special_keys) + key_gap * (len(special_keys) - 1)
    sx = cx - total_w / 2

    for label, w in special_keys:
        key_rect = QRectF(sx, bottom_y, w, key_h)

        if label == "⇧" and state.shift:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(80, 140, 220, int(150 * a)))
        elif label in ("#+=", "ABC") and state.symbols:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(80, 140, 220, int(150 * a)))
        elif label == "GO":
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(60, 160, 100, int(180 * a)))
        else:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(30, 35, 45, int(200 * a)))
        p.drawRoundedRect(key_rect, 4, 4)

        p.setPen(QPen(QColor(80, 100, 140, int(120 * a)), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(key_rect, 4, 4)

        sp_font = QFont("DejaVu Sans", max(9, int(mind * 0.016)))
        sp_font.setBold(True)
        p.setFont(sp_font)
        p.setPen(QColor(210, 225, 245, int(220 * a)))
        p.drawText(key_rect, Qt.AlignCenter, label)

        sx += w + key_gap


# ---------------------------------------------------------------------------
# Touch handling
# ---------------------------------------------------------------------------

def handle_wifi_tap(x: float, y: float, cx: float, cy: float,
                    mind: float, state: WifiPageState) -> Optional[str]:
    """Handle a tap on the WiFi page. Returns action string or None.

    Actions: "back", "scan", "scroll_up", "scroll_down", None
    Handles network selection, keyboard input, and connection internally.
    """
    R = mind * 0.40

    # Back button (top-left area)
    if (cx - R * 0.85 < x < cx - R * 0.55 and
            cy - R * 0.88 < y < cy - R * 0.73):
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
    row_h = mind * 0.055
    max_visible = 7

    # Scan button
    btn_rect = QRectF(cx - R * 0.25, cy + R * 0.60, R * 0.50, R * 0.12)
    if btn_rect.contains(QPointF(x, y)):
        state.trigger_scan()
        return "scan"

    # Scroll up
    if (cy - R * 0.65 - row_h * 0.6 < y < cy - R * 0.65 and
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
        row_rect = QRectF(cx - R * 0.80, row_y, R * 1.60, row_h)
        if row_rect.contains(QPointF(x, y)):
            net = state.networks[idx]
            state.selected_ssid = net.ssid
            if net.connected:
                state.status_msg = f"Already connected to {net.ssid}"
                state.status_time = time.time()
            elif net.security:
                # Needs password
                state.show_keyboard = True
                state.password = ""
            else:
                # Open network — connect directly
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
    key_h = mind * 0.048
    key_gap = mind * 0.004
    kb_top = cy - R * 0.46

    # Cancel button (top-right of keyboard area)
    cancel_rect = QRectF(cx + R * 0.45, cy - R * 0.74, R * 0.30, R * 0.10)
    if cancel_rect.contains(QPointF(x, y)):
        state.show_keyboard = False
        state.password = ""
        state.selected_ssid = None
        return None

    # Keyboard bounds: from SSID label down to bottom of special keys row
    kb_bottom = kb_top + len(rows) * (key_h + key_gap) + key_h
    kb_zone = QRectF(cx - R * 0.80, cy - R * 0.76, R * 1.60, kb_bottom - (cy - R * 0.76))

    # Tap outside keyboard zone — dismiss
    if not kb_zone.contains(QPointF(x, y)):
        state.show_keyboard = False
        state.password = ""
        state.selected_ssid = None
        return None

    # Character rows
    for row_idx, row in enumerate(rows):
        row_y = kb_top + row_idx * (key_h + key_gap)
        total_keys = len(row)
        key_w = min(mind * 0.042, (R * 1.50) / max(total_keys, 1))
        row_w = total_keys * (key_w + key_gap) - key_gap
        row_x = cx - row_w / 2

        if row_y <= y <= row_y + key_h:
            for col_idx, key_char in enumerate(row):
                kx = row_x + col_idx * (key_w + key_gap)
                if kx <= x <= kx + key_w:
                    state.password += key_char
                    return None

    # Special keys row
    bottom_y = kb_top + len(rows) * (key_h + key_gap)
    if bottom_y <= y <= bottom_y + key_h:
        sym_label = "ABC" if state.symbols else "+#="
        special_keys = [("⇧", R * 0.18), (sym_label, R * 0.18), ("SPACE", R * 0.42), ("⌫", R * 0.18), ("GO", R * 0.22)]
        total_w = sum(w for _, w in special_keys) + key_gap * (len(special_keys) - 1)
        sx = cx - total_w / 2
        for label, w in special_keys:
            if sx <= x <= sx + w:
                if label == "⇧":
                    state.shift = not state.shift
                    state.symbols = False
                elif label in ("#+=", "ABC"):
                    state.symbols = not state.symbols
                    state.shift = False
                elif label == "SPACE":
                    state.password += " "
                elif label == "⌫":
                    state.password = state.password[:-1]
                elif label == "GO":
                    state.trigger_connect()
                return None
            sx += w + key_gap

    return None

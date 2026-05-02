"""
gui.auraconnect_page -- AuraConnect BLE sub-page for the Settings overlay.

Patek Philippe Grand Complications aesthetic: engine-turned guilloché,
polished triple bezel, chapter-ring ticks, champagne gold accents.

Draws a status panel and toggle for the Bluetooth GATT peripheral that
lets the macOS AuraConnect app pair with this device.
"""

from __future__ import annotations

import asyncio
import math
import os
import threading
import time
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QLinearGradient, QPen,
    QRadialGradient, QPainterPath,
)

from gui.renderer import clamp

# ── Palette ──────────────────────────────────────────────────────────
_CHAMPAGNE   = lambda a=255: QColor(218, 200, 155, a)
_IVORY       = lambda a=255: QColor(240, 234, 218, a)
_DEEP_NAVY   = lambda a=255: QColor(6, 9, 22, a)
_DIM_GOLD    = lambda a=255: QColor(165, 152, 118, a)
_ACCENT_TEAL = lambda a=255: QColor(80, 200, 165, a)
_ROSE        = lambda a=255: QColor(200, 130, 130, a)


# ---------------------------------------------------------------------------
# BLE GATT server (runs in a background thread with its own asyncio loop)
# ---------------------------------------------------------------------------

_ble_thread: Optional[threading.Thread] = None
_ble_loop: Optional[asyncio.AbstractEventLoop] = None
_ble_stop_event: Optional[asyncio.Event] = None
_ble_running = False
_ble_error: Optional[str] = None
_ble_connected = False

# File transfer state — AuraConnect sends JSON on CTRL: {"cmd":"BEGIN","name":...,"size":...,"sha256":...}
# DATA receives length-prefixed frames: [uint32_le_len][payload]
# CTRL {"cmd":"END","id":...} finalizes and writes to RAG input dir
_file_transfer_id: Optional[str] = None
_file_transfer_name: Optional[str] = None
_file_transfer_chunks: list = []
_file_transfer_size: int = 0
_file_transfer_received: int = 0
_RAG_INPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "input"

# UUIDs must match macOS app and /home/ledger/aura_gatt.py
AURA_SERVICE_UUID = "E7810A71-73AE-499D-8C15-FAA9AEF0C3F2"
CTRL_UUID         = "E7810A71-73AE-499D-8C15-FAA9AEF0C3F3"
DATA_UUID         = "E7810A71-73AE-499D-8C15-FAA9AEF0C3F4"
LOCAL_NAME        = "Aura Puck"


def _run_ble_server():
    """Entry point for the BLE background thread."""
    global _ble_running, _ble_error, _ble_loop, _ble_stop_event

    try:
        from dbus_next.aio import MessageBus
        from dbus_next import Variant, BusType
        from dbus_next.service import (
            ServiceInterface, method, dbus_property, PropertyAccess,
        )
    except ImportError as e:
        _ble_error = f"dbus_next not installed: {e}"
        _ble_running = False
        return

    BLUEZ = "org.bluez"
    OM_IFACE = "org.freedesktop.DBus.ObjectManager"
    PROP_IFACE = "org.freedesktop.DBus.Properties"
    LE_ADV_MGR = "org.bluez.LEAdvertisingManager1"
    GATT_MGR = "org.bluez.GattManager1"
    ADAPTER_IFACE = "org.bluez.Adapter1"
    LE_ADV_IFACE = "org.bluez.LEAdvertisement1"
    GATT_SVC_IFACE = "org.bluez.GattService1"
    GATT_CHR_IFACE = "org.bluez.GattCharacteristic1"

    BASE = "/org/bluez/aura"
    APP_PATH = f"{BASE}/app"
    SVC_PATH = f"{APP_PATH}/service0"
    CTRL_PATH = f"{SVC_PATH}/ctrl"
    DATA_PATH = f"{SVC_PATH}/data"
    ADV_PATH = f"{BASE}/advertisement0"

    class AuraAdvertisement(ServiceInterface):
        def __init__(self):
            super().__init__(LE_ADV_IFACE)

        @method()
        def Release(self):
            return

        @dbus_property(access=PropertyAccess.READ)
        def Type(self) -> "s":
            return "peripheral"

        @dbus_property(access=PropertyAccess.READ)
        def ServiceUUIDs(self) -> "as":
            return [AURA_SERVICE_UUID]

        @dbus_property(access=PropertyAccess.READ)
        def LocalName(self) -> "s":
            return LOCAL_NAME

        @dbus_property(access=PropertyAccess.READ)
        def Includes(self) -> "as":
            return ["local-name"]

    class AuraService(ServiceInterface):
        def __init__(self):
            super().__init__(GATT_SVC_IFACE)

        @dbus_property(access=PropertyAccess.READ)
        def UUID(self) -> "s":
            return AURA_SERVICE_UUID

        @dbus_property(access=PropertyAccess.READ)
        def Primary(self) -> "b":
            return True

    class CtrlCharacteristic(ServiceInterface):
        def __init__(self):
            super().__init__(GATT_CHR_IFACE)
            self._value = b""
            self._notifying = False

        @dbus_property(access=PropertyAccess.READ)
        def UUID(self) -> "s":
            return CTRL_UUID

        @dbus_property(access=PropertyAccess.READ)
        def Service(self) -> "o":
            return SVC_PATH

        @dbus_property(access=PropertyAccess.READ)
        def Flags(self) -> "as":
            return ["write", "notify"]

        @dbus_property(access=PropertyAccess.READ)
        def Value(self) -> "ay":
            return self._value

        @method()
        def ReadValue(self, options: "a{sv}") -> "ay":
            return self._value

        @method()
        def WriteValue(self, value: "ay", options: "a{sv}"):
            global _ble_connected, _file_transfer_id, _file_transfer_name
            global _file_transfer_chunks, _file_transfer_size, _file_transfer_received
            data = value if isinstance(value, (bytes, bytearray)) else bytes(value)
            self._value = data
            _ble_connected = True

            # AuraConnect sends JSON commands on CTRL
            try:
                import json as _json
                text = data.decode("utf-8", errors="replace").strip()
                msg = _json.loads(text)
                cmd = msg.get("cmd", "")
            except Exception:
                text = data.decode("utf-8", errors="replace").strip()
                print(f"[auraconnect] CTRL (non-JSON): {text}")
                if self._notifying:
                    self.emit_properties_changed({"Value": self._value}, [])
                return

            print(f"[auraconnect] CTRL cmd={cmd}")

            if cmd == "BEGIN":
                fname = os.path.basename(msg.get("name", ""))
                if fname:
                    _file_transfer_id = msg.get("id", "")
                    _file_transfer_name = fname
                    _file_transfer_size = msg.get("size", 0)
                    _file_transfer_chunks = []
                    _file_transfer_received = 0
                    print(f"[auraconnect] Transfer BEGIN: {fname} ({_file_transfer_size} bytes, sha={msg.get('sha256', '?')[:12]})")

            elif cmd == "END" and _file_transfer_name:
                _RAG_INPUT_DIR.mkdir(parents=True, exist_ok=True)
                out_path = _RAG_INPUT_DIR / _file_transfer_name
                file_data = b"".join(_file_transfer_chunks)
                out_path.write_bytes(file_data)
                print(f"[auraconnect] Transfer END: {out_path} ({len(file_data)} bytes) → RAG auto-ingest")
                _file_transfer_id = None
                _file_transfer_name = None
                _file_transfer_chunks = []
                _file_transfer_size = 0
                _file_transfer_received = 0

            elif cmd == "ABORT":
                print(f"[auraconnect] Transfer ABORT — discarding {_file_transfer_name}")
                _file_transfer_id = None
                _file_transfer_name = None
                _file_transfer_chunks = []
                _file_transfer_size = 0
                _file_transfer_received = 0

            if self._notifying:
                self.emit_properties_changed({"Value": self._value}, [])

        @method()
        def StartNotify(self):
            global _ble_connected
            self._notifying = True
            _ble_connected = True
            print("[auraconnect] Notify enabled — Mac connected")

        @method()
        def StopNotify(self):
            global _ble_connected
            self._notifying = False
            _ble_connected = False
            print("[auraconnect] Notify disabled")

    class DataCharacteristic(ServiceInterface):
        def __init__(self):
            super().__init__(GATT_CHR_IFACE)
            self.total_bytes = 0

        @dbus_property(access=PropertyAccess.READ)
        def UUID(self) -> "s":
            return DATA_UUID

        @dbus_property(access=PropertyAccess.READ)
        def Service(self) -> "o":
            return SVC_PATH

        @dbus_property(access=PropertyAccess.READ)
        def Flags(self) -> "as":
            return ["write-without-response"]

        @dbus_property(access=PropertyAccess.READ)
        def Value(self) -> "ay":
            return b""

        @method()
        def WriteValue(self, value: "ay", options: "a{sv}"):
            global _file_transfer_chunks, _file_transfer_received
            data = value if isinstance(value, (bytes, bytearray)) else bytes(value)
            self.total_bytes += len(data)

            # AuraConnect sends length-prefixed frames: [uint32_le][payload]
            if _file_transfer_name and len(data) > 4:
                import struct
                payload_len = struct.unpack_from("<I", data, 0)[0]
                payload = data[4:4 + payload_len]
                _file_transfer_chunks.append(payload)
                _file_transfer_received += len(payload)
            elif _file_transfer_name:
                # Runt frame — still buffer it
                _file_transfer_chunks.append(data)

    class AuraObjectManager(ServiceInterface):
        def __init__(self):
            super().__init__(OM_IFACE)

        @method()
        def GetManagedObjects(self) -> "a{oa{sa{sv}}}":
            return {
                APP_PATH: {OM_IFACE: {}},
                SVC_PATH: {
                    GATT_SVC_IFACE: {
                        "UUID": Variant("s", AURA_SERVICE_UUID),
                        "Primary": Variant("b", True),
                        "Includes": Variant("ao", []),
                        "Characteristics": Variant("ao", [CTRL_PATH, DATA_PATH]),
                    }
                },
                CTRL_PATH: {
                    GATT_CHR_IFACE: {
                        "UUID": Variant("s", CTRL_UUID),
                        "Service": Variant("o", SVC_PATH),
                        "Flags": Variant("as", ["write", "notify"]),
                        "Value": Variant("ay", b""),
                    }
                },
                DATA_PATH: {
                    GATT_CHR_IFACE: {
                        "UUID": Variant("s", DATA_UUID),
                        "Service": Variant("o", SVC_PATH),
                        "Flags": Variant("as", ["write-without-response"]),
                        "Value": Variant("ay", b""),
                    }
                },
            }

    def _ensure_bluetooth_experimental():
        """Ensure bluetoothd is running with --experimental (required for LE advertising).
        Auto-fixes the service file and restarts if needed. One-time self-healing."""
        import subprocess

        # Check if bluetoothd already has --experimental
        try:
            result = subprocess.run(
                ["systemctl", "show", "bluetooth", "--property=ExecStart"],
                capture_output=True, text=True, timeout=5,
            )
            if "--experimental" in result.stdout:
                return  # Already configured
        except Exception:
            pass

        print("[auraconnect] bluetoothd missing --experimental flag — fixing...")
        # Try to patch the service file
        patch_cmds = [
            "sudo sed -i 's|ExecStart=.*bluetoothd.*|& --experimental|' /lib/systemd/system/bluetooth.service",
            "sudo systemctl daemon-reload",
            "sudo systemctl restart bluetooth",
        ]
        for cmd in patch_cmds:
            try:
                subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
            except Exception as e:
                print(f"[auraconnect] Fix attempt failed: {e}")
                return

        import time
        time.sleep(2)
        print("[auraconnect] bluetoothd restarted with --experimental")

    def _ensure_btmgmt_sudoers():
        """Ensure btmgmt can run without password prompt."""
        import subprocess
        sudoers_file = "/etc/sudoers.d/aura-btmgmt"
        try:
            result = subprocess.run(
                ["test", "-f", sudoers_file],
                capture_output=True, timeout=3,
            )
            if result.returncode == 0:
                return  # Already exists
        except Exception:
            pass

        print("[auraconnect] Adding passwordless sudo for btmgmt...")
        try:
            # Get current username
            import getpass
            user = getpass.getuser()
            rule = f"{user} ALL=(ALL) NOPASSWD: /usr/bin/btmgmt"
            subprocess.run(
                f"echo '{rule}' | sudo tee {sudoers_file} && sudo chmod 440 {sudoers_file}",
                shell=True, capture_output=True, timeout=5,
            )
        except Exception as e:
            print(f"[auraconnect] sudoers fix failed: {e}")

    async def _server_main():
        global _ble_running, _ble_error, _ble_stop_event, _ble_connected

        _ble_stop_event = asyncio.Event()

        try:
            import subprocess

            # Self-heal: ensure system is configured for BLE advertising
            _ensure_bluetooth_experimental()
            _ensure_btmgmt_sudoers()

            # Ensure LE advertising is enabled, adapter discoverable, and name set
            try:
                subprocess.run(["sudo", "btmgmt", "le", "on"], capture_output=True, timeout=5)
                subprocess.run(["sudo", "btmgmt", "advertising", "on"], capture_output=True, timeout=5)
                subprocess.run(["sudo", "btmgmt", "name", LOCAL_NAME], capture_output=True, timeout=5)
                subprocess.run(["sudo", "btmgmt", "discov", "on"], capture_output=True, timeout=5)
                subprocess.run(["sudo", "btmgmt", "connectable", "on"], capture_output=True, timeout=5)
            except Exception as e:
                print(f"[auraconnect] btmgmt setup warning: {e}")

            bus = await MessageBus(bus_type=BusType.SYSTEM).connect()

            # Find adapter
            intro = await bus.introspect(BLUEZ, "/")
            obj = bus.get_proxy_object(BLUEZ, "/", intro)
            om = obj.get_interface(OM_IFACE)
            objects = await om.call_get_managed_objects()

            adv_mgr_path = gatt_mgr_path = None
            for path, ifaces in objects.items():
                if LE_ADV_MGR in ifaces and adv_mgr_path is None:
                    adv_mgr_path = path
                if GATT_MGR in ifaces and gatt_mgr_path is None:
                    gatt_mgr_path = path

            if not adv_mgr_path or not gatt_mgr_path:
                _ble_error = "No Bluetooth adapter found"
                _ble_running = False
                return

            # Power on adapter and make discoverable
            intro2 = await bus.introspect(BLUEZ, adv_mgr_path)
            obj2 = bus.get_proxy_object(BLUEZ, adv_mgr_path, intro2)
            props = obj2.get_interface(PROP_IFACE)
            await props.call_set(ADAPTER_IFACE, "Powered", Variant("b", True))
            await props.call_set(ADAPTER_IFACE, "Discoverable", Variant("b", True))
            await props.call_set(ADAPTER_IFACE, "Alias", Variant("s", LOCAL_NAME))

            # Export GATT tree
            bus.export(APP_PATH, AuraObjectManager())
            bus.export(SVC_PATH, AuraService())
            bus.export(CTRL_PATH, CtrlCharacteristic())
            bus.export(DATA_PATH, DataCharacteristic())
            bus.export(ADV_PATH, AuraAdvertisement())

            # Register GATT application
            intro3 = await bus.introspect(BLUEZ, gatt_mgr_path)
            obj3 = bus.get_proxy_object(BLUEZ, gatt_mgr_path, intro3)
            gatt_mgr = obj3.get_interface(GATT_MGR)
            await gatt_mgr.call_register_application(APP_PATH, {})
            print(f"[auraconnect] GATT application registered at {APP_PATH}")

            # Register LE advertisement
            intro4 = await bus.introspect(BLUEZ, adv_mgr_path)
            obj4 = bus.get_proxy_object(BLUEZ, adv_mgr_path, intro4)
            adv_mgr = obj4.get_interface(LE_ADV_MGR)
            await adv_mgr.call_register_advertisement(ADV_PATH, {})
            print(f"[auraconnect] Advertisement registered at {ADV_PATH}")

            _ble_running = True
            _ble_error = None
            print(f"[auraconnect] Advertising '{LOCAL_NAME}' — waiting for Mac app")

            await _ble_stop_event.wait()

            # Cleanup
            _ble_connected = False
            try:
                await adv_mgr.call_unregister_advertisement(ADV_PATH)
            except Exception:
                pass
            try:
                await gatt_mgr.call_unregister_application(APP_PATH)
            except Exception:
                pass

            print("[auraconnect] BLE server stopped")

        except Exception as e:
            _ble_error = str(e)
            print(f"[auraconnect] BLE error: {e}")
        finally:
            _ble_running = False

    loop = asyncio.new_event_loop()
    _ble_loop = loop
    try:
        loop.run_until_complete(_server_main())
    finally:
        loop.close()
        _ble_loop = None


def start_ble():
    """Start the BLE GATT server in a background thread."""
    global _ble_thread, _ble_running, _ble_error, _ble_connected
    if _ble_thread and _ble_thread.is_alive():
        return  # already running
    _ble_error = None
    _ble_connected = False
    _ble_thread = threading.Thread(target=_run_ble_server, daemon=True)
    _ble_thread.start()


def stop_ble():
    """Signal the BLE server to stop."""
    global _ble_stop_event, _ble_loop
    if _ble_stop_event and _ble_loop:
        _ble_loop.call_soon_threadsafe(_ble_stop_event.set)


def is_ble_running() -> bool:
    return _ble_running


def is_ble_connected() -> bool:
    return _ble_connected


def get_ble_error() -> Optional[str]:
    return _ble_error


# ---------------------------------------------------------------------------
# Drawing — Patek Philippe Grand Complications aesthetic
# ---------------------------------------------------------------------------

_WIFI_R_FRAC = 0.33  # match wifi_page radius for consistent sub-page sizing


def draw_auraconnect_page(p, cx, cy, mind, t, trans, state=None):
    """Draw the AuraConnect BLE status page — Patek Philippe aesthetic."""
    a = clamp(float(trans), 0.0, 1.0)
    if a <= 0.002:
        return

    R = mind * _WIFI_R_FRAC

    p.save()
    p.setRenderHint(p.Antialiasing, True)

    # ── Clip to circle ───────────────────────────────────────────
    clip = QPainterPath()
    clip.addEllipse(QPointF(cx, cy), R, R)
    p.setClipPath(clip)

    # ── Deep lacquer background with vignette ────────────────────
    bg = QRadialGradient(cx, cy, R)
    bg.setColorAt(0.00, QColor(10, 14, 32, int(252 * a)))
    bg.setColorAt(0.55, QColor(6, 10, 24, int(248 * a)))
    bg.setColorAt(0.85, QColor(3, 5, 14, int(245 * a)))
    bg.setColorAt(1.00, QColor(1, 2, 8, int(240 * a)))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(bg))
    p.drawEllipse(QPointF(cx, cy), R, R)

    # ── Engine-turned guilloché (radial + concentric) ────────────
    gu_pen = QPen(_CHAMPAGNE(int(12 * a)), max(0.4, R * 0.0015))
    p.setPen(gu_pen)
    p.setBrush(Qt.NoBrush)
    for i in range(72):
        angle = (2 * math.pi * i) / 72
        wave = 1.0 + 0.006 * math.sin(i * 5 + t * 0.2)
        x1 = cx + R * 0.10 * math.cos(angle)
        y1 = cy + R * 0.10 * math.sin(angle)
        x2 = cx + R * 0.88 * wave * math.cos(angle)
        y2 = cy + R * 0.88 * wave * math.sin(angle)
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    ring_pen = QPen(_CHAMPAGNE(int(8 * a)), max(0.3, R * 0.001))
    p.setPen(ring_pen)
    for frac in (0.22, 0.38, 0.54, 0.70, 0.84):
        rr = R * frac
        p.drawEllipse(QPointF(cx, cy), rr, rr)

    # ── Triple bezel ─────────────────────────────────────────────
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(_CHAMPAGNE(int(160 * a)), max(3.0, mind * 0.005)))
    p.drawEllipse(QPointF(cx, cy), R * 0.97, R * 0.97)
    p.setPen(QPen(_DIM_GOLD(int(55 * a)), max(1.5, mind * 0.0025)))
    p.drawEllipse(QPointF(cx, cy), R * 0.94, R * 0.94)
    p.setPen(QPen(_CHAMPAGNE(int(90 * a)), max(1.2, mind * 0.002)))
    p.drawEllipse(QPointF(cx, cy), R * 0.91, R * 0.91)

    # ── Chapter ring (60 ticks) ──────────────────────────────────
    for i in range(60):
        ang = (i / 60.0) * 2.0 * math.pi - math.pi / 2.0
        is_major = (i % 5 == 0)
        tick_out = R * 0.965
        tick_in = tick_out - (R * 0.055 if is_major else R * 0.030)
        tw = (mind * 0.003 if is_major else mind * 0.0015)
        col = _CHAMPAGNE(int((150 if is_major else 70) * a))
        p.setPen(QPen(col, tw, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(
            QPointF(cx + tick_in * math.cos(ang), cy + tick_in * math.sin(ang)),
            QPointF(cx + tick_out * math.cos(ang), cy + tick_out * math.sin(ang)),
        )

    # ── Beveled depth shadow ─────────────────────────────────────
    depth = QRadialGradient(cx, cy, R * 0.91)
    depth.setColorAt(0.70, QColor(0, 0, 0, 0))
    depth.setColorAt(0.92, QColor(0, 0, 0, int(55 * a)))
    depth.setColorAt(1.00, QColor(0, 0, 0, int(95 * a)))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(depth))
    p.drawEllipse(QPointF(cx, cy), R * 0.91, R * 0.91)

    # ── Crystal reflection ───────────────────────────────────────
    hl = QRadialGradient(cx - R * 0.15, cy - R * 0.25, R * 0.7)
    hl.setColorAt(0.0, QColor(255, 255, 255, int(7 * a)))
    hl.setColorAt(0.5, QColor(255, 255, 255, int(2 * a)))
    hl.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.setBrush(QBrush(hl))
    p.drawEllipse(QPointF(cx, cy), R * 0.90, R * 0.90)

    # ── Header: "AURACONNECT" ────────────────────────────────────
    hdr_font = QFont("DejaVu Serif", max(10, int(mind * 0.018)))
    hdr_font.setLetterSpacing(QFont.AbsoluteSpacing, mind * 0.010)
    hdr_font.setBold(True)
    p.setFont(hdr_font)
    p.setPen(QColor(0, 0, 0, int(120 * a)))
    p.drawText(QRectF(cx - R, cy - R * 0.84 + 1, 2 * R, R * 0.10),
               Qt.AlignCenter, "AURACONNECT")
    p.setPen(_CHAMPAGNE(int(235 * a)))
    p.drawText(QRectF(cx - R, cy - R * 0.84, 2 * R, R * 0.10),
               Qt.AlignCenter, "AURACONNECT")

    # Thin gold separator
    sep_y = cy - R * 0.72
    p.setPen(QPen(_CHAMPAGNE(int(40 * a)), max(0.6, mind * 0.001)))
    p.drawLine(QPointF(cx - R * 0.48, sep_y), QPointF(cx + R * 0.48, sep_y))

    # ── BLE Status ───────────────────────────────────────────────
    running = is_ble_running()
    connected = is_ble_connected()
    error = get_ble_error()

    # ── Central Bluetooth icon — engine-turned sub-dial ──────────
    sub_r = R * 0.22
    sub_cy = cy - R * 0.18

    # Sub-dial background
    sub_bg = QRadialGradient(cx, sub_cy, sub_r)
    sub_bg.setColorAt(0.0, QColor(16, 20, 40, int(200 * a)))
    sub_bg.setColorAt(0.8, QColor(8, 12, 28, int(220 * a)))
    sub_bg.setColorAt(1.0, QColor(4, 6, 16, int(230 * a)))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(sub_bg))
    p.drawEllipse(QPointF(cx, sub_cy), sub_r, sub_r)

    # Sub-dial bezel
    p.setPen(QPen(_CHAMPAGNE(int(100 * a)), max(1.5, mind * 0.0025)))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(cx, sub_cy), sub_r, sub_r)
    p.setPen(QPen(_DIM_GOLD(int(40 * a)), max(0.8, mind * 0.0012)))
    p.drawEllipse(QPointF(cx, sub_cy), sub_r * 0.90, sub_r * 0.90)

    # Sub-dial guilloché (fine radial lines)
    for i in range(24):
        ang = (2 * math.pi * i) / 24
        p.setPen(QPen(_CHAMPAGNE(int(10 * a)), 0.3))
        p.drawLine(
            QPointF(cx + sub_r * 0.15 * math.cos(ang), sub_cy + sub_r * 0.15 * math.sin(ang)),
            QPointF(cx + sub_r * 0.85 * math.cos(ang), sub_cy + sub_r * 0.85 * math.sin(ang)),
        )

    # Status-dependent glow
    if running:
        if connected:
            pulse = 0.7 + 0.3 * math.sin(t * 2.0)
            glow_col = QColor(80, 200, 140, int(pulse * 90 * a))
            status_text = "CONNECTED"
            status_col = _ACCENT_TEAL(int(230 * a))
        else:
            pulse = 0.5 + 0.5 * math.sin(t * 3.0)
            glow_col = QColor(218, 200, 155, int(pulse * 60 * a))
            status_text = "ADVERTISING"
            status_col = _CHAMPAGNE(int(220 * a))
    else:
        pulse = 0.3
        glow_col = QColor(80, 80, 100, int(30 * a))
        if error:
            status_text = "ERROR"
            status_col = _ROSE(int(200 * a))
        else:
            status_text = "OFF"
            status_col = QColor(120, 120, 135, int(180 * a))

    # Glow halo in sub-dial
    halo = QRadialGradient(QPointF(cx, sub_cy), sub_r * 1.2)
    halo.setColorAt(0.0, glow_col)
    halo.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(halo))
    p.drawEllipse(QPointF(cx, sub_cy), sub_r * 1.2, sub_r * 1.2)

    # Bluetooth rune (refined)
    _draw_bt_icon(p, cx, sub_cy, sub_r * 0.55, a, connected, running)

    # ── Status text (applied index engraving) ────────────────────
    stat_font = QFont("DejaVu Serif", max(9, int(mind * 0.018)))
    stat_font.setBold(True)
    stat_font.setLetterSpacing(QFont.AbsoluteSpacing, mind * 0.008)
    p.setFont(stat_font)
    # Shadow
    p.setPen(QColor(0, 0, 0, int(100 * a)))
    p.drawText(QRectF(cx - R, cy + R * 0.10 + 1, 2 * R, R * 0.12),
               Qt.AlignCenter, status_text)
    p.setPen(status_col)
    p.drawText(QRectF(cx - R, cy + R * 0.10, 2 * R, R * 0.12),
               Qt.AlignCenter, status_text)

    # ── Device name cartouche ────────────────────────────────────
    name_font = QFont("DejaVu Serif", max(8, int(mind * 0.013)))
    name_font.setLetterSpacing(QFont.AbsoluteSpacing, mind * 0.005)
    p.setFont(name_font)
    p.setPen(_DIM_GOLD(int(130 * a)))
    p.drawText(QRectF(cx - R, cy + R * 0.22, 2 * R, R * 0.09),
               Qt.AlignCenter, f"\u201C{LOCAL_NAME}\u201D")

    # ── Error message ────────────────────────────────────────────
    if error and not running:
        err_font = QFont("DejaVu Sans", max(7, int(mind * 0.011)))
        p.setFont(err_font)
        p.setPen(_ROSE(int(160 * a)))
        err_short = error[:38] + "\u2026" if len(error) > 38 else error
        p.drawText(QRectF(cx - R * 0.75, cy + R * 0.32, R * 1.50, R * 0.08),
                   Qt.AlignCenter, err_short)

    # ── Toggle button — Patek Philippe applied index style ───────
    btn_w = R * 0.60
    btn_h = R * 0.11
    btn_y = cy + R * 0.45
    btn_rect = QRectF(cx - btn_w / 2, btn_y, btn_w, btn_h)
    corner_r = max(3.0, mind * 0.006)

    # Shadow
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(0, 0, 0, int(80 * a)))
    p.drawRoundedRect(btn_rect.adjusted(-1, -1, 1, 2), corner_r + 1, corner_r + 1)

    # Chamfer — gold bevel
    chamfer = QLinearGradient(btn_rect.topLeft(), btn_rect.bottomLeft())
    if running:
        chamfer.setColorAt(0.0, QColor(200, 130, 130, int(130 * a)))
        chamfer.setColorAt(0.5, QColor(160, 90, 90, int(90 * a)))
        chamfer.setColorAt(1.0, QColor(180, 110, 110, int(110 * a)))
    else:
        chamfer.setColorAt(0.0, QColor(200, 195, 150, int(130 * a)))
        chamfer.setColorAt(0.5, QColor(160, 145, 110, int(90 * a)))
        chamfer.setColorAt(1.0, QColor(180, 170, 130, int(110 * a)))
    p.setBrush(QBrush(chamfer))
    p.drawRoundedRect(btn_rect, corner_r, corner_r)

    # Recessed steel face
    face = btn_rect.adjusted(2, 2, -2, -2)
    face_grad = QLinearGradient(face.topLeft(), face.bottomLeft())
    face_grad.setColorAt(0.0, QColor(28, 32, 42, int(230 * a)))
    face_grad.setColorAt(0.5, QColor(22, 26, 36, int(225 * a)))
    face_grad.setColorAt(1.0, QColor(26, 30, 40, int(230 * a)))
    p.setBrush(QBrush(face_grad))
    p.drawRoundedRect(face, corner_r - 1, corner_r - 1)

    # Brushing texture
    brush_pen = QPen(QColor(160, 148, 120, int(7 * a)), 0.4)
    p.setPen(brush_pen)
    for by in range(int(face.top()) + 2, int(face.bottom()) - 1, 2):
        p.drawLine(QPointF(face.left() + 3, by), QPointF(face.right() - 3, by))

    # Button text
    btn_text = "STOP" if running else "START"
    btn_font = QFont("DejaVu Serif", max(8, int(mind * 0.015)))
    btn_font.setLetterSpacing(QFont.AbsoluteSpacing, max(3.0, mind * 0.010))
    p.setFont(btn_font)
    p.setPen(QColor(0, 0, 0, int(120 * a)))
    p.drawText(btn_rect.adjusted(0, 1, 0, 1), Qt.AlignCenter, btn_text)
    if running:
        p.setPen(_ROSE(int(200 * a)))
    else:
        p.setPen(_CHAMPAGNE(int(210 * a)))
    p.drawText(btn_rect, Qt.AlignCenter, btn_text)

    # ── Connection jewel indicator ───────────────────────────────
    if connected:
        jewel_r = max(3.0, mind * 0.006)
        jewel_cy = cy + R * 0.62
        # Halo
        jh = QRadialGradient(cx, jewel_cy, jewel_r * 4)
        jh.setColorAt(0.0, QColor(80, 200, 140, int(50 * a)))
        jh.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(jh))
        p.drawEllipse(QPointF(cx, jewel_cy), jewel_r * 4, jewel_r * 4)
        # Jewel
        jg = QRadialGradient(cx - jewel_r * 0.3, jewel_cy - jewel_r * 0.3, jewel_r)
        jg.setColorAt(0.0, QColor(160, 255, 200, int(240 * a)))
        jg.setColorAt(0.5, QColor(80, 200, 140, int(220 * a)))
        jg.setColorAt(1.0, QColor(40, 120, 80, int(180 * a)))
        p.setBrush(QBrush(jg))
        p.drawEllipse(QPointF(cx, jewel_cy), jewel_r, jewel_r)
        # Spec highlight
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, int(120 * a)))
        p.drawEllipse(QPointF(cx - jewel_r * 0.25, jewel_cy - jewel_r * 0.25),
                       jewel_r * 0.3, jewel_r * 0.2)

    # ── "BLUETOOTH  LOW  ENERGY" engraved at bottom ──────────────
    ble_font = QFont("DejaVu Serif", max(6, int(mind * 0.009)))
    ble_font.setLetterSpacing(QFont.AbsoluteSpacing, mind * 0.006)
    p.setFont(ble_font)
    p.setPen(_DIM_GOLD(int(60 * a)))
    p.drawText(QRectF(cx - R, cy + R * 0.72, 2 * R, R * 0.06),
               Qt.AlignCenter, "BLUETOOTH  LOW  ENERGY")

    # ── Back chevron ─────────────────────────────────────────────
    back_font = QFont("DejaVu Sans", max(13, int(mind * 0.026)))
    p.setFont(back_font)
    p.setPen(_CHAMPAGNE(int(140 * a)))
    p.drawText(QRectF(cx - R * 0.88, cy - R * 0.88, R * 0.25, R * 0.14),
               Qt.AlignCenter, "\u2039")

    p.restore()


def _draw_bt_icon(p, cx, cy, r, a, connected, running):
    """Draw a refined Bluetooth rune with metallic finish."""
    if connected:
        col = _ACCENT_TEAL(int(230 * a))
    elif running:
        col = _CHAMPAGNE(int(200 * a))
    else:
        col = QColor(100, 100, 115, int(150 * a))

    pen = QPen(col)
    pen.setWidthF(max(1.8, r * 0.12))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)

    # Shadow
    shd = QPen(QColor(0, 0, 0, int(80 * a)))
    shd.setWidthF(pen.widthF() * 1.2)
    shd.setCapStyle(Qt.RoundCap)
    shd.setJoinStyle(Qt.RoundJoin)
    off = max(0.5, r * 0.03)

    h = r * 0.85
    w = r * 0.42

    # Shadow pass
    p.setPen(shd)
    p.setBrush(Qt.NoBrush)
    p.drawLine(QPointF(cx + off, cy - h + off), QPointF(cx + off, cy + h + off))
    p.drawLine(QPointF(cx + off, cy - h + off), QPointF(cx + w + off, cy - h * 0.35 + off))
    p.drawLine(QPointF(cx + w + off, cy - h * 0.35 + off), QPointF(cx - w + off, cy + h * 0.35 + off))
    p.drawLine(QPointF(cx + off, cy + h + off), QPointF(cx + w + off, cy + h * 0.35 + off))
    p.drawLine(QPointF(cx + w + off, cy + h * 0.35 + off), QPointF(cx - w + off, cy - h * 0.35 + off))

    # Main icon
    p.setPen(pen)
    p.drawLine(QPointF(cx, cy - h), QPointF(cx, cy + h))
    p.drawLine(QPointF(cx, cy - h), QPointF(cx + w, cy - h * 0.35))
    p.drawLine(QPointF(cx + w, cy - h * 0.35), QPointF(cx - w, cy + h * 0.35))
    p.drawLine(QPointF(cx, cy + h), QPointF(cx + w, cy + h * 0.35))
    p.drawLine(QPointF(cx + w, cy + h * 0.35), QPointF(cx - w, cy - h * 0.35))


def handle_auraconnect_tap(x, y, cx, cy, mind):
    """Handle a tap on the AuraConnect page. Returns 'back' or None."""
    R = mind * _WIFI_R_FRAC

    # Back chevron (top-left)
    if (cx - R * 0.88 < x < cx - R * 0.63 and
            cy - R * 0.88 < y < cy - R * 0.74):
        return "back"

    # Toggle button zone
    btn_w = R * 0.60
    btn_h = R * 0.11
    btn_y = cy + R * 0.45
    if (cx - btn_w / 2 < x < cx + btn_w / 2 and
            btn_y < y < btn_y + btn_h):
        if is_ble_running():
            print("[auraconnect] Stopping BLE...")
            stop_ble()
        else:
            print("[auraconnect] Starting BLE...")
            start_ble()
        return None

    return None

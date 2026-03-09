"""
gui.auraconnect_page -- AuraConnect BLE sub-page for the Settings overlay.

Draws a status panel and toggle for the Bluetooth GATT peripheral that
lets the macOS AuraConnect app pair with this device.

Zero Qt imports beyond QPainter types used for drawing.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from typing import Optional

from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QColor, QFont, QPen, QRadialGradient, QBrush

# ---------------------------------------------------------------------------
# BLE GATT server (runs in a background thread with its own asyncio loop)
# ---------------------------------------------------------------------------

_ble_thread: Optional[threading.Thread] = None
_ble_loop: Optional[asyncio.AbstractEventLoop] = None
_ble_stop_event: Optional[asyncio.Event] = None
_ble_running = False
_ble_error: Optional[str] = None
_ble_connected = False

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
        def TxPower(self) -> "n":
            return 0

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
            global _ble_connected
            data = value if isinstance(value, (bytes, bytearray)) else bytes(value)
            self._value = data
            _ble_connected = True
            try:
                text = data.decode("utf-8", errors="replace")
            except Exception:
                text = repr(data)
            print(f"[auraconnect] CTRL write from Mac: {text}")
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
            data = value if isinstance(value, (bytes, bytearray)) else bytes(value)
            self.total_bytes += len(data)
            print(f"[auraconnect] DATA: {len(data)} bytes (total {self.total_bytes})")

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

    async def _server_main():
        global _ble_running, _ble_error, _ble_stop_event, _ble_connected

        _ble_stop_event = asyncio.Event()

        try:
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

            # Power on adapter
            intro2 = await bus.introspect(BLUEZ, adv_mgr_path)
            obj2 = bus.get_proxy_object(BLUEZ, adv_mgr_path, intro2)
            props = obj2.get_interface(PROP_IFACE)
            await props.call_set(ADAPTER_IFACE, "Powered", Variant("b", True))

            # Export GATT tree
            bus.export(APP_PATH, AuraObjectManager())
            bus.export(SVC_PATH, AuraService())
            bus.export(CTRL_PATH, CtrlCharacteristic())
            bus.export(DATA_PATH, DataCharacteristic())
            bus.export(ADV_PATH, AuraAdvertisement())

            # Register
            intro3 = await bus.introspect(BLUEZ, gatt_mgr_path)
            obj3 = bus.get_proxy_object(BLUEZ, gatt_mgr_path, intro3)
            gatt_mgr = obj3.get_interface(GATT_MGR)
            await gatt_mgr.call_register_application(APP_PATH, {})

            intro4 = await bus.introspect(BLUEZ, adv_mgr_path)
            obj4 = bus.get_proxy_object(BLUEZ, adv_mgr_path, intro4)
            adv_mgr = obj4.get_interface(LE_ADV_MGR)
            await adv_mgr.call_register_advertisement(ADV_PATH, {})

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
# Drawing
# ---------------------------------------------------------------------------

GOLD = lambda a=255: QColor(145, 175, 215, a)  # noqa: E731


def draw_auraconnect_page(p, cx, cy, mind, t, trans, state=None):
    """Draw the AuraConnect BLE status page inside the settings overlay."""
    if trans <= 0.0:
        return

    A = int(240 * trans)
    R = mind * 0.235

    # Background plate
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(8, 9, 12, int(220 * trans)))
    p.drawEllipse(QPointF(cx, cy), R, R)

    # Bezel ring
    pen = QPen(GOLD(int(175 * trans)))
    pen.setWidthF(max(2.0, mind * 0.0042))
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(cx, cy), R * 0.98, R * 0.98)

    # Title
    title_font = QFont("DejaVu Sans", max(9, int(mind * 0.018)))
    title_font.setBold(True)
    title_font.setLetterSpacing(QFont.PercentageSpacing, 120)
    p.setFont(title_font)
    p.setPen(GOLD(A))
    p.drawText(
        int(cx - R), int(cy - R * 0.85), int(2 * R), int(R * 0.2),
        Qt.AlignCenter, "AURACONNECT"
    )

    # Status indicator
    running = is_ble_running()
    connected = is_ble_connected()
    error = get_ble_error()

    # Central Bluetooth icon area — pulsing glow
    icon_r = R * 0.18
    if running:
        if connected:
            pulse = 0.7 + 0.3 * math.sin(t * 2.0)
            glow_col = QColor(80, 200, 140, int(pulse * 180 * trans))
            status_text = "CONNECTED"
            status_col = QColor(80, 200, 140, A)
        else:
            pulse = 0.5 + 0.5 * math.sin(t * 3.0)
            glow_col = QColor(145, 175, 215, int(pulse * 140 * trans))
            status_text = "ADVERTISING"
            status_col = GOLD(A)
    else:
        pulse = 0.3
        glow_col = QColor(100, 100, 120, int(60 * trans))
        if error:
            status_text = "ERROR"
            status_col = QColor(220, 80, 80, A)
        else:
            status_text = "OFF"
            status_col = QColor(150, 150, 160, A)

    # Glow halo
    halo = QRadialGradient(QPointF(cx, cy - R * 0.12), icon_r * 2.5)
    halo.setColorAt(0.0, glow_col)
    halo.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(halo))
    p.drawEllipse(QPointF(cx, cy - R * 0.12), icon_r * 2.5, icon_r * 2.5)

    # Bluetooth rune (simplified ᛒ shape)
    _draw_bt_icon(p, cx, cy - R * 0.12, icon_r, trans, connected)

    # Status text
    stat_font = QFont("DejaVu Sans", max(10, int(mind * 0.022)))
    stat_font.setBold(True)
    stat_font.setLetterSpacing(QFont.PercentageSpacing, 115)
    p.setFont(stat_font)
    p.setPen(status_col)
    p.drawText(
        int(cx - R), int(cy + R * 0.12), int(2 * R), int(R * 0.18),
        Qt.AlignCenter, status_text
    )

    # Device name
    name_font = QFont("DejaVu Sans", max(8, int(mind * 0.014)))
    name_font.setLetterSpacing(QFont.PercentageSpacing, 110)
    p.setFont(name_font)
    p.setPen(QColor(180, 190, 210, int(160 * trans)))
    p.drawText(
        int(cx - R), int(cy + R * 0.28), int(2 * R), int(R * 0.14),
        Qt.AlignCenter, f'"{LOCAL_NAME}"'
    )

    # Error message
    if error and not running:
        err_font = QFont("DejaVu Sans", max(7, int(mind * 0.012)))
        p.setFont(err_font)
        p.setPen(QColor(220, 100, 100, int(180 * trans)))
        # Truncate long errors
        err_short = error[:40] + "..." if len(error) > 40 else error
        p.drawText(
            int(cx - R * 0.9), int(cy + R * 0.40), int(R * 1.8), int(R * 0.14),
            Qt.AlignCenter, err_short
        )

    # Toggle button
    btn_w = R * 0.7
    btn_h = R * 0.16
    btn_y = cy + R * 0.55
    btn_rect = QRectF(cx - btn_w / 2, btn_y, btn_w, btn_h)

    if running:
        btn_col = QColor(180, 60, 60, int(200 * trans))
        btn_text = "STOP"
    else:
        btn_col = QColor(60, 140, 100, int(200 * trans))
        btn_text = "START"

    p.setPen(Qt.NoPen)
    p.setBrush(btn_col)
    p.drawRoundedRect(btn_rect, btn_h * 0.3, btn_h * 0.3)

    btn_font = QFont("DejaVu Sans", max(9, int(mind * 0.016)))
    btn_font.setBold(True)
    p.setFont(btn_font)
    p.setPen(QColor(255, 255, 255, A))
    p.drawText(btn_rect, Qt.AlignCenter, btn_text)

    # Back label at bottom
    back_font = QFont("DejaVu Sans", max(8, int(mind * 0.013)))
    back_font.setLetterSpacing(QFont.PercentageSpacing, 130)
    p.setFont(back_font)
    p.setPen(QColor(180, 190, 210, int(120 * trans)))
    p.drawText(
        int(cx - R), int(cy + R * 0.76), int(2 * R), int(R * 0.14),
        Qt.AlignCenter, "< BACK"
    )


def _draw_bt_icon(p, cx, cy, r, trans, connected):
    """Draw a simple Bluetooth rune."""
    col = QColor(255, 255, 255, int(220 * trans)) if not connected else \
          QColor(80, 200, 140, int(230 * trans))
    pen = QPen(col)
    pen.setWidthF(max(1.5, r * 0.12))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)

    # Bluetooth symbol: vertical line + two arrow tips
    h = r * 0.9
    w = r * 0.45
    # Vertical line
    p.drawLine(QPointF(cx, cy - h), QPointF(cx, cy + h))
    # Top-right arrow
    p.drawLine(QPointF(cx, cy - h), QPointF(cx + w, cy - h * 0.4))
    p.drawLine(QPointF(cx + w, cy - h * 0.4), QPointF(cx - w, cy + h * 0.4))
    # Bottom-right arrow
    p.drawLine(QPointF(cx, cy + h), QPointF(cx + w, cy + h * 0.4))
    p.drawLine(QPointF(cx + w, cy + h * 0.4), QPointF(cx - w, cy - h * 0.4))


def handle_auraconnect_tap(x, y, cx, cy, mind):
    """Handle a tap on the AuraConnect page. Returns 'back' or None."""
    R = mind * 0.235

    # Back button zone (bottom)
    if cy + R * 0.72 < y < cy + R * 0.94:
        return "back"

    # Toggle button zone
    btn_w = R * 0.7
    btn_h = R * 0.16
    btn_y = cy + R * 0.55
    if (cx - btn_w / 2 < x < cx + btn_w / 2 and
            btn_y < y < btn_y + btn_h):
        if is_ble_running():
            print("[auraconnect] Stopping BLE...")
            stop_ble()
        else:
            print("[auraconnect] Starting BLE...")
            start_ble()
        return None

    # Tap outside — ignore (don't close)
    return None

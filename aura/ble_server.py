#!/usr/bin/env python3
"""
ble_server.py — Standalone BLE GATT server for AuraConnect file transfers.

Runs as a separate process so the DBus event loop isn't starved by the
GUI/CUDA threads in aura.py. Receives files over BLE and writes them to
data/input/ for RAG auto-ingest.

Usage:
    python3 ble_server.py          # foreground
    python3 ble_server.py --daemon # background (called by aura.py)
"""

import asyncio
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

# UUIDs must match macOS AuraConnect app
AURA_SERVICE_UUID = "E7810A71-73AE-499D-8C15-FAA9AEF0C3F2"
CTRL_UUID = "E7810A71-73AE-499D-8C15-FAA9AEF0C3F3"
DATA_UUID = "E7810A71-73AE-499D-8C15-FAA9AEF0C3F4"
LOCAL_NAME = "Aura Puck"

# File transfer writes to RAG input dir
RAG_INPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "input"

# Transfer state
_file_transfer_id = None
_file_transfer_name = None
_file_transfer_chunks = []
_file_transfer_size = 0
_file_transfer_received = 0


def _ensure_bluetooth():
    """Bluetooth setup matching the previously-working embedded server."""
    try:
        subprocess.run(["sudo", "btmgmt", "le", "on"], capture_output=True, timeout=5)
        subprocess.run(["sudo", "btmgmt", "advertising", "on"], capture_output=True, timeout=5)
        subprocess.run(["sudo", "btmgmt", "name", LOCAL_NAME], capture_output=True, timeout=5)
        subprocess.run(["sudo", "btmgmt", "discov", "on"], capture_output=True, timeout=5)
        subprocess.run(["sudo", "btmgmt", "connectable", "on"], capture_output=True, timeout=5)
        subprocess.run(["sudo", "btmgmt", "bondable", "on"], capture_output=True, timeout=5)
        subprocess.run(["bluetoothctl", "pairable", "on"], capture_output=True, timeout=5)
    except Exception as e:
        print(f"[ble] btmgmt warning: {e}")


async def main():
    global _file_transfer_id, _file_transfer_name
    global _file_transfer_chunks, _file_transfer_size, _file_transfer_received

    from dbus_next.aio import MessageBus
    from dbus_next import Variant, BusType
    from dbus_next.service import ServiceInterface, method, dbus_property, PropertyAccess

    # Monkey-patch dbus-next to log all messages and handle readonly gracefully
    import dbus_next.message_bus as _mb
    from dbus_next import MessageType
    _orig_on_message = _mb.BaseMessageBus._on_message

    def _patched_on_message(self, msg):
        # Log all incoming method calls to our paths
        if msg.message_type == MessageType.METHOD_CALL and msg.path and '/bluez/aura' in msg.path:
            print(f"[ble] DBus CALL: {msg.path} {msg.interface}.{msg.member} body={msg.body[:2] if msg.body else []}", flush=True)
        try:
            _orig_on_message(self, msg)
        except Exception as e:
            if "readonly" in str(e):
                print(f"[ble] IGNORED readonly Set: path={msg.path} body={msg.body}", flush=True)
            else:
                raise

    _mb.BaseMessageBus._on_message = _patched_on_message

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
        def IncludeTxPower(self) -> "b":
            return False

    class AuraService(ServiceInterface):
        def __init__(self):
            super().__init__(GATT_SVC_IFACE)

        @dbus_property(access=PropertyAccess.READ)
        def UUID(self) -> "s":
            return AURA_SERVICE_UUID

        @dbus_property(access=PropertyAccess.READ)
        def Primary(self) -> "b":
            return True

        @dbus_property(access=PropertyAccess.READ)
        def Characteristics(self) -> "ao":
            return [CTRL_PATH, DATA_PATH]

        @dbus_property(access=PropertyAccess.READ)
        def Includes(self) -> "ao":
            return []

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
        def Notifying(self) -> "b":
            return self._notifying

        @dbus_property(access=PropertyAccess.READ)
        def Value(self) -> "ay":
            return self._value

        @method()
        def ReadValue(self, options: "a{sv}") -> "ay":
            return self._value

        @method()
        def WriteValue(self, value: "ay", options: "a{sv}"):
            global _file_transfer_id, _file_transfer_name
            global _file_transfer_chunks, _file_transfer_size, _file_transfer_received

            data = value if isinstance(value, (bytes, bytearray)) else bytes(value)
            self._value = data

            try:
                text = data.decode("utf-8", errors="replace").strip()
                msg = json.loads(text)
                cmd = msg.get("cmd", "")
            except Exception:
                text = data.decode("utf-8", errors="replace").strip()
                print(f"[ble] CTRL (non-JSON): {text}")
                if self._notifying:
                    self.emit_properties_changed({"Value": self._value}, [])
                return

            print(f"[ble] CTRL cmd={cmd}")

            if cmd == "BEGIN":
                fname = os.path.basename(msg.get("name", ""))
                if fname:
                    _file_transfer_id = msg.get("id", "")
                    _file_transfer_name = fname
                    _file_transfer_size = msg.get("size", 0)
                    _file_transfer_chunks = []
                    _file_transfer_received = 0
                    print(f"[ble] Transfer BEGIN: {fname} ({_file_transfer_size} bytes)")

            elif cmd == "END" and _file_transfer_name:
                RAG_INPUT_DIR.mkdir(parents=True, exist_ok=True)
                out_path = RAG_INPUT_DIR / _file_transfer_name
                file_data = b"".join(_file_transfer_chunks)
                out_path.write_bytes(file_data)
                print(f"[ble] Transfer END: {out_path} ({len(file_data)} bytes) -> RAG auto-ingest")
                _file_transfer_id = None
                _file_transfer_name = None
                _file_transfer_chunks = []
                _file_transfer_size = 0
                _file_transfer_received = 0

            elif cmd == "ABORT":
                print(f"[ble] Transfer ABORT — discarding {_file_transfer_name}")
                _file_transfer_id = None
                _file_transfer_name = None
                _file_transfer_chunks = []
                _file_transfer_size = 0
                _file_transfer_received = 0

            if self._notifying:
                self.emit_properties_changed({"Value": self._value}, [])

        @method()
        def StartNotify(self):
            self._notifying = True
            print("[ble] Notify enabled — Mac connected")

        @method()
        def StopNotify(self):
            self._notifying = False
            print("[ble] Notify disabled")

    class DataCharacteristic(ServiceInterface):
        def __init__(self):
            super().__init__(GATT_CHR_IFACE)

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

            if _file_transfer_name and len(data) > 4:
                payload_len = struct.unpack_from("<I", data, 0)[0]
                payload = data[4:4 + payload_len]
                _file_transfer_chunks.append(payload)
                _file_transfer_received += len(payload)
            elif _file_transfer_name:
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

    # --- Main server logic ---
    _ensure_bluetooth()

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
        print("[ble] ERROR: No Bluetooth adapter found")
        sys.exit(1)

    # Power on adapter, make discoverable, set name as Alias (Mac reads adapter name, not adv LocalName)
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
    print(f"[ble] GATT application registered")

    # Register advertisement
    intro4 = await bus.introspect(BLUEZ, adv_mgr_path)
    obj4 = bus.get_proxy_object(BLUEZ, adv_mgr_path, intro4)
    adv_mgr = obj4.get_interface(LE_ADV_MGR)
    await adv_mgr.call_register_advertisement(ADV_PATH, {})
    print(f"[ble] Advertising '{LOCAL_NAME}' — waiting for connections")

    # Write PID file so aura.py can check/manage us
    pid_file = Path("/tmp/aura_ble.pid")
    pid_file.write_text(str(os.getpid()))

    # Run forever (process DBus calls)
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            await adv_mgr.call_unregister_advertisement(ADV_PATH)
        except Exception:
            pass
        try:
            await gatt_mgr.call_unregister_application(APP_PATH)
        except Exception:
            pass
        pid_file.unlink(missing_ok=True)
        print("[ble] Server stopped")


if __name__ == "__main__":
    if "--daemon" in sys.argv:
        # Detach from parent
        if os.fork() > 0:
            sys.exit(0)
        os.setsid()
    asyncio.run(main())

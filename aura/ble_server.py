#!/usr/bin/env python3
"""
ble_server.py — Standalone BLE GATT server for AuraConnect file transfers.

Runs as a separate process so the DBus event loop isn't starved by the
GUI/CUDA threads in aura.py. Receives files over BLE and writes them to
data/input/ for RAG auto-ingest.

Protocol (matches AuraConnect Mac app):
  CTRL (write + notify):
    Mac -> Puck: JSON {"cmd": "BEGIN", "id", "name", "size", "sha256"}
    Puck -> Mac: JSON {"resp": "begin", "ok": true|false, "msg": "..."}
    Mac -> Puck: JSON {"cmd": "END", "id"}
    Puck -> Mac: JSON {"resp": "end", "ok": true|false, "msg": "..."}
    Mac -> Puck: JSON {"cmd": "ABORT"}
  DATA (write-without-response):
    Mac -> Puck: raw payload bytes (one atomic ATT write per chunk)

Usage:
    python3 ble_server.py          # foreground
    python3 ble_server.py --daemon # background (called by aura.py)
"""

import asyncio
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

AURA_SERVICE_UUID = "E7810A71-73AE-499D-8C15-FAA9AEF0C3F2"
CTRL_UUID         = "E7810A71-73AE-499D-8C15-FAA9AEF0C3F3"
DATA_UUID         = "E7810A71-73AE-499D-8C15-FAA9AEF0C3F4"
LOCAL_NAME        = "Aura Puck"

RAG_INPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "input"
PID_FILE      = Path("/tmp/aura_ble.pid")
LOG_FILE      = Path("/tmp/ble.log")

# Single mutable holder so nested classes can share state without globals
class TransferState:
    def __init__(self):
        self.id = None
        self.name = None
        self.size = 0
        self.sha256 = ""
        self.received = 0
        self.hasher = None
        self.chunks = []

    def reset(self):
        self.__init__()

    def begin(self, fid, name, size, sha):
        self.id = fid
        self.name = os.path.basename(name)
        self.size = int(size)
        self.sha256 = (sha or "").lower()
        self.received = 0
        self.hasher = hashlib.sha256()
        self.chunks = []

    def add(self, payload: bytes):
        self.chunks.append(payload)
        self.received += len(payload)
        if self.hasher is not None:
            self.hasher.update(payload)

    def finalize(self) -> tuple[bool, str, bytes]:
        if not self.name:
            return False, "no transfer in progress", b""
        data = b"".join(self.chunks)
        digest = self.hasher.hexdigest() if self.hasher else hashlib.sha256(data).hexdigest()
        # Diagnostic: dump the first/last bytes so we can see if Mac is still
        # sending framed [u32 len][payload] chunks (look for tiny LE counts at
        # the start of each chunk).
        head = data[:48].hex()
        tail = data[-48:].hex() if len(data) > 48 else ""
        print(
            f"[ble] FINALIZE name={self.name} expected_size={self.size} "
            f"received={len(data)} chunks={len(self.chunks)} "
            f"chunk_sizes={[len(c) for c in self.chunks[:8]]}",
            flush=True,
        )
        print(f"[ble] FINALIZE head={head}", flush=True)
        print(f"[ble] FINALIZE tail={tail}", flush=True)
        if self.sha256 and digest != self.sha256:
            return False, f"sha256 mismatch (got {digest[:12]}, want {self.sha256[:12]})", data
        if self.size and len(data) != self.size:
            return False, f"size mismatch (got {len(data)}, want {self.size})", data
        return True, "ok", data


def _ensure_bluetooth():
    # Order matters: bredr/le toggles require the controller to be powered off
    # first. macOS Core Bluetooth refuses to connect if a dual-mode controller
    # advertises with AD flags that don't declare BR/EDR-not-supported, so we
    # turn BR/EDR off entirely — this puck is BLE-only.
    # NB: do NOT issue `btmgmt le on` after `bredr off` — `le on` flips
    # BR/EDR back on as a side effect and macOS rejects the resulting
    # dual-mode advertisement with "Invalid Dual mode support indication".
    cmds = [
        ["sudo", "btmgmt", "power", "off"],
        ["sudo", "btmgmt", "bredr", "off"],
        ["sudo", "btmgmt", "power", "on"],
        ["sudo", "btmgmt", "advertising", "on"],
        ["sudo", "btmgmt", "name", LOCAL_NAME],
        ["sudo", "btmgmt", "discov", "on"],
        ["sudo", "btmgmt", "connectable", "on"],
        ["sudo", "btmgmt", "bondable", "on"],
        ["bluetoothctl", "pairable", "on"],
    ]
    for cmd in cmds:
        try:
            subprocess.run(cmd, capture_output=True, timeout=5)
        except Exception as e:
            print(f"[ble] {cmd[0]} {cmd[-1]} warning: {e}", flush=True)


def _check_existing_daemon() -> bool:
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        try:
            PID_FILE.unlink()
        except OSError:
            pass
        return False


async def main():
    state = TransferState()

    import logging

    from dbus_next.aio import MessageBus
    from dbus_next import Variant, BusType, MessageType
    from dbus_next.service import ServiceInterface, method, dbus_property, PropertyAccess

    # BlueZ on some versions calls Properties.Set on advertisement TxPower.
    # The library logs this through the root logger before sending the
    # error reply back. Filter the message so the log isn't drowned in it.
    class _SuppressTxPower(logging.Filter):
        def filter(self, record):
            try:
                return "TxPower" not in record.getMessage()
            except Exception:
                return True
    logging.getLogger().addFilter(_SuppressTxPower())

    BLUEZ          = "org.bluez"
    OM_IFACE       = "org.freedesktop.DBus.ObjectManager"
    PROP_IFACE     = "org.freedesktop.DBus.Properties"
    LE_ADV_MGR     = "org.bluez.LEAdvertisingManager1"
    GATT_MGR       = "org.bluez.GattManager1"
    ADAPTER_IFACE  = "org.bluez.Adapter1"
    LE_ADV_IFACE   = "org.bluez.LEAdvertisement1"
    GATT_SVC_IFACE = "org.bluez.GattService1"
    GATT_CHR_IFACE = "org.bluez.GattCharacteristic1"

    BASE      = "/org/bluez/aura"
    APP_PATH  = f"{BASE}/app"
    SVC_PATH  = f"{APP_PATH}/service0"
    CTRL_PATH = f"{SVC_PATH}/ctrl"
    DATA_PATH = f"{SVC_PATH}/data"
    ADV_PATH  = f"{BASE}/advertisement0"

    # ---- Advertisement -----------------------------------------------------

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

    # ---- GATT service & characteristics -----------------------------------

    class AuraService(ServiceInterface):
        def __init__(self):
            super().__init__(GATT_SVC_IFACE)

        @dbus_property(access=PropertyAccess.READ)
        def UUID(self) -> "s": return AURA_SERVICE_UUID

        @dbus_property(access=PropertyAccess.READ)
        def Primary(self) -> "b": return True

        @dbus_property(access=PropertyAccess.READ)
        def Characteristics(self) -> "ao": return [CTRL_PATH, DATA_PATH]

        @dbus_property(access=PropertyAccess.READ)
        def Includes(self) -> "ao": return []

    class CtrlCharacteristic(ServiceInterface):
        def __init__(self):
            super().__init__(GATT_CHR_IFACE)
            self._value = b""
            self._notifying = False

        @dbus_property(access=PropertyAccess.READ)
        def UUID(self) -> "s": return CTRL_UUID

        @dbus_property(access=PropertyAccess.READ)
        def Service(self) -> "o": return SVC_PATH

        @dbus_property(access=PropertyAccess.READ)
        def Flags(self) -> "as": return ["write", "notify"]

        @dbus_property(access=PropertyAccess.READ)
        def Notifying(self) -> "b": return self._notifying

        @dbus_property(access=PropertyAccess.READ)
        def Value(self) -> "ay": return self._value

        @method()
        def ReadValue(self, options: "a{sv}") -> "ay": return self._value

        def _ack(self, resp: str, ok: bool, msg: str = ""):
            payload = json.dumps({"resp": resp, "ok": ok, "msg": msg}).encode("utf-8")
            self._value = payload
            if self._notifying:
                self.emit_properties_changed({"Value": self._value}, [])
            print(f"[ble] ack {resp} ok={ok} msg={msg!r}", flush=True)

        @method()
        def WriteValue(self, value: "ay", options: "a{sv}"):
            data = bytes(value)
            try:
                obj = json.loads(data.decode("utf-8", errors="replace"))
                cmd = obj.get("cmd", "")
            except Exception:
                print(f"[ble] CTRL non-JSON ({len(data)}B)", flush=True)
                return

            print(f"[ble] CTRL cmd={cmd}", flush=True)

            if cmd == "BEGIN":
                fname = os.path.basename(obj.get("name", ""))
                if not fname:
                    self._ack("begin", False, "missing filename")
                    return
                state.begin(obj.get("id", ""), fname,
                            obj.get("size", 0), obj.get("sha256", ""))
                print(f"[ble] BEGIN {fname} size={state.size}", flush=True)
                self._ack("begin", True, f"ready for {fname}")

            elif cmd == "END":
                if not state.name:
                    self._ack("end", False, "no transfer in progress")
                    return
                ok, msg, data_blob = state.finalize()
                if ok:
                    try:
                        RAG_INPUT_DIR.mkdir(parents=True, exist_ok=True)
                        out = RAG_INPUT_DIR / state.name
                        out.write_bytes(data_blob)
                        print(f"[ble] WROTE {out} ({len(data_blob)} bytes)", flush=True)
                        self._ack("end", True, f"saved {state.name}")
                    except Exception as e:
                        self._ack("end", False, f"write failed: {e}")
                else:
                    self._ack("end", False, msg)
                state.reset()

            elif cmd == "ABORT":
                print(f"[ble] ABORT {state.name}", flush=True)
                state.reset()
                self._ack("abort", True, "")

            else:
                self._ack("err", False, f"unknown cmd {cmd}")

        @method()
        def StartNotify(self):
            self._notifying = True
            print("[ble] Notify enabled — Mac connected", flush=True)

        @method()
        def StopNotify(self):
            self._notifying = False
            print("[ble] Notify disabled", flush=True)

    class DataCharacteristic(ServiceInterface):
        def __init__(self):
            super().__init__(GATT_CHR_IFACE)

        @dbus_property(access=PropertyAccess.READ)
        def UUID(self) -> "s": return DATA_UUID

        @dbus_property(access=PropertyAccess.READ)
        def Service(self) -> "o": return SVC_PATH

        @dbus_property(access=PropertyAccess.READ)
        def Flags(self) -> "as": return ["write", "write-without-response"]

        @dbus_property(access=PropertyAccess.READ)
        def Value(self) -> "ay": return b""

        @method()
        def WriteValue(self, value: "ay", options: "a{sv}"):
            if not state.name:
                return
            # Each ATT write is one atomic chunk — no length prefix needed.
            state.add(bytes(value))

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
                        "Flags": Variant("as", ["write", "write-without-response"]),
                        "Value": Variant("ay", b""),
                    }
                },
            }

    # ---- Wire up ----------------------------------------------------------

    _ensure_bluetooth()

    bus = await MessageBus(bus_type=BusType.SYSTEM).connect()

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
        print("[ble] ERROR: No Bluetooth adapter found", flush=True)
        sys.exit(1)

    intro_a = await bus.introspect(BLUEZ, adv_mgr_path)
    obj_a = bus.get_proxy_object(BLUEZ, adv_mgr_path, intro_a)
    props = obj_a.get_interface(PROP_IFACE)
    await props.call_set(ADAPTER_IFACE, "Powered", Variant("b", True))
    await props.call_set(ADAPTER_IFACE, "Discoverable", Variant("b", True))
    await props.call_set(ADAPTER_IFACE, "Alias", Variant("s", LOCAL_NAME))

    bus.export(APP_PATH,  AuraObjectManager())
    bus.export(SVC_PATH,  AuraService())
    bus.export(CTRL_PATH, CtrlCharacteristic())
    bus.export(DATA_PATH, DataCharacteristic())
    bus.export(ADV_PATH,  AuraAdvertisement())

    intro_g = await bus.introspect(BLUEZ, gatt_mgr_path)
    obj_g = bus.get_proxy_object(BLUEZ, gatt_mgr_path, intro_g)
    gatt_mgr = obj_g.get_interface(GATT_MGR)
    await gatt_mgr.call_register_application(APP_PATH, {})
    print("[ble] GATT application registered", flush=True)

    adv_mgr = obj_a.get_interface(LE_ADV_MGR)
    # Defensive: clear any stale registration left over from a prior session
    # that was killed via SIGTERM before its cleanup could run.
    try:
        await adv_mgr.call_unregister_advertisement(ADV_PATH)
        print("[ble] Cleared stale advertisement from prior session", flush=True)
    except Exception:
        pass
    await adv_mgr.call_register_advertisement(ADV_PATH, {})
    print(f"[ble] Advertising '{LOCAL_NAME}' — waiting for connections", flush=True)

    PID_FILE.write_text(str(os.getpid()))

    # Translate SIGTERM into a clean asyncio shutdown so finally runs.
    import signal
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):
            pass

    try:
        await stop_event.wait()
    finally:
        try: await adv_mgr.call_unregister_advertisement(ADV_PATH)
        except Exception: pass
        try: await gatt_mgr.call_unregister_application(APP_PATH)
        except Exception: pass
        PID_FILE.unlink(missing_ok=True)
        print("[ble] Server stopped", flush=True)


if __name__ == "__main__":
    if "--daemon" in sys.argv:
        if _check_existing_daemon():
            print("[ble] Already running, exiting", flush=True)
            sys.exit(0)
        if os.fork() > 0:
            sys.exit(0)
        os.setsid()
        sys.stdout = open(LOG_FILE, "a", buffering=1)
        sys.stderr = sys.stdout
        print(f"\n[ble] === session start pid={os.getpid()} ===", flush=True)
    asyncio.run(main())

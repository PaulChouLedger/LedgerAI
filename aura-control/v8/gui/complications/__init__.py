"""
gui.complications -- Complication registry and dock manager.

Auto-discovers all BaseComplication subclasses in this package and makes
them available to the GUI and Topics Center.

Usage:
    from gui.complications import registry

    registry.load()                        # scan & instantiate all complications
    docked = registry.get_docked()         # list of active perimeter complications
    all_   = registry.get_all()            # every registered complication
    vol    = registry.get("Volume")        # by name
    registry.dock("Medical", slot=2)       # user pins Medical to slot 2
    registry.undock("Alerts")              # user removes Alerts from dock
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, List, Optional, Type

from core.bus import bus
from core.state import state
from gui.complications.base import BaseComplication


class _Registry:
    """Singleton that holds all discovered complications."""

    def __init__(self) -> None:
        self._by_name: Dict[str, BaseComplication] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Import every module in this package (and domains/) and register
        any BaseComplication subclasses found."""
        if self._loaded:
            return
        self._loaded = True

        pkg_dir = Path(__file__).resolve().parent

        # Walk this package and the domains sub-package
        for finder, mod_name, is_pkg in pkgutil.walk_packages(
            path=[str(pkg_dir)],
            prefix=__name__ + ".",
        ):
            if mod_name.endswith(".__init__") or mod_name.endswith(".base"):
                continue
            try:
                mod = importlib.import_module(mod_name)
            except Exception as exc:
                print(f"[registry] skip {mod_name}: {exc}")
                continue

            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, BaseComplication)
                    and obj is not BaseComplication
                    and getattr(obj, "name", "")
                ):
                    self.register_class(obj)

    def register_class(self, cls: Type[BaseComplication]) -> None:
        """Instantiate and register a complication class."""
        if cls.name in self._by_name:
            return  # already registered
        try:
            instance = cls(bus=bus)
            self._by_name[instance.name] = instance
        except Exception as exc:
            print(f"[registry] failed to instantiate {cls.__name__}: {exc}")

    def register(self, instance: BaseComplication) -> None:
        """Register an already-instantiated complication."""
        self._by_name[instance.name] = instance

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[BaseComplication]:
        return self._by_name.get(name)

    def get_all(self) -> List[BaseComplication]:
        """All registered complications (for Topics Center browser)."""
        return list(self._by_name.values())

    def get_docked(self) -> List[BaseComplication]:
        """Complications on the perimeter ring, in dock order.

        Reads from ``state.dock`` so changes are instant.
        Appends any ``always_available`` complications that aren't in the dock.
        """
        docked: List[BaseComplication] = []
        for name in state.dock:
            comp = self._by_name.get(name)
            if comp is not None:
                docked.append(comp)
        # Always-available complications (e.g. Topics Center) appended if missing
        for comp in self._by_name.values():
            if comp.always_available and comp not in docked:
                docked.append(comp)
        return docked

    def get_by_category(self) -> Dict[str, List[BaseComplication]]:
        """Group all dockable complications by category."""
        cats: Dict[str, List[BaseComplication]] = {}
        for comp in self._by_name.values():
            if comp.dockable:
                cats.setdefault(comp.category, []).append(comp)
        return cats

    # ------------------------------------------------------------------
    # Dock management  (thin wrappers around state.dock_*)
    # ------------------------------------------------------------------

    def dock(self, name: str, slot: int | None = None) -> bool:
        ok = state.dock_add(name, slot)
        if ok:
            bus.emit("dock.changed", dock=state.dock)
        return ok

    def undock(self, name: str) -> bool:
        ok = state.dock_remove(name)
        if ok:
            bus.emit("dock.changed", dock=state.dock)
        return ok

    def swap(self, old: str, new: str) -> bool:
        ok = state.dock_swap(old, new)
        if ok:
            bus.emit("dock.changed", dock=state.dock)
        return ok


# Module-level singleton
registry = _Registry()

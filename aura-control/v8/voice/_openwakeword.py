"""
Thin wrapper that exposes create_detector for the v2/v7 voice package.

Delegates to the original core/openwakeword_wake_word.py module which lives
outside the v7 tree at aura-control/core/.
"""
from __future__ import annotations

import importlib.util
import os

_MOD_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "core", "openwakeword_wake_word.py"
)
_MOD_PATH = os.path.normpath(_MOD_PATH)

def create_detector(**kwargs):
    spec = importlib.util.spec_from_file_location("openwakeword_wake_word", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.create_openwakeword_detector(**kwargs)

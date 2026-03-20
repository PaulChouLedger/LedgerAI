"""
Thin wrapper that exposes create_detector for the voice package.
"""
from voice.openwakeword_wake_word import create_openwakeword_detector


def create_detector(**kwargs):
    return create_openwakeword_detector(**kwargs)

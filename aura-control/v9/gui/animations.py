"""
gui.animations -- Dataclasses for visual elements (stars, loops, particles).

Extracted from carbon_demo.py's Star, LoopParams, and particle generation.
Will be populated in migration step 2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Star:
    r: float       # 0..1 radius fraction in dial
    th: float      # angle radians
    size: float    # pixel-ish scale factor
    base_a: int    # base alpha
    tw: float      # twinkle frequency
    ph: float      # twinkle phase
    hue: int       # 0 = white, 1 = gold


@dataclass
class LoopParams:
    a: float
    b: float
    k1: float
    k2: float
    p1: float
    p2: float
    thick: float
    rot_deg: float
    hue_shift: float = 0.0

"""
gui.renderer -- Shared QPainter drawing primitives.

All non-complication visual layers extracted from carbon_demo.py:
  - Background textures (blue dial + mute red, cached to QPixmap)
  - Celestial starfield (twinkling ivory/gold stars)
  - Chapter ticks (watch-style perimeter ring)
  - Center encircling ring (iris ring around loops)
  - Mist / gold dust particles
  - Inner rings (hero 4-loop harmonic animation, planet palette)
  - Perimeter sweep (optional moving arcs)

Every function takes a QPainter + geometry args. No self/state —
the window passes in whatever state is needed.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import numpy as np

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import (
    QBrush, QColor, QPainter, QPainterPath, QPen,
    QFont, QLinearGradient, QPixmap, QRadialGradient,
)

from gui.animations import LoopParams, Star

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * clamp(t, 0.0, 1.0)


def ease_in_out(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def ease_out(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return 1.0 - (1.0 - x) ** 3


# ---------------------------------------------------------------------------
# Star / particle factories
# ---------------------------------------------------------------------------

def make_celestial_stars(n_white: int = 520, n_gold: int = 240) -> List[Star]:
    stars: List[Star] = []
    rng = random.Random(1337)

    for _ in range(n_white):
        r = (rng.random() ** 0.65) * 0.98
        th = rng.random() * 2 * math.pi
        size = 0.8 + 2.0 * rng.random()
        base_a = int(22 + 55 * rng.random())
        tw = 0.10 + 0.35 * rng.random()
        ph = rng.random() * 2 * math.pi
        stars.append(Star(r=r, th=th, size=size, base_a=base_a, tw=tw, ph=ph, hue=0))

    # Gold dust
    for _ in range(n_gold):
        r = (rng.random() ** 0.65) * 0.98
        th = rng.random() * 2 * math.pi
        size = 0.6 + 1.4 * rng.random()
        base_a = int(16 + 40 * rng.random())
        tw = 0.08 + 0.30 * rng.random()
        ph = rng.random() * 2 * math.pi
        stars.append(Star(r=r, th=th, size=size, base_a=base_a, tw=tw, ph=ph, hue=1))

    return stars


def make_particles(n: int) -> List[Tuple[float, float, float]]:
    pts = []
    for _ in range(n):
        r = random.random()
        th = random.random() * 2 * math.pi
        pts.append((r, th, random.random()))
    return pts


# ---------------------------------------------------------------------------
# Background textures (cached to QPixmap)
# ---------------------------------------------------------------------------

class BackgroundCache:
    """Holds cached background pixmaps; rebuilds on resize or scheme change."""

    def __init__(self) -> None:
        self._cache: Optional[QPixmap] = None
        self._key: Optional[tuple] = None
        self._red_cache: Optional[QPixmap] = None
        self._red_key: Optional[Tuple[int, int]] = None

    def get(self, W: int, H: int, mind: float, scheme: dict) -> QPixmap:
        key = (W, H, id(scheme))
        if self._cache is not None and self._key == key:
            return self._cache
        self._cache = _build_background(W, H, mind, scheme)
        self._key = key
        return self._cache

    def get_muted(self, W: int, H: int, mind: float) -> QPixmap:
        """Muted background is always red regardless of scheme."""
        key = (W, H)
        if self._red_cache is not None and self._red_key == key:
            return self._red_cache
        self._red_cache = _build_red_background(W, H, mind)
        self._red_key = key
        return self._red_cache

    def invalidate(self):
        self._cache = None
        self._key = None


def _build_background(W: int, H: int, mind: float, scheme: dict) -> QPixmap:
    """Build a themed background based on scheme['bg_style']."""
    style = scheme.get("bg_style", "lacquer")
    if style == "radial":
        return _build_radial_background(W, H, mind, scheme)
    return _build_lacquer_background(W, H, mind, scheme)


def _build_lacquer_background(W: int, H: int, mind: float, scheme: dict) -> QPixmap:
    pm = QPixmap(W, H)
    pm.fill(QColor(0, 0, 0))
    q = QPainter(pm)
    try:
        q.setRenderHint(QPainter.Antialiasing)
        bg_base = scheme["bg_base"]
        base = QColor(*bg_base)
        q.fillRect(0, 0, W, H, base)

        # Fine horizontal emboss
        bg_emboss = scheme["bg_emboss"]
        step = 2
        for y in range(0, H, step):
            wob = 0.5 + 0.5 * math.sin(y * 0.035)
            lift = int(4 + 9 * wob)
            col = QColor(bg_emboss[0] + lift, bg_emboss[1] + lift, bg_emboss[2] + lift, 16)
            q.setPen(QPen(col, 1))
            q.drawLine(0, y, W, y)

        # Silver threads
        bg_thread = scheme["bg_thread"]
        bg_thread_s = scheme["bg_thread_strong"]
        for y in range(2, H, 24):
            a = 8 + ((y * 7) % 6)
            q.setPen(QPen(QColor(*bg_thread, a), 1))
            q.drawLine(0, y, W, y)
        for y in range(11, H, 96):
            q.setPen(QPen(QColor(*bg_thread_s, 16), 1))
            q.drawLine(0, y, W, y)

        # Ultra-fine sheen
        for y in range(1, H, 6):
            q.setPen(QPen(QColor(255, 255, 255, 4), 1))
            q.drawLine(0, y, W, y)

        # Lacquer vignette + center lift
        cx, cy = W * 0.5, H * 0.5
        q.save()
        q.translate(cx, cy)
        R = min(W, H) * 0.56

        for i in range(22):
            rr = R * (1.0 - i * 0.035)
            a = int(12 + i * 9)
            q.setPen(QPen(QColor(0, 0, 0, a), max(1.0, mind * 0.002)))
            q.setBrush(Qt.NoBrush)
            q.drawEllipse(int(-rr), int(-rr), int(2 * rr), int(2 * rr))

        for i in range(14):
            rr = R * (0.18 + i * 0.030)
            a = int(12 - i)
            if a <= 0:
                break
            q.setPen(QPen(QColor(255, 255, 255, a), 1))
            q.setBrush(Qt.NoBrush)
            q.drawEllipse(int(-rr), int(-rr), int(2 * rr), int(2 * rr))

        q.restore()

        # Micro grain
        q.setPen(Qt.NoPen)
        for i in range(900):
            x = (i * 73) % W
            y = (i * 191) % H
            q.setBrush(QBrush(QColor(255, 255, 255, 5)))
            q.drawEllipse(int(x), int(y), 1, 1)

        # Dial plate inset
        q.setPen(Qt.NoPen)
        q.setBrush(QBrush(QColor(0, 0, 0, 22)))
        dial_r = int(min(W, H) * 0.44)
        q.drawEllipse(int(cx - dial_r), int(cy - dial_r), int(2 * dial_r), int(2 * dial_r))
    finally:
        q.end()
    return pm


def _build_radial_background(W: int, H: int, mind: float, scheme: dict) -> QPixmap:
    pm = QPixmap(W, H)
    pm.fill(QColor(0, 0, 0))
    q = QPainter(pm)
    try:
        q.setRenderHint(QPainter.Antialiasing)
        bg_base = scheme["bg_base"]
        base = QColor(*bg_base)
        q.fillRect(0, 0, W, H, base)

        bg_emboss = scheme["bg_emboss"]
        line_alpha = 18
        y_step = max(2, int(mind * 0.010))
        pen1 = QPen(QColor(*bg_emboss, line_alpha))
        pen2 = QPen(QColor(0, 0, 0, 22))
        for y in range(0, H, y_step):
            q.setPen(pen1); q.drawLine(0, y, W, y)
            q.setPen(pen2); q.drawLine(0, y + 1, W, y + 1)

        cx, cy = W * 0.5, H * 0.5
        R = min(W, H) * 0.58
        g = QRadialGradient(QPointF(cx, cy), R * 1.12)
        g.setColorAt(0.00, QColor(255, 70, 90, 35))
        g.setColorAt(0.50, QColor(90, 10, 18, 10))
        g.setColorAt(1.00, QColor(0, 0, 0, 180))
        q.setPen(Qt.NoPen)
        q.setBrush(QBrush(g))
        q.drawEllipse(QPointF(cx, cy), R * 1.05, R * 1.05)

        bg_thread = scheme["bg_thread"]
        rng = random.Random(1337)
        dots = int((W * H) * 0.00010)
        for _ in range(dots):
            x = rng.randint(0, W - 1)
            y = rng.randint(0, H - 1)
            a = rng.randint(6, 18)
            q.setPen(QColor(*bg_thread, a))
            q.drawPoint(x, y)
    finally:
        q.end()
    return pm


def _build_red_background(W: int, H: int, mind: float) -> QPixmap:
    pm = QPixmap(W, H)
    pm.fill(QColor(0, 0, 0))
    q = QPainter(pm)
    try:
        q.setRenderHint(QPainter.Antialiasing)
        base = QColor(54, 10, 16)
        q.fillRect(0, 0, W, H, base)

        line_alpha = 18
        y_step = max(2, int(mind * 0.010))
        pen1 = QPen(QColor(255, 210, 210, line_alpha))
        pen2 = QPen(QColor(0, 0, 0, 22))
        for y in range(0, H, y_step):
            q.setPen(pen1); q.drawLine(0, y, W, y)
            q.setPen(pen2); q.drawLine(0, y + 1, W, y + 1)

        cx, cy = W * 0.5, H * 0.5
        R = min(W, H) * 0.58
        g = QRadialGradient(QPointF(cx, cy), R * 1.12)
        g.setColorAt(0.00, QColor(255, 70, 90, 35))
        g.setColorAt(0.50, QColor(90, 10, 18, 10))
        g.setColorAt(1.00, QColor(0, 0, 0, 180))
        q.setPen(Qt.NoPen)
        q.setBrush(QBrush(g))
        q.drawEllipse(QPointF(cx, cy), R * 1.05, R * 1.05)

        rng = random.Random(1337)
        dots = int((W * H) * 0.00010)
        for _ in range(dots):
            x = rng.randint(0, W - 1)
            y = rng.randint(0, H - 1)
            a = rng.randint(6, 18)
            q.setPen(QColor(255, 190, 190, a))
            q.drawPoint(x, y)
    finally:
        q.end()
    return pm


# ---------------------------------------------------------------------------
# Nebula — drifting particles
# ---------------------------------------------------------------------------

_EMBER_SEEDS: List[Tuple[float, float, float, float, float, float]] = []

def _ensure_ember_seeds(n: int = 60) -> None:
    global _EMBER_SEEDS
    if _EMBER_SEEDS:
        return
    rng = random.Random(42)
    for _ in range(n):
        _EMBER_SEEDS.append((
            rng.random() * 2 * math.pi,      # base angle
            0.08 + rng.random() * 0.82,       # base radius (0.08..0.90)
            0.015 + rng.random() * 0.035,     # drift speed
            rng.random() * 2 * math.pi,       # phase
            1.5 + rng.random() * 4.5,         # size (px at mind=1080)
            0.4 + rng.random() * 0.6,         # brightness factor
        ))


def draw_nebula(
    p: QPainter, cx: float, cy: float, mind: float, t: float,
    alpha: float = 1.0,
    scheme: Optional[dict] = None,
) -> None:
    """Drifting particles — cool sparks floating across the dark dial."""
    alpha = clamp(alpha, 0.0, 1.0)
    if alpha <= 0.0:
        return

    _ensure_ember_seeds()

    # Resolve colors from scheme or fall back to defaults
    neb_core   = scheme["nebula_core"]   if scheme else (60, 130, 220)
    neb_mid    = scheme["nebula_mid"]    if scheme else (35, 80, 160)
    neb_deep   = scheme["nebula_deep"]   if scheme else (15, 35, 90)
    neb_edge   = scheme["nebula_edge"]   if scheme else (10, 20, 55)
    neb_bright = scheme["nebula_bright"] if scheme else (100, 180, 255)

    p.save()
    try:
        R = mind * 0.46
        p.setPen(Qt.NoPen)

        for base_ang, base_r, speed, phase, sz, bright in _EMBER_SEEDS:
            # Slow orbital drift
            ang = base_ang + t * speed + phase
            # Gentle radial breathing
            r_frac = base_r + 0.06 * math.sin(t * speed * 1.5 + phase * 2.0)
            r_frac = clamp(r_frac, 0.05, 0.95)

            x = cx + R * r_frac * math.cos(ang)
            y = cy + R * r_frac * math.sin(ang)

            # Pulsing glow
            pulse = 0.5 + 0.5 * math.sin(t * (0.4 + speed * 8) + phase * 3.0)
            a0 = int(120 * alpha * bright * (0.5 + 0.5 * pulse))

            # Core fading to dark
            rad = sz * (mind / 1080.0)
            g = QRadialGradient(QPointF(x, y), max(1.0, rad * 2.5))
            g.setColorAt(0.0, QColor(*neb_core, a0))
            g.setColorAt(0.3, QColor(*neb_mid, int(a0 * 0.6)))
            g.setColorAt(0.7, QColor(*neb_deep, int(a0 * 0.2)))
            g.setColorAt(1.0, QColor(*neb_edge, 0))
            p.setBrush(QBrush(g))
            p.drawEllipse(QPointF(x, y), rad * 2.5, rad * 2.5)

            # Bright core dot
            if a0 > 40:
                p.setBrush(QColor(*neb_bright, int(a0 * 0.8)))
                p.drawEllipse(QPointF(x, y), rad * 0.5, rad * 0.5)

    finally:
        p.restore()


# ---------------------------------------------------------------------------
# Celestial starfield
# ---------------------------------------------------------------------------

def draw_celestial(
    p: QPainter, cx: float, cy: float, mind: float, t: float,
    stars: List[Star],
    scheme: Optional[dict] = None,
) -> None:
    t = t * 2.5
    R = mind * 0.46
    rot = t * (2 * math.pi / (9 * 60.0))

    star_white = scheme["star_white"] if scheme else (210, 225, 248)
    star_gold  = scheme["star_gold"]  if scheme else (180, 210, 245)

    p.save()
    p.setCompositionMode(QPainter.CompositionMode_Screen)

    for s in stars:
        th = s.th + rot
        rr = s.r * R
        x = cx + rr * math.cos(th)
        y = cy + rr * math.sin(th)

        tw = 0.78 + 0.22 * (0.5 + 0.5 * math.sin(t * (0.35 + s.tw) + s.ph))
        a = int(s.base_a * tw)

        if s.hue == 0:
            jitter = int((s.ph % 1.0) * 8.0)
            col = QColor(
                int(clamp(star_white[0] + jitter, 0, 255)),
                int(clamp(star_white[1] + jitter, 0, 255)),
                int(clamp(star_white[2], 0, 255)),
                a,
            )
        else:
            col = QColor(*star_gold, a)

        rad = max(1, int(s.size))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(col))
        p.drawEllipse(int(x - rad), int(y - rad), int(2 * rad), int(2 * rad))

        if rad >= 2:
            halo_a = int(col.alpha() * 0.35)
            halo_col = QColor(col.red(), col.green(), col.blue(), halo_a)
            p.setBrush(QBrush(halo_col))
            hr = rad * 1.8
            p.drawEllipse(int(x - hr), int(y - hr), int(2 * hr), int(2 * hr))

    p.restore()


# ---------------------------------------------------------------------------
# Chapter ticks + Patek-style complication bezel
# ---------------------------------------------------------------------------

def draw_chapter_ticks(
    p: QPainter, cx: float, cy: float, mind: float, t: float,
    alpha: float = 1.0, perim_margin_frac: float = 0.038,
    scheme: Optional[dict] = None,
) -> None:
    alpha = clamp(alpha, 0.0, 1.0)
    if alpha <= 0.0:
        return

    tick_color = scheme["tick_color"] if scheme else (195, 215, 240)
    bezel_hi   = scheme["bezel_hi"]  if scheme else (200, 215, 235)

    p.save()
    try:
        perim_margin = mind * perim_margin_frac
        edge_pad = mind * 0.008
        r_outer = (mind * 0.5) - edge_pad

        # --- Ticks (varied lengths) ---
        _rng = random.Random(7777)
        tick_vars = [_rng.uniform(-0.12, 0.12) for _ in range(60)]

        r_inner_minor = r_outer - perim_margin * 0.55
        r_inner_major = r_outer - perim_margin * 0.85

        base_a = int(90 * alpha)
        glint_ang = (-math.pi / 2) + (t * 0.18) % (2 * math.pi)
        glint_span = math.radians(22)

        for i in range(60):
            ang = -math.pi / 2 + (2 * math.pi) * (i / 60.0)
            major = (i % 5 == 0)

            if major:
                rin = r_inner_major
            else:
                base_rin = r_inner_minor
                variation = tick_vars[i] * perim_margin * 0.3
                rin = base_rin + variation

            d = (ang - glint_ang + math.pi) % (2 * math.pi) - math.pi
            w = max(0.0, 1.0 - (abs(d) / glint_span))
            a = base_a + int(90 * w * alpha)

            col = QColor(*tick_color, clamp(a, 0, 255))
            pen = QPen(col)
            pen.setWidthF(max(1.0, mind * (0.0038 if major else 0.0024)))
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)

            x1 = cx + rin * math.cos(ang)
            y1 = cy + rin * math.sin(ang)
            x2 = cx + r_outer * math.cos(ang)
            y2 = cy + r_outer * math.sin(ang)
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # --- Patek-style bezel ring BELOW the ticks ---
        bezel_r_outer = r_inner_major - mind * 0.003
        bezel_r_inner = bezel_r_outer - mind * 0.005

        # Outer shadow (dark groove edge)
        pen_shadow = QPen(QColor(0, 0, 0, int(100 * alpha)))
        pen_shadow.setWidthF(max(1.0, mind * 0.003))
        p.setPen(pen_shadow)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), bezel_r_outer, bezel_r_outer)

        # Recessed channel
        channel_r = (bezel_r_outer + bezel_r_inner) * 0.5
        pen_ch = QPen(QColor(5, 10, 25, int(65 * alpha)))
        pen_ch.setWidthF(max(1.0, (bezel_r_outer - bezel_r_inner) * 0.9))
        p.setPen(pen_ch)
        p.drawEllipse(QPointF(cx, cy), channel_r, channel_r)

        # Inner highlight (subtle silver catch)
        pen_hi = QPen(QColor(*bezel_hi, int(30 * alpha)))
        pen_hi.setWidthF(max(0.8, mind * 0.002))
        p.setPen(pen_hi)
        p.drawEllipse(QPointF(cx, cy), bezel_r_inner, bezel_r_inner)
    finally:
        p.restore()


# ---------------------------------------------------------------------------
# Perpetual second hand — elegant sweeping indicator
# ---------------------------------------------------------------------------

def draw_perpetual_hand(
    p: QPainter, cx: float, cy: float, mind: float, t: float,
    alpha: float = 1.0, scheme: Optional[dict] = None,
) -> None:
    """Draw a thin, sweeping second hand when Aura Perpetual is active.

    Completes one full revolution every 60 seconds — a real second hand.
    Rendered as a hairspring-thin blued steel needle with a jeweled
    counterweight and luminous tip.
    """
    alpha = clamp(alpha, 0.0, 1.0)
    if alpha <= 0.0:
        return

    tick_color = scheme["tick_color"] if scheme else (195, 215, 240)
    accent = scheme.get("accent", (145, 175, 215)) if scheme else (145, 175, 215)

    p.save()
    try:
        p.setRenderHint(p.Antialiasing, True)

        # Hand sweeps once per 60 seconds (smooth, not ticking)
        import time as _time
        frac = (_time.time() % 60.0) / 60.0
        ang = -math.pi / 2.0 + frac * 2.0 * math.pi  # 12 o'clock start

        # Dimensions
        edge_pad = mind * 0.008
        r_outer = (mind * 0.5) - edge_pad
        perim_margin = mind * 0.038
        hand_len = r_outer - perim_margin * 0.85  # reach to inner bezel ring
        tail_len = hand_len * 0.18                 # counterweight tail
        hand_w = max(1.0, mind * 0.0018)           # hairspring thin

        tip_x = cx + hand_len * math.cos(ang)
        tip_y = cy + hand_len * math.sin(ang)
        tail_x = cx - tail_len * math.cos(ang)
        tail_y = cy - tail_len * math.sin(ang)

        # Shadow (subtle depth)
        shd_off = max(0.5, mind * 0.001)
        shd_pen = QPen(QColor(0, 0, 0, int(80 * alpha)))
        shd_pen.setWidthF(hand_w * 1.5)
        shd_pen.setCapStyle(Qt.RoundCap)
        p.setPen(shd_pen)
        p.drawLine(QPointF(tail_x + shd_off, tail_y + shd_off),
                   QPointF(tip_x + shd_off, tip_y + shd_off))

        # Main hand — blued steel color
        hand_col = QColor(*accent, int(180 * alpha))
        hand_pen = QPen(hand_col)
        hand_pen.setWidthF(hand_w)
        hand_pen.setCapStyle(Qt.RoundCap)
        p.setPen(hand_pen)
        p.drawLine(QPointF(tail_x, tail_y), QPointF(tip_x, tip_y))

        # Luminous tip dot
        tip_r = max(1.5, mind * 0.004)
        p.setPen(Qt.NoPen)
        glow = QRadialGradient(QPointF(tip_x, tip_y), tip_r * 3)
        glow.setColorAt(0.0, QColor(*accent, int(60 * alpha)))
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(glow))
        p.drawEllipse(QPointF(tip_x, tip_y), tip_r * 3, tip_r * 3)
        p.setBrush(QColor(*tick_color, int(220 * alpha)))
        p.drawEllipse(QPointF(tip_x, tip_y), tip_r, tip_r)

        # Counterweight circle (small, at tail)
        cw_r = max(1.8, mind * 0.005)
        p.setBrush(QColor(*accent, int(140 * alpha)))
        p.drawEllipse(QPointF(tail_x, tail_y), cw_r, cw_r)

        # Center jewel pivot
        pivot_r = max(2.5, mind * 0.006)
        jg = QRadialGradient(QPointF(cx - pivot_r * 0.2, cy - pivot_r * 0.2), pivot_r)
        jg.setColorAt(0.0, QColor(255, 255, 255, int(200 * alpha)))
        jg.setColorAt(0.4, QColor(*accent, int(180 * alpha)))
        jg.setColorAt(1.0, QColor(40, 30, 60, int(160 * alpha)))
        p.setBrush(QBrush(jg))
        p.drawEllipse(QPointF(cx, cy), pivot_r, pivot_r)

    finally:
        p.restore()


# ---------------------------------------------------------------------------
# Center encircling ring
# ---------------------------------------------------------------------------

def draw_center_ring(
    p: QPainter, cx: float, cy: float, mind: float, t: float,
    alpha: float = 1.0,
    farsight_active: bool = False,
    scheme: Optional[dict] = None,
) -> None:
    """Draw the iris ring around the center loops.

    When farsight_active is True, the ring shifts toward gold/amber
    to indicate remote GPU processing.
    """
    alpha = clamp(alpha, 0.0, 1.0)
    if alpha <= 0.0:
        return

    ring_main = scheme["ring_main"] if scheme else (145, 175, 215)
    ring_hi   = scheme["ring_hi"]   if scheme else (200, 220, 245)

    p.save()
    try:
        r = mind * 0.265 * 0.80
        breathe = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(t * 0.65))

        penS = QPen(QColor(0, 0, 0, int(120 * alpha)))
        penS.setWidthF(max(1.0, mind * 0.0046))
        penS.setCapStyle(Qt.RoundCap)
        p.setPen(penS)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r, r)

        if farsight_active:
            # Gold/amber ring when offloading to RTX
            fs_pulse = 0.6 + 0.4 * math.sin(t * 2.5)
            col_main = QColor(
                int(180 + 20 * fs_pulse), int(155 + 20 * fs_pulse), int(70 + 20 * fs_pulse),
                int((140 + 60 * breathe) * alpha),
            )
        else:
            col_main = QColor(*ring_main, int((120 + 55 * breathe) * alpha))

        pen = QPen(col_main)
        pen.setWidthF(max(0.8, mind * 0.0027))
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawEllipse(QPointF(cx, cy), r * 0.998, r * 0.998)

        col_hi = QColor(*ring_hi, int(38 * alpha))
        penH = QPen(col_hi)
        penH.setWidthF(max(0.8, mind * 0.0024))
        penH.setCapStyle(Qt.RoundCap)
        p.setPen(penH)
        p.drawEllipse(QPointF(cx, cy), r * 0.972, r * 0.972)
    finally:
        p.restore()


# ---------------------------------------------------------------------------
# Constellation jewels — peer puck indicators on the main dial
# ---------------------------------------------------------------------------

def draw_constellation_jewels(
    p: QPainter, cx: float, cy: float, mind: float, t: float,
    peers: list,
    alpha: float = 1.0,
) -> None:
    """Draw small jewels near 4 o'clock showing other online pucks.

    Each peer is a dict with at least 'color' and 'status' keys.
    Jewels are positioned in a tight arc just inside the chapter ticks.
    """
    alpha = clamp(alpha, 0.0, 1.0)
    if alpha <= 0.0 or not peers:
        return

    p.save()
    try:
        p.setRenderHint(QPainter.Antialiasing, True)

        n = len(peers)
        jewel_r = mind * 0.010
        orbit_r = mind * 0.38          # just inside the chapter ring
        base_angle = math.pi * 0.28    # ~4 o'clock position
        spread = math.pi * 0.06        # angular spread per jewel

        for i, peer in enumerate(peers):
            # Distribute evenly around the base angle
            offset = (i - (n - 1) / 2.0) * spread
            angle = base_angle + offset

            jx = cx + orbit_r * math.cos(angle)
            jy = cy + orbit_r * math.sin(angle)

            # Parse color
            col_hex = peer.get("color", "#23A5FF")
            try:
                cr = int(col_hex[1:3], 16)
                cg = int(col_hex[3:5], 16)
                cb = int(col_hex[5:7], 16)
            except (ValueError, IndexError):
                cr, cg, cb = 35, 165, 255

            is_online = peer.get("status", "offline") != "offline"
            j_alpha = alpha if is_online else alpha * 0.3

            # Glow halo
            pulse = 0.6 + 0.4 * math.sin(t * 1.8 + i * 1.5)
            glow_a = int(40 * j_alpha * pulse)
            halo = QRadialGradient(QPointF(jx, jy), jewel_r * 3.5)
            halo.setColorAt(0.0, QColor(cr, cg, cb, glow_a))
            halo.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(halo))
            p.drawEllipse(QPointF(jx, jy), jewel_r * 3.5, jewel_r * 3.5)

            # Jewel body
            jg = QRadialGradient(
                QPointF(jx - jewel_r * 0.3, jy - jewel_r * 0.3),
                jewel_r,
            )
            jg.setColorAt(0.0, QColor(
                min(255, cr + 60), min(255, cg + 60), min(255, cb + 60),
                int(230 * j_alpha),
            ))
            jg.setColorAt(0.6, QColor(cr, cg, cb, int(210 * j_alpha)))
            jg.setColorAt(1.0, QColor(
                max(0, cr - 40), max(0, cg - 40), max(0, cb - 40),
                int(180 * j_alpha),
            ))
            p.setBrush(QBrush(jg))
            p.drawEllipse(QPointF(jx, jy), jewel_r, jewel_r)

            # Highlight spec
            p.setBrush(QColor(255, 255, 255, int(80 * j_alpha)))
            p.drawEllipse(
                QPointF(jx - jewel_r * 0.25, jy - jewel_r * 0.25),
                jewel_r * 0.28, jewel_r * 0.28,
            )

            # Bezel ring
            p.setPen(QPen(
                QColor(200, 205, 218, int(60 * j_alpha)),
                max(0.6, mind * 0.001),
            ))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(jx, jy), jewel_r * 1.15, jewel_r * 1.15)
    finally:
        p.restore()


# ---------------------------------------------------------------------------
# Mist / gold dust
# ---------------------------------------------------------------------------

def draw_mist(
    p: QPainter, cx: float, cy: float, mind: float,
    particles: List[Tuple[float, float, float]],
    strength: float = 1.0,
    scheme: Optional[dict] = None,
) -> None:
    mist_color = scheme["mist_color"] if scheme else (150, 185, 225)
    for (r, th, s) in particles:
        rr = (mind * 0.5) * (r ** 0.65)
        x = cx + rr * math.cos(th)
        y = cy + rr * math.sin(th)
        a = int(80 * strength * (0.3 + 0.7 * s) * (1.0 - r))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(*mist_color, a)))
        rad = int(1 + 4 * (0.3 + s) * (1.0 - r))
        p.drawEllipse(int(x - rad), int(y - rad), int(2 * rad), int(2 * rad))


# ---------------------------------------------------------------------------
# Inner rings (hero animation — planet palette)
# ---------------------------------------------------------------------------

def _ring_color(
    loops: List[LoopParams], loop_idx: int, seg_u: float, tsec: float, a255: int,
    palette: str = "blue",
) -> QColor:
    """Planet tribute palette with per-ring phase offset."""

    def clamp01(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    def smoothstep(x: float) -> float:
        x = clamp01(x)
        return x * x * (3 - 2 * x)

    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    def mix(c0, c1, t):
        return (int(_lerp(c0[0], c1[0], t)), int(_lerp(c0[1], c1[1], t)), int(_lerp(c0[2], c1[2], t)))

    def grad(stops, u):
        u = u % 1.0
        for i in range(len(stops) - 1):
            p0, c0 = stops[i]
            p1, c1 = stops[i + 1]
            if p0 <= u <= p1:
                return mix(c0, c1, smoothstep((u - p0) / (p1 - p0)))
        return stops[-1][1]

    BLUE_PLANETS = [
        ("Deep",    [(0.0, (12, 28, 68)), (0.5, (40, 110, 195)), (1.0, (8, 22, 55))]),
        ("Arctic",  [(0.0, (18, 45, 95)), (0.4, (75, 165, 235)), (0.7, (140, 200, 245)),
                     (1.0, (18, 45, 95))]),
        ("Abyss",   [(0.0, (6, 16, 42)),  (0.35, (30, 85, 165)), (0.65, (55, 140, 210)),
                     (0.85, (90, 175, 235)), (1.0, (6, 16, 42))]),
        ("Silver",  [(0.0, (35, 45, 65)), (0.3, (120, 145, 185)), (0.6, (170, 190, 220)),
                     (0.85, (100, 130, 175)), (1.0, (35, 45, 65))]),
        ("Cobalt",  [(0.0, (15, 32, 72)), (0.4, (50, 120, 200)), (0.7, (85, 160, 230)),
                     (1.0, (15, 32, 72))]),
    ]

    RED_PLANETS = [
        ("Ember",   [(0.0, (68, 12, 12)), (0.5, (195, 55, 40)), (1.0, (55, 8, 8))]),
        ("Flame",   [(0.0, (95, 18, 18)), (0.4, (235, 75, 40)), (0.7, (245, 120, 60)),
                     (1.0, (95, 18, 18))]),
        ("Garnet",  [(0.0, (42, 6, 16)),  (0.35, (165, 30, 55)), (0.65, (210, 55, 65)),
                     (0.85, (235, 90, 75)), (1.0, (42, 6, 16))]),
        ("Burgundy",[(0.0, (55, 10, 18)), (0.3, (160, 35, 45)), (0.6, (200, 60, 55)),
                     (0.85, (170, 40, 40)), (1.0, (55, 10, 18))]),
        ("Crimson", [(0.0, (72, 15, 15)), (0.4, (200, 50, 50)), (0.7, (230, 85, 70)),
                     (1.0, (72, 15, 15))]),
    ]

    PLANETS = RED_PLANETS if palette == "red" else BLUE_PLANETS

    PLANET_DUR = 16.0
    FADE = 4.0
    total = PLANET_DUR * len(PLANETS)
    tt = tsec % total
    idx = int(tt // PLANET_DUR)
    u_time = (tt % PLANET_DUR) / PLANET_DUR

    _, g0 = PLANETS[idx]
    _, g1 = PLANETS[(idx + 1) % len(PLANETS)]

    fade = 0.0
    if u_time > 1.0 - (FADE / PLANET_DUR):
        fade = smoothstep((u_time - (1.0 - FADE / PLANET_DUR)) / (FADE / PLANET_DUR))

    shift = getattr(loops[loop_idx], "hue_shift", 0.0) / 360.0
    flow = (seg_u + shift + tsec * (0.010 + 0.004 * loop_idx)) % 1.0

    c0 = grad(g0, flow)
    c1 = grad(g1, flow)
    r, g, b = mix(c0, c1, fade)

    breathe = 0.92 + 0.08 * math.sin(tsec * 0.22 + loop_idx * 1.3)
    r = int(clamp01((r / 255) * breathe) * 255)
    g = int(clamp01((g / 255) * breathe) * 255)
    b = int(clamp01((b / 255) * breathe) * 255)

    return QColor(r, g, b, a255)


# Mute-mode red color
MUTE_RED = (220, 75, 65)


def draw_rings(
    p: QPainter, cx: float, cy: float, mind: float, t: float,
    loops: List[LoopParams],
    base_speed: float = 0.20,
    loop_scale: float = 2.53,
    alpha_scale: float = 1.0,
    pixelate: float = 0.0,
    speaking: bool = False,
    muted: bool = False,
    scheme: Optional[dict] = None,
) -> None:
    """Draw the 4-loop hero harmonic animation.

    Optimised: QPainterPath batching replaces per-segment drawLine calls.
    Points reduced from 64/96 → 48/72 (Lissajous curves are smooth enough).
    """
    _palette = scheme["ring_palette"] if scheme else "blue"
    _speak_hi = scheme["speak_hi"] if scheme else (190, 218, 248)
    iris_R = mind * 0.24 * 0.80
    N = 72 if speaking else 48

    circ_blend = 0.18
    circ_target = 0.205
    circ_min, circ_max = 0.175, 0.235
    streak_thresh2 = (mind * 0.22) ** 2

    pz = clamp((pixelate - 0.20) / 0.80, 0.0, 1.0)
    grid = max(1, int(1 + (pz ** 0.85) * (mind * 0.055)))
    drop_p = clamp(0.72 * (pz ** 1.10), 0.0, 0.78)

    _col_cache = {}
    _sin = math.sin
    _pi2 = 2.0 * math.pi

    def _build_path(points):
        """Build a QPainterPath from point list, skipping streak jumps."""
        path = QPainterPath()
        n = len(points)
        if n < 2:
            return path
        need_move = True
        for i in range(n):
            x0, y0 = points[i]
            x1, y1 = points[(i + 1) % n]
            dx = x0 - x1
            dy = y0 - y1
            if dx * dx + dy * dy >= streak_thresh2:
                need_move = True
                continue
            if need_move:
                path.moveTo(x0, y0)
                need_move = False
            path.lineTo(x1, y1)
        return path

    def draw_loop_batched(loop_idx: int, points, alpha_base: int, width: float):
        a = int(alpha_base * alpha_scale)
        if a <= 0:
            return

        w_eff = max(1.0, width * (1.0 - 0.25 * pz))

        if muted:
            # Muted: single red colour for the whole loop
            shimmer = 0.5 + 0.5 * _sin(t * 0.55 + loop_idx * 1.3)
            rr = int(MUTE_RED[0] + 20 * shimmer)
            gg = int(MUTE_RED[1] - 10 * shimmer)
            bb = int(MUTE_RED[2] - 10 * shimmer)
            col = QColor(rr, gg, bb, a)
            path = _build_path(points)
            p.setPen(QPen(col, w_eff, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
            return

        if drop_p <= 0.0:
            # No pixelation: batch the entire loop with a single representative colour
            mid_u = 0.5
            key = (loop_idx, int(mid_u * N), a)
            col = _col_cache.get(key)
            if col is None:
                col = _ring_color(loops, loop_idx, mid_u, t, a, palette=_palette)
                _col_cache[key] = col
            path = _build_path(points)
            p.setPen(QPen(col, w_eff, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
        else:
            # Pixelation mode: per-segment with drop probability (rare)
            n = len(points)
            for i in range(n):
                x0, y0 = points[i]
                x1, y1 = points[(i + 1) % n]
                dx = x0 - x1
                dy = y0 - y1
                if dx * dx + dy * dy >= streak_thresh2:
                    continue
                block = 7 if grid <= 10 else 5
                bi = i // block
                h = (bi * 1103515245 + 12345 + (loop_idx * 1013904223)) & 0xFFFFFFFF
                u = ((h >> 8) & 0xFF) / 255.0
                if u < drop_p:
                    continue
                seg_u = i / float(max(1, n - 1))
                key = (loop_idx, int(seg_u * N), a)
                col = _col_cache.get(key)
                if col is None:
                    col = _ring_color(loops, loop_idx, seg_u, t, a, palette=_palette)
                    _col_cache[key] = col
                p.setPen(QPen(col, w_eff, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                p.drawLine(x0, y0, x1, y1)

    phi = 1.61803398875
    w1 = base_speed * 1.00
    w2 = base_speed * phi
    w3 = base_speed * 0.50

    depth = 0.006
    loop_scale_back = loop_scale * (1.0 - depth)
    loop_scale_front = loop_scale * (1.0 + depth)

    # Pre-compute shared u array once (all loops use same N)
    _u_arr = np.linspace(0.0, _pi2, N, endpoint=False)

    for loop_idx, lp in enumerate(loops):
        off = loop_idx * 0.35

        # Vectorised Lissajous: all N points in one pass
        a1 = lp.k1 * _u_arr + lp.p1 + t * w1
        a2 = 0.5 * lp.k1 * _u_arr + 0.7 * lp.p1 + t * w2 + off
        a3 = lp.k1 * _u_arr + 0.3 * lp.p1 - t * w3 - off * 0.8

        b1 = lp.k2 * _u_arr + lp.p2 - t * w1 * 0.90
        b2 = 0.5 * lp.k2 * _u_arr + 0.7 * lp.p2 - t * w2 * 0.62 - off
        b3 = lp.k2 * _u_arr + 0.3 * lp.p2 + t * w3 * 0.78 + off * 0.6

        x = lp.a * np.sin(a1) + (lp.a * 0.11) * np.sin(a2) + (lp.a * 0.06) * np.sin(a3)
        y = lp.b * np.sin(b1) + (lp.b * 0.11) * np.sin(b2) + (lp.b * 0.06) * np.sin(b3)

        # Circular blend
        r0 = np.sqrt(x * x + y * y) + 1e-6
        nx = x / r0
        ny = y / r0
        x = (1.0 - circ_blend) * x + circ_blend * (nx * circ_target)
        y = (1.0 - circ_blend) * y + circ_blend * (ny * circ_target)

        # Radial clamp
        r2 = np.sqrt(x * x + y * y) + 1e-6
        need_clamp = (r2 < circ_min) | (r2 > circ_max)
        rr = np.clip(r2, circ_min, circ_max)
        scale = np.where(need_clamp, rr / r2, 1.0)
        x *= scale
        y *= scale

        # Screen coordinates (back + front depth layers)
        Xb = (cx + x * iris_R * loop_scale_back).astype(np.int32)
        Yb = (cy + y * iris_R * loop_scale_back).astype(np.int32)
        Xf = (cx + x * iris_R * loop_scale_front).astype(np.int32)
        Yf = (cy + y * iris_R * loop_scale_front).astype(np.int32)

        if grid > 1:
            Xb = (Xb // grid) * grid
            Yb = (Yb // grid) * grid
            Xf = (Xf // grid) * grid
            Yf = (Yf // grid) * grid

        # Convert to Python list of tuples for QPainterPath
        pts_back = list(zip(Xb.tolist(), Yb.tolist()))
        pts_front = list(zip(Xf.tolist(), Yf.tolist()))

        do_glow = (pixelate <= 0.01)

        if do_glow:
            draw_loop_batched(loop_idx, pts_back, 95, 6.0)
        draw_loop_batched(loop_idx, pts_back, 175, 3.6)

        if do_glow:
            draw_loop_batched(loop_idx, pts_front, 130, 5.0)
        draw_loop_batched(loop_idx, pts_front, 245, 3.0)

        # Speaking micro-highlights
        if speaking and do_glow:
            step = max(1, len(pts_front) // 8)
            a = int(42 * alpha_scale)
            if a > 0:
                p.save()
                try:
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(*_speak_hi, a))
                    rpx = max(1.0, mind * 0.0022)
                    for k in range(0, len(pts_front), step):
                        xx, yy = pts_front[k]
                        p.drawEllipse(QPointF(xx, yy), rpx, rpx)
                finally:
                    p.restore()


# ---------------------------------------------------------------------------
# Muted lacquer wash overlay
# ---------------------------------------------------------------------------

def draw_mute_wash(p: QPainter, cx: float, cy: float, mind: float, W: int, H: int) -> None:
    p.save()
    try:
        p.setCompositionMode(QPainter.CompositionMode_Multiply)
        p.setOpacity(0.72)
        g = QRadialGradient(QPointF(cx, cy), mind * 0.62)
        g.setColorAt(0.00, QColor(255, 80, 90, 160))
        g.setColorAt(0.55, QColor(170, 20, 30, 210))
        g.setColorAt(1.00, QColor(90, 0, 10, 245))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(g))
        p.drawEllipse(QPointF(cx, cy), mind * 0.52, mind * 0.52)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        p.setOpacity(0.20)
        step = max(2, int(mind * 0.010))
        penA = QPen(QColor(255, 210, 210, 28))
        penB = QPen(QColor(0, 0, 0, 26))
        for yy in range(0, int(H), step):
            p.setPen(penA); p.drawLine(0, yy, W, yy)
            p.setPen(penB); p.drawLine(0, yy + 1, W, yy + 1)
    finally:
        p.restore()

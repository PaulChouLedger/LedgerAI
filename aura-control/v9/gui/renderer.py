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

def make_celestial_stars(n_white: int = 320, n_gold: int = 140) -> List[Star]:
    stars: List[Star] = []
    rng = random.Random(1337)

    for _ in range(n_white):
        r = (rng.random() ** 0.65) * 0.98
        th = rng.random() * 2 * math.pi
        size = 0.6 + 1.4 * rng.random()
        base_a = int(14 + 35 * rng.random())    # dimmer — luxury dark dial
        tw = 0.10 + 0.35 * rng.random()
        ph = rng.random() * 2 * math.pi
        stars.append(Star(r=r, th=th, size=size, base_a=base_a, tw=tw, ph=ph, hue=0))

    # Platinum dust (subtle)
    for _ in range(n_gold):
        r = (rng.random() ** 0.65) * 0.98
        th = rng.random() * 2 * math.pi
        size = 0.5 + 1.0 * rng.random()
        base_a = int(10 + 28 * rng.random())    # dimmer
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
    """Holds cached background pixmaps; rebuilds on resize."""

    def __init__(self) -> None:
        self._blue_cache: Optional[QPixmap] = None
        self._blue_key: Optional[Tuple[int, int]] = None
        self._red_cache: Optional[QPixmap] = None
        self._red_key: Optional[Tuple[int, int]] = None

    def get_blue(self, W: int, H: int, mind: float) -> QPixmap:
        key = (W, H)
        if self._blue_cache is not None and self._blue_key == key:
            return self._blue_cache
        self._blue_cache = _build_blue_background(W, H, mind)
        self._blue_key = key
        return self._blue_cache

    def get_red(self, W: int, H: int, mind: float) -> QPixmap:
        key = (W, H)
        if self._red_cache is not None and self._red_key == key:
            return self._red_cache
        self._red_cache = _build_red_background(W, H, mind)
        self._red_key = key
        return self._red_cache


def _build_blue_background(W: int, H: int, mind: float) -> QPixmap:
    """Super dark red dial — deep oxblood, nearly black."""
    pm = QPixmap(W, H)
    pm.fill(QColor(140, 22, 18))
    q = QPainter(pm)
    try:
        q.setRenderHint(QPainter.Antialiasing)
        cx, cy = W * 0.5, H * 0.5
        R = min(W, H) * 0.5

        # Ruby glow in center
        gv = QRadialGradient(QPointF(cx, cy), R * 0.85)
        gv.setColorAt(0.0, QColor(170, 30, 28, 60))
        gv.setColorAt(0.5, QColor(130, 22, 20, 30))
        gv.setColorAt(1.0, QColor(0, 0, 0, 0))
        q.setPen(Qt.NoPen)
        q.setBrush(QBrush(gv))
        q.drawRect(0, 0, W, H)

        # Edge darkening
        gv2 = QRadialGradient(QPointF(cx, cy), R * 1.1)
        gv2.setColorAt(0.00, QColor(0, 0, 0, 0))
        gv2.setColorAt(0.75, QColor(0, 0, 0, 0))
        gv2.setColorAt(0.95, QColor(0, 0, 0, 40))
        gv2.setColorAt(1.00, QColor(0, 0, 0, 90))
        q.setBrush(QBrush(gv2))
        q.drawRect(0, 0, W, H)

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
# Drifting ember particles (visible dynamic texture on dark red background)
# ---------------------------------------------------------------------------

# Pre-seeded ember data: (base_angle, base_radius, drift_speed, phase, size, brightness)
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
) -> None:
    """Drifting ember particles — bright red sparks floating across the dark dial."""
    alpha = clamp(alpha, 0.0, 1.0)
    if alpha <= 0.0:
        return

    _ensure_ember_seeds()

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

            # Ember color: hot red core fading to dark
            rad = sz * (mind / 1080.0)
            g = QRadialGradient(QPointF(x, y), max(1.0, rad * 2.5))
            g.setColorAt(0.0, QColor(220, 50, 40, a0))
            g.setColorAt(0.3, QColor(160, 25, 20, int(a0 * 0.6)))
            g.setColorAt(0.7, QColor(90, 10, 10, int(a0 * 0.2)))
            g.setColorAt(1.0, QColor(140, 22, 18, 0))
            p.setBrush(QBrush(g))
            p.drawEllipse(QPointF(x, y), rad * 2.5, rad * 2.5)

            # Bright core dot
            if a0 > 40:
                p.setBrush(QColor(255, 100, 60, int(a0 * 0.8)))
                p.drawEllipse(QPointF(x, y), rad * 0.5, rad * 0.5)

    finally:
        p.restore()


# ---------------------------------------------------------------------------
# Celestial starfield
# ---------------------------------------------------------------------------

def draw_celestial(
    p: QPainter, cx: float, cy: float, mind: float, t: float,
    stars: List[Star],
) -> None:
    t = t * 2.5
    R = mind * 0.46
    rot = t * (2 * math.pi / (9 * 60.0))

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
            jitter = int((s.ph % 1.0) * 6.0)
            col = QColor(
                int(clamp(218 + jitter, 0, 255)),
                int(clamp(222 + jitter, 0, 255)),
                int(clamp(235 + jitter, 0, 255)),
                a,
            )
        else:
            col = QColor(195, 200, 215, a)  # cool platinum dust

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
) -> None:
    alpha = clamp(alpha, 0.0, 1.0)
    if alpha <= 0.0:
        return

    p.save()
    try:
        perim_margin = mind * perim_margin_frac
        edge_pad = mind * 0.008
        r_outer = (mind * 0.5) - edge_pad

        # --- Ticks (white/silver, varied lengths) ---
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

            # Clean white ticks — crisp against the red dial
            col = QColor(235, 235, 240, clamp(a, 0, 255))
            pen = QPen(col)
            pen.setWidthF(max(1.0, mind * (0.0038 if major else 0.0024)))
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)

            x1 = cx + rin * math.cos(ang)
            y1 = cy + rin * math.sin(ang)
            x2 = cx + r_outer * math.cos(ang)
            y2 = cy + r_outer * math.sin(ang)
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # --- Patek-style bezel ring BELOW the ticks (encompassing them) ---
        bezel_r_outer = r_inner_major - mind * 0.003
        bezel_r_inner = bezel_r_outer - mind * 0.005

        # Outer shadow (dark groove edge — gives depth)
        pen_shadow = QPen(QColor(0, 0, 0, int(100 * alpha)))
        pen_shadow.setWidthF(max(1.0, mind * 0.003))
        p.setPen(pen_shadow)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), bezel_r_outer, bezel_r_outer)

        # Recessed channel (dark band for depth illusion)
        channel_r = (bezel_r_outer + bezel_r_inner) * 0.5
        pen_ch = QPen(QColor(8, 2, 2, int(65 * alpha)))
        pen_ch.setWidthF(max(1.0, (bezel_r_outer - bezel_r_inner) * 0.9))
        p.setPen(pen_ch)
        p.drawEllipse(QPointF(cx, cy), channel_r, channel_r)

        # Inner highlight (subtle silver catch on lower lip)
        pen_hi = QPen(QColor(220, 220, 225, int(30 * alpha)))
        pen_hi.setWidthF(max(0.8, mind * 0.002))
        p.setPen(pen_hi)
        p.drawEllipse(QPointF(cx, cy), bezel_r_inner, bezel_r_inner)
    finally:
        p.restore()


# ---------------------------------------------------------------------------
# Center encircling ring
# ---------------------------------------------------------------------------

def draw_center_ring(
    p: QPainter, cx: float, cy: float, mind: float, t: float,
    alpha: float = 1.0,
) -> None:
    alpha = clamp(alpha, 0.0, 1.0)
    if alpha <= 0.0:
        return

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

        col_main = QColor(120, 20, 25, int((120 + 55 * breathe) * alpha))  # dark ruby ring
        pen = QPen(col_main)
        pen.setWidthF(max(0.8, mind * 0.0027))
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawEllipse(QPointF(cx, cy), r * 0.998, r * 0.998)

        col_hi = QColor(200, 205, 218, int(38 * alpha))  # platinum highlight
        penH = QPen(col_hi)
        penH.setWidthF(max(0.8, mind * 0.0024))
        penH.setCapStyle(Qt.RoundCap)
        p.setPen(penH)
        p.drawEllipse(QPointF(cx, cy), r * 0.972, r * 0.972)
    finally:
        p.restore()


# ---------------------------------------------------------------------------
# Mist / gold dust
# ---------------------------------------------------------------------------

def draw_mist(
    p: QPainter, cx: float, cy: float, mind: float,
    particles: List[Tuple[float, float, float]],
    strength: float = 1.0,
) -> None:
    for (r, th, s) in particles:
        rr = (mind * 0.5) * (r ** 0.65)
        x = cx + rr * math.cos(th)
        y = cy + rr * math.sin(th)
        a = int(45 * strength * (0.3 + 0.7 * s) * (1.0 - r))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(140, 30, 25, a)))  # dark red dust
        rad = int(1 + 4 * (0.3 + s) * (1.0 - r))
        p.drawEllipse(int(x - rad), int(y - rad), int(2 * rad), int(2 * rad))


# ---------------------------------------------------------------------------
# Inner rings (hero animation — planet palette)
# ---------------------------------------------------------------------------

def _ring_color(
    loops: List[LoopParams], loop_idx: int, seg_u: float, tsec: float, a255: int,
) -> QColor:
    """Subtle cool blue-white — softer against warm rose background."""

    def clamp01(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    shift = getattr(loops[loop_idx], "hue_shift", 0.0) / 360.0
    flow = (seg_u + shift + tsec * (0.008 + 0.003 * loop_idx)) % 1.0

    # Subtle cool blue-white shimmer
    shimmer = 0.5 + 0.5 * math.sin(flow * 2 * math.pi)
    r = int(185 + 20 * shimmer)    # 185-205  (muted)
    g = int(192 + 18 * shimmer)    # 192-210  (slightly blue-shifted)
    b = int(210 + 15 * shimmer)    # 210-225  (cool blue tint)

    breathe = 0.90 + 0.10 * math.sin(tsec * 0.22 + loop_idx * 1.3)
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
    hand_angle: Optional[float] = None,
    hand_strength: float = 0.0,
) -> None:
    """Draw the 4-loop hero harmonic animation."""
    iris_R = mind * 0.24 * 0.80
    N = 96 if speaking else 64

    circ_blend = 0.18
    circ_target = 0.205
    circ_min, circ_max = 0.175, 0.235
    streak_thresh2 = (mind * 0.22) ** 2

    pz = clamp((pixelate - 0.20) / 0.80, 0.0, 1.0)
    grid = max(1, int(1 + (pz ** 0.85) * (mind * 0.055)))
    drop_p = clamp(0.72 * (pz ** 1.10), 0.0, 0.78)

    _col_cache = {}

    def draw_loop(loop_idx: int, points, alpha_base: int, width: float):
        for i in range(len(points)):
            x0, y0 = points[i]
            x1, y1 = points[(i + 1) % len(points)]

            dx = x0 - x1
            dy = y0 - y1
            if dx * dx + dy * dy >= streak_thresh2:
                continue

            a = int(alpha_base * alpha_scale)
            if a <= 0:
                continue

            if drop_p > 0.0:
                block = 7 if grid <= 10 else 5
                bi = i // block
                h = (bi * 1103515245 + 12345 + (loop_idx * 1013904223)) & 0xFFFFFFFF
                u = ((h >> 8) & 0xFF) / 255.0
                if u < drop_p:
                    continue

            seg_u = i / float(max(1, len(points) - 1))

            if muted:
                shimmer = 0.5 + 0.5 * math.sin(t * 0.55 + loop_idx * 1.3 + seg_u * 6.0)
                rr = int(MUTE_RED[0] + 20 * shimmer)
                gg = int(MUTE_RED[1] - 10 * shimmer)
                bb = int(MUTE_RED[2] - 10 * shimmer)
                col = QColor(rr, gg, bb, a)
            else:
                key = (loop_idx, int(seg_u * N), a)
                col = _col_cache.get(key)
                if col is None:
                    col = _ring_color(loops, loop_idx, seg_u, t, a)
                    _col_cache[key] = col

            w_eff = max(1.0, width * (1.0 - 0.25 * pz))
            p.setPen(QPen(col, w_eff, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawLine(x0, y0, x1, y1)

    phi = 1.61803398875
    w1 = base_speed * 1.00
    w2 = base_speed * phi
    w3 = base_speed * 0.50

    depth = 0.006
    loop_scale_back = loop_scale * (1.0 - depth)
    loop_scale_front = loop_scale * (1.0 + depth)

    for loop_idx, lp in enumerate(loops):
        pts_back = []
        pts_front = []
        off = loop_idx * 0.35

        for i in range(N):
            u = (2 * math.pi) * (i / N)

            a1 = lp.k1 * u + lp.p1 + t * w1
            a2 = 0.5 * lp.k1 * u + 0.7 * lp.p1 + t * w2 + off
            a3 = lp.k1 * u + 0.3 * lp.p1 - t * w3 - off * 0.8

            b1 = lp.k2 * u + lp.p2 - t * w1 * 0.90
            b2 = 0.5 * lp.k2 * u + 0.7 * lp.p2 - t * w2 * 0.62 - off
            b3 = lp.k2 * u + 0.3 * lp.p2 + t * w3 * 0.78 + off * 0.6

            x = lp.a * math.sin(a1) + (lp.a * 0.11) * math.sin(a2) + (lp.a * 0.06) * math.sin(a3)
            y = lp.b * math.sin(b1) + (lp.b * 0.11) * math.sin(b2) + (lp.b * 0.06) * math.sin(b3)

            r0 = math.sqrt(x * x + y * y) + 1e-6
            nx, ny = x / r0, y / r0
            x = (1 - circ_blend) * x + circ_blend * (nx * circ_target)
            y = (1 - circ_blend) * y + circ_blend * (ny * circ_target)

            r2 = math.sqrt(x * x + y * y) + 1e-6
            if r2 < circ_min or r2 > circ_max:
                rr = clamp(r2, circ_min, circ_max)
                x *= rr / r2
                y *= rr / r2

            # Subtle "hand" bias — extend loops toward active complication
            if hand_angle is not None and hand_strength > 0.001:
                pt_ang = math.atan2(y, x)
                alignment = 0.5 + 0.5 * math.cos(pt_ang - hand_angle)
                boost = 1.0 + hand_strength * (alignment ** 2)
                x *= boost
                y *= boost

            Xb = int(cx + x * iris_R * loop_scale_back)
            Yb = int(cy + y * iris_R * loop_scale_back)
            Xf = int(cx + x * iris_R * loop_scale_front)
            Yf = int(cy + y * iris_R * loop_scale_front)

            if grid > 1:
                Xb = (Xb // grid) * grid
                Yb = (Yb // grid) * grid
                Xf = (Xf // grid) * grid
                Yf = (Yf // grid) * grid

            pts_back.append((Xb, Yb))
            pts_front.append((Xf, Yf))

        do_glow = (pixelate <= 0.01)

        if do_glow:
            draw_loop(loop_idx, pts_back, 95, 6.0)
        draw_loop(loop_idx, pts_back, 175, 3.6)

        if do_glow:
            draw_loop(loop_idx, pts_front, 130, 5.0)
        draw_loop(loop_idx, pts_front, 245, 3.0)

        # Speaking micro-highlights
        if speaking and do_glow:
            step = max(1, len(pts_front) // 8)
            a = int(42 * alpha_scale)
            if a > 0:
                p.save()
                try:
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(210, 215, 230, a))  # platinum highlight
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

"""
gui.boot_renderer -- Falcon boot animation rendering.

Stateless draw functions extracted from falcon.py. The window calls these
during boot mode; the orchestrator drives progress/phase via bus events.

All functions take a QPainter + geometry args + time/progress state.
No self/state — pure rendering.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import (
    QBrush, QColor, QPainter, QPen, QFont, QRadialGradient,
)

from core.config import FIXED_ROTATION_DEG, COLOR_SCHEMES, DEFAULT_COLOR_SCHEME


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


# ---------------------------------------------------------------------------
# Boot visual state (created once by window, mutated by tick/bus events)
# ---------------------------------------------------------------------------

@dataclass
class FalconStar:
    """A single star in the falcon starfield."""
    x: float
    y: float
    layer: int
    base_r: float
    tw: float  # twinkle phase


@dataclass
class BootVisuals:
    """All mutable state for the falcon boot animation."""
    stars: List[FalconStar] = field(default_factory=list)
    progress: float = 0.0          # 0..1
    phase_text: str = ""           # shown below title
    phase_bounds: List[float] = field(default_factory=lambda: [0.0, 1.0])

    # Timing
    t0: float = 0.0

    # Transition
    fade_alpha: int = 0            # 0..255 for fade-to-black

    # Falcon loop params (same as falcon.py)
    loops: List[Tuple[float, float, float, float, float, float, float, float, float]] = field(
        default_factory=lambda: [
            # (a, b, k1, k2, p1, p2, thick, rot_deg, hue_shift)
            (0.23, 0.20, 3.0, 2.0, 0.2, 1.1, 0.07, 0,   0),
            (0.21, 0.24, 2.0, 3.0, 2.1, 0.6, 0.06, 10,  80),
            (0.19, 0.18, 5.0, 3.0, 1.4, 2.7, 0.09, -8,  160),
            (0.17, 0.22, 4.0, 5.0, 2.8, 1.9, 0.08, 6,   240),
        ]
    )
    N: int = 72  # points per loop


def make_falcon_stars(count: int = 520, layers: int = 3) -> List[FalconStar]:
    """Create the initial starfield for boot animation."""
    stars = []
    for _ in range(count):
        stars.append(FalconStar(
            x=random.random(),
            y=random.random(),
            layer=random.randint(0, layers - 1),
            base_r=random.uniform(0.6, 1.5) + random.randint(0, layers - 1) * 0.25,
            tw=random.uniform(0.0, math.tau),
        ))
    return stars


def make_phase_bounds(n_phases: int, seed: int = 7) -> List[float]:
    """Generate random-ish progress chunk boundaries for n_phases."""
    rng = random.Random(seed)
    weights = [rng.uniform(0.10, 0.22) for _ in range(n_phases)]
    s = sum(weights)
    weights = [w / s for w in weights]
    bounds = [0.0]
    acc = 0.0
    for w in weights:
        acc += w
        bounds.append(acc)
    bounds[-1] = 1.0
    return bounds


# ---------------------------------------------------------------------------
# Ring color (scheme-aware starlight palette)
# ---------------------------------------------------------------------------

# Gradient stops per palette
_RING_GRADIENTS = {
    "blue": {
        "G0": [(0.0, (40, 130, 255)), (0.5, (210, 240, 255)), (1.0, (40, 130, 255))],
        "G1": [(0.0, (150, 210, 255)), (0.5, (240, 245, 255)), (1.0, (110, 170, 255))],
    },
    "red": {
        "G0": [(0.0, (255, 80, 40)), (0.5, (255, 200, 140)), (1.0, (255, 80, 40))],
        "G1": [(0.0, (255, 160, 100)), (0.5, (255, 230, 180)), (1.0, (255, 120, 70))],
    },
}

# Boot accent colors per palette (stars, ticks, dial, arc, text)
_BOOT_ACCENTS = {
    "blue": {
        "star":     (230, 236, 255),
        "tick":     (220, 230, 255),
        "tick_hi":  (220, 230, 255),
        "arc":      (210, 235, 255),
        "phase":    (190, 225, 255),
        "text":     (240, 245, 255),
        "bg":       (0, 0, 0),
    },
    "red": {
        "star":     (255, 220, 210),
        "tick":     (255, 210, 190),
        "tick_hi":  (255, 220, 200),
        "arc":      (255, 190, 140),
        "phase":    (255, 180, 130),
        "text":     (255, 240, 230),
        "bg":       (30, 4, 6),
    },
}


def _ring_color(loop_idx: int, seg_u: float, tsec: float, a255: int,
                hue_shift: float, palette: str = "blue") -> QColor:
    """Per-segment color for a falcon loop ring."""
    def clamp01(x):
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    def smoothstep(x):
        x = clamp01(x)
        return x * x * (3 - 2 * x)

    def _lerp(a, b, t):
        return a + (b - a) * t

    def mix(c0, c1, t):
        return (
            int(_lerp(c0[0], c1[0], t)),
            int(_lerp(c0[1], c1[1], t)),
            int(_lerp(c0[2], c1[2], t)),
        )

    grads = _RING_GRADIENTS.get(palette, _RING_GRADIENTS["blue"])
    G0 = grads["G0"]
    G1 = grads["G1"]

    def grad(stops, u):
        u = u % 1.0
        for i in range(len(stops) - 1):
            p0, c0 = stops[i]
            p1, c1 = stops[i + 1]
            if p0 <= u <= p1:
                return mix(c0, c1, smoothstep((u - p0) / (p1 - p0)))
        return stops[-1][1]

    fade = 0.5 + 0.5 * math.sin(tsec * 0.08 + loop_idx * 0.7)
    shift = hue_shift / 360.0
    flow = (seg_u + shift + tsec * (0.010 + 0.003 * loop_idx)) % 1.0

    c0 = grad(G0, flow)
    c1 = grad(G1, flow)
    r, g, b = mix(c0, c1, fade)

    breathe = 0.92 + 0.08 * math.sin(tsec * 0.22 + loop_idx * 1.3)
    r = int(clamp01((r / 255) * breathe) * 255)
    g = int(clamp01((g / 255) * breathe) * 255)
    b = int(clamp01((b / 255) * breathe) * 255)

    return QColor(r, g, b, a255)


# ---------------------------------------------------------------------------
# Draw functions (all stateless — take QPainter + geometry + boot state)
# ---------------------------------------------------------------------------

def draw_falcon_stars(p: QPainter, W: int, H: int, t: float,
                      stars: List[FalconStar],
                      palette: str = "blue") -> None:
    """Draw twinkling starfield with 3-layer parallax drift."""
    sc = _BOOT_ACCENTS.get(palette, _BOOT_ACCENTS["blue"])["star"]
    drift_x = 0.0006 * math.cos(t * 0.05)
    drift_y = 0.0010 * math.sin(t * 0.04)

    p.setPen(Qt.NoPen)
    for s in stars:
        sp = (s.layer + 1) * 0.16
        s.x = (s.x + drift_x * sp) % 1.0
        s.y = (s.y + drift_y * sp) % 1.0

        twk = 0.55 + 0.45 * math.sin(t * (0.65 + 0.18 * s.layer) + s.tw)
        alpha = int(_clamp(30 + 140 * twk, 18, 175))
        r = s.base_r * (0.85 + 0.25 * twk)

        p.setBrush(QColor(sc[0], sc[1], sc[2], alpha))
        p.drawEllipse(QPointF(s.x * W, s.y * H), r, r)


def draw_falcon_vignette(p: QPainter, cx: float, cy: float, mind: float,
                         W: int, H: int) -> None:
    """Radial dark gradient for depth."""
    vg = QRadialGradient(QPointF(cx, cy), mind * 0.62)
    vg.setColorAt(0.0, QColor(0, 0, 0, 0))
    vg.setColorAt(0.7, QColor(0, 0, 0, 70))
    vg.setColorAt(1.0, QColor(0, 0, 0, 165))
    p.setBrush(QBrush(vg))
    p.setPen(Qt.NoPen)
    p.drawRect(0, 0, W, H)


def draw_falcon_dial(p: QPainter, cx: float, cy: float, mind: float,
                     progress: float, phase_bounds: List[float],
                     pulse_amt: float = 1.0,
                     palette: str = "blue") -> None:
    """Patek-style progress arc with hairline ring, 60 ticks, phase boundaries."""
    acc = _BOOT_ACCENTS.get(palette, _BOOT_ACCENTS["blue"])
    tc = acc["tick"]
    pc = acc["phase"]
    ac = acc["arc"]

    outer_r = mind * 0.485
    dial_w = max(2.0, mind * 0.0045)
    dial_rect = QRectF(cx - outer_r, cy - outer_r, 2 * outer_r, 2 * outer_r)

    # Hairline ring
    p.setPen(QPen(QColor(tc[0], tc[1], tc[2], 35), dial_w, Qt.SolidLine, Qt.RoundCap))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(dial_rect)

    # Tick marks (60)
    ticks = 60
    for i in range(ticks):
        ang = (i / ticks) * math.tau
        a = ang - math.pi / 2.0
        long_tick = (i % 5 == 0)
        r0 = outer_r - (mind * (0.018 if long_tick else 0.010))
        r1 = outer_r - (mind * 0.004)
        col = QColor(tc[0], tc[1], tc[2], 70 if long_tick else 40)
        p.setPen(QPen(col, max(1.0, dial_w * (1.4 if long_tick else 0.9)),
                       Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(cx + math.cos(a) * r0, cy + math.sin(a) * r0),
                   QPointF(cx + math.cos(a) * r1, cy + math.sin(a) * r1))

    # Phase boundary markers
    for b in phase_bounds[1:-1]:
        a = (b * math.tau) - math.pi / 2.0
        r0 = outer_r - mind * 0.030
        r1 = outer_r - mind * 0.006
        p.setPen(QPen(QColor(pc[0], pc[1], pc[2], 60), max(1.0, dial_w * 1.0),
                       Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(cx + math.cos(a) * r0, cy + math.sin(a) * r0),
                   QPointF(cx + math.cos(a) * r1, cy + math.sin(a) * r1))

    # Animated bezel arc (clockwise)
    prog_draw = min(progress, 0.999)
    start_deg = -270.0
    span_deg = -360.0 * prog_draw

    arc_alpha = int(_clamp(170 * pulse_amt, 120, 235))
    arc_width = max(2.0, dial_w * 1.8) * pulse_amt
    p.setPen(QPen(QColor(ac[0], ac[1], ac[2], arc_alpha), arc_width, Qt.SolidLine, Qt.RoundCap))

    arc_rect = dial_rect.adjusted(mind * 0.006, mind * 0.006, -mind * 0.006, -mind * 0.006)
    p.drawArc(arc_rect, int(start_deg * 16), int(span_deg * 16))


def draw_falcon_loops(p: QPainter, cx: float, cy: float, mind: float,
                      t: float, vis: BootVisuals,
                      palette: str = "blue") -> None:
    """4 Lissajous loop rings with starlight color palette."""
    iris_R = mind * 0.42
    loop_scale = 1.08
    base_speed = 0.55
    circ_blend = 0.18
    circ_target = 0.205
    circ_min, circ_max = 0.175, 0.235

    N = vis.N

    for loop_idx, lp in enumerate(vis.loops):
        a_lp, b_lp, k1, k2, p1, p2, thick, rot_deg, hue_shift = lp
        pts: List[Tuple[int, int]] = []

        for i in range(N):
            u = (2 * math.pi) * (i / N)
            x = a_lp * math.sin(k1 * u + p1 + t * base_speed)
            y = b_lp * math.sin(k2 * u + p2 - t * base_speed * 0.9)

            r0 = math.sqrt(x * x + y * y) + 1e-6
            nx, ny = x / r0, y / r0
            x = (1 - circ_blend) * x + circ_blend * (nx * circ_target)
            y = (1 - circ_blend) * y + circ_blend * (ny * circ_target)

            r2 = math.sqrt(x * x + y * y) + 1e-6
            if r2 < circ_min or r2 > circ_max:
                rr = _clamp(r2, circ_min, circ_max)
                x *= rr / r2
                y *= rr / r2

            X = int(cx + x * iris_R * loop_scale)
            Y = int(cy + y * iris_R * loop_scale)
            pts.append((X, Y))

        glow_w = max(2.0, mind * 0.0065)
        core_w = max(1.5, mind * 0.0036)

        # Glow pass
        _draw_loop_segments(p, loop_idx, pts, t, 105, glow_w, hue_shift, palette)
        # Core pass
        _draw_loop_segments(p, loop_idx, pts, t, 235, core_w, hue_shift, palette)


def _draw_loop_segments(p: QPainter, loop_idx: int,
                        pts: List[Tuple[int, int]], t: float,
                        alpha: int, width: float, hue_shift: float,
                        palette: str = "blue") -> None:
    """Draw a single loop as colored line segments."""
    N = len(pts)
    for i in range(N):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % N]
        seg_u = i / max(1, (N - 1))
        col = _ring_color(loop_idx, seg_u, t, alpha, hue_shift, palette)
        p.setPen(QPen(col, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawLine(x0, y0, x1, y1)


def draw_falcon_text(p: QPainter, W: int, H: int, mind: float,
                     phase_text: str, pct: int,
                     palette: str = "blue") -> None:
    """'A U R A V I S I O N' title + phase text + percentage."""
    tc = _BOOT_ACCENTS.get(palette, _BOOT_ACCENTS["blue"])["text"]

    def _move_toward_center_y(y_frac: float, amount: float = 0.30) -> float:
        return 0.5 + (y_frac - 0.5) * (1.0 - amount)

    title_y = _move_toward_center_y(0.82, 0.30)
    phase_y = _move_toward_center_y(0.92, 0.30)

    # Title
    title_font = QFont("Helvetica Neue", max(18, int(mind * 0.030)))
    title_font.setWeight(QFont.Medium)
    title_font.setLetterSpacing(QFont.AbsoluteSpacing, 1.8)
    p.setFont(title_font)
    p.setPen(QPen(QColor(tc[0], tc[1], tc[2], 140), 1))
    p.drawText(QRectF(W * 0.10, H * title_y, W * 0.80, H * 0.07),
               Qt.AlignHCenter | Qt.AlignVCenter,
               "A U R A V I S I O N")

    # Phase + percentage
    phase_font = QFont("Helvetica Neue", max(16, int(mind * 0.0255)))
    phase_font.setWeight(QFont.Normal)
    p.setFont(phase_font)
    p.setPen(QPen(QColor(tc[0], tc[1], tc[2], 190), 1))
    display_text = f"{phase_text}  \u00b7  {pct}%" if phase_text else f"{pct}%"
    p.drawText(QRectF(W * 0.10, H * phase_y, W * 0.80, H * 0.07),
               Qt.AlignHCenter | Qt.AlignVCenter,
               display_text)


def draw_falcon_fade(p: QPainter, W: int, H: int, alpha: int) -> None:
    """Black overlay for fade-to-black transition."""
    if alpha <= 0:
        return
    p.save()
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(0, 0, 0, min(255, alpha)))
    p.drawRect(0, 0, W, H)
    p.restore()


# ---------------------------------------------------------------------------
# Composite: draw all falcon layers in order
# ---------------------------------------------------------------------------

def paint_boot_frame(p: QPainter, W: int, H: int, t: float,
                     vis: BootVisuals,
                     palette: str = "blue") -> None:
    """Draw a complete falcon boot frame.

    Called from AuraWindow._paint_boot(). The QPainter is already set up
    with the display rotation.
    """
    cx, cy = W * 0.5, H * 0.5
    mind = min(W, H)
    prog = _clamp(vis.progress, 0.0, 1.0)
    pct = int(round(prog * 100))

    # Completion pulse
    done = prog >= 0.999
    if done:
        done_t = t - vis.t0
        pulse = 0.5 + 0.5 * math.sin(done_t * 2.0 * math.pi * 0.9)
        pulse_amt = 0.75 + 0.55 * pulse
    else:
        pulse_amt = 1.0

    # Clear — use scheme background
    bg = _BOOT_ACCENTS.get(palette, _BOOT_ACCENTS["blue"])["bg"]
    p.fillRect(0, 0, W, H, QColor(bg[0], bg[1], bg[2]))

    # Stars
    draw_falcon_stars(p, W, H, t, vis.stars, palette)

    # Vignette
    draw_falcon_vignette(p, cx, cy, mind, W, H)

    # Progress dial
    draw_falcon_dial(p, cx, cy, mind, prog, vis.phase_bounds, pulse_amt, palette)

    # Loops
    draw_falcon_loops(p, cx, cy, mind, t, vis, palette)

    # Text
    draw_falcon_text(p, W, H, mind, vis.phase_text, pct, palette)

    # Fade overlay
    draw_falcon_fade(p, W, H, vis.fade_alpha)

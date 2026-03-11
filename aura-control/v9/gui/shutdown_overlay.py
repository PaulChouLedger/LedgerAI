"""
gui.shutdown_overlay -- Patek Philippe-grade power-down countdown.

A luxury timepiece power-reserve complication: the hand sweeps across a
beveled arc from FULL to EMPTY.  Triple bezel rings, engine-turned
guilloché, applied indices with polished facets, and a Breguet-style
blued-steel hand.  10 discrete second ticks — not continuous — each
tick snaps with authority.

Bus events:
    listens:  "shutdown.begin"   — starts the countdown
              "shutdown.abort"   — cancels (also triggered by tap)
    emits:    "shutdown.execute" — fires when countdown hits 0
"""

from __future__ import annotations

import math
import time

from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QPainter, QPen,
    QRadialGradient, QConicalGradient, QLinearGradient,
)

from core.bus import bus

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COUNTDOWN_SECONDS = 10.0

# Arc geometry (in Qt degrees: 0=3-o'clock, counter-clockwise positive)
# Power reserve runs from 7-o'clock to 5-o'clock (bottom arc)
_ARC_CENTER_DEG = 90.0     # 6-o'clock (bottom)
_ARC_HALF_SPAN  = 120.0    # 120° each side = 240° total arc
_ARC_START = _ARC_CENTER_DEG + _ARC_HALF_SPAN   # 210° (7-o'clock)
_ARC_END   = _ARC_CENTER_DEG - _ARC_HALF_SPAN   # -30° (5-o'clock, wraps)


def _frac_to_angle(frac: float) -> float:
    """Map fraction 0..1 to angle in radians (1.0=full=left, 0.0=empty=right)."""
    deg = _ARC_START - frac * (_ARC_HALF_SPAN * 2)
    return math.radians(deg)


class ShutdownOverlay:
    """Manages state + rendering for the shutdown countdown."""

    def __init__(self) -> None:
        self.active = False
        self._start_ts: float = 0.0
        self._trans: float = 0.0
        self._aborted = False
        self._last_tick_sec = -1  # for discrete ticking

        bus.on("shutdown.begin", self._on_begin)
        bus.on("shutdown.abort", self._on_abort)

    def _on_begin(self, **_kw):
        if self.active:
            return
        self.active = True
        self._start_ts = time.time()
        self._trans = 0.0
        self._aborted = False
        self._last_tick_sec = int(COUNTDOWN_SECONDS)

    def _on_abort(self, **_kw):
        self._aborted = True

    def tick(self, dt: float) -> bool:
        if not self.active and self._trans <= 0.0:
            return False

        if self.active and not self._aborted:
            self._trans = min(1.0, self._trans + dt * 3.0)
            elapsed = time.time() - self._start_ts
            if elapsed >= COUNTDOWN_SECONDS:
                self.active = False
                bus.emit("shutdown.execute")
                return True
        elif self._aborted:
            self._trans = max(0.0, self._trans - dt * 4.0)
            if self._trans <= 0.0:
                self.active = False
                self._aborted = False
                return False

        return self._trans > 0.001

    def handle_tap(self) -> bool:
        if self.active and not self._aborted:
            bus.emit("shutdown.abort")
            return True
        return False

    def remaining(self) -> float:
        if not self.active:
            return 0.0
        return max(0.0, COUNTDOWN_SECONDS - (time.time() - self._start_ts))

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(self, p: QPainter, cx: float, cy: float, mind: float, t: float):
        tr = self._trans
        if tr < 0.001:
            return

        remaining = self.remaining()
        frac = remaining / COUNTDOWN_SECONDS   # 1→0
        # Snap to discrete seconds for the hand
        secs_left = int(math.ceil(remaining))
        discrete_frac = secs_left / COUNTDOWN_SECONDS

        R = mind * 0.40   # main arc radius

        p.save()

        # ================================================================
        # BACKGROUND — deep obsidian with warm vignette
        # ================================================================
        bg = QRadialGradient(cx, cy, mind * 0.52)
        bg.setColorAt(0.0, QColor(12, 8, 6, int(240 * tr)))
        bg.setColorAt(0.4, QColor(6, 4, 3, int(248 * tr)))
        bg.setColorAt(1.0, QColor(0, 0, 0, int(255 * tr)))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(bg))
        p.drawEllipse(QPointF(cx, cy), mind * 0.50, mind * 0.50)

        # ================================================================
        # TRIPLE BEZEL — polished / brushed / polished
        # ================================================================
        for i, (r_frac, width_frac, alpha) in enumerate([
            (1.06, 0.008, 70),   # outer polished
            (1.02, 0.012, 35),   # middle brushed
            (0.98, 0.006, 55),   # inner polished
        ]):
            pen = QPen(QColor(200, 175, 130, int(alpha * tr)))
            pen.setWidthF(max(1.0, mind * width_frac))
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            rr = R * r_frac
            p.drawEllipse(QPointF(cx, cy), rr, rr)

        # ================================================================
        # ENGINE-TURNED GUILLOCHÉ — concentric dotted arcs
        # ================================================================
        for ring_i in range(8):
            ring_r = R * (0.25 + ring_i * 0.09)
            n_dots = 48 + ring_i * 12
            dot_alpha = int((8 + ring_i * 2) * tr)
            p.setPen(Qt.NoPen)
            for d in range(n_dots):
                a = 2 * math.pi * d / n_dots + t * 0.008 * (1 if ring_i % 2 == 0 else -1)
                dx = cx + ring_r * math.cos(a)
                dy = cy + ring_r * math.sin(a)
                dot_r = max(0.5, mind * 0.0015)
                p.setBrush(QColor(180, 155, 110, dot_alpha))
                p.drawEllipse(QPointF(dx, dy), dot_r, dot_r)

        # ================================================================
        # CHAPTER RING — 60 fine tick marks
        # ================================================================
        for i in range(60):
            a = math.radians(i * 6)
            major = (i % 5 == 0)
            inner = R * (0.88 if major else 0.92)
            outer = R * 0.96
            w = mind * (0.004 if major else 0.002)
            alpha = int((50 if major else 25) * tr)
            p.setPen(QPen(QColor(200, 175, 130, alpha), w, Qt.SolidLine, Qt.RoundCap))
            x0 = cx + inner * math.cos(a)
            y0 = cy - inner * math.sin(a)
            x1 = cx + outer * math.cos(a)
            y1 = cy - outer * math.sin(a)
            p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        # ================================================================
        # POWER RESERVE ARC — track (dark) + lit portion
        # ================================================================
        arc_r = R * 0.72
        arc_w = max(6.0, mind * 0.032)

        # Dark track
        track_pen = QPen(QColor(40, 25, 15, int(140 * tr)))
        track_pen.setWidthF(arc_w)
        track_pen.setCapStyle(Qt.FlatCap)
        p.setPen(track_pen)
        p.setBrush(Qt.NoBrush)
        arc_rect = QRectF(cx - arc_r, cy - arc_r, 2 * arc_r, 2 * arc_r)
        # Qt drawArc uses 1/16th degree units, and angles go CCW from 3-o'clock
        # We want our arc from 7-o'clock (210°) sweeping CW to 5-o'clock (-30°/330°)
        qt_start = int(_ARC_START * 16)
        qt_span = int(-_ARC_HALF_SPAN * 2 * 16)
        p.drawArc(arc_rect, qt_start, qt_span)

        # Lit portion (color shifts green→amber→red)
        if discrete_frac > 0.005:
            if discrete_frac > 0.6:
                lit_col = QColor(60, 180, 80, int(210 * tr))
            elif discrete_frac > 0.3:
                lit_col = QColor(220, 170, 40, int(220 * tr))
            else:
                lit_col = QColor(210, 50, 30, int(230 * tr))
            lit_pen = QPen(lit_col)
            lit_pen.setWidthF(arc_w)
            lit_pen.setCapStyle(Qt.FlatCap)
            p.setPen(lit_pen)
            lit_span = int(-discrete_frac * _ARC_HALF_SPAN * 2 * 16)
            p.drawArc(arc_rect, qt_start, lit_span)

        # ================================================================
        # APPLIED INDICES — 11 polished markers along the arc (0–10)
        # ================================================================
        for i in range(11):
            seg = i / 10.0
            a = _frac_to_angle(seg)
            # Outer marker (polished bevel effect)
            mx = cx + arc_r * 1.22 * math.cos(a)
            my = cy - arc_r * 1.22 * math.sin(a)

            is_five = (i % 5 == 0)
            marker_h = mind * (0.030 if is_five else 0.018)
            marker_w = mind * (0.010 if is_five else 0.006)

            p.save()
            p.translate(mx, my)
            p.rotate(-math.degrees(a) + 90)

            # Shadow
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, int(60 * tr)))
            p.drawRoundedRect(QRectF(-marker_w/2 + 1, -marker_h/2 + 1,
                                      marker_w, marker_h), 1, 1)
            # Polished face
            face_grad = QLinearGradient(0, -marker_h/2, 0, marker_h/2)
            face_grad.setColorAt(0.0, QColor(240, 225, 190, int(200 * tr)))
            face_grad.setColorAt(0.5, QColor(180, 155, 110, int(160 * tr)))
            face_grad.setColorAt(1.0, QColor(220, 200, 160, int(190 * tr)))
            p.setBrush(QBrush(face_grad))
            p.drawRoundedRect(QRectF(-marker_w/2, -marker_h/2,
                                      marker_w, marker_h), 1, 1)
            p.restore()

            # Numeral at 0, 5, 10
            if is_five:
                num = 10 - i
                nf = QFont("DejaVu Serif", max(7, int(mind * 0.022)))
                nf.setBold(True)
                p.setFont(nf)
                nx = cx + arc_r * 0.56 * math.cos(a)
                ny = cy - arc_r * 0.56 * math.sin(a)
                p.setPen(QColor(200, 175, 130, int(150 * tr)))
                p.drawText(QRectF(nx - 16, ny - 10, 32, 20),
                           Qt.AlignCenter, str(num))

        # ================================================================
        # BREGUET HAND — blued steel with counterweight
        # ================================================================
        hand_a = _frac_to_angle(discrete_frac)
        hx = cx + arc_r * 1.05 * math.cos(hand_a)
        hy = cy - arc_r * 1.05 * math.sin(hand_a)
        # Counterweight (opposite side)
        cwx = cx - arc_r * 0.20 * math.cos(hand_a)
        cwy = cy + arc_r * 0.20 * math.sin(hand_a)

        # Hand shadow
        p.setPen(QPen(QColor(0, 0, 0, int(100 * tr)), max(3.0, mind * 0.007),
                       Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(cwx + 1.5, cwy + 1.5), QPointF(hx + 1.5, hy + 1.5))

        # Blued steel hand
        hand_col = QColor(60, 80, 140, int(230 * tr))
        p.setPen(QPen(hand_col, max(2.0, mind * 0.005), Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(cwx, cwy), QPointF(hx, hy))

        # Breguet moon tip (open circle near tip)
        moon_r = max(3.0, mind * 0.012)
        moon_dist = arc_r * 0.85
        moon_x = cx + moon_dist * math.cos(hand_a)
        moon_y = cy - moon_dist * math.sin(hand_a)
        p.setPen(QPen(hand_col, max(1.5, mind * 0.004)))
        p.setBrush(QColor(12, 8, 6, int(240 * tr)))  # hollow
        p.drawEllipse(QPointF(moon_x, moon_y), moon_r, moon_r)

        # Jewel pivot — ruby with chamfer
        jr = max(5.0, mind * 0.016)
        # Shadow
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, int(80 * tr)))
        p.drawEllipse(QPointF(cx + 1.5, cy + 1.5), jr, jr)
        # Chamfered ring
        pivot_grad = QRadialGradient(cx, cy, jr)
        pivot_grad.setColorAt(0.0, QColor(240, 225, 195, int(200 * tr)))
        pivot_grad.setColorAt(0.6, QColor(180, 155, 110, int(170 * tr)))
        pivot_grad.setColorAt(1.0, QColor(100, 80, 50, int(140 * tr)))
        p.setBrush(QBrush(pivot_grad))
        p.drawEllipse(QPointF(cx, cy), jr, jr)
        # Ruby center
        ruby_r = jr * 0.45
        ruby_grad = QRadialGradient(cx - ruby_r * 0.3, cy - ruby_r * 0.3, ruby_r)
        ruby_grad.setColorAt(0.0, QColor(200, 40, 40, int(220 * tr)))
        ruby_grad.setColorAt(1.0, QColor(120, 15, 15, int(200 * tr)))
        p.setBrush(QBrush(ruby_grad))
        p.drawEllipse(QPointF(cx, cy), ruby_r, ruby_r)

        # ================================================================
        # "POWER RESERVE" engraved text (top)
        # ================================================================
        f_title = QFont("DejaVu Serif", max(8, int(mind * 0.024)))
        f_title.setBold(False)
        f_title.setLetterSpacing(QFont.AbsoluteSpacing, max(2.0, mind * 0.006))
        p.setFont(f_title)
        p.setPen(QColor(180, 155, 110, int(130 * tr)))
        p.drawText(QRectF(cx - 100, cy - R * 0.48, 200, 22),
                   Qt.AlignCenter, "POWER RESERVE")

        # ================================================================
        # Large countdown number (center)
        # ================================================================
        f_num = QFont("DejaVu Serif", max(32, int(mind * 0.12)))
        f_num.setBold(True)
        p.setFont(f_num)

        # Pulse urgency in final 3 seconds
        if secs_left <= 3 and secs_left > 0:
            pulse = (1.0 + math.sin(t * 10.0)) * 0.5
            r_c = int(210 + 45 * pulse)
            num_col = QColor(r_c, 50, 30, int(240 * tr))
        elif secs_left == 0:
            num_col = QColor(210, 30, 20, int(240 * tr))
        else:
            num_col = QColor(220, 200, 165, int(220 * tr))

        # Shadow
        p.setPen(QColor(0, 0, 0, int(150 * tr)))
        p.drawText(QRectF(cx - 60 + 2, cy - 35 + 2, 120, 60),
                   Qt.AlignCenter, str(secs_left))
        p.setPen(num_col)
        p.drawText(QRectF(cx - 60, cy - 35, 120, 60),
                   Qt.AlignCenter, str(secs_left))

        # ================================================================
        # "TAP TO ABORT" — gentle breathing glow
        # ================================================================
        breath = (1.0 + math.sin(t * 2.5)) * 0.5
        abort_alpha = int((80 + 50 * breath) * tr)
        f_abort = QFont("DejaVu Sans", max(7, int(mind * 0.020)))
        f_abort.setLetterSpacing(QFont.AbsoluteSpacing, max(1.5, mind * 0.004))
        p.setFont(f_abort)
        p.setPen(QColor(180, 160, 130, abort_alpha))
        p.drawText(QRectF(cx - 80, cy + R * 0.35, 160, 22),
                   Qt.AlignCenter, "TAP TO ABORT")

        p.restore()

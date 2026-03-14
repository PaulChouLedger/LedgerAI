"""
gui.shutdown_overlay -- Patek Philippe power-reserve shutdown countdown.

Inspired by the Patek Philippe 5078 power reserve indicator: a refined
arc with applied gold indices, Breguet blued-steel hand, and a clean
engraved dial.  10-second countdown with discrete second ticks.

Bus events:
    listens:  "shutdown.begin"   — starts the countdown
              "shutdown.abort"   — cancels (also triggered by tap)
    emits:    "shutdown.tick"    — each second (secs_left=N)
              "shutdown.execute" — fires when countdown hits 0
"""

from __future__ import annotations

import math
import time

from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QPainter, QPen,
    QRadialGradient, QLinearGradient, QPainterPath,
)

from core.bus import bus

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COUNTDOWN_SECONDS = 10.0

# Arc geometry — power reserve runs across the upper dial
# Using radians directly for the hand/markers; Qt arcs in 1/16° units.
_ARC_START_DEG = 210.0   # 7-o'clock
_ARC_END_DEG   = 330.0   # 5-o'clock (clockwise sweep through bottom)
_ARC_SWEEP_DEG = 240.0   # total sweep


def _frac_to_rad(frac: float) -> float:
    """Map fraction 0..1 to angle in radians.  1.0 = full (left), 0.0 = empty (right)."""
    deg = _ARC_START_DEG - frac * _ARC_SWEEP_DEG
    return math.radians(deg)


class ShutdownOverlay:
    """Manages state + rendering for the shutdown countdown."""

    def __init__(self) -> None:
        self.active = False
        self._start_ts: float = 0.0
        self._trans: float = 0.0
        self._aborted = False
        self._last_tick_sec = -1

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
        # Emit initial tick so voice can start
        bus.emit("shutdown.tick", secs_left=int(COUNTDOWN_SECONDS))

    def _on_abort(self, **_kw):
        self._aborted = True

    def tick(self, dt: float) -> bool:
        if not self.active and self._trans <= 0.0:
            return False

        if self.active and not self._aborted:
            self._trans = min(1.0, self._trans + dt * 3.0)
            elapsed = time.time() - self._start_ts
            secs_left = max(0, int(math.ceil(COUNTDOWN_SECONDS - elapsed)))

            # Emit tick on each new second
            if secs_left != self._last_tick_sec and secs_left >= 0:
                self._last_tick_sec = secs_left
                bus.emit("shutdown.tick", secs_left=secs_left)

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
        secs_left = int(math.ceil(remaining))
        discrete_frac = secs_left / COUNTDOWN_SECONDS

        R = mind * 0.40

        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)

        # ================================================================
        # BACKGROUND — deep obsidian with warm vignette
        # ================================================================
        bg = QRadialGradient(cx, cy, mind * 0.52)
        bg.setColorAt(0.0, QColor(14, 10, 8, int(235 * tr)))
        bg.setColorAt(0.5, QColor(8, 5, 4, int(245 * tr)))
        bg.setColorAt(1.0, QColor(0, 0, 0, int(255 * tr)))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(bg))
        p.drawEllipse(QPointF(cx, cy), mind * 0.50, mind * 0.50)

        # ================================================================
        # BEZEL — single polished ring with subtle chamfer
        # ================================================================
        for r_frac, w_frac, alpha in [
            (1.04, 0.005, 40),
            (1.00, 0.010, 65),
            (0.97, 0.004, 35),
        ]:
            pen = QPen(QColor(195, 170, 125, int(alpha * tr)))
            pen.setWidthF(max(1.0, mind * w_frac))
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            rr = R * r_frac
            p.drawEllipse(QPointF(cx, cy), rr, rr)

        # ================================================================
        # MINUTE TRACK — subtle chapter ring
        # ================================================================
        for i in range(60):
            a = math.radians(i * 6)
            major = (i % 5 == 0)
            inner = R * (0.90 if major else 0.93)
            outer = R * 0.96
            w = mind * (0.003 if major else 0.0015)
            alpha = int((40 if major else 18) * tr)
            p.setPen(QPen(QColor(180, 155, 115, alpha), w, Qt.SolidLine, Qt.RoundCap))
            x0 = cx + inner * math.cos(a)
            y0 = cy - inner * math.sin(a)
            x1 = cx + outer * math.cos(a)
            y1 = cy - outer * math.sin(a)
            p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        # ================================================================
        # POWER RESERVE ARC — track + lit portion
        # ================================================================
        arc_r = R * 0.70
        arc_w = max(5.0, mind * 0.026)

        # Qt arc parameters (1/16° units, CCW from 3-o'clock)
        qt_start = int(_ARC_START_DEG * 16)
        qt_span = int(-_ARC_SWEEP_DEG * 16)

        # Dark track
        track_pen = QPen(QColor(35, 22, 12, int(120 * tr)))
        track_pen.setWidthF(arc_w)
        track_pen.setCapStyle(Qt.RoundCap)
        p.setPen(track_pen)
        p.setBrush(Qt.NoBrush)
        arc_rect = QRectF(cx - arc_r, cy - arc_r, 2 * arc_r, 2 * arc_r)
        p.drawArc(arc_rect, qt_start, qt_span)

        # Lit portion — colour shifts: green → amber → crimson
        if discrete_frac > 0.005:
            if discrete_frac > 0.5:
                lit_col = QColor(55, 170, 75, int(200 * tr))
            elif discrete_frac > 0.25:
                lit_col = QColor(210, 165, 35, int(210 * tr))
            else:
                lit_col = QColor(185, 40, 25, int(220 * tr))
            lit_pen = QPen(lit_col)
            lit_pen.setWidthF(arc_w)
            lit_pen.setCapStyle(Qt.RoundCap)
            p.setPen(lit_pen)
            lit_span = int(-discrete_frac * _ARC_SWEEP_DEG * 16)
            p.drawArc(arc_rect, qt_start, lit_span)

        # ================================================================
        # APPLIED INDICES — 11 polished gold markers (0–10)
        # ================================================================
        for i in range(11):
            seg = i / 10.0
            a = _frac_to_rad(seg)
            mx = cx + arc_r * 1.20 * math.cos(a)
            my = cy - arc_r * 1.20 * math.sin(a)

            is_major = (i % 5 == 0)
            marker_h = mind * (0.026 if is_major else 0.015)
            marker_w = mind * (0.008 if is_major else 0.005)

            p.save()
            p.translate(mx, my)
            p.rotate(-math.degrees(a) + 90)
            p.setPen(Qt.NoPen)

            # Polished face
            face_grad = QLinearGradient(0, -marker_h / 2, 0, marker_h / 2)
            face_grad.setColorAt(0.0, QColor(230, 215, 180, int(190 * tr)))
            face_grad.setColorAt(0.5, QColor(170, 145, 105, int(150 * tr)))
            face_grad.setColorAt(1.0, QColor(210, 190, 150, int(180 * tr)))
            p.setBrush(QBrush(face_grad))
            p.drawRoundedRect(
                QRectF(-marker_w / 2, -marker_h / 2, marker_w, marker_h), 1, 1)
            p.restore()

            # Numeral at 0, 5, 10
            if is_major:
                num = 10 - i
                nf = QFont("Helvetica Neue", max(7, int(mind * 0.020)))
                nf.setWeight(QFont.Light)
                p.setFont(nf)
                nx = cx + arc_r * 0.52 * math.cos(a)
                ny = cy - arc_r * 0.52 * math.sin(a)
                p.setPen(QColor(190, 165, 125, int(140 * tr)))
                p.drawText(QRectF(nx - 14, ny - 10, 28, 20),
                           Qt.AlignCenter, str(num))

        # ================================================================
        # BREGUET HAND — blued steel with counterweight + moon tip
        # ================================================================
        hand_a = _frac_to_rad(discrete_frac)
        hx = cx + arc_r * 1.02 * math.cos(hand_a)
        hy = cy - arc_r * 1.02 * math.sin(hand_a)
        cwx = cx - arc_r * 0.18 * math.cos(hand_a)
        cwy = cy + arc_r * 0.18 * math.sin(hand_a)

        # Shadow
        p.setPen(QPen(QColor(0, 0, 0, int(80 * tr)),
                       max(2.5, mind * 0.006), Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(cwx + 1.2, cwy + 1.2),
                   QPointF(hx + 1.2, hy + 1.2))

        # Blued steel
        hand_col = QColor(55, 75, 135, int(225 * tr))
        p.setPen(QPen(hand_col, max(1.8, mind * 0.004),
                       Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(cwx, cwy), QPointF(hx, hy))

        # Breguet moon tip
        moon_r = max(2.5, mind * 0.010)
        moon_d = arc_r * 0.82
        moon_x = cx + moon_d * math.cos(hand_a)
        moon_y = cy - moon_d * math.sin(hand_a)
        p.setPen(QPen(hand_col, max(1.2, mind * 0.003)))
        p.setBrush(QColor(14, 10, 8, int(235 * tr)))
        p.drawEllipse(QPointF(moon_x, moon_y), moon_r, moon_r)

        # Jewel pivot
        jr = max(4.0, mind * 0.013)
        pivot_grad = QRadialGradient(cx, cy, jr)
        pivot_grad.setColorAt(0.0, QColor(230, 215, 185, int(190 * tr)))
        pivot_grad.setColorAt(0.7, QColor(170, 145, 105, int(160 * tr)))
        pivot_grad.setColorAt(1.0, QColor(90, 70, 45, int(130 * tr)))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(pivot_grad))
        p.drawEllipse(QPointF(cx, cy), jr, jr)
        # Ruby center
        ruby_r = jr * 0.42
        ruby_grad = QRadialGradient(cx - ruby_r * 0.3, cy - ruby_r * 0.3, ruby_r)
        ruby_grad.setColorAt(0.0, QColor(190, 35, 35, int(210 * tr)))
        ruby_grad.setColorAt(1.0, QColor(110, 12, 12, int(190 * tr)))
        p.setBrush(QBrush(ruby_grad))
        p.drawEllipse(QPointF(cx, cy), ruby_r, ruby_r)

        # ================================================================
        # "POWER RESERVE" — engraved text above arc
        # ================================================================
        f_title = QFont("Helvetica Neue", max(7, int(mind * 0.019)))
        f_title.setWeight(QFont.Light)
        f_title.setLetterSpacing(QFont.AbsoluteSpacing, max(2.0, mind * 0.008))
        p.setFont(f_title)
        p.setPen(QColor(170, 148, 108, int(120 * tr)))
        p.drawText(QRectF(cx - 90, cy - R * 0.50, 180, 20),
                   Qt.AlignCenter, "POWER RESERVE")

        # ================================================================
        # LARGE COUNTDOWN — center numeral
        # ================================================================
        f_num = QFont("Helvetica Neue", max(36, int(mind * 0.14)))
        f_num.setWeight(QFont.Thin)
        p.setFont(f_num)

        # Urgency colour: calm → amber → red with pulse
        if secs_left <= 3 and secs_left > 0:
            pulse = (1.0 + math.sin(t * 8.0)) * 0.5
            r_c = int(190 + 50 * pulse)
            num_col = QColor(r_c, 45, 25, int(235 * tr))
        elif secs_left == 0:
            num_col = QColor(200, 28, 18, int(240 * tr))
        else:
            num_col = QColor(215, 195, 160, int(215 * tr))

        # Shadow
        p.setPen(QColor(0, 0, 0, int(120 * tr)))
        p.drawText(QRectF(cx - 50 + 1.5, cy - 30 + 1.5, 100, 50),
                   Qt.AlignCenter, str(secs_left))
        p.setPen(num_col)
        p.drawText(QRectF(cx - 50, cy - 30, 100, 50),
                   Qt.AlignCenter, str(secs_left))

        # ================================================================
        # "SHUTTING DOWN" — subtle label below numeral
        # ================================================================
        f_sub = QFont("Helvetica Neue", max(6, int(mind * 0.016)))
        f_sub.setWeight(QFont.Light)
        f_sub.setLetterSpacing(QFont.AbsoluteSpacing, max(1.5, mind * 0.005))
        p.setFont(f_sub)
        p.setPen(QColor(160, 140, 105, int(100 * tr)))
        p.drawText(QRectF(cx - 80, cy + mind * 0.06, 160, 18),
                   Qt.AlignCenter, "SHUTTING DOWN")

        # ================================================================
        # "TAP TO CANCEL" — breathing glow at bottom
        # ================================================================
        breath = (1.0 + math.sin(t * 2.5)) * 0.5
        abort_alpha = int((70 + 50 * breath) * tr)
        f_abort = QFont("Helvetica Neue", max(7, int(mind * 0.018)))
        f_abort.setLetterSpacing(QFont.AbsoluteSpacing, max(1.5, mind * 0.004))
        p.setFont(f_abort)
        p.setPen(QColor(175, 155, 120, abort_alpha))
        p.drawText(QRectF(cx - 80, cy + R * 0.38, 160, 20),
                   Qt.AlignCenter, "TAP TO CANCEL")

        p.restore()

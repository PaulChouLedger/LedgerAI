"""
gui.complications.mute -- Mute toggle complication.

Extracted from carbon_demo.py `_draw_comp_mute`.
Blued needle oscillates when live; locks to 1.0 when muted.
Aperture window shows MUTED / LIVE; sound waves animate when live.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt5.QtGui import QPainter

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QBrush, QColor, QFont, QPen

from gui.complications.base import BaseComplication
from gui.renderer import clamp


class MuteComplication(BaseComplication):
    name = "Mute"
    label = "Mute"
    category = "System"
    has_overlay = False

    def __init__(self, bus):
        super().__init__(bus)
        self.muted = False
        # Live mic RMS, smoothed for needle motion. Updated by the
        # bus.on("mic.level") subscription below; falls to zero when
        # the listener is silent so the needle decays naturally.
        self._mic_rms_smoothed = 0.0
        # Empirical RMS range — anything below ~0.005 reads as silence,
        # ~0.10 is loud speech right next to the mic. Map this to the
        # needle's 0..1 sweep.
        self._rms_floor = 0.005
        self._rms_ceil  = 0.10
        bus.on("mic.level", self._on_mic_level)

    def _on_mic_level(self, rms: float, **_kw) -> None:
        """Subscribe target — listener emits per-chunk RMS while active.

        Smooth aggressively enough that the needle doesn't twitch on
        single noisy frames, but stays responsive within ~150 ms.
        """
        denom = max(self._rms_ceil - self._rms_floor, 1e-6)
        v = (float(rms) - self._rms_floor) / denom
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        # Asymmetric smoothing — fast attack, slower release. Feels
        # like a real VU meter.
        a = 0.55 if v > self._mic_rms_smoothed else 0.20
        self._mic_rms_smoothed = (1 - a) * self._mic_rms_smoothed + a * v

    # ------------------------------------------------------------------
    def draw_content(self, p: "QPainter", inner: float, t: float, accent: QColor) -> None:
        muted = self.muted

        # --- Curved "MUTE" along top arc — always red, signals the
        # function regardless of current state. ---
        text_col = QColor(220, 75, 65, 240)
        _draw_curved_text(p, "MUTE", inner * 0.58, top=True,
                          color=text_col, inner=inner)

        # Aperture window (date-like)
        win = QRectF(-inner * 0.42, inner * 0.18, inner * 0.84, inner * 0.26)
        p.setPen(QPen(QColor(0, 0, 0, 140), max(1.0, inner * 0.020)))
        p.setBrush(QBrush(QColor(10, 12, 18, 180)))
        p.drawRoundedRect(win, inner * 0.06, inner * 0.06)

        txt = "MUTED" if muted else "LIVE"
        f = QFont("Helvetica", max(8, int(inner * 0.21)))
        f.setBold(True)
        p.setFont(f)
        off = max(1.0, inner * 0.018)
        p.setPen(QColor(0, 0, 0, 175))
        p.drawText(win.translated(off, off), Qt.AlignCenter, txt)
        p.setPen(QColor(0, 0, 0, 145))
        p.drawText(win.translated(-off, off), Qt.AlignCenter, txt)
        if muted:
            p.setPen(QColor(220, 75, 65, 245))    # red when muted
        else:
            p.setPen(QColor(90, 220, 140, 245))    # green when live
        p.drawText(win, Qt.AlignCenter, txt)

        # Needle — actually represents live mic RMS now (was a fake
        # sin/cos oscillation). When muted the needle pegs at full
        # deflection because nothing's getting through.
        if muted:
            v = 1.0
        else:
            v = clamp(self._mic_rms_smoothed, 0.0, 1.0)
        ang = math.radians(210 + 120 * v - 90.0)
        hx = (inner * 0.60) * math.cos(ang)
        hy = (inner * 0.60) * math.sin(ang)

        # Shadow
        penS = QPen(QColor(0, 0, 0, 85))
        penS.setWidthF(max(1.4, inner * 0.040))
        penS.setCapStyle(Qt.RoundCap)
        p.setPen(penS)
        p.drawLine(QPointF(inner * 0.012, inner * 0.012),
                   QPointF(hx + inner * 0.012, hy + inner * 0.012))

        # Hand — green when live, red when muted
        if muted:
            col = QColor(220, 75, 65, 240)
        else:
            col = QColor(90, 200, 130, 210)
        penH = QPen(col)
        penH.setWidthF(max(1.2, inner * 0.035))
        penH.setCapStyle(Qt.RoundCap)
        p.setPen(penH)
        p.drawLine(QPointF(0, 0), QPointF(hx, hy))

        # Center jewel — state-colored
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 120)))
        p.drawEllipse(QPointF(0, 0), inner * 0.095, inner * 0.095)
        if muted:
            p.setBrush(QBrush(QColor(220, 75, 65, 210)))
        else:
            p.setBrush(QBrush(QColor(90, 200, 130, 180)))
        p.drawEllipse(QPointF(0, 0), inner * 0.060, inner * 0.060)

        # Sound waves when live
        if not muted:
            for i, rr in enumerate([0.26, 0.34, 0.42]):
                a = 0.45 + 0.35 * math.sin(t * 2.4 + i * 1.2)
                pen = QPen(QColor(220, 245, 255, int(40 + 60 * a)))
                pen.setWidthF(max(1.0, inner * 0.018))
                pen.setCapStyle(Qt.RoundCap)
                p.setPen(pen)
                rect = QRectF(-inner * rr, -inner * rr, 2 * inner * rr, 2 * inner * rr)
                p.drawArc(rect, int(300 * 16), int(100 * 16))

    # ------------------------------------------------------------------
    def on_tap(self):
        self.muted = not self.muted
        self.bus.emit("mute.toggled", muted=self.muted)
        return True


# ---------------------------------------------------------------------------
# Helper: draw text curved along a circular arc
# ---------------------------------------------------------------------------

def _draw_curved_text(p, text, radius, top, color, inner):
    font = QFont("Helvetica", max(7, int(inner * 0.16)))
    font.setBold(True)
    font.setLetterSpacing(QFont.PercentageSpacing, 145)
    p.setFont(font)

    fm = p.fontMetrics()
    char_widths = [fm.horizontalAdvance(ch) for ch in text]
    total_w = sum(char_widths)
    span_rad = total_w / max(1.0, radius)

    if top:
        start_ang = -math.pi / 2 - span_rad / 2
        direction = 1.0
    else:
        start_ang = math.pi / 2 + span_rad / 2
        direction = -1.0

    shd_off = max(0.8, inner * 0.014)
    current_ang = start_ang
    baseline_shift = fm.ascent() * 0.15

    for ch in text:
        cw = char_widths[0]
        char_widths = char_widths[1:]
        current_ang += direction * (cw * 0.5) / radius

        x = radius * math.cos(current_ang)
        y = radius * math.sin(current_ang)

        p.save()
        p.translate(x, y)
        rot = math.degrees(current_ang) + (90 if top else -90)
        p.rotate(rot)

        ty = -fm.ascent() + baseline_shift
        rect = QRectF(-cw, ty, cw * 2, fm.height())

        p.setPen(QColor(0, 0, 0, 130))
        p.drawText(rect.translated(0, shd_off), Qt.AlignCenter, ch)
        p.setPen(color)
        p.drawText(rect, Qt.AlignCenter, ch)
        p.restore()

        current_ang += direction * (cw * 0.5) / radius

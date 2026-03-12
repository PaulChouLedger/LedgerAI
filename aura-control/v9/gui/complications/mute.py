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

    # ------------------------------------------------------------------
    def draw_content(self, p: "QPainter", inner: float, t: float, accent: QColor) -> None:
        muted = self.muted

        # --- Curved "MUTE" along top arc — tinted by state ---
        if muted:
            text_col = QColor(225, 195, 195, 235)   # muted red tint
        else:
            text_col = QColor(195, 240, 215, 235)   # cool green tint
        _draw_curved_text(p, "MUTE", inner * 0.58, top=True,
                          color=text_col, inner=inner)

        # Aperture window (date-like)
        win = QRectF(-inner * 0.42, inner * 0.18, inner * 0.84, inner * 0.26)
        p.setPen(QPen(QColor(0, 0, 0, 140), max(1.0, inner * 0.020)))
        p.setBrush(QBrush(QColor(14, 8, 8, 180)))
        p.drawRoundedRect(win, inner * 0.06, inner * 0.06)

        txt = "MUTED" if muted else "LIVE"
        f = QFont("Helvetica", max(8, int(inner * 0.25)))
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

        # Needle (oscillates when live, locked at 1.0 when muted)
        v = 1.0 if muted else (0.40 + 0.25 * math.sin(t * 1.8) + 0.12 * math.sin(t * 3.1))
        v = clamp(v, 0.0, 1.0)
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
                pen = QPen(QColor(215, 220, 232, int(40 + 60 * a)))  # platinum sound waves
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

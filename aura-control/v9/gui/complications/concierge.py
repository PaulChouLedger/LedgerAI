"""
gui.complications.concierge -- Aura Concierge complication.

The general-purpose personal assistant hub: chat, schedule, call an Uber,
run errands, etc.  An umbrella of everyday services vs. the specialized
domain glyphs.

Patek Philippe-styled with "AURA" curved along the top arc and
"CONCIERGE" curved along the bottom arc.  Central icon is a classic
concierge bell / service key.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt5.QtGui import QPainter

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QBrush, QColor, QFont, QPen, QPainterPath

from gui.complications.base import BaseComplication


class ConciergeComplication(BaseComplication):
    name = "Aura Concierge"
    label = "Aura Concierge"
    category = "System"
    dockable = True
    has_overlay = True

    def __init__(self, bus):
        super().__init__(bus)

    # ------------------------------------------------------------------
    def draw_content(self, p: "QPainter", inner: float, t: float, accent: QColor) -> None:
        # --- Curved "AURA" along top arc ---
        _draw_curved_text(p, "AURA", inner * 0.58, top=True,
                          color=QColor(235, 225, 248, 235), inner=inner)

        # --- Curved "CONCIERGE" along bottom arc ---
        _draw_curved_text(p, "CONCIERGE", inner * 0.58, top=False,
                          color=QColor(235, 225, 248, 215), inner=inner)

        # --- Central concierge key icon ---
        _draw_concierge_icon(p, inner, t, accent)

    # ------------------------------------------------------------------
    def draw_overlay(self, p, cx, cy, mind, t, trans):
        # TODO: concierge service menu (chat, schedule, transport, etc.)
        pass


# ---------------------------------------------------------------------------
# Concierge icon: ornate skeleton key
# ---------------------------------------------------------------------------

def _draw_concierge_icon(p: "QPainter", inner: float, t: float, accent: QColor):
    """Draw a stylised skeleton key (classic concierge symbol)."""
    kr = inner * 0.38  # overall icon scale

    # Slow gentle rotation
    breathe = math.sin(t * 0.8) * 2.0
    p.save()
    p.rotate(breathe)

    # Violet-tinted key
    col = QColor(155, 120, 210, 200)
    hi_col = QColor(195, 170, 235, 160)

    pen = QPen(col)
    pen.setWidthF(max(1.4, kr * 0.09))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)

    # Key bow (ornate ring at top)
    bow_r = kr * 0.38
    bow_cy = -kr * 0.35

    # Shadow
    p.setPen(QPen(QColor(0, 0, 0, 70), pen.widthF() * 1.1))
    off = max(0.6, kr * 0.03)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(off, bow_cy + off), bow_r, bow_r)

    # Main bow ring
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(0, bow_cy), bow_r, bow_r)

    # Inner decorative ring
    inner_pen = QPen(QColor(col.red(), col.green(), col.blue(), 100))
    inner_pen.setWidthF(max(0.8, kr * 0.04))
    p.setPen(inner_pen)
    p.drawEllipse(QPointF(0, bow_cy), bow_r * 0.55, bow_r * 0.55)

    # Shaft (vertical line from bow to bit)
    shaft_top = bow_cy + bow_r
    shaft_bot = kr * 0.70
    p.setPen(pen)
    p.drawLine(QPointF(0, shaft_top), QPointF(0, shaft_bot))

    # Key bit teeth (2 horizontal notches at bottom)
    tooth_w = kr * 0.22
    for i, ty in enumerate([shaft_bot - kr * 0.12, shaft_bot]):
        p.drawLine(QPointF(0, ty), QPointF(tooth_w, ty))
        # Small vertical drop on each tooth
        drop = kr * 0.08
        p.drawLine(QPointF(tooth_w, ty), QPointF(tooth_w, ty - drop))

    # Highlight pass
    hi_pen = QPen(hi_col)
    hi_pen.setWidthF(max(0.6, kr * 0.03))
    hi_pen.setCapStyle(Qt.RoundCap)
    p.setPen(hi_pen)
    p.drawArc(QRectF(-bow_r, bow_cy - bow_r, bow_r * 2, bow_r * 2),
              45 * 16, 90 * 16)  # top-right quarter highlight

    p.restore()


# ---------------------------------------------------------------------------
# Helper: draw text curved along a circular arc
# ---------------------------------------------------------------------------

def _draw_curved_text(p: "QPainter", text: str, radius: float, top: bool,
                      color: QColor, inner: float):
    """Draw *text* along a circular arc.  top=True arcs upward, False downward."""
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

    for i, ch in enumerate(text):
        cw = char_widths[i]
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

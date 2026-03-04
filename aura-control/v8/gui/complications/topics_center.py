"""
gui.complications.topics_center -- Topics Center complication.

Patek Philippe-styled complication with "TOPICS" curved along the top arc
and "CENTER" curved along the bottom arc.  Central icon is a 4-point
compass rose representing the browsable domain catalogue.

Always visible on the perimeter ring via ``always_available = True``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt5.QtGui import QPainter

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QBrush, QColor, QFont, QPen

from gui.complications.base import BaseComplication


class TopicsCenterComplication(BaseComplication):
    name = "Topics Center"
    label = "Topics Center"
    category = "System"
    dockable = True
    always_available = True
    has_overlay = True

    def __init__(self, bus):
        super().__init__(bus)

    # ------------------------------------------------------------------
    def draw_content(self, p: "QPainter", inner: float, t: float, accent: QColor) -> None:
        # --- Curved "TOPICS" along top arc ---
        _draw_curved_text(p, "TOPICS", inner * 0.58, top=True,
                          color=QColor(248, 238, 218, 235), inner=inner)

        # --- Curved "CENTER" along bottom arc ---
        _draw_curved_text(p, "CENTER", inner * 0.58, top=False,
                          color=QColor(248, 238, 218, 215), inner=inner)

        # --- Central compass rose icon ---
        cr = inner * 0.30
        # Outer ring
        pen = QPen(QColor(accent.red(), accent.green(), accent.blue(), 120))
        pen.setWidthF(max(1.0, inner * 0.020))
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(0, 0), cr * 1.15, cr * 1.15)

        # 4 diamond points (N, E, S, W) — gold tones
        gold_point = QColor(208, 178, 112)
        for i in range(4):
            ang = -math.pi / 2 + i * math.pi / 2
            tip_r = cr * 1.05
            side_r = cr * 0.35
            tip_x = tip_r * math.cos(ang)
            tip_y = tip_r * math.sin(ang)

            perp = ang + math.pi / 2
            sx1 = side_r * math.cos(perp)
            sy1 = side_r * math.sin(perp)

            # Primary: gold for N/S, faded for E/W
            primary = (i % 2 == 0)
            a = 200 if primary else 110
            col = QColor(gold_point.red(), gold_point.green(), gold_point.blue(), a)

            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(col))
            from PyQt5.QtGui import QPainterPath, QPolygonF
            tri = QPainterPath()
            tri.moveTo(tip_x, tip_y)
            tri.lineTo(sx1, sy1)
            tri.lineTo(-sx1, -sy1)
            tri.closeSubpath()
            p.drawPath(tri)

        # Center jewel — amber
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 140)))
        p.drawEllipse(QPointF(0, 0), cr * 0.22, cr * 0.22)
        pulse = 0.5 + 0.5 * math.sin(t * 1.8)
        p.setBrush(QBrush(QColor(218, 175, 85, int(140 + 80 * pulse))))
        p.drawEllipse(QPointF(0, 0), cr * 0.16, cr * 0.16)

    # ------------------------------------------------------------------
    def draw_overlay(self, p, cx, cy, mind, t, trans):
        # TODO: scrollable grid of all available topics
        pass

    def on_tap(self):
        self.open_overlay()
        return True


# ---------------------------------------------------------------------------
# Helper: draw text curved along a circular arc
# ---------------------------------------------------------------------------

def _draw_curved_text(p: "QPainter", text: str, radius: float, top: bool,
                      color: QColor, inner: float):
    """Draw *text* along a circular arc.  top=True arcs upward, False arcs downward."""
    font = QFont("Helvetica", max(7, int(inner * 0.16)))
    font.setBold(True)
    font.setLetterSpacing(QFont.PercentageSpacing, 145)
    p.setFont(font)

    fm = p.fontMetrics()
    char_widths = [fm.horizontalAdvance(ch) for ch in text]
    total_w = sum(char_widths)

    # Angular span: each pixel of text width ~ 1/radius radians
    span_rad = total_w / max(1.0, radius)

    if top:
        # Arc from left to right across the top (centered at -pi/2)
        start_ang = -math.pi / 2 - span_rad / 2
        direction = 1.0
    else:
        # Arc from right to left across the bottom (centered at pi/2)
        start_ang = math.pi / 2 + span_rad / 2
        direction = -1.0

    shd_off = max(0.8, inner * 0.014)
    current_ang = start_ang

    # Vertical offset: push text body inward so outer edge hugs the radius.
    # For top text, after rotation the "up" direction is radially outward,
    # so shift down (positive y) by half the ascent to tuck text inside.
    # For bottom text, "up" is also radially outward after rotation, same shift.
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

        # Shadow
        p.setPen(QColor(0, 0, 0, 130))
        p.drawText(rect.translated(0, shd_off), Qt.AlignCenter, ch)
        # Main
        p.setPen(color)
        p.drawText(rect, Qt.AlignCenter, ch)
        p.restore()

        current_ang += direction * (cw * 0.5) / radius

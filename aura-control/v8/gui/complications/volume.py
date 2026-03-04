"""
gui.complications.volume -- Volume complication + overlay.

Extracted from carbon_demo.py `_draw_comp_volume`.
Stepped ring shows 12-segment arc level; numeric window displays percentage.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt5.QtGui import QPainter

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QBrush, QColor, QFont, QPen

from gui.complications.base import BaseComplication
from gui.renderer import clamp


class VolumeComplication(BaseComplication):
    name = "Volume"
    label = "Volume"
    category = "System"
    has_overlay = True

    def __init__(self, bus):
        super().__init__(bus)
        self.level = 50  # 0..100

    # ------------------------------------------------------------------
    def draw_content(self, p: "QPainter", inner: float, t: float, accent: QColor) -> None:
        vol = clamp(self.level / 100.0, 0.0, 1.0)
        steps = 12
        lit = int(round(vol * steps))

        # Stepped ring
        rr = inner * 0.70
        for i in range(steps):
            a0 = -90.0 + (i / steps) * 360.0
            a1 = -90.0 + ((i + 1) / steps) * 360.0
            rect = QRectF(-rr, -rr, 2 * rr, 2 * rr)

            on = i < lit
            alpha = 40 if not on else (85 + int(120 * (i / max(1, steps - 1))))
            col = QColor(accent.red(), accent.green(), accent.blue(), alpha)
            pen = QPen(col)
            pen.setWidthF(max(1.6, inner * 0.085))
            pen.setCapStyle(Qt.FlatCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawArc(rect, int(a0 * 16), int((a1 - a0) * 16))

        # Numeric window
        win = QRectF(-inner * 0.34, inner * 0.18, inner * 0.68, inner * 0.26)
        p.setPen(QPen(QColor(0, 0, 0, 140), max(1.0, inner * 0.020)))
        p.setBrush(QBrush(QColor(10, 12, 18, 180)))
        p.drawRoundedRect(win, inner * 0.06, inner * 0.06)

        # Value text (shadow + outline for readability)
        txt = f"{int(vol * 100):02d}"
        f = QFont("Helvetica", max(8, int(inner * 0.28)))
        f.setBold(True)
        p.setFont(f)
        off = max(1.0, inner * 0.018)
        p.setPen(QColor(0, 0, 0, 170))
        p.drawText(win.translated(off, off), Qt.AlignCenter, txt)
        p.setPen(QColor(0, 0, 0, 140))
        p.drawText(win.translated(-off, off), Qt.AlignCenter, txt)
        p.setPen(QColor(0, 0, 0, 140))
        p.drawText(win.translated(off, -off), Qt.AlignCenter, txt)
        p.setPen(QColor(255, 255, 255, 235))
        p.drawText(win, Qt.AlignCenter, txt)

        # Title
        f2 = QFont("Helvetica", max(8, int(inner * 0.22)))
        f2.setBold(True)
        p.setFont(f2)
        hdr = QRectF(-inner, -inner * 0.62, 2 * inner, inner * 0.26)
        p.setPen(QColor(0, 0, 0, 155))
        p.drawText(hdr.translated(off, off), Qt.AlignCenter, "VOLUME")
        p.setPen(QColor(235, 242, 255, 215))
        p.drawText(hdr, Qt.AlignCenter, "VOLUME")

    # ------------------------------------------------------------------
    def draw_overlay(self, p, cx, cy, mind, t, trans):
        """Horology-style volume: power-reserve arc + jewel knob + polished hand."""
        from gui.touch import vol_arc_angles

        R = mind * 0.34 * 0.75
        start_deg, sweep_deg = vol_arc_angles()

        # Lacquer dial plate + vignette
        plate_a = int(210 * trans)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(10, 18, 34, plate_a)))
        p.drawEllipse(int(cx - R), int(cy - R), int(2 * R), int(2 * R))

        for i in range(10):
            rr2 = R * (1.00 - i * 0.055)
            a2 = int((10 + i * 10) * trans)
            p.setPen(QPen(QColor(0, 0, 0, a2), max(1.0, mind * 0.002)))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(int(cx - rr2), int(cy - rr2), int(2 * rr2), int(2 * rr2))

        # Guilloché radial brushing
        from PyQt5.QtCore import QPointF
        p.save()
        p.translate(cx, cy)
        p.setPen(QPen(QColor(245, 235, 205, int(10 * trans)), 1))
        for i in range(56):
            ang = (2 * math.pi) * (i / 56) + t * 0.06
            x0 = (R * 0.18) * math.cos(ang); y0 = (R * 0.18) * math.sin(ang)
            x1 = (R * 0.98) * math.cos(ang); y1 = (R * 0.98) * math.sin(ang)
            p.drawLine(int(x0), int(y0), int(x1), int(y1))
        p.restore()

        # Chapter ring
        p.setPen(QPen(QColor(245, 235, 205, int(90 * trans)), max(1.0, mind * 0.004)))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(int(cx - R * 0.98), int(cy - R * 0.98), int(2 * R * 0.98), int(2 * R * 0.98))

        # Arc track
        track = QPen(QColor(245, 235, 205, int(120 * trans)))
        track.setWidthF(max(2.0, mind * 0.010)); track.setCapStyle(Qt.RoundCap)
        p.setPen(track); p.setBrush(Qt.NoBrush)
        p.drawArc(int(cx - R), int(cy - R), int(2 * R), int(2 * R),
                  int(start_deg * 16), int(sweep_deg * 16))

        # Ticks
        tick_pen = QPen(QColor(245, 235, 205, int(190 * trans)))
        tick_pen.setWidthF(max(1.0, mind * 0.004)); tick_pen.setCapStyle(Qt.RoundCap)
        p.setPen(tick_pen)
        for i in range(31):
            frac = i / 30.0
            deg = start_deg + frac * sweep_deg
            a = math.radians(deg)
            x1 = cx + (R * 0.88) * math.cos(a); y1 = cy + (R * 0.88) * math.sin(a)
            major = (i % 5 == 0)
            x2 = cx + (R * (0.98 if major else 0.95)) * math.cos(a)
            y2 = cy + (R * (0.98 if major else 0.95)) * math.sin(a)
            p.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Knob + hand
        v_deg = start_deg + (self.level / 100.0) * sweep_deg
        a = math.radians(v_deg)
        kx = cx + (R * 0.98) * math.cos(a); ky = cy + (R * 0.98) * math.sin(a)

        hand = QPen(QColor(245, 235, 205, int(210 * trans)))
        hand.setWidthF(max(2.0, mind * 0.006)); hand.setCapStyle(Qt.RoundCap)
        p.setPen(hand)
        p.drawLine(int(cx), int(cy), int(kx), int(ky))

        # Jewel knob
        kr = max(6.0, mind * 0.018)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, int(140 * trans))))
        p.drawEllipse(int(kx - kr + 2), int(ky - kr + 3), int(2 * kr), int(2 * kr))
        p.setBrush(QBrush(QColor(245, 235, 205, int(190 * trans))))
        p.drawEllipse(int(kx - kr), int(ky - kr), int(2 * kr), int(2 * kr))
        p.setBrush(QBrush(QColor(255, 255, 255, int(90 * trans))))
        p.drawEllipse(int(kx - kr * 0.45), int(ky - kr * 0.45),
                      int(2 * kr * 0.45), int(2 * kr * 0.45))

        # Center number
        f = QFont("DejaVu Sans", max(14, int(mind * 0.055))); f.setBold(True)
        p.setFont(f)
        txt = str(self.level)
        p.setPen(QColor(0, 0, 0, int(150 * trans)))
        p.drawText(int(cx - 140) + 1, int(cy - 60) + 1, 280, 120, Qt.AlignCenter, txt)
        p.setPen(QColor(245, 235, 205, int(235 * trans)))
        p.drawText(int(cx - 140), int(cy - 60), 280, 120, Qt.AlignCenter, txt)

        # Label
        f2 = QFont("DejaVu Sans", max(10, int(mind * 0.020))); f2.setBold(True)
        p.setFont(f2)
        p.setPen(QColor(245, 235, 205, int(120 * trans)))
        p.drawText(int(cx - 140), int(cy + 50), 280, 40, Qt.AlignCenter, "VOLUME")

    def on_drag(self, dx, dy):
        self.level = max(0, min(100, self.level - int(dy * 0.5)))
        self.bus.emit("volume.changed", level=self.level)
        return True

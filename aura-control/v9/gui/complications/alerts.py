"""
gui.complications.alerts -- Alerts / notifications complication.

Extracted from carbon_demo.py `_draw_comp_alerts`.
Severity sector arc (blue→red interpolation), danger hand with jitter,
pulsing alarm-heart sapphire, and count window.
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


class AlertsComplication(BaseComplication):
    name = "Alerts"
    label = "Alerts"
    category = "System"
    has_overlay = True

    def __init__(self, bus):
        super().__init__(bus)
        self.count = 0
        self.severity = 0.0       # 0..1
        self._alert_pulse = 0.0   # 0..1 flare intensity

    # ------------------------------------------------------------------
    def draw_content(self, p: "QPainter", inner: float, t: float, accent: QColor) -> None:
        sev = clamp(self.severity, 0.0, 1.0)
        flare = clamp(self._alert_pulse, 0.0, 1.0)

        # --- Severity sector (register) ---
        rr = inner * 0.78
        rect = QRectF(-rr, -rr, 2 * rr, 2 * rr)
        start = 218.0
        span = 140.0

        # Base track (cool steel)
        base_pen = QPen(QColor(255, 255, 255, 22))
        base_pen.setWidthF(max(2.0, inner * 0.082))
        base_pen.setCapStyle(Qt.RoundCap)
        p.setPen(base_pen)
        p.setBrush(Qt.NoBrush)
        p.drawArc(rect, int(start * 16), int(span * 16))

        # Severity arc (blue → red interpolation)
        w = max(2.0, inner * (0.082 + 0.022 * sev))
        hot = QColor(255, 90, 80)
        cool = QColor(60, 175, 255)
        cr = int(cool.red()   + (hot.red()   - cool.red())   * sev)
        cg = int(cool.green() + (hot.green() - cool.green()) * sev)
        cb = int(cool.blue()  + (hot.blue()  - cool.blue())  * sev)
        a  = int(70 + 140 * sev + 140 * flare)
        pen2 = QPen(QColor(cr, cg, cb, min(255, a)))
        pen2.setWidthF(w)
        pen2.setCapStyle(Qt.RoundCap)
        p.setPen(pen2)
        p.drawArc(rect, int(start * 16), int((span * sev) * 16))

        # Danger hand (jitters more when severe)
        jitter = (0.5 + 1.8 * sev) * (0.6 + 0.7 * flare)
        ang = math.radians(start + span * sev - 90.0
                           + jitter * math.sin(t * (3.0 + 5.0 * sev)))
        hx = (inner * 0.60) * math.cos(ang)
        hy = (inner * 0.60) * math.sin(ang)

        # Shadow
        penS = QPen(QColor(0, 0, 0, 90))
        penS.setWidthF(max(1.4, inner * 0.040))
        penS.setCapStyle(Qt.RoundCap)
        p.setPen(penS)
        p.drawLine(QPointF(inner * 0.010, inner * 0.010),
                   QPointF(hx + inner * 0.010, hy + inner * 0.010))

        # Hand
        penH = QPen(QColor(255, 105, 95, 220))
        penH.setWidthF(max(1.2, inner * 0.034))
        penH.setCapStyle(Qt.RoundCap)
        p.setPen(penH)
        p.drawLine(QPointF(0, 0), QPointF(hx, hy))

        # --- Pulsing alarm-heart sapphire ---
        pulse_a = int(60 + 160 * flare)
        halo = inner * (0.20 + 0.08 * flare)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(255, 70, 70, int(55 + 120 * flare))))
        p.drawEllipse(QPointF(0, 0), halo, halo)

        core = inner * 0.11
        p.setBrush(QBrush(QColor(0, 0, 0, 140)))
        p.drawEllipse(QPointF(0, 0), core * 1.25, core * 1.25)
        p.setBrush(QBrush(QColor(255, 90, 80, pulse_a)))
        p.drawEllipse(QPointF(0, 0), core, core)

        # Applied "!" severity marker
        p.setPen(QPen(QColor(235, 240, 250, 200), max(1.2, inner * 0.028)))
        p.drawLine(QPointF(0, -inner * 0.18), QPointF(0, -inner * 0.05))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(235, 240, 250, 210)))
        p.drawEllipse(QPointF(0, -inner * 0.02), inner * 0.018, inner * 0.018)

        # --- Count window (date style) ---
        win = QRectF(-inner * 0.44, inner * 0.20, inner * 0.88, inner * 0.24)
        p.setPen(QPen(QColor(0, 0, 0, 140), max(1.0, inner * 0.018)))
        p.setBrush(QBrush(QColor(10, 12, 18, 180)))
        p.drawRoundedRect(win, inner * 0.06, inner * 0.06)

        cnt = int(1 + round(9 * sev + 3 * flare))
        off = max(1.0, inner * 0.018)

        f = QFont("Helvetica", max(8, int(inner * 0.19)))
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(0, 0, 0, 175))
        p.drawText(win.translated(off, off), Qt.AlignCenter, f"ALERTS {cnt}")
        p.setPen(QColor(255, 255, 255, 240))
        p.drawText(win, Qt.AlignCenter, f"ALERTS {cnt}")

        # Title + severity label
        f2 = QFont("Helvetica", max(8, int(inner * 0.22)))
        f2.setBold(True)
        p.setFont(f2)
        lvl = "OK" if sev < 0.2 else ("ELEV" if sev < 0.55 else "SEVERE")
        hdr = QRectF(-inner, -inner * 0.74, 2 * inner, inner * 0.26)
        p.setPen(QColor(0, 0, 0, 150))
        p.drawText(hdr.translated(off, off), Qt.AlignCenter, lvl)
        p.setPen(QColor(235, 242, 255, 235))
        p.drawText(hdr, Qt.AlignCenter, lvl)

    # ------------------------------------------------------------------
    def draw_overlay(self, p, cx, cy, mind, t, trans):
        # TODO: extract full alerts overlay from carbon_demo.py
        pass

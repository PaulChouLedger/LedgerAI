"""
gui.complications.balance -- Ledger Balance complication + overlay.

Extracted from carbon_demo.py `_draw_comp_balance`.
Power-reserve arc with blued hand, jewel hub, and numeric token display.
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


class BalanceComplication(BaseComplication):
    name = "Ledger Balance"
    label = "Balance"
    category = "Finance"
    has_overlay = True

    def __init__(self, bus):
        super().__init__(bus)
        self.balance_amt = 1250.0
        self.burn_per_day = 160.0
        self._value = 0.55  # 0..1 normalized reserve

    # ------------------------------------------------------------------
    def draw_content(self, p: "QPainter", inner: float, t: float, accent: QColor) -> None:
        v = clamp(self._value, 0.0, 1.0)

        rr = inner * 0.78
        rect = QRectF(-rr, -rr, 2 * rr, 2 * rr)

        start = 215.0
        span = 110.0

        # Base track
        base_pen = QPen(QColor(255, 255, 255, 28))
        base_pen.setWidthF(max(2.0, inner * 0.11))
        base_pen.setCapStyle(Qt.RoundCap)
        p.setPen(base_pen)
        p.setBrush(Qt.NoBrush)
        p.drawArc(rect, int(start * 16), int(span * 16))

        # Filled reserve
        a = int((70 + 120 * v) * (1.0 + 0.20 * v))
        fill_pen = QPen(QColor(accent.red(), accent.green(), accent.blue(), a))
        fill_pen.setWidthF(max(2.2, inner * (0.11 + 0.02 * v)))
        fill_pen.setCapStyle(Qt.RoundCap)
        p.setPen(fill_pen)
        p.drawArc(rect, int(start * 16), int((span * v) * 16))

        # Blued hand
        ang = math.radians(start + span * v - 90.0)
        hx = (inner * 0.58) * math.cos(ang)
        hy = (inner * 0.58) * math.sin(ang)
        penH = QPen(QColor(180, 220, 255, 210))
        penH.setWidthF(max(1.4, inner * 0.035))
        penH.setCapStyle(Qt.RoundCap)
        p.setPen(penH)
        p.drawLine(QPointF(0, 0), QPointF(hx, hy))

        # Jewel hub
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 120)))
        p.drawEllipse(QPointF(0, 0), inner * 0.10, inner * 0.10)
        p.setBrush(QBrush(QColor(70, 190, 255, 165)))
        p.drawEllipse(QPointF(0, 0), inner * 0.06, inner * 0.06)

        # Labels
        off = max(1.0, inner * 0.018)

        f = QFont("Helvetica", max(8, int(inner * 0.24)))
        f.setBold(True)
        p.setFont(f)
        r1 = QRectF(-inner, -inner * 0.20, 2 * inner, inner * 0.40)
        p.setPen(QColor(0, 0, 0, 150))
        p.drawText(r1.translated(off, off), Qt.AlignCenter, "LEDGER")
        p.setPen(QColor(235, 242, 255, 230))
        p.drawText(r1, Qt.AlignCenter, "LEDGER")

        # Numeric token amount
        f2 = QFont("Helvetica", max(8, int(inner * 0.28)))
        f2.setBold(True)
        p.setFont(f2)
        amt = int(1000 + 9000 * v)
        r2 = QRectF(-inner, inner * 0.26, 2 * inner, inner * 0.40)
        p.setPen(QColor(0, 0, 0, 175))
        p.drawText(r2.translated(off, off), Qt.AlignCenter, f"{amt:,}")
        p.setPen(QColor(255, 255, 255, 238))
        p.drawText(r2, Qt.AlignCenter, f"{amt:,}")

    # ------------------------------------------------------------------
    def draw_overlay(self, p, cx, cy, mind, t, trans):
        """Ledger Reserve overlay: power-reserve arc, days left, burn/day, crown refill."""
        bal = max(0.0, self.balance_amt)
        burn = max(1e-6, self.burn_per_day)
        days = bal / burn
        target_days = 10.0
        pct = clamp(days / target_days, 0.0, 1.0)

        # Color grade (champagne → amber → red)
        if days >= 7.0:
            c_main = QColor(245, 235, 205, int(220 * trans))
            c_fill = QColor(225, 205, 150, int(230 * trans))
        elif days >= 3.0:
            c_main = QColor(245, 210, 140, int(220 * trans))
            c_fill = QColor(245, 185, 110, int(235 * trans))
        else:
            c_main = QColor(230, 95, 80, int(230 * trans))
            c_fill = QColor(220, 75, 65, int(240 * trans))

        R = mind * 0.28

        # Plate
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, int(190 * trans))))
        p.drawEllipse(int(cx - R), int(cy - R), int(2 * R), int(2 * R))

        # Chapter ring
        p.setPen(QPen(QColor(245, 235, 205, int(80 * trans)), max(1.0, mind * 0.004)))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(int(cx - R * 0.98), int(cy - R * 0.98),
                      int(2 * R * 0.98), int(2 * R * 0.98))

        # Arc geometry
        start_deg = 210.0; span_deg = 120.0
        rect = (int(cx - R * 0.86), int(cy - R * 0.86),
                int(2 * R * 0.86), int(2 * R * 0.86))

        # Track
        track = QPen(QColor(208, 178, 112, int(70 * trans)))
        track.setWidthF(max(2.0, mind * 0.010)); track.setCapStyle(Qt.RoundCap)
        p.setPen(track); p.drawArc(*rect, int(start_deg * 16), int(span_deg * 16))

        # Fill
        fillpen = QPen(c_fill)
        fillpen.setWidthF(max(2.0, mind * 0.010)); fillpen.setCapStyle(Qt.RoundCap)
        p.setPen(fillpen)
        p.drawArc(*rect, int(start_deg * 16), int((span_deg * pct) * 16))

        # Ticks
        p.setPen(QPen(c_main, max(1.0, mind * 0.003), Qt.SolidLine, Qt.RoundCap))
        for i in range(19):
            frac = i / 18.0
            deg = start_deg + frac * span_deg
            a = math.radians(deg)
            x1 = cx + (R * 0.62) * math.cos(a); y1 = cy + (R * 0.62) * math.sin(a)
            x2 = cx + (R * (0.70 if i % 3 == 0 else 0.67)) * math.cos(a)
            y2 = cy + (R * (0.70 if i % 3 == 0 else 0.67)) * math.sin(a)
            p.drawLine(int(x1), int(y1), int(x2), int(y2))

        # RESERVE title
        f = QFont("DejaVu Sans", max(10, int(mind * 0.020))); f.setBold(True)
        p.setFont(f); p.setPen(QColor(245, 235, 205, int(140 * trans)))
        p.drawText(int(cx - R), int(cy - R * 0.30), int(2 * R), int(R * 0.25),
                   Qt.AlignCenter, "RESERVE")

        # Days remaining (hero number)
        f2 = QFont("DejaVu Sans", max(14, int(mind * 0.060))); f2.setBold(True)
        p.setFont(f2)
        days_txt = f"{days:0.1f}"
        p.setPen(QColor(0, 0, 0, int(150 * trans)))
        p.drawText(int(cx - R) + 1, int(cy - R * 0.10) + 1, int(2 * R), int(R * 0.40),
                   Qt.AlignCenter, days_txt)
        p.setPen(QColor(c_main.red(), c_main.green(), c_main.blue(), int(235 * trans)))
        p.drawText(int(cx - R), int(cy - R * 0.10), int(2 * R), int(R * 0.40),
                   Qt.AlignCenter, days_txt)

        # "DAYS LEFT" label
        f3 = QFont("DejaVu Sans", max(10, int(mind * 0.018))); f3.setBold(True)
        p.setFont(f3); p.setPen(QColor(245, 235, 205, int(120 * trans)))
        p.drawText(int(cx - R), int(cy + R * 0.18), int(2 * R), int(R * 0.18),
                   Qt.AlignCenter, "DAYS LEFT")

        # Burn/day
        f4 = QFont("DejaVu Sans", max(9, int(mind * 0.016))); f4.setBold(True)
        p.setFont(f4); p.setPen(QColor(245, 235, 205, int(110 * trans)))
        p.drawText(int(cx - R), int(cy + R * 0.34), int(2 * R), int(R * 0.16),
                   Qt.AlignCenter, f"~{burn:0.0f} LEDGER / DAY")

        # Crown glyph + REFILL
        crown_y = cy + R * 0.52
        dot_r = max(1, int(mind * 0.006))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(245, 235, 205, int(160 * trans))))
        p.drawEllipse(int(cx - dot_r * 3.2), int(crown_y - dot_r), dot_r * 2, dot_r * 2)
        p.drawEllipse(int(cx - dot_r), int(crown_y - dot_r * 1.6), dot_r * 2, dot_r * 2)
        p.drawEllipse(int(cx + dot_r * 1.2), int(crown_y - dot_r), dot_r * 2, dot_r * 2)
        p.setBrush(QBrush(QColor(245, 235, 205, int(130 * trans))))
        p.drawRect(int(cx - dot_r * 3.0), int(crown_y + dot_r * 0.8),
                   int(dot_r * 6.0), int(dot_r * 1.2))

        f5 = QFont("DejaVu Sans", max(10, int(mind * 0.018))); f5.setBold(True)
        p.setFont(f5); p.setPen(QColor(245, 235, 205, int(120 * trans)))
        p.drawText(int(cx - R), int(cy + R * 0.58), int(2 * R), int(R * 0.18),
                   Qt.AlignCenter, "REFILL")

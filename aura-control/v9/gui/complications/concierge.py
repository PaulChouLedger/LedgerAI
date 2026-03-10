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
from PyQt5.QtGui import QBrush, QColor, QFont, QPen, QPainterPath, QRadialGradient

from gui.complications.base import BaseComplication
from gui.renderer import clamp


# Service definitions: (label, angle_degrees, active)
_SERVICES = [
    ("CHAT",      -90.0,  True),    # 12 o'clock
    ("SCHEDULE",  -30.0,  False),   # 2 o'clock
    ("TRANSPORT",  30.0,  False),   # 4 o'clock
    ("MUSIC",      90.0,  False),   # 6 o'clock
    ("WEATHER",   150.0,  False),   # 8 o'clock
    ("MEMORY",    210.0,  True),    # 10 o'clock
]


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
        """Concierge service menu — radial dashboard of 6 service categories."""
        a = clamp(float(trans), 0.0, 1.0)
        if a <= 0.002:
            return

        p.save()
        try:
            p.setRenderHint(p.Antialiasing, True)

            # --- DEEP ENAMEL BACKDROP (dark violet-black) ---
            R_bg = mind * 0.414
            bg_grad = QRadialGradient(cx, cy, R_bg)
            bg_grad.setColorAt(0.00, QColor(18, 8, 32, int(252 * a)))
            bg_grad.setColorAt(0.40, QColor(12, 6, 26, int(240 * a)))
            bg_grad.setColorAt(0.75, QColor(6, 2, 14, int(180 * a)))
            bg_grad.setColorAt(0.92, QColor(2, 0, 8, int(90 * a)))
            bg_grad.setColorAt(1.00, QColor(0, 0, 0, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(bg_grad))
            p.drawEllipse(QPointF(cx, cy), R_bg, R_bg)

            R = mind * 0.266  # base scale unit

            # --- OUTER DECORATIVE RING (slowly rotating) ---
            outer_r = R * 1.32
            ring_rot = (t * 3.0) % 360.0
            p.save()
            try:
                p.translate(cx, cy)
                p.rotate(ring_rot)
                p.setPen(QPen(QColor(155, 120, 210, int(40 * a)), 0.7))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(0, 0), outer_r, outer_r)
                # Tick marks around the ring
                for ti in range(36):
                    ang = 2.0 * math.pi * ti / 36
                    is_major = (ti % 6 == 0)
                    r_in = outer_r * (0.93 if is_major else 0.97)
                    r_out = outer_r * (1.07 if is_major else 1.03)
                    tick_a = int((65 if is_major else 30) * a)
                    p.setPen(QPen(QColor(155, 120, 210, tick_a), 0.6))
                    p.drawLine(
                        QPointF(r_in * math.cos(ang), r_in * math.sin(ang)),
                        QPointF(r_out * math.cos(ang), r_out * math.sin(ang)),
                    )
            finally:
                p.restore()

            # --- "AURA CONCIERGE" HEADLINE ---
            headline_a = int(210 * a)
            if headline_a > 4:
                p.setPen(QColor(210, 185, 245, headline_a))
                f = QFont("Helvetica", max(10, int(mind * 0.020)))
                f.setBold(True)
                f.setLetterSpacing(QFont.AbsoluteSpacing, 4.5)
                p.setFont(f)
                p.drawText(
                    QRectF(cx - R * 1.2, cy - R * 1.22, R * 2.4, R * 0.28),
                    Qt.AlignCenter, "AURA CONCIERGE",
                )

            # --- CENTER STATUS: "Ready" with breathing glow ---
            breathe = 0.6 + 0.4 * math.sin(t * 1.8)
            glow_r = R * 0.22 * (0.9 + 0.1 * breathe)

            # Outer glow halo
            glow_grad = QRadialGradient(cx, cy, glow_r * 2.8)
            glow_grad.setColorAt(0.00, QColor(155, 120, 210, int(55 * a * breathe)))
            glow_grad.setColorAt(0.50, QColor(120, 80, 180, int(25 * a * breathe)))
            glow_grad.setColorAt(1.00, QColor(80, 40, 140, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(glow_grad))
            p.drawEllipse(QPointF(cx, cy), glow_r * 2.8, glow_r * 2.8)

            # Inner core dot
            core_grad = QRadialGradient(cx, cy - glow_r * 0.15, glow_r * 0.4)
            core_grad.setColorAt(0.00, QColor(220, 200, 255, int(200 * a * breathe)))
            core_grad.setColorAt(0.50, QColor(155, 120, 210, int(140 * a * breathe)))
            core_grad.setColorAt(1.00, QColor(100, 60, 170, int(40 * a)))
            p.setBrush(QBrush(core_grad))
            p.drawEllipse(QPointF(cx, cy), glow_r, glow_r)

            # "Ready" text
            ready_a = int(180 * a * (0.7 + 0.3 * breathe))
            if ready_a > 4:
                p.setPen(QColor(210, 195, 240, ready_a))
                rf = QFont("Helvetica", max(8, int(mind * 0.014)))
                rf.setLetterSpacing(QFont.AbsoluteSpacing, 2.0)
                p.setFont(rf)
                p.drawText(
                    QRectF(cx - R * 0.5, cy + glow_r * 1.3, R * 1.0, R * 0.18),
                    Qt.AlignCenter, "Ready",
                )

            # --- RADIAL SERVICE MENU (6 categories on chapter ring) ---
            orbit_r = R * 0.82  # radius of the service circle
            cell_r = R * 0.18   # radius of each service cell
            icon_s = cell_r * 0.52  # icon scale

            for label, angle_deg, active in _SERVICES:
                ang_rad = math.radians(angle_deg)
                sx = cx + orbit_r * math.cos(ang_rad)
                sy = cy + orbit_r * math.sin(ang_rad)

                # Cell background
                cell_grad = QRadialGradient(sx, sy, cell_r)
                if active:
                    cell_grad.setColorAt(0.00, QColor(40, 20, 70, int(180 * a)))
                    cell_grad.setColorAt(0.70, QColor(25, 10, 50, int(150 * a)))
                    cell_grad.setColorAt(1.00, QColor(12, 4, 28, int(80 * a)))
                else:
                    cell_grad.setColorAt(0.00, QColor(22, 12, 38, int(150 * a)))
                    cell_grad.setColorAt(0.70, QColor(14, 6, 26, int(120 * a)))
                    cell_grad.setColorAt(1.00, QColor(6, 2, 14, int(60 * a)))
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(cell_grad))
                p.drawEllipse(QPointF(sx, sy), cell_r, cell_r)

                # Cell border
                border_a = int((120 if active else 55) * a)
                border_col = QColor(155, 120, 210, border_a)
                p.setPen(QPen(border_col, max(0.8, cell_r * 0.06)))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(sx, sy), cell_r, cell_r)

                # Active glow ring
                if active:
                    glow_pulse = 0.6 + 0.4 * math.sin(t * 2.2 + ang_rad)
                    ga = int(50 * a * glow_pulse)
                    p.setPen(QPen(QColor(155, 120, 210, ga), max(0.5, cell_r * 0.08)))
                    p.drawEllipse(QPointF(sx, sy), cell_r * 1.18, cell_r * 1.18)

                # --- Draw icon inside cell ---
                icon_col = QColor(195, 170, 235, int((220 if active else 140) * a))
                icon_pen = QPen(icon_col)
                icon_pen.setWidthF(max(0.8, icon_s * 0.14))
                icon_pen.setCapStyle(Qt.RoundCap)
                icon_pen.setJoinStyle(Qt.RoundJoin)
                p.setPen(icon_pen)
                p.setBrush(Qt.NoBrush)

                _draw_service_icon(p, sx, sy - cell_r * 0.12, icon_s, label, icon_col, a)

                # Label text below icon
                lbl_a = int((185 if active else 110) * a)
                if lbl_a > 4:
                    p.setPen(QColor(195, 170, 235, lbl_a))
                    lf = QFont("Helvetica", max(6, int(mind * 0.010)))
                    lf.setBold(True)
                    lf.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
                    p.setFont(lf)
                    p.drawText(
                        QRectF(sx - cell_r * 1.1, sy + cell_r * 0.38, cell_r * 2.2, cell_r * 0.52),
                        Qt.AlignCenter, label,
                    )

        finally:
            p.restore()


# ---------------------------------------------------------------------------
# Service icon drawing (simple geometric primitives)
# ---------------------------------------------------------------------------

def _draw_service_icon(p, cx, cy, s, label, col, a):
    """Draw a minimal geometric icon for the given service label."""
    pen = QPen(col)
    pen.setWidthF(max(0.7, s * 0.13))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)

    if label == "CHAT":
        # Speech bubble: rounded rectangle + small tail
        bw, bh = s * 0.8, s * 0.55
        p.drawRoundedRect(
            QRectF(cx - bw, cy - bh, bw * 2, bh * 2),
            s * 0.18, s * 0.18,
        )
        # Tail: small triangle at bottom-left
        tail = QPainterPath()
        tail.moveTo(cx - bw * 0.25, cy + bh)
        tail.lineTo(cx - bw * 0.65, cy + bh + s * 0.35)
        tail.lineTo(cx + bw * 0.15, cy + bh)
        p.drawPath(tail)

    elif label == "SCHEDULE":
        # Clock: circle + two hands
        r = s * 0.72
        p.drawEllipse(QPointF(cx, cy), r, r)
        # Hour hand (short, pointing to ~10)
        p.drawLine(QPointF(cx, cy),
                   QPointF(cx - r * 0.32, cy - r * 0.48))
        # Minute hand (long, pointing to ~12)
        p.drawLine(QPointF(cx, cy),
                   QPointF(cx, cy - r * 0.68))
        # Center dot
        p.setPen(Qt.NoPen)
        p.setBrush(col)
        p.drawEllipse(QPointF(cx, cy), s * 0.08, s * 0.08)

    elif label == "TRANSPORT":
        # Car silhouette: body rectangle + roof + two wheels
        bw, bh = s * 0.85, s * 0.28
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        # Body
        p.drawRoundedRect(
            QRectF(cx - bw, cy - bh * 0.3, bw * 2, bh * 2),
            s * 0.12, s * 0.12,
        )
        # Roof (smaller rect on top)
        rw, rh = bw * 0.55, bh * 1.0
        p.drawRoundedRect(
            QRectF(cx - rw, cy - bh * 0.3 - rh, rw * 2, rh),
            s * 0.10, s * 0.10,
        )
        # Wheels
        wh_r = s * 0.14
        p.setPen(Qt.NoPen)
        p.setBrush(col)
        p.drawEllipse(QPointF(cx - bw * 0.55, cy + bh * 1.7), wh_r, wh_r)
        p.drawEllipse(QPointF(cx + bw * 0.55, cy + bh * 1.7), wh_r, wh_r)

    elif label == "MUSIC":
        # Music note: circle head + vertical stem + flag
        nr = s * 0.24
        head_cx = cx - s * 0.15
        head_cy = cy + s * 0.35
        p.setPen(Qt.NoPen)
        p.setBrush(col)
        p.drawEllipse(QPointF(head_cx, head_cy), nr * 1.2, nr)
        # Stem
        p.setPen(pen)
        stem_x = head_cx + nr * 1.1
        p.drawLine(QPointF(stem_x, head_cy),
                   QPointF(stem_x, cy - s * 0.55))
        # Flag (small arc at top)
        flag_path = QPainterPath()
        flag_path.moveTo(stem_x, cy - s * 0.55)
        flag_path.cubicTo(
            stem_x + s * 0.35, cy - s * 0.45,
            stem_x + s * 0.30, cy - s * 0.15,
            stem_x + s * 0.05, cy - s * 0.10,
        )
        p.setBrush(Qt.NoBrush)
        p.drawPath(flag_path)

    elif label == "WEATHER":
        # Sun: circle + radiating lines
        sr = s * 0.35
        p.drawEllipse(QPointF(cx, cy), sr, sr)
        # Rays
        n_rays = 8
        for i in range(n_rays):
            ang = 2.0 * math.pi * i / n_rays
            r_in = sr * 1.35
            r_out = sr * 1.85
            p.drawLine(
                QPointF(cx + r_in * math.cos(ang), cy + r_in * math.sin(ang)),
                QPointF(cx + r_out * math.cos(ang), cy + r_out * math.sin(ang)),
            )

    elif label == "MEMORY":
        # Brain: wavy circle (sinusoidal distortion)
        br = s * 0.6
        path = QPainterPath()
        steps = 48
        for i in range(steps + 1):
            ang = 2.0 * math.pi * i / steps
            wave = 1.0 + 0.15 * math.sin(ang * 5) + 0.08 * math.cos(ang * 3)
            px = cx + br * wave * math.cos(ang)
            py = cy + br * wave * math.sin(ang)
            if i == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)
        p.drawPath(path)
        # Central dividing line (brain hemispheres)
        p.drawLine(QPointF(cx, cy - br * 0.55),
                   QPointF(cx, cy + br * 0.55))


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

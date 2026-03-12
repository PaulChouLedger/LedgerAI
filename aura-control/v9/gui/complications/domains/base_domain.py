"""
gui.complications.domains.base_domain -- Abstract base for domain/topic complications.

Domain complications are installable specialized topics (Medical, Finance,
Trading, Nutrition, etc.) that users discover in Topics Center and pin to
their dock.

Each domain provides:
  - A distinctive domain glyph (per-domain metallic bezel, dial texture, icon)
  - Optional overlay content (educational visualizations, graphs, etc.)
  - Bus event subscriptions for domain-specific data

Domain glyphs are visually differentiated from system complications:
  - System complications → Nautilus bezel (shared BaseComplication.draw_glyph)
  - Domain glyphs → Per-domain metallic finish, colored dial texture, unique icons
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt5.QtGui import QPainter

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import (
    QBrush, QColor, QLinearGradient, QPainterPath,
    QPen, QRadialGradient,
)

from gui.complications.base import BaseComplication


# ---------------------------------------------------------------------------
# Per-domain accent colours — unified ice-blue family with subtle hue shifts
# ---------------------------------------------------------------------------

DOMAIN_COLOR = {
    "Education":  QColor(235, 165, 60),    # warm amber
    "Medical":    QColor(195, 85, 75),     # muted crimson
    "Financial":  QColor(185, 155, 60),    # warm gold
    "AuraNet":    QColor(230, 140, 80),    # warm copper
    "Business":   QColor(185, 155, 60),    # warm gold   (legacy)
    "Counsel":    QColor(200, 135, 100),   # warm rose    (legacy)
}


# ---------------------------------------------------------------------------
# Per-domain metal finishes — same platinum family as system complications,
# differentiated only by texture.
# ---------------------------------------------------------------------------

_DOMAIN_METAL = {
    "metal_hi":   QColor(230, 234, 244),
    "metal_mid":  QColor(168, 175, 190),
    "metal_dark": QColor(52, 56, 66),
    "metal_edge": QColor(238, 242, 250),
}

def _make_finish(name: str) -> dict:
    col = DOMAIN_COLOR.get(name, QColor(220, 155, 90))
    textures = {
        "Education":  "guilloche",
        "Medical":    "pulse",
        "Financial":  "sunburst",
        "AuraNet":    "matte",
        "Business":   "sunburst",
        "Counsel":    "stripe",
    }
    return {
        **_DOMAIN_METAL,
        "texture": textures.get(name, "stripe"),
        "accent": col,
    }


class BaseDomainComplication(BaseComplication):
    """Specialized base for domain/topic complications.

    Overrides ``draw_glyph()`` with a per-domain metallic watchface
    (colored bezel, dial texture, chapter ring, crystal gloss, unique icon)
    so domain glyphs are visually distinct from system Nautilus complications.
    """

    category: str = "Topics"       # default; subclasses override
    dockable: bool = True
    has_overlay: bool = True

    def __init__(self, bus):
        super().__init__(bus)

    # ------------------------------------------------------------------
    # draw_glyph override — per-domain metallic watchface
    # ------------------------------------------------------------------

    def draw_glyph(self, p: "QPainter", size: float, t: float) -> None:
        """Draw domain glyph with per-domain metallic bezel + icon."""
        name = self.name
        r = size * 0.5
        inner = r * 0.84
        col = DOMAIN_COLOR.get(name, QColor(180, 180, 180))

        fin = _make_finish(name)
        M_HI = fin["metal_hi"]
        M_MID = fin["metal_mid"]
        M_DARK = fin["metal_dark"]
        M_EDGE = fin["metal_edge"]
        TEX = fin["texture"]
        ACCENT = fin["accent"]

        # Dial field picks up a hint of the domain's signature colour
        _ar, _ag, _ab = ACCENT.red(), ACCENT.green(), ACCENT.blue()
        BLUE_DARK = QColor(
            int(18 + _ar * 0.06), int(10 + _ag * 0.04), int(8 + _ab * 0.03))
        BLUE_MID = QColor(
            int(42 + _ar * 0.10), int(18 + _ag * 0.06), int(14 + _ab * 0.04))
        TINT = QColor(
            int(0.35 * 210 + 0.65 * _ar),
            int(0.35 * 120 + 0.65 * _ag),
            int(0.35 * 80 + 0.65 * _ab),
        )

        def bezel_path(rr):
            path = QPainterPath()
            path.addEllipse(QPointF(0, 0), rr, rr)
            return path

        outer = bezel_path(r)
        inner_path = bezel_path(r * 0.92)

        # ---- 1) Bezel (metallic) ----
        p.save()
        try:
            g = QLinearGradient(QPointF(-r, -r), QPointF(r, r))
            g.setColorAt(0.0, M_EDGE)
            g.setColorAt(0.22, M_HI)
            g.setColorAt(0.55, M_DARK)
            g.setColorAt(0.82, M_MID)
            g.setColorAt(1.0, M_EDGE)
            p.setBrush(QBrush(g))
            p.setPen(QPen(QColor(0, 0, 0, 140), max(0.35, r * 0.0055)))
            p.drawPath(outer)

            # Inner chamfer highlight
            p.setPen(QPen(QColor(255, 255, 255, 45), max(0.3, r * 0.0037)))
            p.drawPath(inner_path)
        finally:
            p.restore()

        # ---- 2) Dial field (deep warm tinted) ----
        p.save()
        try:
            p.setClipPath(inner_path)
            rg = QRadialGradient(QPointF(-r * 0.18, -r * 0.22), r * 1.20)
            rg.setColorAt(0.0, QColor(BLUE_MID.red(), BLUE_MID.green(), BLUE_MID.blue(), 255))
            rg.setColorAt(0.55, QColor(BLUE_DARK.red(), BLUE_DARK.green(), BLUE_DARK.blue(), 255))
            rg.setColorAt(1.0, QColor(14, 6, 5, 255))
            p.fillRect(QRectF(-r, -r, 2 * r, 2 * r), QBrush(rg))

            # Dial texture
            if TEX == "guilloche":
                step = max(2.0, r * 0.085)
                pen = QPen(QColor(255, 255, 255, 14))
                pen.setWidthF(max(0.6, r * 0.010))
                pen.setCapStyle(Qt.RoundCap)
                p.setPen(pen)
                yy = -r
                while yy <= r:
                    p.drawLine(QPointF(-r, yy), QPointF(r, yy + r * 0.10))
                    p.drawLine(QPointF(-r, yy), QPointF(r, yy - r * 0.10))
                    yy += step
            elif TEX == "pulse":
                step = max(2.0, r * 0.085)
                pulse_u = 0.5 + 0.5 * math.sin(t * 2.1)
                pen = QPen(QColor(255, 255, 255, int(8 + 6 * pulse_u)))
                pen.setWidthF(max(0.8, r * 0.012))
                p.setPen(pen)
                yy = -r
                while yy <= r:
                    wob = 0.9 * math.sin((yy / step) * 0.55 + t * 0.6)
                    p.drawLine(QPointF(-r, yy + wob), QPointF(r, yy - wob))
                    yy += step
            elif TEX == "sunburst":
                step = max(2.0, r * 0.085)
                pen = QPen(QColor(255, 255, 255, 10))
                pen.setWidthF(max(0.8, r * 0.012))
                p.setPen(pen)
                yy = -r
                while yy <= r:
                    wob = 0.9 * math.sin((yy / step) * 0.55 + t * 0.6)
                    p.drawLine(QPointF(-r, yy + wob), QPointF(r, yy - wob))
                    yy += step
            else:
                # stripe
                step = max(2.0, r * 0.075)
                pen = QPen(QColor(255, 255, 255, 16))
                pen.setWidthF(max(0.8, r * 0.011))
                p.setPen(pen)
                yy = -r
                while yy <= r:
                    p.drawLine(QPointF(-r, yy), QPointF(r, yy))
                    yy += step

            # Color wash / lift
            burst = QRadialGradient(QPointF(0, 0), r * 1.05)
            if TEX == "pulse":
                pulse_u2 = 0.5 + 0.5 * math.sin(t * 2.1)
                a0 = int(30 + 35 * pulse_u2)
                burst.setColorAt(0.0, QColor(_ar, _ag, _ab, a0))
                burst.setColorAt(0.45, QColor(_ar, _ag, _ab, int(12 + 14 * pulse_u2)))
            else:
                burst.setColorAt(0.0, QColor(TINT.red(), TINT.green(), TINT.blue(), 55))
                burst.setColorAt(0.40, QColor(TINT.red(), TINT.green(), TINT.blue(), 16))
            burst.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.fillRect(QRectF(-r, -r, 2 * r, 2 * r), QBrush(burst))
        finally:
            p.restore()

        # ---- 3) Chapter ring + applied indices ----
        p.save()
        try:
            p.setClipPath(inner_path)
            rr_ch = inner * 0.96
            pen_ch = QPen(QColor(TINT.red(), TINT.green(), TINT.blue(), 32))
            pen_ch.setWidthF(max(0.50, inner * 0.0114))
            p.setPen(pen_ch)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(0, 0), rr_ch, rr_ch)

            # Applied baton indices (8 marks)
            for i in range(8):
                ang = -math.pi / 2 + i * (2 * math.pi / 8.0)
                major = (i % 2 == 0)
                rr0 = inner * (0.78 if major else 0.82)
                rr1 = inner * (0.93 if major else 0.92)
                x0 = rr0 * math.cos(ang)
                y0 = rr0 * math.sin(ang)
                x1 = rr1 * math.cos(ang)
                y1 = rr1 * math.sin(ang)

                # Shadow
                penS = QPen(QColor(0, 0, 0, 60))
                penS.setWidthF(max(1.2, inner * (0.040 if major else 0.028)))
                penS.setCapStyle(Qt.RoundCap)
                p.setPen(penS)
                off = inner * 0.012
                p.drawLine(QPointF(x0 + off, y0 + off), QPointF(x1 + off, y1 + off))

                # Metal
                grad = QLinearGradient(QPointF(x0, y0), QPointF(x1, y1))
                grad.setColorAt(0.0, QColor(228, 232, 242))
                grad.setColorAt(0.5, QColor(175, 180, 195))
                grad.setColorAt(1.0, QColor(95, 100, 112))
                penM = QPen(QBrush(grad), max(1.0, inner * (0.036 if major else 0.024)))
                penM.setCapStyle(Qt.RoundCap)
                p.setPen(penM)
                p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

            # Minute dots — tinted with domain colour
            dot_r = int(0.5 * 255 + 0.5 * _ar)
            dot_g = int(0.5 * 255 + 0.5 * _ag)
            dot_b = int(0.5 * 255 + 0.5 * _ab)
            for i in range(32):
                ang = -math.pi / 2 + i * (2 * math.pi / 32.0)
                rr0 = inner * 0.955
                dx = rr0 * math.cos(ang)
                dy = rr0 * math.sin(ang)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(dot_r, dot_g, dot_b, 35 if (i % 4) else 60))
                d = max(0.8, inner * (0.016 if (i % 4) else 0.024))
                p.drawEllipse(QPointF(dx, dy), d * 0.5, d * 0.5)
        finally:
            p.restore()

        # ---- 4) Inner glass rim ----
        p.save()
        try:
            p.setClipPath(inner_path)
            rr_gl = inner * 0.88
            pen_gl = QPen(QColor(255, 255, 255, 20))
            pen_gl.setWidthF(max(0.50, inner * 0.0097))
            p.setPen(pen_gl)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(0, 0), rr_gl, rr_gl)
        finally:
            p.restore()

        # ---- 5) Crystal gloss (sapphire dome) ----
        p.save()
        try:
            p.setClipPath(inner_path)
            gloss = QLinearGradient(QPointF(0, -r * 0.6), QPointF(0, r * 0.3))
            gloss.setColorAt(0.0, QColor(255, 255, 255, 28))
            gloss.setColorAt(0.35, QColor(255, 255, 255, 8))
            gloss.setColorAt(0.65, QColor(255, 255, 255, 0))
            gloss.setColorAt(1.0, QColor(255, 255, 255, 6))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(gloss))
            p.drawEllipse(QPointF(0, 0), inner * 0.86, inner * 0.86)
        finally:
            p.restore()

        # ---- 6) Icon (per-domain colour, larger, more detailed) ----
        icon_r = inner * 0.72

        # Shadow pass
        p.save()
        try:
            shd_off = max(0.8, icon_r * 0.035)
            p.translate(shd_off, shd_off)
            _draw_domain_icon(p, name, icon_r, t, shadow=True)
        finally:
            p.restore()

        # Main icon pass
        _draw_domain_icon(p, name, icon_r, t, shadow=False)

        # Highlight pass (subtle top-lit edge)
        p.save()
        try:
            p.translate(0, -max(0.5, icon_r * 0.018))
            _draw_domain_icon(p, name, icon_r, t, shadow=False, highlight=True)
        finally:
            p.restore()


# ---------------------------------------------------------------------------
# Per-domain signature colours (used INSIDE the icon only)
# ---------------------------------------------------------------------------

_ICON_COLORS = {
    "Medical":   (QColor(235, 75, 65),  QColor(255, 130, 120)),   # red
    "Education": (QColor(245, 175, 65), QColor(255, 215, 150)),    # amber
    "Financial": (QColor(205, 175, 60), QColor(235, 210, 130)),    # warm gold
    "AuraNet":   (QColor(240, 155, 80), QColor(255, 200, 155)),    # copper
    "Business":  (QColor(205, 175, 60), QColor(235, 210, 130)),
    "Counsel":   (QColor(215, 150, 110), QColor(240, 195, 165)),
}


# ---------------------------------------------------------------------------
# Domain icon shapes — detailed, full-circle designs
# ---------------------------------------------------------------------------

def _draw_domain_icon(p, name: str, ir: float, t: float,
                      shadow: bool = False, highlight: bool = False):
    """Draw the actual icon shape for a domain glyph.  ir = icon radius."""
    main_col, hi_col = _ICON_COLORS.get(name, (QColor(195, 200, 215), QColor(225, 230, 240)))

    if shadow:
        pen = QPen(QColor(0, 0, 0, 70))
        pen.setWidthF(max(1.4, ir * 0.065))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
    elif highlight:
        pen = QPen(QColor(hi_col.red(), hi_col.green(), hi_col.blue(), 70))
        pen.setWidthF(max(0.6, ir * 0.025))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
    else:
        pen = QPen(main_col)
        pen.setWidthF(max(1.2, ir * 0.055))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)

    if name == "Medical":
        _draw_medical_icon(p, ir, t, main_col, hi_col, shadow, highlight)
    elif name == "Education":
        _draw_education_icon(p, ir, t, main_col, hi_col, shadow, highlight)
    elif name == "Financial":
        _draw_financial_icon(p, ir, t, main_col, hi_col, shadow, highlight)
    elif name == "AuraNet":
        _draw_auranet_icon(p, ir, t, main_col, hi_col, shadow, highlight)
    elif name == "Business":
        _draw_financial_icon(p, ir, t, main_col, hi_col, shadow, highlight)
    elif name == "Counsel":
        # Simple speech bubble fallback
        bw, bh = ir * 0.75, ir * 0.55
        bubble = QPainterPath()
        bubble.addRoundedRect(QRectF(-bw, -bh, bw * 2, bh * 1.55), ir * 0.20, ir * 0.20)
        tail = QPainterPath()
        tail.moveTo(-bw * 0.12, bh * 0.55)
        tail.lineTo(-bw * 0.38, bh * 0.92)
        tail.lineTo(bw * 0.18, bh * 0.55)
        tail.closeSubpath()
        bubble.addPath(tail)
        p.drawPath(bubble)


def _draw_medical_icon(p, ir, t, col, hi, shadow, highlight):
    """Red cross with pulsing ring + animated EKG waveform."""
    # Outer pulse ring
    pulse = 0.5 + 0.5 * math.sin(t * 2.5)
    ring_r = ir * (0.85 + 0.08 * pulse)
    if not shadow and not highlight:
        ring_pen = QPen(QColor(col.red(), col.green(), col.blue(), int(40 + 35 * pulse)))
        ring_pen.setWidthF(max(0.8, ir * 0.020))
        p.setPen(ring_pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(0, 0), ring_r, ring_r)

    # Cross (filled)
    arm_w = ir * 0.18
    arm_h = ir * 0.52
    cross = QPainterPath()
    cross.addRoundedRect(QRectF(-arm_w, -arm_h, arm_w * 2, arm_h * 2),
                         arm_w * 0.30, arm_w * 0.30)
    cross.addRoundedRect(QRectF(-arm_h, -arm_w, arm_h * 2, arm_w * 2),
                         arm_w * 0.30, arm_w * 0.30)
    if shadow:
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 50))
    elif highlight:
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(hi.red(), hi.green(), hi.blue(), 35))
    else:
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(col.red(), col.green(), col.blue(), 200))
    p.drawPath(cross.simplified())

    # Inner cross highlight
    if not shadow and not highlight:
        inner_arm_w = arm_w * 0.55
        inner_arm_h = arm_h * 0.65
        inner_cross = QPainterPath()
        inner_cross.addRoundedRect(QRectF(-inner_arm_w, -inner_arm_h,
                                          inner_arm_w * 2, inner_arm_h * 2),
                                   inner_arm_w * 0.3, inner_arm_w * 0.3)
        inner_cross.addRoundedRect(QRectF(-inner_arm_h, -inner_arm_w,
                                          inner_arm_h * 2, inner_arm_w * 2),
                                   inner_arm_w * 0.3, inner_arm_w * 0.3)
        p.setBrush(QColor(255, 255, 255, 45))
        p.drawPath(inner_cross.simplified())

    # EKG waveform across the middle
    if not shadow and not highlight:
        ekg_pen = QPen(QColor(255, 240, 240, 220))
        ekg_pen.setWidthF(max(1.2, ir * 0.042))
        ekg_pen.setCapStyle(Qt.RoundCap)
        ekg_pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(ekg_pen)
        # More detailed waveform: P wave, QRS complex, T wave
        pts = [
            QPointF(-ir * 0.82, 0),
            QPointF(-ir * 0.55, 0),
            QPointF(-ir * 0.45, -ir * 0.06),   # P wave
            QPointF(-ir * 0.35, 0),
            QPointF(-ir * 0.22, ir * 0.05),     # Q dip
            QPointF(-ir * 0.12, -ir * 0.38),    # R peak
            QPointF( ir * 0.02, ir * 0.18),     # S valley
            QPointF( ir * 0.12, 0),
            QPointF( ir * 0.28, -ir * 0.10),    # T wave
            QPointF( ir * 0.42, 0),
            QPointF( ir * 0.82, 0),
        ]
        path = QPainterPath()
        path.moveTo(pts[0])
        for pt in pts[1:]:
            path.lineTo(pt)
        p.drawPath(path)

    # Small heart at top-right
    if not shadow and not highlight:
        hx, hy = ir * 0.52, -ir * 0.50
        hs = ir * 0.13
        heart = QPainterPath()
        heart.moveTo(hx, hy + hs * 0.35)
        heart.cubicTo(QPointF(hx, hy - hs * 0.25),
                      QPointF(hx - hs, hy - hs * 0.25),
                      QPointF(hx - hs, hy + hs * 0.10))
        heart.cubicTo(QPointF(hx - hs, hy + hs * 0.55),
                      QPointF(hx, hy + hs * 0.75),
                      QPointF(hx, hy + hs * 0.95))
        heart.cubicTo(QPointF(hx, hy + hs * 0.75),
                      QPointF(hx + hs, hy + hs * 0.55),
                      QPointF(hx + hs, hy + hs * 0.10))
        heart.cubicTo(QPointF(hx + hs, hy - hs * 0.25),
                      QPointF(hx, hy - hs * 0.25),
                      QPointF(hx, hy + hs * 0.35))
        p.setPen(Qt.NoPen)
        beat = 0.5 + 0.5 * math.sin(t * 3.0)
        p.setBrush(QColor(col.red(), col.green(), col.blue(), int(120 + 80 * beat)))
        p.drawPath(heart)


def _draw_education_icon(p, ir, t, col, hi, shadow, highlight):
    """Open book with visible pages + graduation cap above."""
    # --- Graduation cap ---
    cap_y = -ir * 0.42
    cap_w = ir * 0.55
    cap_h = ir * 0.18

    # Mortarboard (diamond shape)
    cap_path = QPainterPath()
    cap_path.moveTo(0, cap_y - cap_h * 0.6)        # top
    cap_path.lineTo(-cap_w, cap_y)                  # left
    cap_path.lineTo(0, cap_y + cap_h * 0.4)         # bottom
    cap_path.lineTo(cap_w, cap_y)                    # right
    cap_path.closeSubpath()

    if not shadow and not highlight:
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(col.red(), col.green(), col.blue(), 160))
        p.drawPath(cap_path)
        # Cap highlight
        cap_hi = QPainterPath()
        cap_hi.moveTo(0, cap_y - cap_h * 0.6)
        cap_hi.lineTo(cap_w * 0.5, cap_y - cap_h * 0.15)
        cap_hi.lineTo(0, cap_y + cap_h * 0.15)
        cap_hi.lineTo(-cap_w * 0.5, cap_y - cap_h * 0.15)
        cap_hi.closeSubpath()
        p.setBrush(QColor(hi.red(), hi.green(), hi.blue(), 55))
        p.drawPath(cap_hi)
    else:
        p.drawPath(cap_path)

    # Tassel (hangs from center-right)
    if not shadow and not highlight:
        tassel_pen = QPen(QColor(col.red(), col.green(), col.blue(), 180))
        tassel_pen.setWidthF(max(0.8, ir * 0.028))
        tassel_pen.setCapStyle(Qt.RoundCap)
        p.setPen(tassel_pen)
        p.drawLine(QPointF(0, cap_y), QPointF(ir * 0.32, cap_y + ir * 0.22))
        # Tassel end
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(col.red(), col.green(), col.blue(), 200))
        p.drawEllipse(QPointF(ir * 0.32, cap_y + ir * 0.24), ir * 0.035, ir * 0.050)

    # --- Open book (lower portion) ---
    bw = ir * 0.88
    spine_top = -ir * 0.08
    spine_bot = ir * 0.72

    # Spine
    spine_pen = QPen(p.pen().color() if not highlight else
                     QColor(hi.red(), hi.green(), hi.blue(), 70))
    spine_pen.setWidthF(max(1.0, ir * 0.040))
    spine_pen.setCapStyle(Qt.RoundCap)
    p.setPen(spine_pen)
    p.drawLine(QPointF(0, spine_top), QPointF(0, spine_bot))

    # Left page
    left = QPainterPath()
    left.moveTo(0, spine_top)
    left.cubicTo(QPointF(-bw * 0.35, spine_top - ir * 0.18),
                 QPointF(-bw * 0.80, spine_top - ir * 0.12),
                 QPointF(-bw, spine_top + ir * 0.05))
    left.lineTo(-bw, spine_bot - ir * 0.08)
    left.cubicTo(QPointF(-bw * 0.70, spine_bot - ir * 0.18),
                 QPointF(-bw * 0.30, spine_bot - ir * 0.10),
                 QPointF(0, spine_bot))
    p.drawPath(left)

    # Right page
    right = QPainterPath()
    right.moveTo(0, spine_top)
    right.cubicTo(QPointF(bw * 0.35, spine_top - ir * 0.18),
                  QPointF(bw * 0.80, spine_top - ir * 0.12),
                  QPointF(bw, spine_top + ir * 0.05))
    right.lineTo(bw, spine_bot - ir * 0.08)
    right.cubicTo(QPointF(bw * 0.70, spine_bot - ir * 0.18),
                  QPointF(bw * 0.30, spine_bot - ir * 0.10),
                  QPointF(0, spine_bot))
    p.drawPath(right)

    # Page fill (subtle)
    if not shadow and not highlight:
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(col.red(), col.green(), col.blue(), 18))
        p.drawPath(left)
        p.drawPath(right)

    # Text lines on pages
    if not shadow and not highlight:
        line_pen = QPen(QColor(col.red(), col.green(), col.blue(), 100))
        line_pen.setWidthF(max(0.6, ir * 0.018))
        line_pen.setCapStyle(Qt.RoundCap)
        p.setPen(line_pen)
        for j in range(4):
            yy = spine_top + ir * 0.12 + j * ir * 0.13
            # Left page lines
            p.drawLine(QPointF(-bw * 0.78, yy), QPointF(-ir * 0.12, yy))
            # Right page lines
            p.drawLine(QPointF(ir * 0.12, yy), QPointF(bw * 0.78, yy))


def _draw_financial_icon(p, ir, t, col, hi, shadow, highlight):
    """Dollar sign with ascending chart bars + trend arrow."""
    # --- Background chart bars (ascending) ---
    n_bars = 6
    bar_w = ir * 0.14
    bar_gap = ir * 0.24
    base_y = ir * 0.65
    heights = [0.22, 0.35, 0.28, 0.48, 0.42, 0.68]

    for i, h in enumerate(heights):
        bx = -ir * 0.60 + i * bar_gap
        bar_h = ir * h * 1.1
        bar_top = base_y - bar_h

        if shadow:
            p.drawRoundedRect(QRectF(bx - bar_w * 0.5, bar_top, bar_w, bar_h),
                              ir * 0.015, ir * 0.015)
        elif highlight:
            p.drawRoundedRect(QRectF(bx - bar_w * 0.5, bar_top, bar_w, bar_h),
                              ir * 0.015, ir * 0.015)
        else:
            # Bar fill with gradient
            is_up = (i == 0) or (heights[i] >= heights[i - 1])
            bar_a = 140 if is_up else 80
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(col.red(), col.green(), col.blue(), bar_a))
            p.drawRoundedRect(QRectF(bx - bar_w * 0.5, bar_top, bar_w, bar_h),
                              ir * 0.015, ir * 0.015)
            # Bar outline
            outline = QPen(QColor(col.red(), col.green(), col.blue(), 200))
            outline.setWidthF(max(0.6, ir * 0.016))
            p.setPen(outline)
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(bx - bar_w * 0.5, bar_top, bar_w, bar_h),
                              ir * 0.015, ir * 0.015)

    # --- Trend arrow (sweeping upward curve) ---
    if not shadow:
        trend = QPainterPath()
        trend.moveTo(-ir * 0.65, ir * 0.45)
        trend.cubicTo(QPointF(-ir * 0.20, ir * 0.40),
                      QPointF(ir * 0.15, ir * 0.05),
                      QPointF(ir * 0.62, -ir * 0.35))
        trend_pen = QPen(QColor(255, 255, 255, 160) if not highlight
                         else QColor(hi.red(), hi.green(), hi.blue(), 60))
        trend_pen.setWidthF(max(1.0, ir * 0.035))
        trend_pen.setCapStyle(Qt.RoundCap)
        p.setPen(trend_pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(trend)

        # Arrowhead
        if not highlight:
            arr_pen = QPen(QColor(255, 255, 255, 180))
            arr_pen.setWidthF(max(1.0, ir * 0.032))
            arr_pen.setCapStyle(Qt.RoundCap)
            p.setPen(arr_pen)
            p.drawLine(QPointF(ir * 0.62, -ir * 0.35),
                       QPointF(ir * 0.48, -ir * 0.28))
            p.drawLine(QPointF(ir * 0.62, -ir * 0.35),
                       QPointF(ir * 0.58, -ir * 0.18))

    # --- Central dollar sign ---
    if not shadow and not highlight:
        from PyQt5.QtGui import QFont
        f = QFont("Helvetica", max(8, int(ir * 0.56)))
        f.setBold(True)
        p.setFont(f)
        # Shadow
        p.setPen(QColor(0, 0, 0, 100))
        p.drawText(QRectF(-ir * 0.5, -ir * 0.72, ir, ir * 0.8),
                   Qt.AlignCenter, "$")
        # Main
        p.setPen(QColor(col.red(), col.green(), col.blue(), 240))
        off = max(0.5, ir * 0.012)
        p.drawText(QRectF(-ir * 0.5 - off, -ir * 0.72 - off, ir, ir * 0.8),
                   Qt.AlignCenter, "$")

    # --- Baseline ---
    if not shadow:
        base_pen = QPen(QColor(col.red(), col.green(), col.blue(), 60)
                        if not highlight else
                        QColor(hi.red(), hi.green(), hi.blue(), 30))
        base_pen.setWidthF(max(0.6, ir * 0.018))
        p.setPen(base_pen)
        p.drawLine(QPointF(-ir * 0.72, base_y), QPointF(ir * 0.72, base_y))


def _draw_auranet_icon(p, ir, t, col, hi, shadow, highlight):
    """Globe wireframe with connection arcs + orbiting nodes."""
    # --- Globe wireframe ---
    globe_r = ir * 0.55

    # Outer circle
    p.drawEllipse(QPointF(0, 0), globe_r, globe_r)

    # Equator (ellipse)
    p.drawEllipse(QRectF(-globe_r, -globe_r * 0.18, globe_r * 2, globe_r * 0.36))

    # Meridians (two vertical ellipses at different tilts)
    p.save()
    p.drawEllipse(QRectF(-globe_r * 0.32, -globe_r, globe_r * 0.64, globe_r * 2))
    p.restore()
    p.save()
    p.drawEllipse(QRectF(-globe_r * 0.70, -globe_r, globe_r * 1.40, globe_r * 2))
    p.restore()

    # Latitude lines
    for lat_f in [-0.52, 0.52]:
        lat_y = globe_r * lat_f
        # Width narrows at poles
        w_factor = math.sqrt(max(0.01, 1.0 - lat_f * lat_f))
        p.drawEllipse(QRectF(-globe_r * w_factor, lat_y - globe_r * 0.08,
                              globe_r * w_factor * 2, globe_r * 0.16))

    # --- Orbiting connection nodes ---
    if not shadow and not highlight:
        n_nodes = 5
        orbit_r = ir * 0.82
        for i in range(n_nodes):
            ang = (2 * math.pi * i / n_nodes) + t * 0.4
            nx = orbit_r * math.cos(ang)
            ny = orbit_r * math.sin(ang)
            node_r = ir * 0.055

            # Connection line from globe surface to node
            surf_x = globe_r * 0.92 * math.cos(ang)
            surf_y = globe_r * 0.92 * math.sin(ang)
            conn_pen = QPen(QColor(col.red(), col.green(), col.blue(), 55))
            conn_pen.setWidthF(max(0.5, ir * 0.012))
            conn_pen.setCapStyle(Qt.RoundCap)
            p.setPen(conn_pen)
            p.drawLine(QPointF(surf_x, surf_y), QPointF(nx, ny))

            # Node glow
            pulse_v = 0.5 + 0.5 * math.sin(t * 2.2 + i * 1.3)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(col.red(), col.green(), col.blue(),
                              int(25 + 30 * pulse_v)))
            p.drawEllipse(QPointF(nx, ny), node_r * 1.8, node_r * 1.8)

            # Node body
            p.setBrush(QColor(col.red(), col.green(), col.blue(), 200))
            p.drawEllipse(QPointF(nx, ny), node_r, node_r)

            # Bright core
            p.setBrush(QColor(255, 255, 255, int(100 + 60 * pulse_v)))
            p.drawEllipse(QPointF(nx, ny), node_r * 0.35, node_r * 0.35)

        # Data pulse arcs between adjacent nodes
        for i in range(n_nodes):
            j = (i + 1) % n_nodes
            a0 = (2 * math.pi * i / n_nodes) + t * 0.4
            a1 = (2 * math.pi * j / n_nodes) + t * 0.4
            mid_ang = (a0 + a1) * 0.5
            # Only draw a few active arcs
            if i % 2 == 0:
                arc_pen = QPen(QColor(col.red(), col.green(), col.blue(), 35))
                arc_pen.setWidthF(max(0.5, ir * 0.010))
                p.setPen(arc_pen)
                x0 = orbit_r * math.cos(a0)
                y0 = orbit_r * math.sin(a0)
                x1 = orbit_r * math.cos(a1)
                y1 = orbit_r * math.sin(a1)
                # Control point pushed outward
                ctrl_r = orbit_r * 1.25
                cx_a = ctrl_r * math.cos(mid_ang)
                cy_a = ctrl_r * math.sin(mid_ang)
                arc_path = QPainterPath()
                arc_path.moveTo(x0, y0)
                arc_path.quadTo(QPointF(cx_a, cy_a), QPointF(x1, y1))
                p.drawPath(arc_path)

    # --- Center glow ---
    if not shadow and not highlight:
        center_pulse = 0.5 + 0.5 * math.sin(t * 1.5)
        glow_r = ir * 0.18
        p.setPen(Qt.NoPen)
        glow = QRadialGradient(QPointF(0, 0), glow_r)
        glow.setColorAt(0.0, QColor(col.red(), col.green(), col.blue(),
                                    int(80 + 50 * center_pulse)))
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(glow))
        p.drawEllipse(QPointF(0, 0), glow_r, glow_r)

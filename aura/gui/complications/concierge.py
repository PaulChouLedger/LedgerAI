"""
gui.complications.concierge -- Aura Concierge complication.

Patek Philippe Grand Complications aesthetic: the concierge hub is rendered
as a perpetual calendar complication.  Each satellite service is a true
miniature Nautilus complication — bezel, metal gradient, dial field,
engine-turned guilloché, chapter-ring indices, and jeweled accent.
A central tourbillon cage rotates slowly at the heart of the dial.

Subscribes to bus events:
  - "system.metrics" → live service health for sub-dial active states
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt5.QtGui import QPainter

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QLinearGradient,
    QPen, QPainterPath, QRadialGradient,
)

from gui.complications.base import BaseComplication
from gui.renderer import clamp

# ── Palette ──────────────────────────────────────────────────────────
_CHAMPAGNE  = lambda a=255: QColor(218, 200, 155, a)
_IVORY      = lambda a=255: QColor(240, 234, 218, a)
_DIM_GOLD   = lambda a=255: QColor(165, 152, 118, a)
_AMETHYST   = lambda a=255: QColor(155, 120, 210, a)
_DEEP_VIOLET = lambda a=255: QColor(80, 50, 140, a)

# Unified platinum-violet palette
_METAL_HI   = QColor(228, 224, 240)   # platinum highlight
_METAL_MID  = QColor(152, 142, 185)   # violet-silver mid
_METAL_DARK = QColor(50, 44, 72)      # deep violet shadow
_DIAL_DARK  = QColor(10, 7, 26)       # universal dial dark
_DIAL_MID   = QColor(24, 16, 50)      # universal dial mid

# Per-service accent jewel (violet family)
_SVC_ACCENT = {
    "VOICE":     QColor(170, 145, 230),   # soft violet
    "SCHEDULE":  QColor(185, 165, 215),   # warm lavender
    "CHAT":      QColor(155, 150, 210),   # cool periwinkle
    "BRAIN":     QColor(175, 150, 220),   # mid violet
    "SPEAK":     QColor(160, 155, 215),   # blue-violet
    "FARSIGHT":  QColor(180, 170, 210),   # silver-lavender
}

# Service definitions: (label, angle_degrees, default_active)
_SERVICES = [
    ("VOICE",     -90.0,  True),    # 12 o'clock — Whisper STT
    ("SCHEDULE",  -30.0,  False),   # 2 o'clock  — verbal reminders (planned)
    ("CHAT",       30.0,  False),   # 4 o'clock  — Twilio calls (planned)
    ("BRAIN",      90.0,  True),    # 6 o'clock  — LLM inference
    ("SPEAK",     150.0,  True),    # 8 o'clock  — TTS engine
    ("FARSIGHT",  210.0,  False),   # 10 o'clock — RTX offload
]


class ConciergeComplication(BaseComplication):
    name = "Aura Concierge"
    label = "Aura Concierge"
    category = "System"
    dockable = True
    has_overlay = True

    def __init__(self, bus):
        super().__init__(bus)
        self._svc_active = {
            "VOICE": True, "SCHEDULE": False, "CHAT": False,
            "BRAIN": True, "SPEAK": True, "FARSIGHT": False,
        }
        bus.on("system.metrics", self._on_metrics)

    def _on_metrics(self, services=None, farsight_ok=False, **_kw):
        if services:
            self._svc_active["VOICE"] = services.get("whisper", False)
            self._svc_active["BRAIN"] = services.get("llm", False)
            self._svc_active["FARSIGHT"] = farsight_ok
        # SPEAK always True (local Piper), SCHEDULE/CHAT stay False (planned)

    # ------------------------------------------------------------------
    def draw_content(self, p: "QPainter", inner: float, t: float, accent: QColor) -> None:
        _draw_curved_text(p, "AURA", inner * 0.58, top=True,
                          color=QColor(235, 225, 248, 235), inner=inner)
        _draw_curved_text(p, "CONCIERGE", inner * 0.58, top=False,
                          color=QColor(235, 225, 248, 215), inner=inner)
        _draw_concierge_icon(p, inner, t, accent)

    # ------------------------------------------------------------------
    def draw_overlay(self, p, cx, cy, mind, t, trans):
        """Concierge: grand complication with six satellite Nautilus sub-dials."""
        a = clamp(float(trans), 0.0, 1.0)
        if a <= 0.002:
            return

        p.save()
        try:
            p.setRenderHint(p.Antialiasing, True)

            R_bg = mind * 0.333
            R = mind * 0.234

            # ── Deep enamel backdrop ─────────────────────────────────
            bg = QRadialGradient(cx, cy, R_bg)
            bg.setColorAt(0.00, QColor(16, 10, 34, int(254 * a)))
            bg.setColorAt(0.35, QColor(12, 6, 28, int(252 * a)))
            bg.setColorAt(0.65, QColor(8, 4, 20, int(240 * a)))
            bg.setColorAt(0.85, QColor(4, 2, 12, int(180 * a)))
            bg.setColorAt(1.00, QColor(0, 0, 0, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(bg))
            p.drawEllipse(QPointF(cx, cy), R_bg, R_bg)

            # ── Engine-turned guilloché (slowly rotating radial sun-ray) ──
            gu_rot = (t * 1.5) % 360.0
            p.save()
            p.translate(cx, cy)
            p.rotate(gu_rot)
            gu_pen = QPen(QColor(155, 130, 200, int(14 * a)), max(0.4, R * 0.002))
            p.setPen(gu_pen)
            for i in range(72):
                angle = (2 * math.pi * i) / 72
                wave = 1.0 + 0.008 * math.sin(i * 5 + t * 0.3)
                x1 = R * 0.06 * math.cos(angle)
                y1 = R * 0.06 * math.sin(angle)
                x2 = R * 1.35 * wave * math.cos(angle)
                y2 = R * 1.35 * wave * math.sin(angle)
                p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

            # Concentric rings (counter-rotating)
            ring_pen = QPen(QColor(140, 110, 190, int(10 * a)), 0.4)
            p.setPen(ring_pen)
            for frac in (0.25, 0.45, 0.65, 0.85, 1.05, 1.25):
                rr = R * frac
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(0, 0), rr, rr)
            p.restore()

            # ── Outer triple bezel ────────────────────────────────────
            bezel_r = R * 1.38
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(_AMETHYST(int(130 * a)), max(2.8, mind * 0.005)))
            p.drawEllipse(QPointF(cx, cy), bezel_r, bezel_r)
            p.setPen(QPen(_DEEP_VIOLET(int(60 * a)), max(1.5, mind * 0.003)))
            p.drawEllipse(QPointF(cx, cy), bezel_r * 0.97, bezel_r * 0.97)
            p.setPen(QPen(_AMETHYST(int(80 * a)), max(1.0, mind * 0.002)))
            p.drawEllipse(QPointF(cx, cy), bezel_r * 0.94, bezel_r * 0.94)

            # ── Chapter ring (60 ticks, slowly rotating) ──────────────
            ring_rot = (t * 2.5) % 360.0
            p.save()
            p.translate(cx, cy)
            p.rotate(ring_rot)
            for i in range(60):
                ang = (2 * math.pi * i) / 60
                is_major = (i % 5 == 0)
                tick_out = bezel_r * 0.935
                tick_in = tick_out - (R * 0.07 if is_major else R * 0.03)
                tw = mind * (0.0035 if is_major else 0.0012)
                ta = int((110 if is_major else 40) * a)
                p.setPen(QPen(_CHAMPAGNE(ta), tw, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(
                    QPointF(tick_in * math.cos(ang), tick_in * math.sin(ang)),
                    QPointF(tick_out * math.cos(ang), tick_out * math.sin(ang)),
                )
            p.restore()

            # ── Beveled depth shadow ──────────────────────────────────
            depth = QRadialGradient(cx, cy, bezel_r * 0.94)
            depth.setColorAt(0.70, QColor(0, 0, 0, 0))
            depth.setColorAt(0.90, QColor(0, 0, 0, int(50 * a)))
            depth.setColorAt(1.00, QColor(0, 0, 0, int(90 * a)))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(depth))
            p.drawEllipse(QPointF(cx, cy), bezel_r * 0.94, bezel_r * 0.94)

            # ── Crystal reflection ────────────────────────────────────
            refl = QRadialGradient(cx - R * 0.4, cy - R * 0.5, R * 1.2)
            refl.setColorAt(0.0, QColor(255, 255, 255, int(8 * a)))
            refl.setColorAt(0.3, QColor(255, 255, 255, int(3 * a)))
            refl.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setBrush(QBrush(refl))
            p.drawEllipse(QPointF(cx, cy), bezel_r * 0.94, bezel_r * 0.94)

            # ── "AURA" / "CONCIERGE" curved along bezel arcs ─────────
            _draw_overlay_arc_text(p, cx, cy, bezel_r * 0.82, "AURA", top=True,
                                   color=_IVORY(int(245 * a)),
                                   shadow=QColor(0, 0, 0, int(160 * a)),
                                   font_size=max(10, int(mind * 0.020)),
                                   spacing=mind * 0.018)
            _draw_overlay_arc_text(p, cx, cy, bezel_r * 0.82, "CONCIERGE", top=False,
                                   color=_IVORY(int(230 * a)),
                                   shadow=QColor(0, 0, 0, int(140 * a)),
                                   font_size=max(8, int(mind * 0.015)),
                                   spacing=mind * 0.012)

            # ── Central tourbillon cage ───────────────────────────────
            _draw_tourbillon(p, cx, cy, R, mind, t, a)

            # ── Six satellite Nautilus sub-dials (live active state) ──
            orbit_r = R * 0.76
            cell_r = R * 0.22
            icon_s = cell_r * 0.72

            for label, angle_deg, default_active in _SERVICES:
                ang_rad = math.radians(angle_deg)
                sx = cx + orbit_r * math.cos(ang_rad)
                sy = cy + orbit_r * math.sin(ang_rad)
                active = self._svc_active.get(label, default_active)
                _draw_subdial(p, sx, sy, cell_r, icon_s, label, active, t, a, mind)

        finally:
            p.restore()


# =====================================================================
# Tourbillon — rotating cage with breathing jewel
# =====================================================================

def _draw_tourbillon(p, cx, cy, R, mind, t, a):
    cage_r = R * 0.18
    cage_rot = (t * 15.0) % 360.0

    p.save()
    p.translate(cx, cy)

    # Recessed sub-dial
    cage_bg = QRadialGradient(0, 0, cage_r * 1.4)
    cage_bg.setColorAt(0.0, QColor(22, 14, 42, int(230 * a)))
    cage_bg.setColorAt(0.6, QColor(14, 8, 30, int(220 * a)))
    cage_bg.setColorAt(1.0, QColor(6, 3, 16, int(200 * a)))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(cage_bg))
    p.drawEllipse(QPointF(0, 0), cage_r * 1.4, cage_r * 1.4)

    # Guilloché inside cage
    p.save()
    p.rotate(-cage_rot * 0.3)
    gu_pen = QPen(QColor(140, 110, 190, int(12 * a)), 0.3)
    p.setPen(gu_pen)
    for i in range(24):
        ang = (2 * math.pi * i) / 24
        p.drawLine(QPointF(0, 0),
                   QPointF(cage_r * 1.3 * math.cos(ang),
                           cage_r * 1.3 * math.sin(ang)))
    p.restore()

    # Cage bezel (polished)
    bezel_g = QLinearGradient(QPointF(-cage_r * 1.4, -cage_r * 1.4),
                              QPointF(cage_r * 1.4, cage_r * 1.4))
    bezel_g.setColorAt(0.0, QColor(200, 180, 240, int(140 * a)))
    bezel_g.setColorAt(0.5, QColor(90, 60, 150, int(80 * a)))
    bezel_g.setColorAt(1.0, QColor(180, 160, 220, int(120 * a)))
    p.setPen(QPen(QBrush(bezel_g), max(2.0, mind * 0.003)))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(0, 0), cage_r * 1.4, cage_r * 1.4)

    # Rotating cage lines
    p.save()
    p.rotate(cage_rot)
    cage_pen = QPen(_AMETHYST(int(55 * a)), max(0.7, mind * 0.0012))
    p.setPen(cage_pen)
    for i in range(6):
        ang = math.pi * i / 6
        p.drawLine(
            QPointF(-cage_r * math.cos(ang), -cage_r * math.sin(ang)),
            QPointF(cage_r * math.cos(ang), cage_r * math.sin(ang)),
        )
    p.drawEllipse(QPointF(0, 0), cage_r * 0.55, cage_r * 0.55)

    p.rotate(30)
    cage_pen2 = QPen(_AMETHYST(int(30 * a)), max(0.4, mind * 0.0008))
    p.setPen(cage_pen2)
    for i in range(6):
        ang = math.pi * i / 6
        p.drawLine(
            QPointF(-cage_r * 0.7 * math.cos(ang), -cage_r * 0.7 * math.sin(ang)),
            QPointF(cage_r * 0.7 * math.cos(ang), cage_r * 0.7 * math.sin(ang)),
        )
    p.restore()

    p.restore()

    # Breathing jewel
    breathe = 0.55 + 0.45 * math.sin(t * 1.8)
    jewel_r = max(4.0, mind * 0.011)
    jg = QRadialGradient(cx - jewel_r * 0.3, cy - jewel_r * 0.3, jewel_r)
    jg.setColorAt(0.0, QColor(230, 210, 255, int(245 * a * breathe)))
    jg.setColorAt(0.4, QColor(175, 140, 230, int(210 * a * breathe)))
    jg.setColorAt(1.0, QColor(90, 55, 160, int(150 * a)))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(jg))
    p.drawEllipse(QPointF(cx, cy), jewel_r, jewel_r)
    p.setBrush(QColor(255, 255, 255, int(120 * a * breathe)))
    p.drawEllipse(QPointF(cx - jewel_r * 0.25, cy - jewel_r * 0.3),
                   jewel_r * 0.25, jewel_r * 0.15)


# =====================================================================
# Cushion-cut path builder — rounded-square gem shape
# =====================================================================

def _cushion_path(cx, cy, r, corner_frac=0.38):
    s = r
    c = s * corner_frac
    path = QPainterPath()
    path.moveTo(cx, cy - s)
    path.cubicTo(cx + c * 1.1, cy - s, cx + s, cy - c * 1.1, cx + s, cy)
    path.cubicTo(cx + s, cy + c * 1.1, cx + c * 1.1, cy + s, cx, cy + s)
    path.cubicTo(cx - c * 1.1, cy + s, cx - s, cy + c * 1.1, cx - s, cy)
    path.cubicTo(cx - s, cy - c * 1.1, cx - c * 1.1, cy - s, cx, cy - s)
    path.closeSubpath()
    return path


# =====================================================================
# Cushion-cut sub-dial
# =====================================================================

def _draw_subdial(p, sx, sy, cell_r, icon_s, label, active, t, a, mind):
    M_HI   = _METAL_HI
    M_MID  = _METAL_MID
    M_DARK = _METAL_DARK
    accent = _SVC_ACCENT.get(label, _AMETHYST())
    D_DARK = _DIAL_DARK
    D_MID  = _DIAL_MID

    outer_r = cell_r * 1.10
    inner_r = cell_r * 0.95

    # ── 1) Cushion-cut metal bezel ────────────────────────────────
    bezel_path = _cushion_path(sx, sy, outer_r, 0.40)
    inner_path = _cushion_path(sx, sy, inner_r, 0.38)

    bg = QLinearGradient(QPointF(sx - outer_r, sy - outer_r),
                         QPointF(sx + outer_r, sy + outer_r))
    bg.setColorAt(0.0, QColor(M_HI.red(), M_HI.green(), M_HI.blue(), int(255 * a)))
    bg.setColorAt(0.25, QColor(M_HI.red(), M_HI.green(), M_HI.blue(), int(240 * a)))
    bg.setColorAt(0.55, QColor(M_DARK.red(), M_DARK.green(), M_DARK.blue(), int(255 * a)))
    bg.setColorAt(0.80, QColor(M_MID.red(), M_MID.green(), M_MID.blue(), int(240 * a)))
    bg.setColorAt(1.0, QColor(M_HI.red(), M_HI.green(), M_HI.blue(), int(220 * a)))
    p.setBrush(QBrush(bg))
    p.setPen(QPen(QColor(0, 0, 0, int(130 * a)), max(0.4, outer_r * 0.0098)))
    p.drawPath(bezel_path)

    p.setPen(QPen(QColor(255, 255, 255, int(30 * a)), max(0.3, outer_r * 0.0078)))
    p.setBrush(Qt.NoBrush)
    p.drawPath(inner_path)

    # ── 2) Dial field ─────────────────────────────────────────────
    p.save()
    p.setClipPath(inner_path)

    rg = QRadialGradient(QPointF(sx - inner_r * 0.2, sy - inner_r * 0.2), inner_r * 1.3)
    rg.setColorAt(0.0, QColor(D_MID.red(), D_MID.green(), D_MID.blue(), int(255 * a)))
    rg.setColorAt(0.55, QColor(D_DARK.red(), D_DARK.green(), D_DARK.blue(), int(255 * a)))
    rg.setColorAt(1.0, QColor(max(0, D_DARK.red() - 3), max(0, D_DARK.green() - 2),
                               max(0, D_DARK.blue() - 4), int(255 * a)))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(rg))
    p.drawPath(inner_path)

    # ── 3) Guilloché texture ──────────────────────────────────────
    gu_speed = 0.8 + hash(label) % 5 * 0.2
    gu_rot = (t * gu_speed) % 360.0
    p.save()
    p.translate(sx, sy)
    p.rotate(gu_rot)
    n_rays = 18
    gu_pen = QPen(QColor(accent.red(), accent.green(), accent.blue(), int(10 * a)),
                  max(0.3, inner_r * 0.010))
    p.setPen(gu_pen)
    for i in range(n_rays):
        ang = (2 * math.pi * i) / n_rays
        p.drawLine(QPointF(0, 0),
                   QPointF(inner_r * 0.95 * math.cos(ang),
                           inner_r * 0.95 * math.sin(ang)))
    p.restore()

    # ── 4) Accent lift ────────────────────────────────────────────
    if active:
        pulse = 0.6 + 0.4 * math.sin(t * 1.6 + hash(label) % 10)
        glow_a = int(28 * a * pulse)
    else:
        glow_a = int(12 * a)
    lift = QRadialGradient(QPointF(sx, sy), inner_r)
    lift.setColorAt(0.0, QColor(accent.red(), accent.green(), accent.blue(), glow_a))
    lift.setColorAt(0.4, QColor(accent.red(), accent.green(), accent.blue(), glow_a // 3))
    lift.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.setBrush(QBrush(lift))
    p.setPen(Qt.NoPen)
    p.drawPath(inner_path)

    # ── 5) Corner facet highlights ────────────────────────────────
    for corner_ang in (math.pi * 0.25, math.pi * 0.75, math.pi * 1.25, math.pi * 1.75):
        fx = sx + inner_r * 0.62 * math.cos(corner_ang)
        fy = sy + inner_r * 0.62 * math.sin(corner_ang)
        facet = QRadialGradient(fx, fy, inner_r * 0.25)
        facet.setColorAt(0.0, QColor(255, 255, 255, int(10 * a)))
        facet.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(facet))
        p.drawPath(inner_path)

    # ── 6) Glass rim ──────────────────────────────────────────────
    rim_path = _cushion_path(sx, sy, inner_r * 0.88, 0.36)
    p.setPen(QPen(QColor(255, 255, 255, int(16 * a)), max(0.3, inner_r * 0.012)))
    p.setBrush(Qt.NoBrush)
    p.drawPath(rim_path)

    p.restore()  # end clip

    # ── 7) Active jewel indicator ─────────────────────────────────
    if active:
        pulse = 0.6 + 0.4 * math.sin(t * 2.2 + hash(label) % 7)
        jr = max(2.0, cell_r * 0.10)
        jy = sy - cell_r * 0.78
        jh = QRadialGradient(sx, jy, jr * 3)
        jh.setColorAt(0.0, _AMETHYST(int(40 * a * pulse)))
        jh.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(jh))
        p.drawEllipse(QPointF(sx, jy), jr * 3, jr * 3)
        jg = QRadialGradient(sx - jr * 0.2, jy - jr * 0.2, jr)
        jg.setColorAt(0.0, QColor(255, 255, 255, int(210 * a * pulse)))
        jg.setColorAt(0.4, _AMETHYST(int(180 * a)))
        jg.setColorAt(1.0, _DEEP_VIOLET(int(140 * a)))
        p.setBrush(QBrush(jg))
        p.drawEllipse(QPointF(sx, jy), jr, jr)

    # ── 8) Service icon ───────────────────────────────────────────
    icon_a = int((210 if active else 120) * a)
    icon_col = _IVORY(icon_a)
    _draw_service_icon(p, sx, sy + cell_r * 0.02, icon_s, label, icon_col, a)

    # ── 9) Label below the cushion ────────────────────────────────
    lbl_a = int((210 if active else 115) * a)
    lbl_font = QFont("DejaVu Serif", max(6, int(mind * 0.013)))
    lbl_font.setLetterSpacing(QFont.AbsoluteSpacing, mind * 0.004)
    lbl_font.setBold(True)
    p.setFont(lbl_font)
    lbl_y = sy + outer_r * 1.08
    p.setPen(QColor(0, 0, 0, int(100 * a)))
    p.drawText(QRectF(sx - cell_r * 1.5, lbl_y + 0.5, cell_r * 3.0, cell_r * 0.50),
               Qt.AlignCenter, label)
    p.setPen(_CHAMPAGNE(lbl_a))
    p.drawText(QRectF(sx - cell_r * 1.5, lbl_y, cell_r * 3.0, cell_r * 0.50),
               Qt.AlignCenter, label)


# =====================================================================
# Curved arc text for overlay
# =====================================================================

def _draw_overlay_arc_text(p, cx, cy, radius, text, *, top, color, shadow,
                           font_size, spacing):
    font = QFont("DejaVu Serif", font_size)
    font.setBold(True)
    font.setLetterSpacing(QFont.AbsoluteSpacing, spacing)
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

    shd_off = max(0.8, radius * 0.012)
    current_ang = start_ang
    baseline_shift = fm.ascent() * 0.15

    for i, ch in enumerate(text):
        cw = char_widths[i]
        current_ang += direction * (cw * 0.5) / radius

        x = cx + radius * math.cos(current_ang)
        y = cy + radius * math.sin(current_ang)

        p.save()
        p.translate(x, y)
        rot = math.degrees(current_ang) + (90 if top else -90)
        p.rotate(rot)

        ty = -fm.ascent() + baseline_shift
        rect = QRectF(-cw, ty, cw * 2, fm.height())

        p.setPen(shadow)
        p.drawText(rect.translated(0, shd_off), Qt.AlignCenter, ch)
        p.setPen(color)
        p.drawText(rect, Qt.AlignCenter, ch)
        p.restore()

        current_ang += direction * (cw * 0.5) / radius


# =====================================================================
# Service icon drawing (geometric primitives)
# =====================================================================

def _draw_service_icon(p, cx, cy, s, label, col, a):
    pen = QPen(col)
    pen.setWidthF(max(0.7, s * 0.12))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)

    if label == "VOICE":
        # Microphone: oval head + stem + base
        head_w = s * 0.30
        head_h = s * 0.45
        p.drawEllipse(QPointF(cx, cy - s * 0.15), head_w, head_h)
        # Stem
        p.drawLine(QPointF(cx, cy + s * 0.30), QPointF(cx, cy + s * 0.55))
        # Base
        p.drawLine(QPointF(cx - s * 0.25, cy + s * 0.55),
                   QPointF(cx + s * 0.25, cy + s * 0.55))
        # U-shaped pickup arc
        arc_rect = QRectF(cx - s * 0.42, cy - s * 0.15, s * 0.84, s * 0.80)
        p.drawArc(arc_rect, -180 * 16, 180 * 16)

    elif label == "SCHEDULE":
        # Clock face
        r = s * 0.68
        p.drawEllipse(QPointF(cx, cy), r, r)
        p.drawLine(QPointF(cx, cy), QPointF(cx - r * 0.30, cy - r * 0.45))
        p.drawLine(QPointF(cx, cy), QPointF(cx, cy - r * 0.65))
        p.setPen(Qt.NoPen)
        p.setBrush(col)
        p.drawEllipse(QPointF(cx, cy), s * 0.07, s * 0.07)

    elif label == "CHAT":
        # Phone handset
        p.setPen(pen)
        # Earpiece (top)
        p.drawRoundedRect(QRectF(cx - s * 0.22, cy - s * 0.60,
                                  s * 0.44, s * 0.30), s * 0.10, s * 0.10)
        # Mouthpiece (bottom)
        p.drawRoundedRect(QRectF(cx - s * 0.22, cy + s * 0.30,
                                  s * 0.44, s * 0.30), s * 0.10, s * 0.10)
        # Connecting curve
        path = QPainterPath()
        path.moveTo(cx, cy - s * 0.30)
        path.cubicTo(cx + s * 0.55, cy - s * 0.15,
                     cx + s * 0.55, cy + s * 0.15,
                     cx, cy + s * 0.30)
        p.drawPath(path)

    elif label == "BRAIN":
        # Brain: two hemispheres with folds
        # Left hemisphere
        path_l = QPainterPath()
        path_l.moveTo(cx, cy - s * 0.50)
        path_l.cubicTo(cx - s * 0.55, cy - s * 0.50,
                       cx - s * 0.65, cy + s * 0.10,
                       cx - s * 0.35, cy + s * 0.50)
        path_l.lineTo(cx, cy + s * 0.50)
        p.drawPath(path_l)
        # Right hemisphere
        path_r = QPainterPath()
        path_r.moveTo(cx, cy - s * 0.50)
        path_r.cubicTo(cx + s * 0.55, cy - s * 0.50,
                       cx + s * 0.65, cy + s * 0.10,
                       cx + s * 0.35, cy + s * 0.50)
        path_r.lineTo(cx, cy + s * 0.50)
        p.drawPath(path_r)
        # Central fissure
        p.drawLine(QPointF(cx, cy - s * 0.50), QPointF(cx, cy + s * 0.50))
        # Folds (left)
        fold_pen = QPen(col)
        fold_pen.setWidthF(max(0.4, s * 0.06))
        fold_pen.setCapStyle(Qt.RoundCap)
        p.setPen(fold_pen)
        p.drawArc(QRectF(cx - s * 0.55, cy - s * 0.30, s * 0.50, s * 0.35),
                  0, 180 * 16)
        # Folds (right)
        p.drawArc(QRectF(cx + s * 0.05, cy - s * 0.10, s * 0.50, s * 0.35),
                  0, -180 * 16)

    elif label == "SPEAK":
        # Speaker cone with sound waves
        p.setPen(pen)
        # Cone body (trapezoid approximation)
        cone = QPainterPath()
        cone.moveTo(cx - s * 0.30, cy - s * 0.20)
        cone.lineTo(cx + s * 0.10, cy - s * 0.45)
        cone.lineTo(cx + s * 0.10, cy + s * 0.45)
        cone.lineTo(cx - s * 0.30, cy + s * 0.20)
        cone.closeSubpath()
        p.drawPath(cone)
        # Back plate
        p.drawLine(QPointF(cx - s * 0.30, cy - s * 0.20),
                   QPointF(cx - s * 0.50, cy - s * 0.20))
        p.drawLine(QPointF(cx - s * 0.30, cy + s * 0.20),
                   QPointF(cx - s * 0.50, cy + s * 0.20))
        p.drawLine(QPointF(cx - s * 0.50, cy - s * 0.20),
                   QPointF(cx - s * 0.50, cy + s * 0.20))
        # Sound wave arcs
        wave_pen = QPen(col)
        wave_pen.setWidthF(max(0.5, s * 0.08))
        wave_pen.setCapStyle(Qt.RoundCap)
        p.setPen(wave_pen)
        for wr in (s * 0.28, s * 0.45):
            arc_r = QRectF(cx + s * 0.10 - wr, cy - wr, wr * 2, wr * 2)
            p.drawArc(arc_r, -40 * 16, 80 * 16)

    elif label == "FARSIGHT":
        # Lightning bolt / uplink
        p.setPen(pen)
        bolt = QPainterPath()
        bolt.moveTo(cx + s * 0.10, cy - s * 0.60)
        bolt.lineTo(cx - s * 0.20, cy - s * 0.05)
        bolt.lineTo(cx + s * 0.05, cy - s * 0.05)
        bolt.lineTo(cx - s * 0.10, cy + s * 0.60)
        bolt.lineTo(cx + s * 0.20, cy + s * 0.05)
        bolt.lineTo(cx - s * 0.05, cy + s * 0.05)
        bolt.closeSubpath()
        p.setBrush(Qt.NoBrush)
        p.drawPath(bolt)
        # Signal arcs emanating upward
        wave_pen = QPen(col)
        wave_pen.setWidthF(max(0.4, s * 0.06))
        p.setPen(wave_pen)
        for wr in (s * 0.20, s * 0.35):
            arc_r = QRectF(cx - wr, cy - s * 0.60 - wr, wr * 2, wr * 2)
            p.drawArc(arc_r, 30 * 16, 120 * 16)


# =====================================================================
# Concierge icon: ornate skeleton key
# =====================================================================

def _draw_concierge_icon(p: "QPainter", inner: float, t: float, accent: QColor):
    kr = inner * 0.38
    breathe = math.sin(t * 0.8) * 2.0
    p.save()
    p.rotate(breathe)

    col = QColor(155, 120, 210, 200)
    hi_col = QColor(195, 170, 235, 160)

    pen = QPen(col)
    pen.setWidthF(max(1.4, kr * 0.09))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)

    bow_r = kr * 0.38
    bow_cy = -kr * 0.35

    # Shadow
    p.setPen(QPen(QColor(0, 0, 0, 70), pen.widthF() * 1.1))
    off = max(0.6, kr * 0.03)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(off, bow_cy + off), bow_r, bow_r)

    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(0, bow_cy), bow_r, bow_r)

    inner_pen = QPen(QColor(col.red(), col.green(), col.blue(), 100))
    inner_pen.setWidthF(max(0.8, kr * 0.04))
    p.setPen(inner_pen)
    p.drawEllipse(QPointF(0, bow_cy), bow_r * 0.55, bow_r * 0.55)

    shaft_top = bow_cy + bow_r
    shaft_bot = kr * 0.70
    p.setPen(pen)
    p.drawLine(QPointF(0, shaft_top), QPointF(0, shaft_bot))

    tooth_w = kr * 0.22
    for ty in [shaft_bot - kr * 0.12, shaft_bot]:
        p.drawLine(QPointF(0, ty), QPointF(tooth_w, ty))
        drop = kr * 0.08
        p.drawLine(QPointF(tooth_w, ty), QPointF(tooth_w, ty - drop))

    hi_pen = QPen(hi_col)
    hi_pen.setWidthF(max(0.6, kr * 0.03))
    hi_pen.setCapStyle(Qt.RoundCap)
    p.setPen(hi_pen)
    p.drawArc(QRectF(-bow_r, bow_cy - bow_r, bow_r * 2, bow_r * 2),
              45 * 16, 90 * 16)

    p.restore()


# =====================================================================
# Helper: draw text curved along a circular arc
# =====================================================================

def _draw_curved_text(p: "QPainter", text: str, radius: float, top: bool,
                      color: QColor, inner: float):
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

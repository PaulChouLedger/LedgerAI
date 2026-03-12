"""
gui.complications.base -- Abstract base for all complications.

Includes the shared Nautilus-style bezel/dial frame that every complication
uses. Subclasses override ``draw_content()`` to render their specific
hands, gauges, text, etc. inside the prepared dial field.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt5.QtGui import QPainter

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QLinearGradient, QPainterPath,
    QPen, QRadialGradient,
)


# ---------------------------------------------------------------------------
# Shared palette — Patek 5271P: platinum on oxblood
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fallback palette constants (used as defaults when a complication is unknown)
# ---------------------------------------------------------------------------
BLUE_DARK = QColor(28, 6, 10)       # deep oxblood black
BLUE_MID  = QColor(62, 12, 18)      # rich burgundy
BLUE_LIFT = QColor(190, 25, 30)     # ruby red
STEEL_0   = QColor(42, 46, 52)      # cool platinum shadow
STEEL_1   = QColor(165, 172, 182)   # platinum mid
STEEL_2   = QColor(225, 230, 238)   # platinum highlight

# ---------------------------------------------------------------------------
# Per-complication metal finishes — each has a unique material identity
# ---------------------------------------------------------------------------
_FINISH = {
    "Topics Center": {
        "metal_hi":   QColor(232, 236, 244),   # platinum highlight
        "metal_mid":  QColor(168, 175, 188),   # platinum mid
        "metal_dark": QColor(58, 62, 72),      # platinum shadow
        "metal_edge": QColor(240, 244, 250),   # platinum rim
        "texture": "sunburst", "dots": 48,
    },
    "Settings": {
        "metal_hi":   QColor(228, 232, 240),   # brushed platinum highlight
        "metal_mid":  QColor(160, 168, 180),   # brushed platinum mid
        "metal_dark": QColor(52, 56, 65),      # platinum dark shadow
        "metal_edge": QColor(236, 240, 248),   # platinum rim
        "texture": "guilloche", "dots": 48,
    },
    "Mute": {
        "metal_hi":   QColor(148, 152, 160),   # gunmetal platinum highlight
        "metal_mid":  QColor(82, 86, 94),      # dark platinum mid
        "metal_dark": QColor(24, 26, 30),      # near-black platinum shadow
        "metal_edge": QColor(165, 170, 178),   # gunmetal rim
        "texture": "matte", "dots": 36,
    },
    "Aura Concierge": {
        "metal_hi":   QColor(235, 238, 246),   # bright platinum highlight
        "metal_mid":  QColor(172, 178, 190),   # platinum mid
        "metal_dark": QColor(55, 58, 68),      # platinum shadow
        "metal_edge": QColor(242, 245, 252),   # bright platinum rim
        "texture": "sunburst", "dots": 60,
    },
    "Ledger Balance": {
        "metal_hi":   QColor(230, 234, 242),   # platinum highlight
        "metal_mid":  QColor(165, 172, 185),   # platinum mid
        "metal_dark": QColor(52, 56, 66),      # platinum shadow
        "metal_edge": QColor(238, 242, 250),   # platinum rim
        "texture": "guilloche", "dots": 60,
    },
    "Alerts": {
        "metal_hi":   QColor(224, 228, 236),   # cool brushed highlight
        "metal_mid":  QColor(155, 162, 175),   # cool gray mid
        "metal_dark": QColor(48, 52, 60),      # cool steel shadow
        "metal_edge": QColor(232, 236, 244),   # cool steel rim
        "texture": "pulse", "dots": 48,
    },
}

# ---------------------------------------------------------------------------
# Per-complication dial colors (radial gradient: dark → mid → lift)
# ---------------------------------------------------------------------------
_DIAL = {
    "Topics Center":  {"dark": QColor(22, 4, 8),    "mid": QColor(48, 8, 14),   "lift": QColor(190, 28, 35)},
    "Settings":       {"dark": QColor(24, 6, 10),   "mid": QColor(52, 10, 16),  "lift": QColor(175, 180, 192)},
    "Mute":           {"dark": QColor(10, 6, 6),    "mid": QColor(22, 10, 10),  "lift": QColor(90, 200, 130)},
    "Aura Concierge": {"dark": QColor(22, 4, 8),    "mid": QColor(48, 8, 14),   "lift": QColor(200, 30, 38)},
    "Ledger Balance": {"dark": QColor(24, 5, 9),    "mid": QColor(50, 10, 16),  "lift": QColor(195, 32, 40)},
    "Alerts":         {"dark": QColor(24, 6, 10),   "mid": QColor(50, 10, 16),  "lift": QColor(210, 40, 45)},
}

_DEFAULT_DIAL = {"dark": BLUE_DARK, "mid": BLUE_MID, "lift": BLUE_LIFT}

# ---------------------------------------------------------------------------
# Per-complication accent colors — 5271P ruby / platinum family
# ---------------------------------------------------------------------------
_ACCENT = {
    "Topics Center":  QColor(195, 30, 38),     # ruby red
    "Settings":       QColor(175, 180, 192),   # platinum silver
    "Mute":           QColor(90, 200, 130),    # green (live default — functional)
    "Aura Concierge": QColor(200, 30, 38),     # ruby crimson
    "Ledger Balance": QColor(195, 32, 40),     # ruby
    "Alerts":         QColor(210, 40, 45),     # deep ruby-red
}


class BaseComplication(ABC):
    """Abstract base class for a complication."""

    name: str = ""
    label: str = ""
    category: str = "System"
    dockable: bool = True
    always_available: bool = False
    has_overlay: bool = False

    def __init__(self, bus) -> None:
        self.bus = bus
        self.overlay_trans: float = 0.0
        self.overlay_target: float = 0.0
        self._overlay_speed: float = 1.5

    # ----- Per-frame update -----------------------------------------------

    def tick(self, dt: float) -> None:
        if self.overlay_trans != self.overlay_target:
            step = self._overlay_speed * dt
            if self.overlay_target > self.overlay_trans:
                self.overlay_trans = min(self.overlay_target, self.overlay_trans + step)
            else:
                self.overlay_trans = max(self.overlay_target, self.overlay_trans - step)

    # ----- Drawing: shared frame ------------------------------------------

    def draw_glyph(self, p: "QPainter", size: float, t: float) -> None:
        """Draw the full Nautilus complication: bezel + dial + content."""
        r = size * 0.5
        inner = r * 0.78

        fin = _FINISH.get(self.name, {})
        M_HI   = fin.get("metal_hi", STEEL_2)
        M_MID  = fin.get("metal_mid", STEEL_1)
        M_DARK = fin.get("metal_dark", STEEL_0)
        M_EDGE = fin.get("metal_edge", STEEL_2)
        TEX    = fin.get("texture", "stripe")
        DOTS_N = int(fin.get("dots", 48))

        accent = _ACCENT.get(self.name, QColor(210, 120, 45))
        dial = _DIAL.get(self.name, _DEFAULT_DIAL)
        D_DARK = dial["dark"]
        D_MID  = dial["mid"]
        D_LIFT = dial["lift"]

        tint = QColor(
            int(0.45 * D_LIFT.red() + 0.55 * accent.red()),
            int(0.45 * D_LIFT.green() + 0.55 * accent.green()),
            int(0.45 * D_LIFT.blue() + 0.55 * accent.blue()),
        )

        def bezel_path(rr):
            path = QPainterPath()
            path.addEllipse(QPointF(0, 0), rr, rr)
            return path

        # 1) Outer bezel
        outer = bezel_path(r)
        inner_path = bezel_path(r * 0.90)

        p.save()
        try:
            # Side ears
            ear_w = r * 0.18; ear_h = r * 0.30; ear_r = max(2.0, r * 0.10)
            p.setPen(Qt.NoPen)
            ear_col = QColor(int(M_DARK.red() * 0.22), int(M_DARK.green() * 0.22), int(M_DARK.blue() * 0.22), 230)
            p.setBrush(QBrush(ear_col))
            p.drawRoundedRect(QRectF(-r - ear_w * 0.55, -ear_h * 0.5, ear_w, ear_h), ear_r, ear_r)
            p.drawRoundedRect(QRectF(+r - ear_w * 0.45, -ear_h * 0.5, ear_w, ear_h), ear_r, ear_r)

            # Metal fill
            g = QLinearGradient(QPointF(-r, -r), QPointF(r, r))
            g.setColorAt(0.0, M_EDGE); g.setColorAt(0.22, M_HI)
            g.setColorAt(0.55, M_DARK); g.setColorAt(0.82, M_MID); g.setColorAt(1.0, M_EDGE)
            p.setBrush(QBrush(g))
            p.setPen(QPen(QColor(0, 0, 0, 120), max(0.4, r * 0.0049)))
            p.drawPath(outer)

            p.setPen(QPen(QColor(255, 255, 255, 35), max(0.35, r * 0.0031)))
            p.drawPath(inner_path)
        finally:
            p.restore()

        # 2) Dial field
        p.save()
        try:
            p.setClipPath(inner_path)
            rg = QRadialGradient(QPointF(-r * 0.18, -r * 0.22), r * 1.25)
            rg.setColorAt(0.0, QColor(D_MID.red(), D_MID.green(), D_MID.blue(), 255))
            rg.setColorAt(0.55, QColor(D_DARK.red(), D_DARK.green(), D_DARK.blue(), 255))
            rg.setColorAt(1.0, QColor(max(0, D_DARK.red() - 3),
                                       max(0, D_DARK.green() - 2),
                                       max(0, D_DARK.blue() - 6), 255))
            p.fillRect(QRectF(-r, -r, 2 * r, 2 * r), QBrush(rg))

            # Texture
            if TEX == "guilloche":
                p.save()
                p.setClipPath(inner_path)
                step = max(2.0, r * 0.075)
                pen = QPen(QColor(255, 255, 255, 16))
                pen.setWidthF(max(0.8, r * 0.010)); pen.setCapStyle(Qt.RoundCap)
                p.setPen(pen)
                y = -r
                while y <= r:
                    p.drawLine(QPointF(-r, y), QPointF(r, y + r * 0.10))
                    p.drawLine(QPointF(-r, y), QPointF(r, y - r * 0.10))
                    y += step
                p.restore()
            elif TEX == "matte":
                self._draw_emboss(p, inner_path, r, t, 6)
            elif TEX == "sunburst":
                self._draw_emboss(p, inner_path, r, t, 12)
            elif TEX == "pulse":
                self._draw_emboss(p, inner_path, r, t, 10)
            else:
                self._draw_emboss(p, inner_path, r, t, 18)

            # Lift
            burst = QRadialGradient(QPointF(0, 0), r * 1.05)
            if TEX == "pulse":
                pulse_u = 0.5 + 0.5 * math.sin(t * 2.1)
                a0 = int(26 + 34 * pulse_u)
                burst.setColorAt(0.0, QColor(255, 92, 82, a0))
                burst.setColorAt(0.40, QColor(255, 92, 82, int(10 + 14 * pulse_u)))
            elif TEX == "matte":
                burst.setColorAt(0.0, QColor(tint.red(), tint.green(), tint.blue(), 22))
                burst.setColorAt(0.45, QColor(tint.red(), tint.green(), tint.blue(), 8))
            else:
                burst.setColorAt(0.0, QColor(tint.red(), tint.green(), tint.blue(), 46))
                burst.setColorAt(0.35, QColor(tint.red(), tint.green(), tint.blue(), 12))
            burst.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.fillRect(QRectF(-r, -r, 2 * r, 2 * r), QBrush(burst))
        finally:
            p.restore()

        # 3) Chapter ring + indices
        p.save()
        try:
            p.setClipPath(inner_path)
            rr = inner * 0.97
            pen = QPen(QColor(tint.red(), tint.green(), tint.blue(), 34))
            pen.setWidthF(max(0.6, inner * 0.0123))
            p.setPen(pen); p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(0, 0), rr, rr)

            self._draw_indices(p, inner_path, inner, accent)

            for i in range(DOTS_N):
                ang = -math.pi / 2 + i * (2 * math.pi / DOTS_N)
                rr0 = inner * 0.965
                x = rr0 * math.cos(ang); y = rr0 * math.sin(ang)
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(QColor(255, 255, 255, 40 if (i % 4) else 65)))
                d = max(1.2, inner * (0.020 if (i % 4) else 0.028))
                p.drawEllipse(QPointF(x, y), d * 0.5, d * 0.5)
        finally:
            p.restore()

        # 4) Glass rim
        p.save()
        try:
            p.setClipPath(inner_path)
            rr = inner * 0.90
            p.setPen(QPen(QColor(255, 255, 255, 22), max(0.6, inner * 0.0123)))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(0, 0), rr, rr)
        finally:
            p.restore()

        # 5) Content (subclass draws here)
        self.draw_content(p, inner, t, accent)

    @abstractmethod
    def draw_content(self, p: "QPainter", inner: float, t: float, accent: QColor) -> None:
        """Draw complication-specific content inside the prepared dial.

        Coordinate origin is (0,0) at center. *inner* is the usable radius.
        """

    def draw_overlay(self, p, cx, cy, mind, t, trans):
        """Draw fullscreen overlay. Override in subclasses with overlays."""

    # ----- Frame helpers (private) ----------------------------------------

    @staticmethod
    def _draw_emboss(p, clip_path, r, t, alpha):
        p.save()
        try:
            p.setClipPath(clip_path)
            step = max(2.0, r * 0.080)
            pen = QPen(QColor(255, 255, 255, alpha))
            pen.setWidthF(max(1.0, r * 0.012))
            p.setPen(pen)
            y = -r
            while y <= r:
                wob = 0.9 * math.sin((y / step) * 0.55 + t * 0.6)
                p.drawLine(QPointF(-r, y + wob), QPointF(r, y - wob))
                y += step
        finally:
            p.restore()

    @staticmethod
    def _draw_indices(p, clip_path, inner, accent=None):
        if accent is None:
            accent = BLUE_LIFT
        p.save()
        try:
            p.setClipPath(clip_path)
            for i in range(12):
                ang = -math.pi / 2 + i * (2 * math.pi / 12.0)
                major = (i % 3 == 0)
                rr0 = inner * (0.80 if major else 0.82)
                rr1 = inner * (0.94 if major else 0.93)
                x0 = rr0 * math.cos(ang); y0 = rr0 * math.sin(ang)
                x1 = rr1 * math.cos(ang); y1 = rr1 * math.sin(ang)

                penS = QPen(QColor(0, 0, 0, 70))
                penS.setWidthF(max(1.8, inner * (0.045 if major else 0.032)))
                penS.setCapStyle(Qt.RoundCap)
                p.setPen(penS)
                off = inner * 0.012
                p.drawLine(QPointF(x0 + off, y0 + off), QPointF(x1 + off, y1 + off))

                # Accent-tinted baton gradient
                baton_hi = QColor(
                    int(0.75 * STEEL_2.red() + 0.25 * accent.red()),
                    int(0.75 * STEEL_2.green() + 0.25 * accent.green()),
                    int(0.75 * STEEL_2.blue() + 0.25 * accent.blue()),
                )
                baton_mid = QColor(
                    int(0.80 * STEEL_1.red() + 0.20 * accent.red()),
                    int(0.80 * STEEL_1.green() + 0.20 * accent.green()),
                    int(0.80 * STEEL_1.blue() + 0.20 * accent.blue()),
                )
                grad = QLinearGradient(QPointF(x0, y0), QPointF(x1, y1))
                grad.setColorAt(0.0, baton_hi); grad.setColorAt(0.5, baton_mid)
                grad.setColorAt(1.0, QColor(95, 100, 110))
                penM = QPen(QBrush(grad), max(1.6, inner * (0.042 if major else 0.030)))
                penM.setCapStyle(Qt.RoundCap)
                p.setPen(penM)
                p.drawLine(QPointF(x0, y0), QPointF(x1, y1))
        finally:
            p.restore()

    # ----- Input ----------------------------------------------------------

    def on_tap(self) -> bool:
        if self.has_overlay:
            self.overlay_target = 0.0 if self.overlay_target > 0.5 else 1.0
            self.bus.emit("complication.toggle", name=self.name, opening=self.overlay_target > 0.5)
            return True
        return False

    def on_drag(self, dx: float, dy: float) -> bool:
        return False

    def on_bus_event(self, event: str, **kwargs) -> None:
        pass

    def open_overlay(self) -> None:
        self.overlay_target = 1.0
        self.bus.emit("complication.toggle", name=self.name, opening=True)

    def close_overlay(self) -> None:
        self.overlay_target = 0.0
        self.bus.emit("complication.toggle", name=self.name, opening=False)

    @property
    def overlay_open(self) -> bool:
        return self.overlay_trans > 0.01

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"

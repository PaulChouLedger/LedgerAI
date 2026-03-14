"""
gui.complications.settings -- Settings / Identity Seal complication + overlay.

Extracted from carbon_demo.py `_draw_comp_settings`.
Regulator layout with rotating tourbillon cage, three CFG/NET/SYS meter arcs,
and a MODE aperture window.
"""

from __future__ import annotations

import math
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt5.QtGui import QPainter

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QLinearGradient, QPen, QRadialGradient,
)

# Horology steel-blue tone (matches unified palette)
GOLD = lambda a=255: QColor(145, 175, 215, a)  # noqa: E731

from gui.complications.base import BaseComplication
from gui.renderer import clamp
from gui.auraconnect_page import draw_auraconnect_page, handle_auraconnect_tap
from gui.wifi_page import WifiPageState, draw_wifi_page, handle_wifi_tap


class SettingsComplication(BaseComplication):
    name = "Settings"
    label = "Settings"
    category = "System"
    has_overlay = True

    def __init__(self, bus):
        super().__init__(bus)
        self.owner_name = os.environ.get("AURA_OWNER_NAME", "Paul")
        self.owner_phone = os.environ.get("AURA_OWNER_PHONE", "+1 917 555 0137")
        self.emergency_enabled = True
        self._cfg_heat = 0.35
        self._net_flux = 0.25
        self._sys_load = 0.30
        self.settings_page = None       # None = main, "wifi" = WiFi config
        self._wifi_state = WifiPageState()
        # OTA update state
        self._updates_pending = False
        self._updates_count = 0
        bus.on("updates.available", self._on_updates_available)
        bus.on("updates.none", self._on_updates_none)
        bus.on("updates.applied", self._on_updates_none)

    def _on_updates_available(self, count=0, **kw):
        self._updates_pending = True
        self._updates_count = count

    def _on_updates_none(self, **kw):
        self._updates_pending = False
        self._updates_count = 0

    # ------------------------------------------------------------------
    def draw_glyph(self, p: "QPainter", size: float, t: float) -> None:
        """Override to add pulsing glow when OTA updates are pending."""
        super().draw_glyph(p, size, t)
        if self._updates_pending:
            r = size * 0.5
            pulse = 0.5 + 0.5 * math.sin(t * 2.5)
            # Subtle pulsing halo around the entire glyph
            p.setPen(Qt.NoPen)
            glow = QRadialGradient(QPointF(0, 0), r * 1.35)
            glow.setColorAt(0.70, QColor(0, 0, 0, 0))
            glow.setColorAt(0.85, QColor(180, 210, 255, int(30 + 50 * pulse)))
            glow.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setBrush(QBrush(glow))
            p.drawEllipse(QPointF(0, 0), r * 1.35, r * 1.35)

    # ------------------------------------------------------------------
    def draw_content(self, p: "QPainter", inner: float, t: float, accent: QColor) -> None:
        cfg = clamp(self._cfg_heat, 0.0, 1.0)
        net = clamp(self._net_flux, 0.0, 1.0)
        sysl = clamp(self._sys_load, 0.0, 1.0)

        # --- Curved "SETTINGS" along top arc ---
        if self._updates_pending:
            pulse = 0.5 + 0.5 * math.sin(t * 2.5)
            alpha = int(160 + 95 * pulse)
            txt_color = QColor(180, 215, 255, alpha)
        else:
            txt_color = QColor(225, 235, 250, 235)
        _draw_curved_text(p, "SETTINGS", inner * 0.58, top=True,
                          color=txt_color, inner=inner)

        # --- Central rotating cage (tourbillon) ---
        cage_r = inner * 0.34
        p.save()
        p.rotate((t * 26.0) % 360.0)

        # Brushed silver cage ring
        pen = QPen(QColor(170, 195, 225, 165))
        pen.setWidthF(max(1.6, inner * 0.030))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(0, 0), cage_r, cage_r)

        # Bridges
        for i in range(4):
            ang = (i / 4.0) * 2.0 * math.pi
            a = -ang + math.pi / 2
            x = cage_r * 0.92 * math.cos(a)
            y = cage_r * 0.92 * math.sin(a)
            p.drawLine(QPointF(0, 0), QPointF(x, y))

        # Micro teeth shimmer
        teeth = 14
        for i in range(teeth):
            ang = (2 * math.pi) * (i / teeth)
            a = -ang + math.pi / 2
            x1 = (cage_r * 0.92) * math.cos(a)
            y1 = (cage_r * 0.92) * math.sin(a)
            x2 = (cage_r * 1.06) * math.cos(a)
            y2 = (cage_r * 1.06) * math.sin(a)
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # Jewel hub
        hub = cage_r * 0.14
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 140)))
        p.drawEllipse(QPointF(0, 0), hub * 1.15, hub * 1.15)
        p.setBrush(QBrush(QColor(accent.red(), accent.green(), accent.blue(), 155)))
        p.drawEllipse(QPointF(0, 0), hub, hub)

        p.restore()

        # --- Three regulator arcs (CFG / NET / SYS) ---
        def meter(angle_center_deg: float, value: float, label: str, col: QColor):
            rr = inner * 0.74
            span = 64.0
            start = angle_center_deg - span * 0.5
            rect = QRectF(-rr, -rr, 2 * rr, 2 * rr)

            # Base arc
            base_pen = QPen(QColor(255, 255, 255, 26))
            base_pen.setWidthF(max(2.0, inner * 0.070))
            base_pen.setCapStyle(Qt.RoundCap)
            p.setPen(base_pen)
            p.drawArc(rect, int(start * 16), int(span * 16))

            # Value arc
            a = int(65 + 160 * value)
            pen2 = QPen(QColor(col.red(), col.green(), col.blue(), a))
            pen2.setWidthF(max(2.2, inner * (0.072 + 0.018 * value)))
            pen2.setCapStyle(Qt.RoundCap)
            p.setPen(pen2)
            p.drawArc(rect, int(start * 16), int((span * value) * 16))

            # Label
            if label:
                p.setPen(QColor(235, 242, 255, 195))
                f = QFont("Helvetica", max(7, int(inner * 0.14)))
                f.setBold(True)
                p.setFont(f)

                a_mid = math.radians(angle_center_deg - 90.0)
                tx = (inner * 0.42) * math.cos(a_mid)
                ty = (inner * 0.42) * math.sin(a_mid)
                p.drawText(QRectF(tx - inner * 0.26, ty - inner * 0.11,
                                  inner * 0.52, inner * 0.22),
                           Qt.AlignCenter, label)

        meter(310.0, cfg, "", accent)
        meter(90.0, net, "", QColor(accent.red() + 15, accent.green() + 10, accent.blue(), accent.alpha()))
        meter(230.0, sysl, "", QColor(accent.red() - 10, accent.green() + 5, accent.blue() + 5, accent.alpha()))

        # --- OTA update badge (pulsing amber dot at 2 o'clock) ---
        if self._updates_pending:
            badge_r = inner * 0.10
            badge_ang = math.radians(-45)
            bx = inner * 0.52 * math.cos(badge_ang)
            by = inner * 0.52 * math.sin(badge_ang)
            pulse = 0.5 + 0.5 * math.sin(t * 3.0)
            # Halo
            p.setPen(Qt.NoPen)
            halo = QRadialGradient(QPointF(bx, by), badge_r * 3.0)
            halo.setColorAt(0.0, QColor(255, 180, 50, int(60 + 40 * pulse)))
            halo.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setBrush(QBrush(halo))
            p.drawEllipse(QPointF(bx, by), badge_r * 3.0, badge_r * 3.0)
            # Badge
            p.setBrush(QColor(255, 180, 50, int(200 + 55 * pulse)))
            p.drawEllipse(QPointF(bx, by), badge_r, badge_r)
            # Count text
            if self._updates_count > 0:
                bf = QFont("DejaVu Sans", max(5, int(inner * 0.10)))
                bf.setBold(True)
                p.setFont(bf)
                p.setPen(QColor(20, 10, 0))
                p.drawText(QRectF(bx - badge_r, by - badge_r,
                                  badge_r * 2, badge_r * 2),
                           Qt.AlignCenter, str(self._updates_count))

    # ------------------------------------------------------------------
    def draw_overlay(self, p, cx, cy, mind, t, trans):
        """Settings identity-seal overlay: enamel plate, guilloché, name cartouche, emergency jewel."""
        trans = clamp(trans, 0.0, 1.0)
        if trans <= 0.0:
            return

        p.save()
        try:
            p.setRenderHint(p.Antialiasing, True)

            # Sub-page routing
            page = getattr(self, "settings_page", None)
            if page == "wifi":
                draw_wifi_page(p, cx, cy, mind, t, trans, self._wifi_state)
                p.restore()
                return
            elif page == "auraconnect":
                draw_auraconnect_page(p, cx, cy, mind, t, trans)
                p.restore()
                return
            elif page == "profile":
                _draw_profile_page(p, cx, cy, mind, t, trans, self)
                p.restore()
                return
            elif page == "alerts":
                _draw_alerts_page(p, cx, cy, mind, t, trans, self)
                p.restore()
                return
            elif page == "updates":
                _draw_updates_page(p, cx, cy, mind, t, trans, self)
                p.restore()
                return

            # ----- Sizing (matches volume/balance class) -----
            R = mind * 0.235

            r_bezel_outer = R * 0.98
            r_bezel_inner = R * 0.90
            r_chapter_out = R * 0.84
            r_chapter_in  = R * 0.74
            rg = R * 0.64  # guilloché field

            cart_w = R * 1.28
            cart_h = R * 0.40

            # ----- Alpha / colors -----
            A  = int(240 * trans)
            A2 = int(175 * trans)
            A3 = int(120 * trans)

            gold_strong = GOLD(A)
            gold_mid    = GOLD(A2)
            gold_faint  = GOLD(A3)

            base_dark = QColor(8, 9, 12, int(210 * trans))
            mid_dark  = QColor(14, 15, 20, int(200 * trans))

            # ----- Helpers -----
            def draw_radial_glow(radius, alpha):
                grad = QRadialGradient(QPointF(cx, cy), radius)
                grad.setColorAt(0.0, QColor(255, 255, 255, int(alpha * 0.12)))
                grad.setColorAt(0.55, QColor(255, 255, 255, int(alpha * 0.06)))
                grad.setColorAt(1.0, QColor(0, 0, 0, 0))
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(grad))
                p.drawEllipse(QPointF(cx, cy), radius, radius)

            def draw_bezel_depth():
                # Outer bevel highlight
                g1 = QRadialGradient(QPointF(cx, cy), r_bezel_outer)
                g1.setColorAt(0.70, QColor(255, 240, 200, int(34 * trans)))
                g1.setColorAt(0.86, QColor(255, 240, 200, int(62 * trans)))
                g1.setColorAt(1.00, QColor(0, 0, 0, 0))
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(g1))
                p.drawEllipse(QPointF(cx, cy), r_bezel_outer, r_bezel_outer)

                # Inner chamfer shadow
                g2 = QRadialGradient(QPointF(cx, cy), r_bezel_inner)
                g2.setColorAt(0.60, QColor(0, 0, 0, 0))
                g2.setColorAt(0.92, QColor(0, 0, 0, int(80 * trans)))
                g2.setColorAt(1.00, QColor(0, 0, 0, int(110 * trans)))
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(g2))
                p.drawEllipse(QPointF(cx, cy), r_bezel_inner, r_bezel_inner)

            def draw_arc_text(text, radius, angle_deg, font, color, letter_spacing_deg=6.0):
                """Draw text along an arc centered at (cx, cy)."""
                if not text:
                    return
                p.save()
                p.setFont(font)
                n = len(text)
                span = (n - 1) * letter_spacing_deg if n > 1 else 0.0
                start = angle_deg - span * 0.5
                shadow_col = QColor(0, 0, 0, int(150 * trans))

                for i, ch in enumerate(text):
                    a = math.radians(start + i * letter_spacing_deg)
                    x = cx + radius * math.cos(a)
                    y = cy + radius * math.sin(a)
                    p.save()
                    p.translate(x, y)
                    rot = math.degrees(a) + 90.0
                    p.rotate(rot)
                    rect = QRectF(-50, -18, 100, 36)
                    p.setPen(shadow_col)
                    p.drawText(rect.translated(0, 1), Qt.AlignCenter, ch)
                    p.setPen(color)
                    p.drawText(rect, Qt.AlignCenter, ch)
                    p.restore()
                p.restore()

            # =========================================================
            # Base plate (deep enamel feel)
            # =========================================================
            p.setPen(Qt.NoPen)
            p.setBrush(base_dark)
            p.drawEllipse(QPointF(cx, cy), R, R)

            # Crystal gloss / reflection
            draw_radial_glow(R * 0.98, A)

            # Depth bevels
            draw_bezel_depth()

            # Gold bezel rings
            bezel_pen = QPen(gold_mid)
            bezel_pen.setWidthF(max(2.0, mind * 0.0042))
            p.setPen(bezel_pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx, cy), r_bezel_outer, r_bezel_outer)

            inner_pen = QPen(gold_faint)
            inner_pen.setWidthF(max(1.2, mind * 0.0026))
            p.setPen(inner_pen)
            p.drawEllipse(QPointF(cx, cy), r_bezel_inner, r_bezel_inner)

            # =========================================================
            # Chapter ring (60 fine ticks)
            # =========================================================
            for i in range(60):
                ang = (i / 60.0) * 2.0 * math.pi - math.pi / 2.0
                is_major = (i % 5 == 0)
                tick_len = (R * 0.080) if is_major else (R * 0.045)
                tick_w = (mind * 0.0038) if is_major else (mind * 0.0022)

                rr_out = r_chapter_out
                rr_in = rr_out - tick_len
                x1 = cx + rr_in * math.cos(ang)
                y1 = cy + rr_in * math.sin(ang)
                x2 = cx + rr_out * math.cos(ang)
                y2 = cy + rr_out * math.sin(ang)

                col = GOLD(int((170 if is_major else 115) * trans))
                p.setPen(QPen(col, tick_w, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

            # Inner chapter ring boundary
            p.setPen(QPen(gold_faint, max(1.0, mind * 0.0018)))
            p.drawEllipse(QPointF(cx, cy), r_chapter_in, r_chapter_in)

            # =========================================================
            # Guilloché center (engine-turned concentric ellipses)
            # =========================================================
            p.setPen(Qt.NoPen)
            p.setBrush(mid_dark)
            p.drawEllipse(QPointF(cx, cy), rg, rg)

            gu_pen = QPen(QColor(255, 255, 255, int(24 * trans)))
            gu_pen.setWidthF(max(1.0, mind * 0.0016))
            p.setPen(gu_pen)
            p.setBrush(Qt.NoBrush)

            steps = 46
            for k in range(steps):
                frac = k / (steps - 1) if steps > 1 else 0.0
                rad = rg * (0.18 + 0.82 * frac)
                wob = 1.0 + 0.018 * math.sin(t * 0.9 + k * 0.55)
                p.drawEllipse(QPointF(cx, cy), rad * wob, rad)

            # Center cap
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, int(72 * trans)))
            p.drawEllipse(QPointF(cx, cy), rg * 0.10, rg * 0.10)

            # =========================================================
            # Signature microtext (12 o'clock)
            # =========================================================
            micro_font = QFont("DejaVu Sans", max(8, int(mind * 0.014)))
            micro_font.setLetterSpacing(QFont.PercentageSpacing, 108)
            p.setFont(micro_font)
            p.setPen(gold_faint)
            p.drawText(
                int(cx - R), int(cy - R * 0.95), int(2 * R), int(R * 0.22),
                Qt.AlignCenter, "AURA  \u2022  SETTINGS"
            )

            # =========================================================
            # Perimeter menu (curved arc labels)
            # =========================================================
            menu_font = QFont("DejaVu Sans", max(9, int(mind * 0.016)))
            menu_font.setBold(True)
            menu_font.setLetterSpacing(QFont.PercentageSpacing, 112)

            menu_r = (r_chapter_in + r_chapter_out) * 0.5
            menu_col = QColor(255, 244, 220, int(220 * trans))

            labels = [
                ("WI-FI",        -90.0),
                ("AURACONNECT",  -18.0),
                ("PROFILE",       54.0),
                ("ALERTS",       126.0),
                ("UPDATES",      198.0),
            ]
            # Flash the UPDATES label when updates are pending
            updates_flash = self._updates_pending
            for txt, a_deg in labels:
                if txt == "UPDATES" and updates_flash:
                    flash_pulse = 0.5 + 0.5 * math.sin(t * 3.5)
                    flash_col = QColor(255, 200, 80, int((160 + 95 * flash_pulse) * trans))
                    draw_arc_text(txt, menu_r, a_deg - 90.0, menu_font, flash_col,
                                  letter_spacing_deg=7.0)
                else:
                    draw_arc_text(txt, menu_r, a_deg - 90.0, menu_font, menu_col,
                                  letter_spacing_deg=7.0)

            # =========================================================
            # Name cartouche (applied plaque)
            # =========================================================
            cart_rect = QRectF(cx - cart_w / 2.0, cy - cart_h * 0.62, cart_w, cart_h)

            # Shadow
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, int(85 * trans)))
            p.drawRoundedRect(cart_rect.translated(0, mind * 0.004),
                              cart_h * 0.18, cart_h * 0.18)

            # Plate gradient
            plate_grad = QLinearGradient(cart_rect.topLeft(), cart_rect.bottomLeft())
            plate_grad.setColorAt(0.0, QColor(34, 34, 40, int(220 * trans)))
            plate_grad.setColorAt(0.5, QColor(18, 18, 22, int(220 * trans)))
            plate_grad.setColorAt(1.0, QColor(10, 10, 12, int(220 * trans)))

            p.setBrush(QBrush(plate_grad))
            p.setPen(QPen(gold_faint, max(1.2, mind * 0.0022)))
            p.drawRoundedRect(cart_rect, cart_h * 0.18, cart_h * 0.18)

            # Owner name
            name = str(self.owner_name)
            name_font = QFont("DejaVu Sans", max(12, int(mind * 0.040)))
            name_font.setBold(True)
            p.setFont(name_font)
            p.setPen(gold_strong)
            p.drawText(cart_rect, Qt.AlignCenter, name)

            # Phone line
            phone = str(self.owner_phone)
            phone_font = QFont("DejaVu Sans", max(10, int(mind * 0.022)))
            phone_font.setLetterSpacing(QFont.PercentageSpacing, 112)
            p.setFont(phone_font)
            p.setPen(gold_mid)
            p.drawText(
                int(cx - R), int(cy + R * 0.22), int(2 * R), int(R * 0.25),
                Qt.AlignCenter, phone
            )

            # Micro divider
            p.setPen(QPen(gold_faint, max(1.0, mind * 0.0016)))
            p.drawLine(
                QPointF(cx - R * 0.52, cy + R * 0.13),
                QPointF(cx + R * 0.52, cy + R * 0.13)
            )

            # =========================================================
            # Emergency jewel at 6 o'clock
            # =========================================================
            dot_r = mind * 0.011
            dot_x = cx
            dot_y = cy + R * 0.80

            if self.emergency_enabled:
                pulse = 0.55 + 0.45 * math.sin(time.time() * 2.0)
                dot_a = int((120 + 110 * pulse) * trans)
                halo_a = int((65 + 55 * pulse) * trans)
            else:
                dot_a = int(70 * trans)
                halo_a = int(20 * trans)

            # Halo glow
            halo = QRadialGradient(QPointF(dot_x, dot_y), dot_r * 3.2)
            halo.setColorAt(0.0, QColor(255, 215, 140, halo_a))
            halo.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(halo))
            p.drawEllipse(QPointF(dot_x, dot_y), dot_r * 3.2, dot_r * 3.2)

            # Jewel body
            p.setBrush(GOLD(dot_a))
            p.drawEllipse(QPointF(dot_x, dot_y), dot_r, dot_r)

            # Highlight spec
            p.setBrush(QColor(255, 255, 255, int(90 * trans)))
            p.drawEllipse(QPointF(dot_x - dot_r * 0.35, dot_y - dot_r * 0.35),
                          dot_r * 0.28, dot_r * 0.28)

            # =========================================================
            # Ultra-subtle usage hints (only near fully open)
            # =========================================================
            if trans > 0.72:
                hint_font = QFont("DejaVu Sans", max(8, int(mind * 0.012)))
                hint_font.setLetterSpacing(QFont.PercentageSpacing, 120)
                p.setFont(hint_font)

                a = int(45 * (trans - 0.72) / 0.28)
                a = max(0, min(45, a))

                p.setPen(QColor(255, 215, 140, a))
                p.drawText(
                    int(cx - R), int(cy - R * 0.08), int(R * 0.95), int(R * 0.18),
                    Qt.AlignRight | Qt.AlignVCenter, "TAP NAME"
                )
                p.drawText(
                    int(cx), int(cy - R * 0.08), int(R * 0.95), int(R * 0.18),
                    Qt.AlignLeft | Qt.AlignVCenter, "TAP NUMBER"
                )

        finally:
            p.restore()


    # ------------------------------------------------------------------
    # Overlay tap handling
    # ------------------------------------------------------------------

    def handle_overlay_tap(self, x, y, cx, cy, mind):
        """Handle a tap inside the settings overlay area.

        Returns True if the tap was consumed, False to let it propagate.
        """
        R = mind * 0.235

        # Sub-pages: check if tap is outside the sub-page circle → go back
        if self.settings_page in ("wifi", "auraconnect", "profile", "alerts", "updates"):
            from gui.wifi_page import _WIFI_R_FRAC
            sub_r = mind * _WIFI_R_FRAC if self.settings_page == "wifi" else R
            dx = x - cx
            dy = y - cy
            dist = math.sqrt(dx * dx + dy * dy)
            # Tap outside sub-page circle → back to settings main
            if dist > sub_r * 1.05:
                self.settings_page = None
                return True

        # WiFi sub-page active — delegate to WiFi tap handler
        if self.settings_page == "wifi":
            action = handle_wifi_tap(x, y, cx, cy, mind, self._wifi_state)
            if action == "back":
                self.settings_page = None
            return True  # always consume taps when WiFi page is active

        # AuraConnect sub-page
        if self.settings_page == "auraconnect":
            action = handle_auraconnect_tap(x, y, cx, cy, mind)
            if action == "back":
                self.settings_page = None
            return True

        # Profile sub-page
        if self.settings_page == "profile":
            # Back tap: upper 40% of overlay area
            if y < cy - R * 0.3:
                self.settings_page = None
            return True

        # Alerts sub-page
        if self.settings_page == "alerts":
            if y < cy - R * 0.3:
                self.settings_page = None
            return True

        # Updates sub-page
        if self.settings_page == "updates":
            from core.updater import updater
            # "Apply Update" button — lower 30%
            if y > cy + R * 0.3 and updater.available:
                updater.apply_update()
                return True
            # Back tap — upper 40%
            if y < cy - R * 0.3:
                self.settings_page = None
            return True

        # Main settings page — check menu item taps (5 items)
        r_chapter_in = R * 0.74
        r_chapter_out = R * 0.84

        dx = x - cx
        dy = y - cy
        dist = math.sqrt(dx * dx + dy * dy)

        if r_chapter_in * 0.8 < dist < r_chapter_out * 1.2:
            tap_angle = math.degrees(math.atan2(dy, dx))

            # Angles match the labels array: (a_deg - 90) for atan2 coords
            menu_items = [
                ("wifi",        -90.0 - 90.0),
                ("auraconnect", -18.0 - 90.0),
                ("profile",      54.0 - 90.0),
                ("alerts",      126.0 - 90.0),
                ("updates",     198.0 - 90.0),
            ]

            for page_name, item_angle in menu_items:
                diff = tap_angle - item_angle
                while diff > 180:
                    diff -= 360
                while diff < -180:
                    diff += 360

                if abs(diff) < 28:
                    if page_name == "wifi":
                        self.settings_page = "wifi"
                        self._wifi_state.trigger_scan()
                        return True
                    self.settings_page = page_name
                    return True

        return False


# ---------------------------------------------------------------------------
# Profile sub-page renderer
# ---------------------------------------------------------------------------

def _draw_profile_page(p, cx, cy, mind, t, trans, comp):
    """Draw the Profile sub-page: avatar, name, phone, voice status, user ID."""
    from core.state import state

    R = mind * 0.235
    base_dark = QColor(8, 9, 12, int(210 * trans))
    A  = int(240 * trans)
    A2 = int(175 * trans)
    A3 = int(120 * trans)
    gold_strong = GOLD(A)
    gold_mid    = GOLD(A2)
    gold_faint  = GOLD(A3)

    # --- Base plate ---
    p.setPen(Qt.NoPen)
    p.setBrush(base_dark)
    p.drawEllipse(QPointF(cx, cy), R, R)

    # Bezel ring
    bezel_pen = QPen(gold_mid)
    bezel_pen.setWidthF(max(2.0, mind * 0.0042))
    p.setPen(bezel_pen)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(cx, cy), R * 0.98, R * 0.98)

    # Inner ring
    inner_pen = QPen(gold_faint)
    inner_pen.setWidthF(max(1.2, mind * 0.0026))
    p.setPen(inner_pen)
    p.drawEllipse(QPointF(cx, cy), R * 0.90, R * 0.90)

    # --- Chapter ring (subtle, 30 ticks) ---
    r_chapter_out = R * 0.84
    for i in range(30):
        ang = (i / 30.0) * 2.0 * math.pi - math.pi / 2.0
        is_major = (i % 5 == 0)
        tick_len = (R * 0.055) if is_major else (R * 0.030)
        tick_w = (mind * 0.0030) if is_major else (mind * 0.0018)
        rr_out = r_chapter_out
        rr_in = rr_out - tick_len
        x1 = cx + rr_in * math.cos(ang)
        y1 = cy + rr_in * math.sin(ang)
        x2 = cx + rr_out * math.cos(ang)
        y2 = cy + rr_out * math.sin(ang)
        col = GOLD(int((140 if is_major else 80) * trans))
        p.setPen(QPen(col, tick_w, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    # --- Guilloché (subtle concentric ellipses) ---
    rg = R * 0.64
    gu_pen = QPen(QColor(255, 255, 255, int(16 * trans)))
    gu_pen.setWidthF(max(1.0, mind * 0.0014))
    p.setPen(gu_pen)
    p.setBrush(Qt.NoBrush)
    for k in range(20):
        frac = k / 19.0
        rad = rg * (0.25 + 0.75 * frac)
        wob = 1.0 + 0.015 * math.sin(t * 0.7 + k * 0.6)
        p.drawEllipse(QPointF(cx, cy), rad * wob, rad)

    # --- "PROFILE" header ---
    header_font = QFont("DejaVu Sans", max(8, int(mind * 0.014)))
    header_font.setLetterSpacing(QFont.PercentageSpacing, 130)
    header_font.setBold(True)
    p.setFont(header_font)
    p.setPen(gold_faint)
    p.drawText(
        int(cx - R), int(cy - R * 0.92), int(2 * R), int(R * 0.20),
        Qt.AlignCenter, "PROFILE"
    )

    # --- Avatar circle with initials ---
    avatar_r = R * 0.18
    avatar_y = cy - R * 0.42

    # Dark circle
    p.setPen(QPen(gold_mid, max(1.5, mind * 0.003)))
    p.setBrush(QColor(18, 19, 24, int(230 * trans)))
    p.drawEllipse(QPointF(cx, avatar_y), avatar_r, avatar_r)

    # Initials
    name = str(comp.owner_name)
    parts = name.split()
    initials = "".join(w[0].upper() for w in parts if w)[:2]
    if not initials:
        initials = "?"
    init_font = QFont("DejaVu Sans", max(14, int(mind * 0.034)))
    init_font.setBold(True)
    p.setFont(init_font)
    p.setPen(gold_strong)
    p.drawText(
        QRectF(cx - avatar_r, avatar_y - avatar_r, avatar_r * 2, avatar_r * 2),
        Qt.AlignCenter, initials
    )

    # --- Owner name (large, gold) ---
    name_font = QFont("DejaVu Sans", max(12, int(mind * 0.036)))
    name_font.setBold(True)
    p.setFont(name_font)
    p.setPen(gold_strong)
    p.drawText(
        int(cx - R), int(cy - R * 0.18), int(2 * R), int(R * 0.20),
        Qt.AlignCenter, name
    )

    # --- Phone number (smaller, muted) ---
    phone_font = QFont("DejaVu Sans", max(9, int(mind * 0.020)))
    phone_font.setLetterSpacing(QFont.PercentageSpacing, 110)
    p.setFont(phone_font)
    p.setPen(gold_mid)
    p.drawText(
        int(cx - R), int(cy - R * 0.02), int(2 * R), int(R * 0.16),
        Qt.AlignCenter, str(comp.owner_phone)
    )

    # --- Divider ---
    p.setPen(QPen(gold_faint, max(1.0, mind * 0.0016)))
    p.drawLine(
        QPointF(cx - R * 0.48, cy + R * 0.18),
        QPointF(cx + R * 0.48, cy + R * 0.18)
    )

    # --- Voice Profile status ---
    row_y = cy + R * 0.28
    dot_r = mind * 0.008
    user_id = getattr(state, "active_user_id", None)
    enrolled = user_id is not None and str(user_id) != ""

    # Green dot
    dot_col = QColor(80, 210, 120, int(220 * trans)) if enrolled else QColor(120, 120, 130, int(160 * trans))
    p.setPen(Qt.NoPen)
    p.setBrush(dot_col)
    p.drawEllipse(QPointF(cx - R * 0.38, row_y), dot_r, dot_r)

    label_font = QFont("DejaVu Sans", max(9, int(mind * 0.018)))
    p.setFont(label_font)
    p.setPen(gold_mid)
    p.drawText(
        int(cx - R * 0.32), int(row_y - R * 0.06), int(R * 0.80), int(R * 0.12),
        Qt.AlignLeft | Qt.AlignVCenter, "Voice Profile"
    )

    status_text = "Enrolled" if enrolled else "Not Enrolled"
    status_col = QColor(80, 210, 120, int(200 * trans)) if enrolled else QColor(180, 140, 100, int(160 * trans))
    p.setPen(status_col)
    p.drawText(
        int(cx - R * 0.10), int(row_y - R * 0.06), int(R * 0.55), int(R * 0.12),
        Qt.AlignRight | Qt.AlignVCenter, status_text
    )

    # --- User ID ---
    row_y2 = cy + R * 0.46
    p.setPen(gold_mid)
    p.drawText(
        int(cx - R * 0.38), int(row_y2 - R * 0.06), int(R * 0.40), int(R * 0.12),
        Qt.AlignLeft | Qt.AlignVCenter, "User ID"
    )

    uid_str = str(user_id)[:12] if user_id else "\u2014"
    p.setPen(gold_faint)
    id_font = QFont("DejaVu Sans Mono", max(8, int(mind * 0.015)))
    p.setFont(id_font)
    p.drawText(
        int(cx - R * 0.10), int(row_y2 - R * 0.06), int(R * 0.55), int(R * 0.12),
        Qt.AlignRight | Qt.AlignVCenter, uid_str
    )

    # --- Back hint ---
    if trans > 0.72:
        hint_font = QFont("DejaVu Sans", max(7, int(mind * 0.011)))
        hint_font.setLetterSpacing(QFont.PercentageSpacing, 120)
        p.setFont(hint_font)
        a = int(40 * (trans - 0.72) / 0.28)
        a = max(0, min(40, a))
        p.setPen(QColor(255, 215, 140, a))
        p.drawText(
            int(cx - R), int(cy + R * 0.78), int(2 * R), int(R * 0.14),
            Qt.AlignCenter, "\u25C0  BACK"
        )


# ---------------------------------------------------------------------------
# Alerts sub-page renderer
# ---------------------------------------------------------------------------

def _draw_alerts_page(p, cx, cy, mind, t, trans, comp):
    """Draw the Alerts sub-page: toggle rows for system alerts, health checks, SOS."""
    R = mind * 0.235
    base_dark = QColor(8, 9, 12, int(210 * trans))
    A  = int(240 * trans)
    A2 = int(175 * trans)
    A3 = int(120 * trans)
    gold_strong = GOLD(A)
    gold_mid    = GOLD(A2)
    gold_faint  = GOLD(A3)

    # --- Base plate ---
    p.setPen(Qt.NoPen)
    p.setBrush(base_dark)
    p.drawEllipse(QPointF(cx, cy), R, R)

    # Bezel ring
    bezel_pen = QPen(gold_mid)
    bezel_pen.setWidthF(max(2.0, mind * 0.0042))
    p.setPen(bezel_pen)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(cx, cy), R * 0.98, R * 0.98)

    # Inner ring
    inner_pen = QPen(gold_faint)
    inner_pen.setWidthF(max(1.2, mind * 0.0026))
    p.setPen(inner_pen)
    p.drawEllipse(QPointF(cx, cy), R * 0.90, R * 0.90)

    # --- Chapter ring (subtle, 30 ticks) ---
    r_chapter_out = R * 0.84
    for i in range(30):
        ang = (i / 30.0) * 2.0 * math.pi - math.pi / 2.0
        is_major = (i % 5 == 0)
        tick_len = (R * 0.055) if is_major else (R * 0.030)
        tick_w = (mind * 0.0030) if is_major else (mind * 0.0018)
        rr_out = r_chapter_out
        rr_in = rr_out - tick_len
        x1 = cx + rr_in * math.cos(ang)
        y1 = cy + rr_in * math.sin(ang)
        x2 = cx + rr_out * math.cos(ang)
        y2 = cy + rr_out * math.sin(ang)
        col = GOLD(int((140 if is_major else 80) * trans))
        p.setPen(QPen(col, tick_w, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    # --- Guilloché (subtle concentric ellipses) ---
    rg = R * 0.64
    gu_pen = QPen(QColor(255, 255, 255, int(16 * trans)))
    gu_pen.setWidthF(max(1.0, mind * 0.0014))
    p.setPen(gu_pen)
    p.setBrush(Qt.NoBrush)
    for k in range(20):
        frac = k / 19.0
        rad = rg * (0.25 + 0.75 * frac)
        wob = 1.0 + 0.015 * math.sin(t * 0.7 + k * 0.6)
        p.drawEllipse(QPointF(cx, cy), rad * wob, rad)

    # --- "ALERT SETTINGS" header ---
    header_font = QFont("DejaVu Sans", max(8, int(mind * 0.014)))
    header_font.setLetterSpacing(QFont.PercentageSpacing, 130)
    header_font.setBold(True)
    p.setFont(header_font)
    p.setPen(gold_faint)
    p.drawText(
        int(cx - R), int(cy - R * 0.92), int(2 * R), int(R * 0.20),
        Qt.AlignCenter, "ALERT SETTINGS"
    )

    # --- Toggle rows ---
    label_font = QFont("DejaVu Sans", max(9, int(mind * 0.018)))
    dot_r = mind * 0.008

    toggles = [
        ("System Alerts",  True),
        ("Health Checks",  True),
        ("Emergency SOS",  comp.emergency_enabled),
    ]

    base_y = cy - R * 0.42
    row_spacing = R * 0.28

    for idx, (label, enabled) in enumerate(toggles):
        row_y = base_y + idx * row_spacing

        # Status dot
        if enabled:
            dot_col = QColor(80, 210, 120, int(220 * trans))
        else:
            dot_col = QColor(210, 80, 80, int(220 * trans))
        p.setPen(Qt.NoPen)
        p.setBrush(dot_col)
        p.drawEllipse(QPointF(cx - R * 0.42, row_y), dot_r, dot_r)

        # Label
        p.setFont(label_font)
        p.setPen(gold_mid)
        p.drawText(
            int(cx - R * 0.35), int(row_y - R * 0.06), int(R * 0.55), int(R * 0.12),
            Qt.AlignLeft | Qt.AlignVCenter, label
        )

        # ON / OFF value
        status = "ON" if enabled else "OFF"
        status_col = QColor(80, 210, 120, int(200 * trans)) if enabled else QColor(210, 80, 80, int(200 * trans))
        p.setPen(status_col)
        val_font = QFont("DejaVu Sans", max(8, int(mind * 0.016)))
        val_font.setBold(True)
        p.setFont(val_font)
        p.drawText(
            int(cx + R * 0.10), int(row_y - R * 0.06), int(R * 0.38), int(R * 0.12),
            Qt.AlignRight | Qt.AlignVCenter, status
        )

        # Subtle separator after each row (except last)
        if idx < len(toggles) - 1:
            sep_y = row_y + row_spacing * 0.5
            p.setPen(QPen(GOLD(int(50 * trans)), max(0.8, mind * 0.0012)))
            p.drawLine(
                QPointF(cx - R * 0.42, sep_y),
                QPointF(cx + R * 0.42, sep_y)
            )

    # --- Divider before notification sound ---
    div_y = base_y + len(toggles) * row_spacing - row_spacing * 0.35
    p.setPen(QPen(gold_faint, max(1.0, mind * 0.0016)))
    p.drawLine(
        QPointF(cx - R * 0.48, div_y),
        QPointF(cx + R * 0.48, div_y)
    )

    # --- Notification Sound row ---
    sound_y = base_y + len(toggles) * row_spacing
    p.setFont(label_font)
    p.setPen(gold_mid)
    p.drawText(
        int(cx - R * 0.42), int(sound_y - R * 0.06), int(R * 0.60), int(R * 0.12),
        Qt.AlignLeft | Qt.AlignVCenter, "Notification Sound"
    )

    p.setPen(gold_faint)
    val_font2 = QFont("DejaVu Sans", max(8, int(mind * 0.016)))
    p.setFont(val_font2)
    p.drawText(
        int(cx + R * 0.05), int(sound_y - R * 0.06), int(R * 0.42), int(R * 0.12),
        Qt.AlignRight | Qt.AlignVCenter, "Default"
    )

    # --- Back hint ---
    if trans > 0.72:
        hint_font = QFont("DejaVu Sans", max(7, int(mind * 0.011)))
        hint_font.setLetterSpacing(QFont.PercentageSpacing, 120)
        p.setFont(hint_font)
        a = int(40 * (trans - 0.72) / 0.28)
        a = max(0, min(40, a))
        p.setPen(QColor(255, 215, 140, a))
        p.drawText(
            int(cx - R), int(cy + R * 0.78), int(2 * R), int(R * 0.14),
            Qt.AlignCenter, "\u25C0  BACK"
        )


# ---------------------------------------------------------------------------
# Updates sub-page renderer
# ---------------------------------------------------------------------------

def _draw_updates_page(p, cx, cy, mind, t, trans, comp):
    """Draw the Updates sub-page: commit list + apply button."""
    from core.updater import updater
    from PyQt5.QtGui import QRadialGradient

    R = mind * 0.235
    base_dark = QColor(8, 9, 12, int(210 * trans))
    A  = int(240 * trans)
    A2 = int(175 * trans)
    A3 = int(120 * trans)
    gold_strong = GOLD(A)
    gold_mid    = GOLD(A2)
    gold_faint  = GOLD(A3)

    # --- Base plate ---
    p.setPen(Qt.NoPen)
    p.setBrush(base_dark)
    p.drawEllipse(QPointF(cx, cy), R, R)

    # Bezel ring
    bezel_pen = QPen(gold_mid)
    bezel_pen.setWidthF(max(2.0, mind * 0.0042))
    p.setPen(bezel_pen)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(cx, cy), R * 0.98, R * 0.98)

    # Inner ring
    inner_pen = QPen(gold_faint)
    inner_pen.setWidthF(max(1.2, mind * 0.0026))
    p.setPen(inner_pen)
    p.drawEllipse(QPointF(cx, cy), R * 0.90, R * 0.90)

    # --- Header ---
    header_font = QFont("DejaVu Sans", max(8, int(mind * 0.014)))
    header_font.setLetterSpacing(QFont.PercentageSpacing, 130)
    header_font.setBold(True)
    p.setFont(header_font)

    if updater.applying:
        p.setPen(QColor(255, 200, 80, A))
        p.drawText(
            int(cx - R), int(cy - R * 0.92), int(2 * R), int(R * 0.20),
            Qt.AlignCenter, "UPDATING..."
        )
        # Spinner
        spin_r = R * 0.15
        spin_ang = (t * 180.0) % 360.0
        spin_pen = QPen(QColor(255, 200, 80, A2))
        spin_pen.setWidthF(max(2.0, mind * 0.004))
        spin_pen.setCapStyle(Qt.RoundCap)
        p.setPen(spin_pen)
        p.setBrush(Qt.NoBrush)
        p.drawArc(
            QRectF(cx - spin_r, cy - spin_r, spin_r * 2, spin_r * 2),
            int(spin_ang * 16), int(120 * 16),
        )
        return

    commits = updater.commits
    n = len(commits)

    if n == 0:
        # No updates
        p.setPen(gold_faint)
        p.drawText(
            int(cx - R), int(cy - R * 0.92), int(2 * R), int(R * 0.20),
            Qt.AlignCenter, "SOFTWARE  UPDATES"
        )
        check_font = QFont("DejaVu Sans", max(9, int(mind * 0.017)))
        p.setFont(check_font)
        p.setPen(QColor(80, 210, 120, A2))
        p.drawText(
            int(cx - R), int(cy - R * 0.15), int(2 * R), int(R * 0.30),
            Qt.AlignCenter, "Up to date"
        )
    else:
        # Updates available
        flash = 0.5 + 0.5 * math.sin(t * 2.5)
        p.setPen(QColor(255, 200, 80, int((180 + 75 * flash) * trans)))
        p.drawText(
            int(cx - R), int(cy - R * 0.92), int(2 * R), int(R * 0.20),
            Qt.AlignCenter, f"{n}  UPDATE{'S' if n > 1 else ''}"
        )

        # Divider
        div_y = cy - R * 0.68
        p.setPen(QPen(gold_faint, max(1.0, mind * 0.0015)))
        p.drawLine(QPointF(cx - R * 0.55, div_y), QPointF(cx + R * 0.55, div_y))

        # Commit list (show up to 4)
        commit_font = QFont("DejaVu Sans", max(7, int(mind * 0.012)))
        commit_font.setBold(True)
        date_font = QFont("DejaVu Sans", max(6, int(mind * 0.010)))

        base_y = cy - R * 0.58
        row_h = R * 0.22
        max_show = min(n, 4)

        for idx in range(max_show):
            c = commits[idx]
            row_y = base_y + idx * row_h

            # Staggered reveal
            delay = idx * 0.08
            row_trans = clamp((trans - delay) / max(0.01, 1.0 - delay), 0.0, 1.0)
            if row_trans <= 0.0:
                continue

            # Hash badge
            p.setFont(date_font)
            p.setPen(QColor(100, 180, 255, int(180 * row_trans)))
            p.drawText(
                int(cx - R * 0.52), int(row_y),
                int(R * 0.25), int(row_h * 0.5),
                Qt.AlignLeft | Qt.AlignVCenter, c["hash"]
            )

            # Subject (truncate to fit)
            p.setFont(commit_font)
            p.setPen(QColor(225, 220, 210, int(210 * row_trans)))
            subject = c["subject"]
            if len(subject) > 28:
                subject = subject[:26] + "\u2026"
            p.drawText(
                int(cx - R * 0.52), int(row_y + row_h * 0.38),
                int(R * 1.0), int(row_h * 0.55),
                Qt.AlignLeft | Qt.AlignVCenter, subject
            )

            # Date (right-aligned)
            p.setFont(date_font)
            p.setPen(gold_faint)
            p.drawText(
                int(cx + R * 0.05), int(row_y),
                int(R * 0.48), int(row_h * 0.5),
                Qt.AlignRight | Qt.AlignVCenter, c["date"]
            )

            # Separator
            if idx < max_show - 1:
                sep_y = row_y + row_h * 0.92
                p.setPen(QPen(GOLD(int(35 * row_trans)), max(0.6, mind * 0.001)))
                p.drawLine(
                    QPointF(cx - R * 0.48, sep_y),
                    QPointF(cx + R * 0.48, sep_y)
                )

        if n > 4:
            p.setFont(date_font)
            p.setPen(gold_faint)
            more_y = base_y + max_show * row_h
            p.drawText(
                int(cx - R), int(more_y),
                int(2 * R), int(row_h * 0.5),
                Qt.AlignCenter, f"+{n - 4} more"
            )

        # --- "APPLY UPDATE" button ---
        btn_y = cy + R * 0.48
        btn_w = R * 0.70
        btn_h = R * 0.18
        btn_rect = QRectF(cx - btn_w, btn_y, btn_w * 2, btn_h)
        corner = btn_h * 0.35

        # Button glow
        pulse = 0.5 + 0.5 * math.sin(t * 2.0)
        btn_grad = QRadialGradient(
            QPointF(cx, btn_y + btn_h * 0.5), btn_w
        )
        btn_grad.setColorAt(0.0, QColor(60, 180, 120, int((120 + 60 * pulse) * trans)))
        btn_grad.setColorAt(1.0, QColor(30, 90, 60, int(80 * trans)))
        p.setPen(QPen(QColor(80, 220, 140, int(200 * trans)),
                       max(1.2, mind * 0.002)))
        p.setBrush(QBrush(btn_grad))
        p.drawRoundedRect(btn_rect, corner, corner)

        # Button text
        btn_font = QFont("DejaVu Sans", max(8, int(mind * 0.014)))
        btn_font.setBold(True)
        btn_font.setLetterSpacing(QFont.PercentageSpacing, 120)
        p.setFont(btn_font)
        p.setPen(QColor(255, 255, 255, int(240 * trans)))
        p.drawText(btn_rect, Qt.AlignCenter, "APPLY  UPDATE")

    # --- Back hint ---
    if trans > 0.72:
        hint_font = QFont("DejaVu Sans", max(7, int(mind * 0.011)))
        hint_font.setLetterSpacing(QFont.PercentageSpacing, 120)
        p.setFont(hint_font)
        a = int(40 * (trans - 0.72) / 0.28)
        a = max(0, min(40, a))
        p.setPen(QColor(255, 215, 140, a))
        p.drawText(
            int(cx - R), int(cy + R * 0.78), int(2 * R), int(R * 0.14),
            Qt.AlignCenter, "\u25C0  BACK"
        )


# ---------------------------------------------------------------------------
# Helper: draw text curved along a circular arc
# ---------------------------------------------------------------------------

def _draw_curved_text(p, text, radius, top, color, inner):
    """Draw *text* along a circular arc inside the dial."""
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
        rect = QRectF(- cw, ty, cw * 2, fm.height())

        p.setPen(QColor(0, 0, 0, 130))
        p.drawText(rect.translated(0, shd_off), Qt.AlignCenter, ch)
        p.setPen(color)
        p.drawText(rect, Qt.AlignCenter, ch)
        p.restore()

        current_ang += direction * (cw * 0.5) / radius

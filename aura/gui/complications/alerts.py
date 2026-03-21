"""
gui.complications.alerts -- Alerts / notifications complication.

Severity sector arc (blue→red interpolation), danger hand with jitter,
pulsing alarm-heart sapphire, and count window.

Subscribes to bus events:
  - "alerts.update"   → live alert list from SystemMonitor
  - "system.metrics"  → GPU/RAM/INF gauge values
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt5.QtGui import QPainter

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QBrush, QColor, QFont, QPen, QRadialGradient

from gui.complications.base import BaseComplication
from gui.renderer import clamp


# ---------------------------------------------------------------------------
# Severity definitions
# ---------------------------------------------------------------------------

_SEVERITY_INFO = 0
_SEVERITY_WARN = 1
_SEVERITY_CRIT = 2

_SEVERITY_COLORS = {
    _SEVERITY_INFO: QColor(60, 175, 255),     # blue
    _SEVERITY_WARN: QColor(245, 180, 50),      # amber
    _SEVERITY_CRIT: QColor(225, 75, 65),        # red
}

_SEVERITY_LABELS = {
    _SEVERITY_INFO: "INFO",
    _SEVERITY_WARN: "WARN",
    _SEVERITY_CRIT: "CRIT",
}

# ---------------------------------------------------------------------------
# Initial defaults (shown until first real data arrives)
# ---------------------------------------------------------------------------

_DEFAULT_ALERTS = [
    {"msg": "Monitoring starting…",  "sev": _SEVERITY_INFO, "ago": "now"},
]

_DEFAULT_GAUGES = [
    {"label": "GPU",  "value": 0.0, "color": QColor(60, 175, 255)},
    {"label": "MEM",  "value": 0.0, "color": QColor(60, 175, 255)},
    {"label": "INF",  "value": 0.0, "color": QColor(60, 175, 255)},
]


def _gauge_color(val):
    """Green → amber → red based on utilization."""
    if val < 0.5:
        return QColor(0, 255, 136)
    elif val < 0.8:
        return QColor(245, 180, 50)
    else:
        return QColor(225, 75, 65)


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

        # Live data (replaced when bus events arrive)
        self._alerts = list(_DEFAULT_ALERTS)
        self._gauges = list(_DEFAULT_GAUGES)

        # Subscribe to live data
        bus.on("alerts.update", self._on_alerts_update)
        bus.on("system.metrics", self._on_metrics)

    # ------------------------------------------------------------------
    # Bus handlers
    # ------------------------------------------------------------------
    def _on_alerts_update(self, alerts=None, **_kw):
        if alerts:
            self._alerts = alerts[:5]
            max_sev = max(a.get("sev", 0) for a in self._alerts)
            self.severity = max_sev / 2.0  # 0-2 → 0-1
            self.count = len(self._alerts)
            self._alert_pulse = (0.8 if max_sev == _SEVERITY_CRIT
                                 else 0.4 if max_sev == _SEVERITY_WARN
                                 else 0.1)

    def _on_metrics(self, gpu_pct=0, ram_pct=0, **_kw):
        gpu_val = clamp(gpu_pct / 100.0, 0, 1)
        ram_val = clamp(ram_pct / 100.0, 0, 1)
        inf_val = gpu_val * 0.5
        self._gauges = [
            {"label": "GPU",  "value": gpu_val, "color": _gauge_color(gpu_val)},
            {"label": "MEM",  "value": ram_val, "color": _gauge_color(ram_val)},
            {"label": "INF",  "value": inf_val, "color": QColor(60, 175, 255)},
        ]

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
        halo = inner * (0.20 + 0.08 * flare)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(255, 70, 70, int(55 + 120 * flare))))
        p.drawEllipse(QPointF(0, 0), halo, halo)

        core = inner * 0.11
        p.setBrush(QBrush(QColor(0, 0, 0, 140)))
        p.drawEllipse(QPointF(0, 0), core * 1.25, core * 1.25)
        p.setBrush(QBrush(QColor(255, 90, 80, int(60 + 160 * flare))))
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

        cnt = len(self._alerts)
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
        """Alerts dashboard overlay: enamel backdrop, alert list, system health gauges."""
        trans = clamp(trans, 0.0, 1.0)
        if trans <= 0.0:
            return

        p.save()
        try:
            p.setRenderHint(p.Antialiasing, True)

            # ----- Sizing (matches settings overlay proportions) -----
            R = mind * 0.235

            r_bezel_outer = R * 0.98
            r_bezel_inner = R * 0.90
            r_chapter_out = R * 0.84
            r_chapter_in  = R * 0.74
            rg = R * 0.64

            # ----- Alpha / colors -----
            A  = int(240 * trans)
            A2 = int(175 * trans)
            A3 = int(120 * trans)

            coral       = QColor(225, 95, 85, A)
            coral_mid   = QColor(225, 95, 85, A2)
            coral_faint = QColor(225, 95, 85, A3)

            base_dark = QColor(14, 12, 16, int(220 * trans))
            mid_dark  = QColor(24, 20, 26, int(210 * trans))

            # ----- Helpers -----
            def draw_radial_glow(radius, alpha):
                grad = QRadialGradient(QPointF(cx, cy), radius)
                grad.setColorAt(0.0, QColor(225, 95, 85, int(alpha * 0.10)))
                grad.setColorAt(0.45, QColor(225, 95, 85, int(alpha * 0.04)))
                grad.setColorAt(1.0, QColor(0, 0, 0, 0))
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(grad))
                p.drawEllipse(QPointF(cx, cy), radius, radius)

            # =========================================================
            # 1) Deep enamel backdrop
            # =========================================================
            p.setPen(Qt.NoPen)
            p.setBrush(base_dark)
            p.drawEllipse(QPointF(cx, cy), R, R)

            # Radial glow
            draw_radial_glow(R * 0.98, A)

            # Bezel depth highlights
            g1 = QRadialGradient(QPointF(cx, cy), r_bezel_outer)
            g1.setColorAt(0.70, QColor(225, 100, 90, int(28 * trans)))
            g1.setColorAt(0.86, QColor(225, 100, 90, int(48 * trans)))
            g1.setColorAt(1.00, QColor(0, 0, 0, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(g1))
            p.drawEllipse(QPointF(cx, cy), r_bezel_outer, r_bezel_outer)

            g2 = QRadialGradient(QPointF(cx, cy), r_bezel_inner)
            g2.setColorAt(0.60, QColor(0, 0, 0, 0))
            g2.setColorAt(0.92, QColor(0, 0, 0, int(80 * trans)))
            g2.setColorAt(1.00, QColor(0, 0, 0, int(110 * trans)))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(g2))
            p.drawEllipse(QPointF(cx, cy), r_bezel_inner, r_bezel_inner)

            # Bezel rings
            bezel_pen = QPen(coral_mid)
            bezel_pen.setWidthF(max(2.0, mind * 0.0042))
            p.setPen(bezel_pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx, cy), r_bezel_outer, r_bezel_outer)

            inner_pen = QPen(coral_faint)
            inner_pen.setWidthF(max(1.2, mind * 0.0026))
            p.setPen(inner_pen)
            p.drawEllipse(QPointF(cx, cy), r_bezel_inner, r_bezel_inner)

            # =========================================================
            # Chapter ring (60 ticks)
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

                col = QColor(225, 95, 85, int((170 if is_major else 100) * trans))
                p.setPen(QPen(col, tick_w, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

            # Inner chapter boundary
            p.setPen(QPen(coral_faint, max(1.0, mind * 0.0018)))
            p.drawEllipse(QPointF(cx, cy), r_chapter_in, r_chapter_in)

            # =========================================================
            # Animated heartbeat ring (pulses faster with severity)
            # =========================================================
            max_sev = max((a.get("sev", 0) for a in self._alerts), default=0)
            pulse_hz = 1.5 + max_sev * 1.25
            pulse = 0.5 + 0.5 * math.sin(t * pulse_hz * 2.0 * math.pi)

            heartbeat_r = r_chapter_out + R * 0.04
            hb_alpha = int((40 + 60 * pulse) * trans)
            hb_width = max(1.5, mind * 0.003) * (0.8 + 0.4 * pulse)
            hb_pen = QPen(QColor(225, 95, 85, hb_alpha))
            hb_pen.setWidthF(hb_width)
            p.setPen(hb_pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx, cy), heartbeat_r, heartbeat_r)

            # =========================================================
            # Guilloché inner field
            # =========================================================
            p.setPen(Qt.NoPen)
            p.setBrush(mid_dark)
            p.drawEllipse(QPointF(cx, cy), rg, rg)

            gu_pen = QPen(QColor(225, 95, 85, int(18 * trans)))
            gu_pen.setWidthF(max(1.0, mind * 0.0016))
            p.setPen(gu_pen)
            p.setBrush(Qt.NoBrush)
            for k in range(30):
                frac = k / 29.0
                rad = rg * (0.15 + 0.85 * frac)
                wob = 1.0 + 0.015 * math.sin(t * 0.8 + k * 0.6)
                p.drawEllipse(QPointF(cx, cy), rad * wob, rad)

            # =========================================================
            # "ALERTS" headline with severity indicator
            # =========================================================
            micro_font = QFont("DejaVu Sans", max(8, int(mind * 0.014)))
            micro_font.setLetterSpacing(QFont.PercentageSpacing, 108)
            p.setFont(micro_font)
            p.setPen(coral_faint)
            p.drawText(
                int(cx - R), int(cy - R * 0.95), int(2 * R), int(R * 0.18),
                Qt.AlignCenter, "AURA  \u2022  ALERTS"
            )

            # Severity badge
            sev_label = "CRITICAL" if max_sev == _SEVERITY_CRIT else (
                "WARNING" if max_sev == _SEVERITY_WARN else "ALL CLEAR")
            sev_color = _SEVERITY_COLORS.get(max_sev, QColor(60, 175, 255))

            badge_font = QFont("DejaVu Sans", max(9, int(mind * 0.017)))
            badge_font.setBold(True)
            badge_font.setLetterSpacing(QFont.PercentageSpacing, 115)
            p.setFont(badge_font)

            badge_rect = QRectF(cx - R * 0.50, cy - R * 0.76, R * 1.00, R * 0.18)

            # Badge background pill
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, int(120 * trans)))
            p.drawRoundedRect(badge_rect, R * 0.06, R * 0.06)

            # Severity dot
            dot_r = mind * 0.007
            dot_x = badge_rect.left() + R * 0.12
            dot_y = badge_rect.center().y()
            p.setBrush(QColor(sev_color.red(), sev_color.green(), sev_color.blue(),
                              int(220 * trans)))
            p.drawEllipse(QPointF(dot_x, dot_y), dot_r, dot_r)

            # Badge text
            text_rect = QRectF(dot_x + dot_r * 2, badge_rect.top(),
                               badge_rect.width() - R * 0.18, badge_rect.height())
            p.setPen(QColor(sev_color.red(), sev_color.green(), sev_color.blue(),
                            int(220 * trans)))
            p.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, sev_label)

            # =========================================================
            # Alert list rows (live data)
            # =========================================================
            row_font = QFont("DejaVu Sans", max(8, int(mind * 0.013)))
            row_font.setLetterSpacing(QFont.PercentageSpacing, 102)
            time_font = QFont("DejaVu Sans", max(7, int(mind * 0.010)))
            time_font.setLetterSpacing(QFont.PercentageSpacing, 105)

            row_h = R * 0.145
            list_top = cy - R * 0.52
            list_w = R * 1.40
            list_x = cx - list_w * 0.50

            for idx, alert in enumerate(self._alerts):
                row_y = list_top + idx * row_h
                sev_col = _SEVERITY_COLORS.get(alert.get("sev", 0),
                                               QColor(60, 175, 255))
                row_alpha = trans * (0.95 - idx * 0.06)

                # Row separator
                if idx > 0:
                    sep_pen = QPen(QColor(255, 255, 255, int(16 * trans)))
                    sep_pen.setWidthF(max(0.5, mind * 0.0008))
                    p.setPen(sep_pen)
                    p.drawLine(QPointF(list_x + R * 0.08, row_y),
                               QPointF(list_x + list_w - R * 0.08, row_y))

                # Severity dot
                sev_dot_r = mind * 0.005
                sev_dot_x = list_x + R * 0.08
                sev_dot_y = row_y + row_h * 0.50
                p.setPen(Qt.NoPen)

                if alert.get("sev", 0) == _SEVERITY_CRIT:
                    dot_pulse = 0.5 + 0.5 * math.sin(t * 4.0 + idx)
                    sev_dot_a = int((160 + 80 * dot_pulse) * row_alpha)
                else:
                    sev_dot_a = int(190 * row_alpha)

                p.setBrush(QColor(sev_col.red(), sev_col.green(), sev_col.blue(),
                                  sev_dot_a))
                p.drawEllipse(QPointF(sev_dot_x, sev_dot_y), sev_dot_r, sev_dot_r)

                # Message text
                p.setFont(row_font)
                msg_rect = QRectF(sev_dot_x + sev_dot_r * 3, row_y + row_h * 0.10,
                                  list_w * 0.72, row_h * 0.80)
                p.setPen(QColor(0, 0, 0, int(120 * row_alpha)))
                p.drawText(msg_rect.translated(0.8, 0.8),
                           Qt.AlignLeft | Qt.AlignVCenter, alert.get("msg", ""))
                text_a = int(215 * row_alpha)
                p.setPen(QColor(235, 240, 250, text_a))
                p.drawText(msg_rect, Qt.AlignLeft | Qt.AlignVCenter,
                           alert.get("msg", ""))

                # Relative time
                p.setFont(time_font)
                time_rect = QRectF(list_x + list_w - R * 0.36, row_y + row_h * 0.10,
                                   R * 0.30, row_h * 0.80)
                p.setPen(QColor(sev_col.red(), sev_col.green(), sev_col.blue(),
                                int(140 * row_alpha)))
                p.drawText(time_rect, Qt.AlignRight | Qt.AlignVCenter,
                           alert.get("ago", ""))

            # =========================================================
            # Micro divider between alerts and health section
            # =========================================================
            div_y = list_top + len(self._alerts) * row_h + R * 0.04
            p.setPen(QPen(coral_faint, max(1.0, mind * 0.0016)))
            p.drawLine(QPointF(cx - R * 0.52, div_y),
                       QPointF(cx + R * 0.52, div_y))

            # =========================================================
            # System Health section — 3 small gauges (live data)
            # =========================================================
            health_font = QFont("DejaVu Sans", max(7, int(mind * 0.011)))
            health_font.setBold(True)
            health_font.setLetterSpacing(QFont.PercentageSpacing, 120)

            p.setFont(health_font)
            label_y = div_y + R * 0.02
            p.setPen(QColor(225, 95, 85, int(160 * trans)))
            p.drawText(int(cx - R), int(label_y), int(2 * R), int(R * 0.12),
                       Qt.AlignCenter, "SYSTEM HEALTH")

            # Gauges
            gauge_y = label_y + R * 0.14
            gauge_spacing = R * 0.42
            gauge_r = R * 0.12

            val_font = QFont("DejaVu Sans", max(6, int(mind * 0.009)))
            val_font.setBold(True)

            for gi, gauge in enumerate(self._gauges):
                gx = cx + (gi - 1) * gauge_spacing
                gy = gauge_y

                val = clamp(gauge.get("value", 0), 0.0, 1.0)
                gc = gauge.get("color", QColor(60, 175, 255))

                # Gauge track
                track_rect = QRectF(gx - gauge_r, gy - gauge_r,
                                    2 * gauge_r, 2 * gauge_r)
                track_pen = QPen(QColor(255, 255, 255, int(25 * trans)))
                track_pen.setWidthF(max(1.5, mind * 0.003))
                track_pen.setCapStyle(Qt.RoundCap)
                p.setPen(track_pen)
                p.setBrush(Qt.NoBrush)
                arc_start = 225.0
                arc_span = 270.0
                p.drawArc(track_rect, int(arc_start * 16), int(arc_span * 16))

                # Value arc
                val_alpha = int((100 + 140 * val) * trans)
                val_pen = QPen(QColor(gc.red(), gc.green(), gc.blue(), val_alpha))
                val_pen.setWidthF(max(1.8, mind * 0.0035))
                val_pen.setCapStyle(Qt.RoundCap)
                p.setPen(val_pen)
                p.drawArc(track_rect, int(arc_start * 16),
                          int((arc_span * val) * 16))

                # Percentage text
                p.setFont(val_font)
                pct = f"{int(val * 100)}"
                p.setPen(QColor(gc.red(), gc.green(), gc.blue(), int(200 * trans)))
                p.drawText(QRectF(gx - gauge_r, gy - gauge_r * 0.5,
                                  2 * gauge_r, gauge_r),
                           Qt.AlignCenter, pct)

                # Label below gauge
                p.setFont(health_font)
                p.setPen(QColor(235, 240, 250, int(150 * trans)))
                p.drawText(QRectF(gx - gauge_r * 1.5, gy + gauge_r * 0.6,
                                  gauge_r * 3, gauge_r * 0.8),
                           Qt.AlignCenter, gauge.get("label", ""))

        finally:
            p.restore()

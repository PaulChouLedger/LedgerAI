"""
gui.complications.domains.auranet -- AuraNet / Farsight Network complication.

Real-time view of the Aura device mesh: connected pucks, RTX hub,
Telegram community stats, and Farsight offload status.

Overlay renders as a grand complication with proper horology aesthetic:
deep enamel backdrop, engine-turned guilloché, bezel rings, and a live
network constellation with animated data flow.
"""

from __future__ import annotations

import math
import time
import random
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from PyQt5.QtGui import QPainter

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QLinearGradient,
    QPen, QRadialGradient, QPainterPath,
)

from gui.complications.domains.base_domain import BaseDomainComplication
from gui.renderer import clamp


# ---------------------------------------------------------------------------
# Puck network — updated from system.metrics bus events
# ---------------------------------------------------------------------------

# Node types in the constellation
_TYPE_HUB = "hub"          # Farsight RTX server
_TYPE_PUCK = "puck"        # Aura puck device
_TYPE_TG = "telegram"      # Telegram community
_TYPE_SELF = "self"        # This puck

# Colors per node type
_NODE_COLORS = {
    _TYPE_HUB:  QColor(180, 140, 255),    # violet (RTX)
    _TYPE_PUCK: QColor(80, 200, 255),     # cyan (pucks)
    _TYPE_TG:   QColor(100, 220, 170),    # teal (telegram)
    _TYPE_SELF: QColor(255, 220, 100),    # gold (this device)
}

# Default network state — represents the real topology
_DEFAULT_NODES: List[Dict] = [
    {"id": "rtx",    "label": "FARSIGHT HUB",  "type": _TYPE_HUB,  "x":  0.00, "y": -0.52, "w": 1.00, "online": False},
    {"id": "self",   "label": "THIS PUCK",     "type": _TYPE_SELF, "x":  0.00, "y":  0.00, "w": 0.90, "online": True},
    {"id": "puck1",  "label": "PUCK 1",        "type": _TYPE_PUCK, "x": -0.58, "y":  0.10, "w": 0.60, "online": False},
    {"id": "tg",     "label": "TELEGRAM",       "type": _TYPE_TG,  "x":  0.55, "y": -0.20, "w": 0.65, "online": False},
]

_DEFAULT_EDGES = [
    ("self", "rtx"),
    ("puck1", "rtx"),
    ("tg", "rtx"),
    ("self", "puck1"),
]

# Stats that update from real data
_DEFAULT_STATS = {
    "pucks_online": 1,
    "pucks_total": 2,
    "tg_users": 12,
    "tg_groups": 2,
    "farsight_model": "Qwen 72B",
    "farsight_online": False,
    "uptime_hours": 0.0,
    "queries_today": 0,
}


class AuraNetComplication(BaseDomainComplication):
    name = "AuraNet"
    label = "AuraNet"
    category = "Topics"

    def __init__(self, bus):
        super().__init__(bus)
        self._nodes = [dict(n) for n in _DEFAULT_NODES]
        self._edges = list(_DEFAULT_EDGES)
        self._stats = dict(_DEFAULT_STATS)
        self._query_count = 0
        self._last_metrics_ts = 0.0

        bus.on("system.metrics", self._on_metrics)

    def _on_metrics(self, **kw):
        """Update network state from system monitor."""
        services = kw.get("services", {})
        uptime = kw.get("uptime_hours", 0.0)

        self._stats["farsight_online"] = services.get("farsight", False)
        self._stats["uptime_hours"] = uptime

        # Update node online states
        for node in self._nodes:
            if node["id"] == "rtx":
                node["online"] = services.get("farsight", False)
            elif node["id"] == "tg":
                node["online"] = True  # Telegram bot runs on RTX
            elif node["id"] == "self":
                node["online"] = True
            elif node["id"] == "puck1":
                node["online"] = False  # TODO: Farsight hub puck registry

        if self._stats["farsight_online"]:
            self._stats["pucks_online"] = 1  # at minimum this puck
        self._last_metrics_ts = time.time()

    # ------------------------------------------------------------------
    def draw_content(self, p: "QPainter", inner: float, t: float, accent: QColor) -> None:
        pass  # Icon drawn by BaseDomainComplication.draw_glyph

    # ------------------------------------------------------------------
    def draw_overlay(self, p, cx, cy, mind, t, trans):
        """AuraNet: deep enamel grand complication with live network mesh."""
        a = clamp(float(trans), 0.0, 1.0)
        if a < 0.002:
            return

        p.save()
        try:
            p.setRenderHint(p.Antialiasing, True)

            R = mind * 0.38
            R_inner = R * 0.92

            # ── Color palette ──────────────────────────────────────
            INDIGO      = lambda al=255: QColor(85, 125, 220, al)
            INDIGO_FAINT = lambda al=255: QColor(85, 125, 220, al)
            CREAM       = lambda al=255: QColor(215, 228, 248, al)
            A  = int(240 * a)
            A2 = int(175 * a)
            A3 = int(120 * a)

            # ══════════════════════════════════════════════════════
            # 1) Deep enamel backdrop
            # ══════════════════════════════════════════════════════
            bg = QRadialGradient(QPointF(cx, cy), R)
            bg.setColorAt(0.00, QColor(6, 12, 32, int(250 * a)))
            bg.setColorAt(0.40, QColor(4, 8, 24, int(245 * a)))
            bg.setColorAt(0.75, QColor(2, 4, 16, int(230 * a)))
            bg.setColorAt(1.00, QColor(0, 0, 0, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(bg))
            p.drawEllipse(QPointF(cx, cy), R, R)

            # Crystal gloss
            gloss = QRadialGradient(QPointF(cx - R * 0.3, cy - R * 0.35), R * 1.1)
            gloss.setColorAt(0.0, QColor(255, 255, 255, int(10 * a)))
            gloss.setColorAt(0.4, QColor(255, 255, 255, int(4 * a)))
            gloss.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setBrush(QBrush(gloss))
            p.drawEllipse(QPointF(cx, cy), R * 0.98, R * 0.98)

            # ══════════════════════════════════════════════════════
            # 2) Triple bezel rings
            # ══════════════════════════════════════════════════════
            # Outer polished bezel
            bevel = QRadialGradient(QPointF(cx, cy), R)
            bevel.setColorAt(0.82, QColor(85, 125, 220, int(35 * a)))
            bevel.setColorAt(0.94, QColor(85, 125, 220, int(65 * a)))
            bevel.setColorAt(1.00, QColor(0, 0, 0, 0))
            p.setBrush(QBrush(bevel))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(cx, cy), R, R)

            p.setPen(QPen(INDIGO(A2), max(2.2, mind * 0.0044)))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx, cy), R * 0.98, R * 0.98)

            p.setPen(QPen(INDIGO(A3), max(1.2, mind * 0.0026)))
            p.drawEllipse(QPointF(cx, cy), R * 0.93, R * 0.93)

            p.setPen(QPen(INDIGO(int(60 * a)), max(0.8, mind * 0.0018)))
            p.drawEllipse(QPointF(cx, cy), R * 0.88, R * 0.88)

            # Depth shadow
            depth = QRadialGradient(cx, cy, R * 0.94)
            depth.setColorAt(0.70, QColor(0, 0, 0, 0))
            depth.setColorAt(0.92, QColor(0, 0, 0, int(55 * a)))
            depth.setColorAt(1.00, QColor(0, 0, 0, int(90 * a)))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(depth))
            p.drawEllipse(QPointF(cx, cy), R * 0.94, R * 0.94)

            # ══════════════════════════════════════════════════════
            # 3) Rotating chapter ring (60 ticks)
            # ══════════════════════════════════════════════════════
            ring_rot = (t * 2.0) % 360.0
            p.save()
            p.translate(cx, cy)
            p.rotate(ring_rot)
            for i in range(60):
                ang = (2 * math.pi * i) / 60
                is_major = (i % 5 == 0)
                tick_out = R * 0.915
                tick_in = tick_out - (R * 0.055 if is_major else R * 0.025)
                tw = mind * (0.003 if is_major else 0.0012)
                ta = int((140 if is_major else 55) * a)
                p.setPen(QPen(INDIGO(ta), tw, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(
                    QPointF(tick_in * math.cos(ang), tick_in * math.sin(ang)),
                    QPointF(tick_out * math.cos(ang), tick_out * math.sin(ang)),
                )
            p.restore()

            # ══════════════════════════════════════════════════════
            # 4) Engine-turned guilloché
            # ══════════════════════════════════════════════════════
            gu_rot = (t * 1.0) % 360.0
            p.save()
            p.translate(cx, cy)
            p.rotate(gu_rot)
            gu_pen = QPen(QColor(85, 125, 220, int(10 * a)), max(0.3, R * 0.0015))
            p.setPen(gu_pen)
            for i in range(48):
                ang = (2 * math.pi * i) / 48
                wave = 1.0 + 0.006 * math.sin(i * 4 + t * 0.2)
                x1 = R * 0.04 * math.cos(ang)
                y1 = R * 0.04 * math.sin(ang)
                x2 = R * 0.86 * wave * math.cos(ang)
                y2 = R * 0.86 * wave * math.sin(ang)
                p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
            # Concentric rings
            ring_pen = QPen(QColor(85, 125, 220, int(6 * a)), 0.4)
            p.setPen(ring_pen)
            for frac in (0.20, 0.40, 0.60, 0.80):
                rr = R * frac
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(0, 0), rr, rr)
            p.restore()

            # ══════════════════════════════════════════════════════
            # 5) Headline: "AURA · NETWORK"
            # ══════════════════════════════════════════════════════
            micro_font = QFont("DejaVu Sans", max(8, int(mind * 0.015)))
            micro_font.setBold(True)
            micro_font.setLetterSpacing(QFont.PercentageSpacing, 130)
            p.setFont(micro_font)
            # Shadow
            p.setPen(QColor(0, 0, 0, int(140 * a)))
            p.drawText(
                int(cx - R), int(cy - R * 0.87) + 1, int(2 * R), int(R * 0.12),
                Qt.AlignCenter, "AURA  \u2022  NETWORK"
            )
            p.setPen(INDIGO(A))
            p.drawText(
                int(cx - R), int(cy - R * 0.87), int(2 * R), int(R * 0.12),
                Qt.AlignCenter, "AURA  \u2022  NETWORK"
            )

            # ══════════════════════════════════════════════════════
            # 6) Status badge (Farsight online/offline)
            # ══════════════════════════════════════════════════════
            farsight_up = self._stats.get("farsight_online", False)
            badge_text = "FARSIGHT ONLINE" if farsight_up else "LOCAL MODE"
            badge_col = QColor(80, 220, 160, A) if farsight_up else QColor(245, 180, 60, A)
            dot_col = QColor(80, 220, 160) if farsight_up else QColor(245, 180, 60)

            badge_font = QFont("DejaVu Sans", max(7, int(mind * 0.013)))
            badge_font.setBold(True)
            badge_font.setLetterSpacing(QFont.PercentageSpacing, 115)
            p.setFont(badge_font)

            badge_rect = QRectF(cx - R * 0.42, cy - R * 0.75, R * 0.84, R * 0.13)
            # Pill background
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, int(130 * a)))
            p.drawRoundedRect(badge_rect, R * 0.04, R * 0.04)
            # Status dot
            dot_r = mind * 0.006
            dot_x = badge_rect.left() + R * 0.08
            dot_y = badge_rect.center().y()
            p.setBrush(QColor(dot_col.red(), dot_col.green(), dot_col.blue(), A))
            if farsight_up:
                # Pulse the dot
                pulse = 0.5 + 0.5 * math.sin(t * 3.0)
                p.setBrush(QColor(dot_col.red(), dot_col.green(), dot_col.blue(),
                                  int((160 + 80 * pulse) * a)))
            p.drawEllipse(QPointF(dot_x, dot_y), dot_r, dot_r)
            # Text
            text_rect = QRectF(dot_x + dot_r * 2.5, badge_rect.top(),
                               badge_rect.width() - R * 0.14, badge_rect.height())
            p.setPen(badge_col)
            p.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, badge_text)

            # ══════════════════════════════════════════════════════
            # 7) Network constellation (center area)
            # ══════════════════════════════════════════════════════
            graph_cx = cx
            graph_cy = cy - R * 0.10
            graph_r = R * 0.48

            nodes = self._nodes
            edges = self._edges

            # Build lookup for positions
            pos = {}
            for node in nodes:
                pos[node["id"]] = (
                    graph_cx + node["x"] * graph_r,
                    graph_cy + node["y"] * graph_r,
                )

            # Draw edges with gradient glow
            for id0, id1 in edges:
                if id0 not in pos or id1 not in pos:
                    continue
                x0, y0 = pos[id0]
                x1, y1 = pos[id1]
                n0 = next((n for n in nodes if n["id"] == id0), None)
                n1 = next((n for n in nodes if n["id"] == id1), None)
                both_online = (n0 and n0["online"]) and (n1 and n1["online"])

                if both_online:
                    edge_a = int(55 * a)
                    edge_w = max(0.8, R * 0.005)
                else:
                    edge_a = int(18 * a)
                    edge_w = max(0.5, R * 0.003)

                edge_pen = QPen(QColor(120, 160, 230, edge_a))
                edge_pen.setWidthF(edge_w)
                edge_pen.setCapStyle(Qt.RoundCap)
                p.setPen(edge_pen)
                p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

            # Data flow pulses on active edges
            active_edges = [(id0, id1) for id0, id1 in edges
                            if any(n["id"] == id0 and n["online"] for n in nodes)
                            and any(n["id"] == id1 and n["online"] for n in nodes)]
            for pi, (id0, id1) in enumerate(active_edges):
                x0, y0 = pos.get(id0, (cx, cy))
                x1, y1 = pos.get(id1, (cx, cy))
                # Multiple pulses per edge at different phases
                for phase_off in [0.0, 0.33, 0.66]:
                    frac = ((t * 0.6 + pi * 0.4 + phase_off) % 1.0)
                    px = x0 + (x1 - x0) * frac
                    py = y0 + (y1 - y0) * frac
                    pa = int(180 * a * (1.0 - abs(frac - 0.5) * 2.0))
                    if pa > 3:
                        pr = max(1.5, R * 0.010)
                        # Glow
                        p.setPen(Qt.NoPen)
                        p.setBrush(QColor(180, 210, 255, int(pa * 0.3)))
                        p.drawEllipse(QPointF(px, py), pr * 2.5, pr * 2.5)
                        # Core
                        p.setBrush(QColor(220, 240, 255, pa))
                        p.drawEllipse(QPointF(px, py), pr, pr)

            # Draw nodes
            for node in nodes:
                nx, ny = pos[node["id"]]
                w = node["w"]
                online = node["online"]
                nc = _NODE_COLORS.get(node["type"], QColor(120, 180, 255))
                node_r = R * (0.030 + 0.022 * w)

                if online:
                    # Breathing glow halo
                    pulse = 0.5 + 0.5 * math.sin(t * 1.6 + hash(node["id"]) % 10)
                    glow_r = node_r * (2.8 + 0.8 * pulse)
                    glow = QRadialGradient(nx, ny, glow_r)
                    glow.setColorAt(0.0, QColor(nc.red(), nc.green(), nc.blue(),
                                                int(50 * a * pulse)))
                    glow.setColorAt(0.5, QColor(nc.red(), nc.green(), nc.blue(),
                                                int(15 * a * pulse)))
                    glow.setColorAt(1.0, QColor(0, 0, 0, 0))
                    p.setPen(Qt.NoPen)
                    p.setBrush(QBrush(glow))
                    p.drawEllipse(QPointF(nx, ny), glow_r, glow_r)

                # Shadow
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(0, 0, 0, int(110 * a)))
                p.drawEllipse(QPointF(nx + 1, ny + 1), node_r * 1.1, node_r * 1.1)

                # Body
                na = int((200 if online else 80) * a)
                body_g = QRadialGradient(nx - node_r * 0.2, ny - node_r * 0.3, node_r)
                body_g.setColorAt(0.0, QColor(min(255, nc.red() + 40),
                                               min(255, nc.green() + 40),
                                               min(255, nc.blue() + 40), na))
                body_g.setColorAt(0.6, QColor(nc.red(), nc.green(), nc.blue(), na))
                body_g.setColorAt(1.0, QColor(max(0, nc.red() - 30),
                                               max(0, nc.green() - 30),
                                               max(0, nc.blue() - 30), na))
                p.setBrush(QBrush(body_g))
                # Thin bezel ring
                p.setPen(QPen(QColor(255, 255, 255, int(30 * a)), max(0.4, node_r * 0.06)))
                p.drawEllipse(QPointF(nx, ny), node_r, node_r)

                # White specular highlight
                if online and w > 0.5:
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(255, 255, 255, int(140 * a * w)))
                    p.drawEllipse(QPointF(nx - node_r * 0.2, ny - node_r * 0.25),
                                  node_r * 0.30, node_r * 0.22)

                # Offline X mark
                if not online:
                    x_pen = QPen(QColor(200, 80, 70, int(120 * a)))
                    x_pen.setWidthF(max(1.0, node_r * 0.15))
                    x_pen.setCapStyle(Qt.RoundCap)
                    p.setPen(x_pen)
                    x_sz = node_r * 0.45
                    p.drawLine(QPointF(nx - x_sz, ny - x_sz),
                               QPointF(nx + x_sz, ny + x_sz))
                    p.drawLine(QPointF(nx + x_sz, ny - x_sz),
                               QPointF(nx - x_sz, ny + x_sz))

                # Node label
                lbl_font = QFont("DejaVu Sans", max(5, int(mind * 0.010)))
                lbl_font.setBold(True)
                lbl_font.setLetterSpacing(QFont.PercentageSpacing, 110)
                p.setFont(lbl_font)
                lbl_y = ny + node_r * 1.5
                lbl_a = int((180 if online else 90) * a)
                # Shadow
                p.setPen(QColor(0, 0, 0, int(100 * a)))
                p.drawText(QRectF(nx - R * 0.20, lbl_y + 0.5, R * 0.40, R * 0.06),
                           Qt.AlignCenter, node["label"])
                p.setPen(QColor(nc.red(), nc.green(), nc.blue(), lbl_a))
                p.drawText(QRectF(nx - R * 0.20, lbl_y, R * 0.40, R * 0.06),
                           Qt.AlignCenter, node["label"])

            # ══════════════════════════════════════════════════════
            # 8) Stats panel (lower third)
            # ══════════════════════════════════════════════════════
            panel_top = cy + R * 0.42
            panel_w = R * 1.40

            # Divider line
            p.setPen(QPen(INDIGO(A3), max(1.0, mind * 0.0016)))
            p.drawLine(QPointF(cx - panel_w * 0.38, panel_top),
                       QPointF(cx + panel_w * 0.38, panel_top))

            # Stats grid: 2 rows × 3 columns
            stats_font = QFont("DejaVu Sans", max(6, int(mind * 0.011)))
            stats_font.setBold(True)
            val_font = QFont("DejaVu Sans", max(8, int(mind * 0.018)))
            val_font.setBold(True)
            label_font = QFont("DejaVu Sans", max(5, int(mind * 0.008)))
            label_font.setLetterSpacing(QFont.PercentageSpacing, 120)

            stats_items = [
                (str(self._stats["pucks_online"]),
                 f"/{self._stats['pucks_total']} PUCKS",
                 QColor(80, 200, 255, A)),
                (str(self._stats["tg_users"]),
                 "TG USERS",
                 QColor(100, 220, 170, A)),
                (str(self._stats["tg_groups"]),
                 "GROUPS",
                 QColor(100, 220, 170, A)),
                (f"{self._stats['uptime_hours']:.0f}h",
                 "UPTIME",
                 QColor(180, 200, 240, A)),
                (self._stats["farsight_model"] if farsight_up else "—",
                 "LLM MODEL",
                 QColor(180, 140, 255, A) if farsight_up else QColor(120, 120, 140, A2)),
                ("ACTIVE" if farsight_up else "OFFLINE",
                 "OFFLOAD",
                 QColor(80, 220, 160, A) if farsight_up else QColor(245, 180, 60, A)),
            ]

            cols = 3
            rows = 2
            cell_w = panel_w / cols
            cell_h = R * 0.18
            grid_x0 = cx - panel_w * 0.5
            grid_y0 = panel_top + R * 0.04

            for idx, (val_str, label_str, col) in enumerate(stats_items):
                col_i = idx % cols
                row_i = idx // cols
                cell_cx = grid_x0 + (col_i + 0.5) * cell_w
                cell_cy = grid_y0 + row_i * cell_h

                # Staggered reveal
                delay = idx * 0.06
                item_a = clamp((a - delay) * 5.0, 0.0, 1.0)
                if item_a < 0.01:
                    continue

                # Value
                p.setFont(val_font)
                p.setPen(QColor(col.red(), col.green(), col.blue(),
                                int(col.alpha() * item_a)))
                p.drawText(QRectF(cell_cx - cell_w * 0.45, cell_cy - cell_h * 0.35,
                                  cell_w * 0.90, cell_h * 0.55),
                           Qt.AlignCenter, val_str)

                # Label
                p.setFont(label_font)
                p.setPen(QColor(160, 180, 210, int(140 * item_a * a)))
                p.drawText(QRectF(cell_cx - cell_w * 0.45, cell_cy + cell_h * 0.12,
                                  cell_w * 0.90, cell_h * 0.35),
                           Qt.AlignCenter, label_str)

            # ══════════════════════════════════════════════════════
            # 9) Bottom signature
            # ══════════════════════════════════════════════════════
            sig_font = QFont("DejaVu Sans", max(6, int(mind * 0.010)))
            sig_font.setLetterSpacing(QFont.PercentageSpacing, 125)
            p.setFont(sig_font)
            p.setPen(INDIGO(int(60 * a)))
            p.drawText(
                int(cx - R), int(cy + R * 0.82), int(2 * R), int(R * 0.10),
                Qt.AlignCenter, "FARSIGHT  \u2022  MESH"
            )

        finally:
            p.restore()

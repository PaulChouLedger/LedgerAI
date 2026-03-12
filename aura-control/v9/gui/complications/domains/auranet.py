"""
gui.complications.domains.auranet -- AuraNet domain complication.

Global Aura user network: trending topics from users worldwide,
shared interests, and social introductions between Aura owners.

Glyph icon shows an animated constellation of connected nodes
representing the user mesh.  Overlay shows a live network dashboard
with active users, trending topics, and pulsing connection graph.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt5.QtGui import QPainter

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QBrush, QColor, QFont, QPen, QRadialGradient, QPainterPath

from gui.complications.domains.base_domain import BaseDomainComplication
from gui.renderer import clamp


# ---------------------------------------------------------------------------
# Demo data — realistic network snapshot
# ---------------------------------------------------------------------------

_ACTIVE_USERS = 14_273
_REGIONS = 42
_CONNECTIONS_TODAY = 1_847

_TRENDS = [
    ("Cardiac Imaging",     0.92, 342),
    ("Portfolio Rebalance",  0.78, 218),
    ("Quantum Chemistry",   0.65, 175),
    ("Sleep Optimization",  0.58, 143),
    ("Climate Modelling",   0.44, 98),
]

# Network node positions (normalised -1..1) with "importance" weight
_NODES = [
    ( 0.00,  0.00, 1.00),   # hub
    (-0.55, -0.40, 0.70),
    ( 0.50, -0.45, 0.65),
    (-0.62,  0.30, 0.55),
    ( 0.58,  0.35, 0.60),
    ( 0.05, -0.72, 0.50),
    (-0.30,  0.65, 0.45),
    ( 0.35,  0.68, 0.48),
    (-0.75,  0.00, 0.40),
    ( 0.78,  0.00, 0.42),
    ( 0.00,  0.78, 0.38),
    (-0.40, -0.68, 0.35),
]

# Edges (index pairs)
_EDGES = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
    (1, 2), (1, 5), (1, 8), (1, 11),
    (2, 4), (2, 5), (2, 9),
    (3, 6), (3, 8),
    (4, 7), (4, 9),
    (5, 11),
    (6, 7), (6, 10),
    (7, 10),
    (8, 11),
    (9, 4),
]


class AuraNetComplication(BaseDomainComplication):
    name = "AuraNet"
    label = "AuraNet"
    category = "Topics"

    def __init__(self, bus):
        super().__init__(bus)

    # ------------------------------------------------------------------
    def draw_content(self, p: "QPainter", inner: float, t: float, accent: QColor) -> None:
        """Draw content is handled by base_domain's icon system."""
        pass  # Icon drawn by BaseDomainComplication.draw_glyph → _draw_auranet_icon

    # ------------------------------------------------------------------
    def draw_overlay(self, p, cx, cy, mind, t, trans):
        """AuraNet dashboard: network graph, trending topics, user stats."""
        a = clamp(float(trans), 0.0, 1.0)
        if a < 0.002:
            return

        p.save()
        try:
            p.setRenderHint(p.Antialiasing, True)

            R = mind * 0.40
            accent = QColor(255, 165, 95)

            # --- Backdrop ---
            bg = QRadialGradient(cx, cy, R)
            bg.setColorAt(0.00, QColor(18, 8, 6, int(245 * a)))
            bg.setColorAt(0.60, QColor(14, 6, 4, int(225 * a)))
            bg.setColorAt(1.00, QColor(0, 0, 0, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(bg))
            p.drawEllipse(QPointF(cx, cy), R, R)

            # --- Clip ---
            clip = QPainterPath()
            clip.addEllipse(QPointF(cx, cy), R * 0.95, R * 0.95)
            p.setClipPath(clip)

            # --- Headline ---
            f_head = QFont("Helvetica", max(8, int(R * 0.072)))
            f_head.setBold(True)
            p.setFont(f_head)
            p.setPen(QColor(248, 225, 215, int(240 * a)))
            p.drawText(
                QRectF(cx - R * 0.80, cy - R * 0.90, R * 1.60, R * 0.14),
                Qt.AlignCenter, "A U R A N E T"
            )

            # Stats row
            f_stat = QFont("Helvetica", max(6, int(R * 0.050)))
            f_stat.setBold(True)
            p.setFont(f_stat)
            p.setPen(QColor(210, 215, 228, int(200 * a)))
            stats_str = f"{_ACTIVE_USERS:,} users  \u00b7  {_REGIONS} regions  \u00b7  {_CONNECTIONS_TODAY:,} new"
            p.drawText(
                QRectF(cx - R * 0.85, cy - R * 0.76, R * 1.70, R * 0.10),
                Qt.AlignCenter, stats_str
            )

            # --- Network graph (upper half) ---
            graph_cx = cx
            graph_cy = cy - R * 0.22
            graph_r = R * 0.42

            # Draw edges
            for i0, i1 in _EDGES:
                x0_n, y0_n, _ = _NODES[i0]
                x1_n, y1_n, _ = _NODES[i1]
                ex0 = graph_cx + x0_n * graph_r
                ey0 = graph_cy + y0_n * graph_r
                ex1 = graph_cx + x1_n * graph_r
                ey1 = graph_cy + y1_n * graph_r

                edge_pen = QPen(QColor(accent.red(), accent.green(), accent.blue(),
                                       int(35 * a)))
                edge_pen.setWidthF(max(0.5, R * 0.004))
                edge_pen.setCapStyle(Qt.RoundCap)
                p.setPen(edge_pen)
                p.drawLine(QPointF(ex0, ey0), QPointF(ex1, ey1))

            # Data pulses travelling along edges
            n_pulses = 4
            for pi in range(n_pulses):
                edge_idx = (int(t * 1.2) + pi * 7) % len(_EDGES)
                i0, i1 = _EDGES[edge_idx]
                x0_n, y0_n, _ = _NODES[i0]
                x1_n, y1_n, _ = _NODES[i1]
                frac = ((t * 1.2 + pi * 0.25) % 1.0)

                px = graph_cx + (x0_n + (x1_n - x0_n) * frac) * graph_r
                py = graph_cy + (y0_n + (y1_n - y0_n) * frac) * graph_r
                pa = int(200 * a * (1.0 - abs(frac - 0.5) * 2.0))
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(255, 220, 195, pa))
                pr = max(1.5, R * 0.012)
                p.drawEllipse(QPointF(px, py), pr, pr)

            # Draw nodes
            for idx, (nx, ny, w) in enumerate(_NODES):
                node_x = graph_cx + nx * graph_r
                node_y = graph_cy + ny * graph_r
                node_r = R * (0.022 + 0.018 * w)

                # Hub gets a glow
                if idx == 0:
                    pulse = 0.5 + 0.5 * math.sin(t * 1.8)
                    glow_r = node_r * (2.2 + 0.6 * pulse)
                    glow = QRadialGradient(node_x, node_y, glow_r)
                    glow.setColorAt(0.0, QColor(accent.red(), accent.green(),
                                                accent.blue(), int(60 * a * pulse)))
                    glow.setColorAt(1.0, QColor(0, 0, 0, 0))
                    p.setPen(Qt.NoPen)
                    p.setBrush(QBrush(glow))
                    p.drawEllipse(QPointF(node_x, node_y), glow_r, glow_r)

                # Node body
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(0, 0, 0, int(100 * a)))
                p.drawEllipse(QPointF(node_x, node_y), node_r * 1.2, node_r * 1.2)

                na = int((140 + 80 * w) * a)
                p.setBrush(QColor(accent.red(), accent.green(), accent.blue(), na))
                p.drawEllipse(QPointF(node_x, node_y), node_r, node_r)

                # White core for brighter nodes
                if w > 0.6:
                    p.setBrush(QColor(255, 235, 215, int(120 * a * w)))
                    p.drawEllipse(QPointF(node_x, node_y), node_r * 0.4, node_r * 0.4)

            # --- Trending topics list (lower half) ---
            list_top = cy + R * 0.22
            row_h = R * 0.105
            f_topic = QFont("Helvetica", max(5, int(R * 0.046)))
            f_topic.setBold(True)
            f_count = QFont("Helvetica", max(5, int(R * 0.038)))

            for i, (topic, heat, count) in enumerate(_TRENDS):
                ry = list_top + i * row_h
                # Animated reveal
                reveal_delay = i * 0.12
                row_a = clamp((a - reveal_delay) * 4.0, 0.0, 1.0)
                if row_a < 0.01:
                    continue

                # Heat bar background
                bar_x = cx - R * 0.72
                bar_w = R * 1.44
                bar_h = row_h * 0.55
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(65, 38, 28, int(60 * row_a)))
                p.drawRoundedRect(QRectF(bar_x, ry, bar_w, bar_h),
                                  R * 0.008, R * 0.008)

                # Heat bar fill
                fill_w = bar_w * heat * row_a
                bar_col = QColor(accent.red(), accent.green(), accent.blue(),
                                 int(65 + 80 * heat) * int(row_a))
                p.setBrush(QBrush(bar_col))
                p.drawRoundedRect(QRectF(bar_x, ry, fill_w, bar_h),
                                  R * 0.008, R * 0.008)

                # Topic name
                p.setFont(f_topic)
                p.setPen(QColor(248, 225, 215, int(230 * row_a)))
                name_rect = QRectF(bar_x + R * 0.04, ry, bar_w * 0.65, bar_h)
                p.drawText(name_rect, Qt.AlignVCenter | Qt.AlignLeft, topic)

                # Count
                p.setFont(f_count)
                p.setPen(QColor(240, 195, 160, int(170 * row_a)))
                cnt_rect = QRectF(bar_x + bar_w * 0.70, ry, bar_w * 0.26, bar_h)
                p.drawText(cnt_rect, Qt.AlignVCenter | Qt.AlignRight,
                           f"{count} users")

            # --- Outer ring decoration: rotating dashes ---
            ring_r = R * 0.92
            n_dashes = 36
            for i in range(n_dashes):
                ang = (2 * math.pi * i / n_dashes) + t * 0.15
                d_a = int(22 * a * (0.5 + 0.5 * math.sin(t * 0.8 + i * 0.5)))
                if d_a < 2:
                    continue
                dx0 = cx + ring_r * 0.96 * math.cos(ang)
                dy0 = cy + ring_r * 0.96 * math.sin(ang)
                dx1 = cx + ring_r * math.cos(ang)
                dy1 = cy + ring_r * math.sin(ang)
                dash_pen = QPen(QColor(accent.red(), accent.green(),
                                       accent.blue(), d_a))
                dash_pen.setWidthF(max(0.5, R * 0.005))
                dash_pen.setCapStyle(Qt.RoundCap)
                p.setPen(dash_pen)
                p.drawLine(QPointF(dx0, dy0), QPointF(dx1, dy1))

        finally:
            p.restore()

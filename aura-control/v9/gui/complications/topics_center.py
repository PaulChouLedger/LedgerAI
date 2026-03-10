"""
gui.complications.topics_center -- Topics Center complication.

Patek Philippe-styled complication with "TOPICS" curved along the top arc
and "CENTER" curved along the bottom arc.  Central icon is a 4-point
compass rose representing the browsable domain catalogue.

Always visible on the perimeter ring via ``always_available = True``.
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


class TopicsCenterComplication(BaseComplication):
    name = "Topics Center"
    label = "Topics Center"
    category = "System"
    dockable = True
    always_available = True
    has_overlay = True

    def __init__(self, bus):
        super().__init__(bus)

    # ------------------------------------------------------------------
    def draw_content(self, p: "QPainter", inner: float, t: float, accent: QColor) -> None:
        # --- Curved "TOPICS" along top arc ---
        _draw_curved_text(p, "TOPICS", inner * 0.58, top=True,
                          color=QColor(248, 238, 218, 235), inner=inner)

        # --- Curved "CENTER" along bottom arc ---
        _draw_curved_text(p, "CENTER", inner * 0.58, top=False,
                          color=QColor(248, 238, 218, 215), inner=inner)

        # --- Central compass rose icon ---
        cr = inner * 0.30
        # Outer ring
        pen = QPen(QColor(accent.red(), accent.green(), accent.blue(), 120))
        pen.setWidthF(max(1.0, inner * 0.020))
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(0, 0), cr * 1.15, cr * 1.15)

        # 4 diamond points (N, E, S, W) — gold tones
        gold_point = QColor(208, 178, 112)
        for i in range(4):
            ang = -math.pi / 2 + i * math.pi / 2
            tip_r = cr * 1.05
            side_r = cr * 0.35
            tip_x = tip_r * math.cos(ang)
            tip_y = tip_r * math.sin(ang)

            perp = ang + math.pi / 2
            sx1 = side_r * math.cos(perp)
            sy1 = side_r * math.sin(perp)

            # Primary: gold for N/S, faded for E/W
            primary = (i % 2 == 0)
            a = 200 if primary else 110
            col = QColor(gold_point.red(), gold_point.green(), gold_point.blue(), a)

            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(col))
            from PyQt5.QtGui import QPainterPath, QPolygonF
            tri = QPainterPath()
            tri.moveTo(tip_x, tip_y)
            tri.lineTo(sx1, sy1)
            tri.lineTo(-sx1, -sy1)
            tri.closeSubpath()
            p.drawPath(tri)

        # Center jewel — amber
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 140)))
        p.drawEllipse(QPointF(0, 0), cr * 0.22, cr * 0.22)
        pulse = 0.5 + 0.5 * math.sin(t * 1.8)
        p.setBrush(QBrush(QColor(218, 175, 85, int(140 + 80 * pulse))))
        p.drawEllipse(QPointF(0, 0), cr * 0.16, cr * 0.16)

    # ------------------------------------------------------------------
    def draw_overlay(self, p, cx, cy, mind, t, trans):
        """Complication browser: deep enamel backdrop, TOPICS CENTER headline,
        grid of available complications as rounded cards, gold highlight for
        docked items, staggered reveal animation, rotating outer tick ring."""
        from PyQt5.QtGui import QRadialGradient, QPainterPath
        from gui.complications import registry
        from core.state import state

        trans = clamp(trans, 0.0, 1.0)
        if trans <= 0.0:
            return

        p.save()
        try:
            p.setRenderHint(p.Antialiasing, True)

            # ----- Sizing -----
            R = mind * 0.38

            # ----- Color palette (gold / champagne) -----
            GOLD = lambda a=255: QColor(208, 178, 112, a)  # noqa: E731
            CREAM = lambda a=255: QColor(245, 235, 205, a)  # noqa: E731
            A = int(240 * trans)
            A2 = int(175 * trans)
            A3 = int(120 * trans)

            gold_strong = GOLD(A)
            gold_mid = GOLD(A2)
            gold_faint = GOLD(A3)

            # =========================================================
            # 1) Deep enamel backdrop (radial gradient, dark blue-black)
            # =========================================================
            grad = QRadialGradient(QPointF(cx, cy), R)
            grad.setColorAt(0.0, QColor(14, 18, 32, int(225 * trans)))
            grad.setColorAt(0.6, QColor(8, 10, 22, int(235 * trans)))
            grad.setColorAt(1.0, QColor(4, 5, 12, int(240 * trans)))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(grad))
            p.drawEllipse(QPointF(cx, cy), R, R)

            # Crystal gloss
            gloss = QRadialGradient(QPointF(cx, cy), R * 0.98)
            gloss.setColorAt(0.0, QColor(255, 255, 255, int(12 * trans)))
            gloss.setColorAt(0.55, QColor(255, 255, 255, int(6 * trans)))
            gloss.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setBrush(QBrush(gloss))
            p.drawEllipse(QPointF(cx, cy), R * 0.98, R * 0.98)

            # Bezel depth (outer bevel highlight)
            bevel = QRadialGradient(QPointF(cx, cy), R)
            bevel.setColorAt(0.70, QColor(255, 240, 200, int(30 * trans)))
            bevel.setColorAt(0.88, QColor(255, 240, 200, int(55 * trans)))
            bevel.setColorAt(1.00, QColor(0, 0, 0, 0))
            p.setBrush(QBrush(bevel))
            p.drawEllipse(QPointF(cx, cy), R, R)

            # Gold bezel ring
            bezel_pen = QPen(gold_mid)
            bezel_pen.setWidthF(max(2.0, mind * 0.004))
            p.setPen(bezel_pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx, cy), R * 0.98, R * 0.98)

            # Inner ring
            inner_pen = QPen(gold_faint)
            inner_pen.setWidthF(max(1.2, mind * 0.0025))
            p.setPen(inner_pen)
            p.drawEllipse(QPointF(cx, cy), R * 0.92, R * 0.92)

            # =========================================================
            # 7) Outer decorative ring with rotating ticks
            # =========================================================
            tick_count = 72
            rot_offset = t * 0.3  # slow rotation
            for i in range(tick_count):
                ang = (i / tick_count) * 2.0 * math.pi + rot_offset
                is_major = (i % 6 == 0)
                r_out = R * 0.975
                r_in = r_out - (R * 0.055 if is_major else R * 0.030)
                tw = (mind * 0.003) if is_major else (mind * 0.0018)
                ta = int((160 if is_major else 90) * trans)

                x1 = cx + r_in * math.cos(ang)
                y1 = cy + r_in * math.sin(ang)
                x2 = cx + r_out * math.cos(ang)
                y2 = cy + r_out * math.sin(ang)

                tp = QPen(GOLD(ta))
                tp.setWidthF(max(1.0, tw))
                tp.setCapStyle(Qt.RoundCap)
                p.setPen(tp)
                p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

            # =========================================================
            # Guilloché radial brushing (subtle engine-turned texture)
            # =========================================================
            p.save()
            p.translate(cx, cy)
            p.setPen(QPen(QColor(208, 178, 112, int(8 * trans)), 1))
            for i in range(48):
                ang = (2 * math.pi) * (i / 48) + t * 0.04
                x0 = (R * 0.15) * math.cos(ang)
                y0 = (R * 0.15) * math.sin(ang)
                x1 = (R * 0.88) * math.cos(ang)
                y1 = (R * 0.88) * math.sin(ang)
                p.drawLine(int(x0), int(y0), int(x1), int(y1))
            p.restore()

            # =========================================================
            # 2) "TOPICS CENTER" headline at top
            # =========================================================
            micro_font = QFont("DejaVu Sans", max(9, int(mind * 0.016)))
            micro_font.setBold(True)
            micro_font.setLetterSpacing(QFont.PercentageSpacing, 135)
            p.setFont(micro_font)
            p.setPen(QColor(0, 0, 0, int(150 * trans)))
            p.drawText(
                int(cx - R), int(cy - R * 0.88) + 1, int(2 * R), int(R * 0.16),
                Qt.AlignCenter, "TOPICS  CENTER"
            )
            p.setPen(gold_strong)
            p.drawText(
                int(cx - R), int(cy - R * 0.88), int(2 * R), int(R * 0.16),
                Qt.AlignCenter, "TOPICS  CENTER"
            )

            # Decorative divider under headline
            div_y = cy - R * 0.72
            p.setPen(QPen(gold_faint, max(1.0, mind * 0.0015)))
            p.drawLine(QPointF(cx - R * 0.55, div_y), QPointF(cx + R * 0.55, div_y))

            # =========================================================
            # 3-6) Complication grid: 2-column layout with cards
            # =========================================================
            # Gather complications grouped by category
            by_cat = registry.get_by_category()
            docked_names = set(state.dock)

            # Build flat list: (category, complication) pairs
            items = []
            for cat_name in sorted(by_cat.keys()):
                for comp in sorted(by_cat[cat_name], key=lambda c: c.name):
                    items.append((cat_name, comp))

            # Grid layout: 2 columns, up to 4 rows (8 items visible)
            cols = 2
            max_rows = 4
            max_visible = cols * max_rows

            # Trim to fit
            visible_items = items[:max_visible]
            n_items = len(visible_items)
            rows = math.ceil(n_items / cols) if n_items > 0 else 0

            # Card dimensions
            card_w = R * 0.78
            card_h = R * 0.28
            gap_x = R * 0.08
            gap_y = R * 0.07

            # Grid origin (centered horizontally, below headline)
            grid_w = cols * card_w + (cols - 1) * gap_x
            grid_h = rows * card_h + max(0, rows - 1) * gap_y
            grid_x0 = cx - grid_w * 0.5
            grid_y0 = cy - R * 0.62

            for idx, (cat_name, comp) in enumerate(visible_items):
                col = idx % cols
                row = idx // cols

                # 6) Staggered reveal animation
                delay = idx * 0.07
                card_trans = clamp((trans - delay) / max(0.01, 1.0 - delay), 0.0, 1.0)
                if card_trans <= 0.0:
                    continue

                # Slide up from below
                slide_offset = (1.0 - card_trans) * R * 0.08

                cx_card = grid_x0 + col * (card_w + gap_x)
                cy_card = grid_y0 + row * (card_h + gap_y) + slide_offset

                card_rect = QRectF(cx_card, cy_card, card_w, card_h)
                corner_r = card_h * 0.22

                is_docked = comp.name in docked_names
                ca = int(card_trans * 255)

                # 4) Card background
                p.setPen(Qt.NoPen)
                # Shadow
                p.setBrush(QColor(0, 0, 0, int(60 * card_trans)))
                p.drawRoundedRect(card_rect.translated(1.5, 2.0), corner_r, corner_r)

                # Card fill
                card_grad = QRadialGradient(
                    QPointF(cx_card + card_w * 0.5, cy_card + card_h * 0.5),
                    card_w * 0.7
                )
                if is_docked:
                    # 5) Gold highlight for docked items
                    card_grad.setColorAt(0.0, QColor(38, 34, 22, int(210 * card_trans)))
                    card_grad.setColorAt(1.0, QColor(22, 20, 14, int(220 * card_trans)))
                else:
                    card_grad.setColorAt(0.0, QColor(22, 24, 32, int(200 * card_trans)))
                    card_grad.setColorAt(1.0, QColor(12, 14, 20, int(210 * card_trans)))

                p.setBrush(QBrush(card_grad))

                # Card border
                if is_docked:
                    border_col = QColor(208, 178, 112, int(180 * card_trans))
                else:
                    border_col = QColor(208, 178, 112, int(55 * card_trans))
                p.setPen(QPen(border_col, max(1.0, mind * 0.0018)))
                p.drawRoundedRect(card_rect, corner_r, corner_r)

                # Complication name
                name_font = QFont("DejaVu Sans", max(8, int(mind * 0.015)))
                name_font.setBold(True)
                p.setFont(name_font)

                name_rect = QRectF(
                    cx_card + card_w * 0.08,
                    cy_card + card_h * 0.08,
                    card_w * 0.84,
                    card_h * 0.52
                )
                if is_docked:
                    p.setPen(QColor(232, 212, 155, int(240 * card_trans)))
                else:
                    p.setPen(QColor(225, 220, 210, int(200 * card_trans)))
                p.drawText(name_rect, Qt.AlignLeft | Qt.AlignVCenter, comp.name)

                # Category label
                cat_font = QFont("DejaVu Sans", max(6, int(mind * 0.010)))
                cat_font.setLetterSpacing(QFont.PercentageSpacing, 120)
                p.setFont(cat_font)
                cat_rect = QRectF(
                    cx_card + card_w * 0.08,
                    cy_card + card_h * 0.52,
                    card_w * 0.84,
                    card_h * 0.40
                )
                p.setPen(QColor(208, 178, 112, int(130 * card_trans)))
                p.drawText(cat_rect, Qt.AlignLeft | Qt.AlignVCenter, cat_name.upper())

                # Small docked indicator jewel (right side)
                if is_docked:
                    jr = max(2.5, mind * 0.006)
                    jx = cx_card + card_w - card_w * 0.12
                    jy = cy_card + card_h * 0.5
                    pulse = 0.5 + 0.5 * math.sin(t * 2.0 + idx * 0.7)
                    p.setPen(Qt.NoPen)
                    # Halo
                    halo = QRadialGradient(QPointF(jx, jy), jr * 3.0)
                    halo.setColorAt(0.0, QColor(208, 178, 112, int(50 * card_trans * pulse)))
                    halo.setColorAt(1.0, QColor(0, 0, 0, 0))
                    p.setBrush(QBrush(halo))
                    p.drawEllipse(QPointF(jx, jy), jr * 3.0, jr * 3.0)
                    # Jewel
                    p.setBrush(QColor(218, 190, 120, int((160 + 80 * pulse) * card_trans)))
                    p.drawEllipse(QPointF(jx, jy), jr, jr)
                    # Spec highlight
                    p.setBrush(QColor(255, 255, 255, int(70 * card_trans)))
                    p.drawEllipse(QPointF(jx - jr * 0.3, jy - jr * 0.3),
                                  jr * 0.3, jr * 0.3)

            # =========================================================
            # Bottom signature
            # =========================================================
            sig_font = QFont("DejaVu Sans", max(7, int(mind * 0.011)))
            sig_font.setLetterSpacing(QFont.PercentageSpacing, 120)
            p.setFont(sig_font)
            p.setPen(QColor(208, 178, 112, int(70 * trans)))
            p.drawText(
                int(cx - R), int(cy + R * 0.78), int(2 * R), int(R * 0.14),
                Qt.AlignCenter, "AURA  \u2022  COMPLICATIONS"
            )

        finally:
            p.restore()

    def on_tap(self):
        self.open_overlay()
        return True


# ---------------------------------------------------------------------------
# Helper: draw text curved along a circular arc
# ---------------------------------------------------------------------------

def _draw_curved_text(p: "QPainter", text: str, radius: float, top: bool,
                      color: QColor, inner: float):
    """Draw *text* along a circular arc.  top=True arcs upward, False arcs downward."""
    font = QFont("Helvetica", max(7, int(inner * 0.16)))
    font.setBold(True)
    font.setLetterSpacing(QFont.PercentageSpacing, 145)
    p.setFont(font)

    fm = p.fontMetrics()
    char_widths = [fm.horizontalAdvance(ch) for ch in text]
    total_w = sum(char_widths)

    # Angular span: each pixel of text width ~ 1/radius radians
    span_rad = total_w / max(1.0, radius)

    if top:
        # Arc from left to right across the top (centered at -pi/2)
        start_ang = -math.pi / 2 - span_rad / 2
        direction = 1.0
    else:
        # Arc from right to left across the bottom (centered at pi/2)
        start_ang = math.pi / 2 + span_rad / 2
        direction = -1.0

    shd_off = max(0.8, inner * 0.014)
    current_ang = start_ang

    # Vertical offset: push text body inward so outer edge hugs the radius.
    # For top text, after rotation the "up" direction is radially outward,
    # so shift down (positive y) by half the ascent to tuck text inside.
    # For bottom text, "up" is also radially outward after rotation, same shift.
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

        # Shadow
        p.setPen(QColor(0, 0, 0, 130))
        p.drawText(rect.translated(0, shd_off), Qt.AlignCenter, ch)
        # Main
        p.setPen(color)
        p.drawText(rect, Qt.AlignCenter, ch)
        p.restore()

        current_ang += direction * (cw * 0.5) / radius

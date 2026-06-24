"""
gui.demo_renderer -- Visual rendering for the healthcare executive demo.

Each demo stage has its own paint function, called from window.py's
paintEvent when demo pipeline is active. All rendering is stateless
QPainter — receives state via DemoVisualState dataclass.

Visual language: same Patek Philippe watchface aesthetic as the rest of
Aura — guilloché, arc text, beveled bezels, scheme-aware palette. But
purpose-built for the demo narrative.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QLinearGradient, QPainter, QPen,
    QRadialGradient, QPainterPath,
)


# ---------------------------------------------------------------------------
# Visual state (updated each frame by window.py from bus events)
# ---------------------------------------------------------------------------

@dataclass
class KPICard:
    label: str = ""
    value: str = ""
    unit: str = ""
    appear_time: float = 0.0
    duration: float = 5.0


@dataclass
class FileParticle:
    name: str = ""
    appear_time: float = 0.0
    angle: float = 0.0


@dataclass
class FollowupPatient:
    name: str = ""
    age: int = 0
    risk: str = ""
    conditions: str = ""
    action: str = ""
    doctor: str = ""
    specialty: str = ""
    appear_time: float = 0.0


@dataclass
class DemoVisualState:
    stage: str = "WAITING"
    progress: float = 0.0
    text: str = ""

    # File drop
    files: list = field(default_factory=list)  # list[FileParticle]
    file_count: int = 0

    # Analysis
    chunk_count: int = 0
    chunk_total: int = 0

    # Briefing
    brief_segment: int = 0
    brief_total: int = 0
    active_kpis: list = field(default_factory=list)  # list[KPICard]

    # Follow-up
    followup_patients: list = field(default_factory=list)  # list[FollowupPatient]

    # Token counter
    tokens_used: int = 0
    token_last_delta: int = 0
    token_last_op: str = ""
    token_last_time: float = 0.0

    # Timing
    stage_start: float = 0.0


def _scheme_palette(scheme: dict) -> dict:
    """Extract demo-relevant colors from the scheme dict."""
    pal = scheme.get("ring_palette", "blue")
    is_red = pal == "red"

    if is_red:
        return {
            "accent": QColor(220, 180, 120),
            "accent_dim": QColor(180, 150, 95),
            "glow": QColor(255, 200, 160),
            "text": QColor(220, 200, 170),
            "text_dim": QColor(160, 140, 110),
            "bg_dark": QColor(12, 4, 6),
            "card_bg": QColor(30, 12, 16),
            "positive": QColor(110, 200, 150),
            "negative": QColor(220, 130, 120),
            "is_red": True,
        }
    return {
        "accent": QColor(145, 175, 215),
        "accent_dim": QColor(100, 130, 170),
        "glow": QColor(200, 220, 255),
        "text": QColor(200, 210, 230),
        "text_dim": QColor(140, 150, 170),
        "bg_dark": QColor(6, 8, 14),
        "card_bg": QColor(14, 18, 28),
        "positive": QColor(110, 210, 165),
        "negative": QColor(220, 140, 130),
        "is_red": False,
    }


# ---------------------------------------------------------------------------
# Arc text helper (shared with auraconnect_page)
# ---------------------------------------------------------------------------

def _arc_text(p: QPainter, cx: float, cy: float, text: str,
              radius: float, center_deg: float, font: QFont,
              color: QColor, spacing_deg: float = 6.5,
              flip: bool = False) -> None:
    chars = list(text)
    if flip:
        chars = chars[::-1]
    n = len(chars)
    span = (n - 1) * spacing_deg if n > 1 else 0.0
    start = center_deg - span * 0.5
    p.save()
    p.setFont(font)
    fm = p.fontMetrics()
    ch_h = fm.height()
    extra_rot = 180.0 if flip else 0.0
    for i, ch in enumerate(chars):
        a = math.radians(start + i * spacing_deg)
        x = cx + radius * math.cos(a)
        y = cy + radius * math.sin(a)
        ch_w = max(fm.horizontalAdvance(ch), ch_h)
        p.save()
        p.translate(x, y)
        p.rotate(math.degrees(a) + 90.0 + extra_rot)
        rect = QRectF(-ch_w * 0.7, -ch_h * 0.6, ch_w * 1.4, ch_h * 1.2)
        p.setPen(color)
        p.drawText(rect, Qt.AlignCenter, ch)
        p.restore()
    p.restore()


# ---------------------------------------------------------------------------
# Stage renderers
# ---------------------------------------------------------------------------

def paint_demo_overlay(p: QPainter, cx: float, cy: float, mind: float,
                       t: float, vs: DemoVisualState, scheme: dict,
                       alpha: float = 1.0) -> None:
    """Route to the correct stage renderer."""
    if alpha < 0.01:
        return

    stage = vs.stage
    if stage == "FILE_DROP":
        _paint_file_drop(p, cx, cy, mind, t, vs, scheme, alpha)
    elif stage == "ANALYZING":
        _paint_analyzing(p, cx, cy, mind, t, vs, scheme, alpha)
    elif stage == "BRIEFING":
        _paint_briefing(p, cx, cy, mind, t, vs, scheme, alpha)
    elif stage == "FOLLOWUP":
        _paint_followup(p, cx, cy, mind, t, vs, scheme, alpha)
    elif stage == "QA":
        _paint_qa(p, cx, cy, mind, t, vs, scheme, alpha)

    if vs.tokens_used > 0 and stage in ("FILE_DROP", "ANALYZING", "BRIEFING", "FOLLOWUP"):
        _paint_token_counter(p, cx, cy, mind, t, vs, scheme, alpha)


# ---------------------------------------------------------------------------
# File Drop — particles spiral inward, file names in arc text
# ---------------------------------------------------------------------------

def _paint_file_drop(p: QPainter, cx: float, cy: float, mind: float,
                     t: float, vs: DemoVisualState, scheme: dict,
                     alpha: float) -> None:
    pal = _scheme_palette(scheme)
    A = int(255 * alpha)
    R = mind * 0.45

    p.save()
    p.setRenderHint(p.Antialiasing, True)

    # Header arc
    hdr_font = QFont("Helvetica Neue", max(9, int(mind * 0.016)))
    hdr_font.setWeight(QFont.DemiBold)
    hdr_font.setLetterSpacing(QFont.PercentageSpacing, 145)
    hdr_col = QColor(pal["accent"])
    hdr_col.setAlpha(A)
    _arc_text(p, cx, cy, "DATA INGESTION", R * 0.85, -90.0,
              hdr_font, hdr_col, spacing_deg=5.5)

    # File count in center
    count_font = QFont("Helvetica Neue", max(28, int(mind * 0.065)))
    count_font.setWeight(QFont.Light)
    p.setFont(count_font)
    count_col = QColor(pal["text"])
    count_col.setAlpha(A)
    p.setPen(count_col)
    count_text = str(vs.file_count)
    p.drawText(QRectF(cx - R, cy - mind * 0.06, 2 * R, mind * 0.1),
               Qt.AlignCenter, count_text)

    # "FILES" label under count
    label_font = QFont("Helvetica Neue", max(8, int(mind * 0.014)))
    label_font.setWeight(QFont.Medium)
    label_font.setLetterSpacing(QFont.PercentageSpacing, 180)
    p.setFont(label_font)
    dim_col = QColor(pal["text_dim"])
    dim_col.setAlpha(int(180 * alpha))
    p.setPen(dim_col)
    p.drawText(QRectF(cx - R, cy + mind * 0.04, 2 * R, mind * 0.04),
               Qt.AlignCenter, "FILES")

    # Animated file particles spiraling inward
    for fp in vs.files:
        age = t - fp.appear_time
        if age < 0:
            continue

        spiral_dur = 2.0
        frac = min(1.0, age / spiral_dur)
        ease = 1.0 - (1.0 - frac) ** 3  # ease out cubic

        start_r = R * 0.95
        end_r = mind * 0.08
        r = start_r + (end_r - start_r) * ease
        angle = fp.angle + ease * math.pi * 1.5

        px = cx + r * math.cos(angle)
        py = cy + r * math.sin(angle)

        dot_alpha = int((1.0 - ease * 0.6) * 200 * alpha)
        dot_r = mind * (0.012 - 0.006 * ease)

        glow = QRadialGradient(QPointF(px, py), dot_r * 3)
        gc = pal["glow"]
        glow.setColorAt(0.0, QColor(gc.red(), gc.green(), gc.blue(),
                                    int(dot_alpha * 0.6)))
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(glow))
        p.drawEllipse(QPointF(px, py), dot_r * 3, dot_r * 3)

        ac = pal["accent"]
        p.setBrush(QColor(ac.red(), ac.green(), ac.blue(), dot_alpha))
        p.drawEllipse(QPointF(px, py), dot_r, dot_r)

        if frac < 0.7:
            name_font = QFont("Helvetica Neue", max(6, int(mind * 0.010)))
            p.setFont(name_font)
            nc = QColor(pal["text_dim"])
            nc.setAlpha(int(dot_alpha * 0.7))
            p.setPen(nc)
            short = fp.name[:18] + ("…" if len(fp.name) > 18 else "")
            p.drawText(QRectF(px - mind * 0.12, py + dot_r * 2,
                              mind * 0.24, mind * 0.02),
                       Qt.AlignCenter, short)

    # Progress arc at bottom
    _draw_progress_arc(p, cx, cy, R * 0.75, vs.progress, pal, alpha)

    # Status text at bottom
    sub_font = QFont("Helvetica Neue", max(7, int(mind * 0.012)))
    sub_font.setLetterSpacing(QFont.PercentageSpacing, 140)
    p.setFont(sub_font)
    sc = QColor(pal["accent_dim"])
    sc.setAlpha(int(160 * alpha))
    p.setPen(sc)
    _arc_text(p, cx, cy, vs.text.upper()[:30], R * 0.75, 90.0,
              sub_font, sc, spacing_deg=5.0, flip=True)

    p.restore()


# ---------------------------------------------------------------------------
# Analyzing — contracted rings, neural network nodes, chunk counter
# ---------------------------------------------------------------------------

def _paint_analyzing(p: QPainter, cx: float, cy: float, mind: float,
                     t: float, vs: DemoVisualState, scheme: dict,
                     alpha: float) -> None:
    pal = _scheme_palette(scheme)
    A = int(255 * alpha)
    R = mind * 0.45

    p.save()
    p.setRenderHint(p.Antialiasing, True)

    # Header
    hdr_font = QFont("Helvetica Neue", max(9, int(mind * 0.016)))
    hdr_font.setWeight(QFont.DemiBold)
    hdr_font.setLetterSpacing(QFont.PercentageSpacing, 145)
    hdr_col = QColor(pal["accent"])
    hdr_col.setAlpha(A)
    _arc_text(p, cx, cy, "ANALYZING", R * 0.85, -90.0,
              hdr_font, hdr_col, spacing_deg=7.0)

    # Neural network nodes — floating dots with connections
    n_nodes = 12
    for i in range(n_nodes):
        base_angle = (i / n_nodes) * 2 * math.pi
        orbit_r = R * (0.25 + 0.15 * math.sin(t * 0.7 + i * 1.3))
        wobble = math.sin(t * 1.2 + i * 0.8) * 0.1

        nx = cx + orbit_r * math.cos(base_angle + t * 0.3 + wobble)
        ny = cy + orbit_r * math.sin(base_angle + t * 0.3 + wobble)

        # Connections to neighbors
        for j in range(1, 3):
            ni = (i + j) % n_nodes
            na = (ni / n_nodes) * 2 * math.pi
            nr = R * (0.25 + 0.15 * math.sin(t * 0.7 + ni * 1.3))
            nw = math.sin(t * 1.2 + ni * 0.8) * 0.1
            ox = cx + nr * math.cos(na + t * 0.3 + nw)
            oy = cy + nr * math.sin(na + t * 0.3 + nw)

            line_alpha = int(40 * alpha * (0.5 + 0.5 * math.sin(t * 2 + i + j)))
            lc = pal["accent_dim"]
            p.setPen(QPen(QColor(lc.red(), lc.green(), lc.blue(), line_alpha),
                          max(0.5, mind * 0.001)))
            p.drawLine(QPointF(nx, ny), QPointF(ox, oy))

        # Node dot
        pulse = 0.5 + 0.5 * math.sin(t * 2.5 + i * 0.9)
        node_r = mind * (0.006 + 0.003 * pulse)
        nc = pal["accent"]
        node_alpha = int((120 + 80 * pulse) * alpha)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(nc.red(), nc.green(), nc.blue(), node_alpha))
        p.drawEllipse(QPointF(nx, ny), node_r, node_r)

    # Center status text
    status_font = QFont("Helvetica Neue", max(10, int(mind * 0.018)))
    status_font.setWeight(QFont.Light)
    p.setFont(status_font)
    tc = QColor(pal["text"])
    tc.setAlpha(A)
    p.setPen(tc)
    status = vs.text or "Processing"
    dots = "." * (int(t * 2) % 4)
    p.drawText(QRectF(cx - R, cy - mind * 0.02, 2 * R, mind * 0.05),
               Qt.AlignCenter, status + dots)

    # Chunk count below
    if vs.chunk_count > 0:
        chunk_font = QFont("Helvetica Neue", max(7, int(mind * 0.012)))
        chunk_font.setLetterSpacing(QFont.PercentageSpacing, 150)
        p.setFont(chunk_font)
        dc = QColor(pal["text_dim"])
        dc.setAlpha(int(160 * alpha))
        p.setPen(dc)
        p.drawText(QRectF(cx - R, cy + mind * 0.04, 2 * R, mind * 0.03),
                   Qt.AlignCenter, f"{vs.chunk_count} CHUNKS INDEXED")

    # Progress arc
    _draw_progress_arc(p, cx, cy, R * 0.75, vs.progress, pal, alpha)

    p.restore()


# ---------------------------------------------------------------------------
# Briefing — KPI cards fade in/out, segment indicator
# ---------------------------------------------------------------------------

def _paint_briefing(p: QPainter, cx: float, cy: float, mind: float,
                    t: float, vs: DemoVisualState, scheme: dict,
                    alpha: float) -> None:
    pal = _scheme_palette(scheme)
    A = int(255 * alpha)
    R = mind * 0.45

    p.save()
    p.setRenderHint(p.Antialiasing, True)

    # Header
    hdr_font = QFont("Helvetica Neue", max(9, int(mind * 0.016)))
    hdr_font.setWeight(QFont.DemiBold)
    hdr_font.setLetterSpacing(QFont.PercentageSpacing, 145)
    hdr_col = QColor(pal["accent"])
    hdr_col.setAlpha(A)
    _arc_text(p, cx, cy, "STRATEGIC BRIEFING", R * 0.85, -90.0,
              hdr_font, hdr_col, spacing_deg=5.0)

    # Segment indicator dots at bottom
    if vs.brief_total > 0:
        dot_spacing = mind * 0.025
        total_w = (vs.brief_total - 1) * dot_spacing
        start_x = cx - total_w / 2
        dot_y = cy + R * 0.65

        for i in range(vs.brief_total):
            dx = start_x + i * dot_spacing
            is_active = i == vs.brief_segment
            is_done = i < vs.brief_segment
            dr = mind * (0.006 if is_active else 0.004)

            if is_active:
                pulse = 0.6 + 0.4 * math.sin(t * 3)
                dc = pal["accent"]
                da = int((180 + 70 * pulse) * alpha)
            elif is_done:
                dc = pal["accent"]
                da = int(140 * alpha)
            else:
                dc = pal["text_dim"]
                da = int(80 * alpha)

            p.setPen(Qt.NoPen)
            p.setBrush(QColor(dc.red(), dc.green(), dc.blue(), da))
            p.drawEllipse(QPointF(dx, dot_y), dr, dr)

    # KPI cards — the star of the show
    _paint_kpi_cards(p, cx, cy, mind, t, vs.active_kpis, pal, alpha)

    # Progress arc
    _draw_progress_arc(p, cx, cy, R * 0.75, vs.progress, pal, alpha)

    p.restore()


def _paint_kpi_cards(p: QPainter, cx: float, cy: float, mind: float,
                     t: float, kpis: list, pal: dict,
                     alpha: float) -> None:
    """Render KPI metric cards that fade in and out during narration."""
    card_w = mind * 0.30
    card_h = mind * 0.10
    card_r = mind * 0.012

    for kpi in kpis:
        age = t - kpi.appear_time
        life = kpi.duration

        if age < 0 or age > life + 1.0:
            continue

        # Fade envelope: 0.4s in, hold, 0.4s out
        if age < 0.4:
            fade = age / 0.4
        elif age > life - 0.4:
            fade = max(0.0, (life - age) / 0.4)
        else:
            fade = 1.0

        fade = _ease_in_out(fade)
        if fade < 0.01:
            continue

        a = alpha * fade

        # Card position — centered, slight float animation
        card_cx = cx
        card_cy = cy + mind * 0.01 * math.sin(t * 0.8 + hash(kpi.label) * 0.1)

        rx = card_cx - card_w / 2
        ry = card_cy - card_h / 2

        # Card background
        bg = QColor(pal["card_bg"])
        bg.setAlpha(int(200 * a))
        p.setPen(Qt.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(QRectF(rx, ry, card_w, card_h), card_r, card_r)

        # Border
        bc = pal["accent_dim"]
        p.setPen(QPen(QColor(bc.red(), bc.green(), bc.blue(), int(100 * a)),
                      max(0.8, mind * 0.001)))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(rx, ry, card_w, card_h), card_r, card_r)

        # Value (large)
        val_font = QFont("Helvetica Neue", max(16, int(mind * 0.032)))
        val_font.setWeight(QFont.DemiBold)
        p.setFont(val_font)
        vc = QColor(pal["text"])
        vc.setAlpha(int(240 * a))
        p.setPen(vc)
        p.drawText(QRectF(rx, ry + card_h * 0.08, card_w, card_h * 0.45),
                   Qt.AlignCenter, kpi.value)

        # Label (small, above value)
        lbl_font = QFont("Helvetica Neue", max(6, int(mind * 0.010)))
        lbl_font.setWeight(QFont.Medium)
        lbl_font.setLetterSpacing(QFont.PercentageSpacing, 160)
        p.setFont(lbl_font)
        lc = QColor(pal["accent"])
        lc.setAlpha(int(200 * a))
        p.setPen(lc)
        p.drawText(QRectF(rx, ry + card_h * 0.02, card_w, card_h * 0.25),
                   Qt.AlignHCenter | Qt.AlignTop, kpi.label.upper())

        # Unit (small, below value)
        if kpi.unit:
            p.setFont(lbl_font)
            uc = QColor(pal["text_dim"])
            uc.setAlpha(int(160 * a))
            p.setPen(uc)
            p.drawText(QRectF(rx, ry + card_h * 0.60, card_w, card_h * 0.30),
                       Qt.AlignCenter, kpi.unit)


# ---------------------------------------------------------------------------
# Follow-up — patient risk cards
# ---------------------------------------------------------------------------

def _paint_followup(p: QPainter, cx: float, cy: float, mind: float,
                    t: float, vs: DemoVisualState, scheme: dict,
                    alpha: float) -> None:
    pal = _scheme_palette(scheme)
    A = int(255 * alpha)
    R = mind * 0.45

    p.save()
    p.setRenderHint(p.Antialiasing, True)

    # Header
    hdr_font = QFont("Helvetica Neue", max(9, int(mind * 0.016)))
    hdr_font.setWeight(QFont.DemiBold)
    hdr_font.setLetterSpacing(QFont.PercentageSpacing, 145)
    hdr_col = QColor(pal["accent"])
    hdr_col.setAlpha(A)
    _arc_text(p, cx, cy, "PATIENT  FOLLOW-UP", R * 0.85, -90.0,
              hdr_font, hdr_col, spacing_deg=5.0)

    # Show the most recent patient card
    active_patient = None
    for pt in reversed(vs.followup_patients):
        age = t - pt.appear_time
        if age < 20.0:
            active_patient = pt
            break

    if active_patient:
        _paint_patient_card(p, cx, cy, mind, t, active_patient, pal, alpha)

    # Progress dots for patients
    total = max(len(vs.followup_patients), 1)
    dot_spacing = mind * 0.022
    total_w = (total - 1) * dot_spacing
    start_x = cx - total_w / 2
    dot_y = cy + R * 0.70

    for i in range(total):
        dx = start_x + i * dot_spacing
        is_active = i == len(vs.followup_patients) - 1
        dr = mind * (0.005 if is_active else 0.003)

        risk_colors = {"Critical": pal["negative"], "High": QColor(220, 170, 90),
                       "Moderate-High": QColor(200, 180, 100), "Moderate": pal["positive"]}
        if i < len(vs.followup_patients):
            risk = vs.followup_patients[i].risk
            dc = risk_colors.get(risk, pal["accent"])
        else:
            dc = pal["text_dim"]

        da = int((220 if is_active else 130) * alpha)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(dc.red(), dc.green(), dc.blue(), da))
        p.drawEllipse(QPointF(dx, dot_y), dr, dr)

    # Progress arc
    _draw_progress_arc(p, cx, cy, R * 0.75, vs.progress, pal, alpha)

    p.restore()


def _paint_patient_card(p: QPainter, cx: float, cy: float, mind: float,
                        t: float, pt, pal: dict, alpha: float) -> None:
    """Render a single patient risk card in the center."""
    age = t - pt.appear_time

    # Fade envelope
    if age < 0.5:
        fade = age / 0.5
    elif age > 16.0:
        fade = max(0.0, (20.0 - age) / 4.0)
    else:
        fade = 1.0
    fade = _ease_in_out(fade)
    a = alpha * fade

    card_w = mind * 0.38
    card_h = mind * 0.22
    card_r = mind * 0.012
    rx = cx - card_w / 2
    ry = cy - card_h / 2 - mind * 0.02

    # Card background
    bg = QColor(pal["card_bg"])
    bg.setAlpha(int(210 * a))
    p.setPen(Qt.NoPen)
    p.setBrush(bg)
    p.drawRoundedRect(QRectF(rx, ry, card_w, card_h), card_r, card_r)

    # Risk-colored left edge
    risk_colors = {"Critical": pal["negative"], "High": QColor(220, 170, 90),
                   "Moderate-High": QColor(200, 180, 100), "Moderate": pal["positive"]}
    rc = risk_colors.get(pt.risk, pal["accent"])
    edge_w = mind * 0.005
    p.setBrush(QColor(rc.red(), rc.green(), rc.blue(), int(220 * a)))
    p.drawRoundedRect(QRectF(rx, ry + card_r, edge_w, card_h - 2 * card_r),
                      edge_w / 2, edge_w / 2)

    # Border
    bc = pal["accent_dim"]
    p.setPen(QPen(QColor(bc.red(), bc.green(), bc.blue(), int(80 * a)),
                  max(0.6, mind * 0.0008)))
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(QRectF(rx, ry, card_w, card_h), card_r, card_r)

    pad = mind * 0.015

    # Patient name
    name_font = QFont("Helvetica Neue", max(10, int(mind * 0.018)))
    name_font.setWeight(QFont.DemiBold)
    p.setFont(name_font)
    nc = QColor(pal["text"])
    nc.setAlpha(int(240 * a))
    p.setPen(nc)
    p.drawText(QRectF(rx + pad + edge_w, ry + pad * 0.5,
                      card_w - 2 * pad, mind * 0.025),
               Qt.AlignLeft | Qt.AlignVCenter,
               f"{pt.name}, {pt.age}")

    # Risk badge
    risk_font = QFont("Helvetica Neue", max(6, int(mind * 0.009)))
    risk_font.setWeight(QFont.Bold)
    risk_font.setLetterSpacing(QFont.PercentageSpacing, 140)
    p.setFont(risk_font)
    badge_text = pt.risk.upper()
    fm = p.fontMetrics()
    bw = fm.horizontalAdvance(badge_text) + mind * 0.015
    bh = fm.height() * 1.4
    bx = rx + card_w - pad - bw
    by = ry + pad * 0.5
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(rc.red(), rc.green(), rc.blue(), int(50 * a)))
    p.drawRoundedRect(QRectF(bx, by, bw, bh), bh / 2, bh / 2)
    p.setPen(QColor(rc.red(), rc.green(), rc.blue(), int(220 * a)))
    p.drawText(QRectF(bx, by, bw, bh), Qt.AlignCenter, badge_text)

    # Conditions
    cond_font = QFont("Helvetica Neue", max(6, int(mind * 0.010)))
    p.setFont(cond_font)
    cc = QColor(pal["text_dim"])
    cc.setAlpha(int(190 * a))
    p.setPen(cc)
    cond_rect = QRectF(rx + pad + edge_w, ry + mind * 0.035,
                       card_w - 2 * pad - edge_w, mind * 0.045)
    p.drawText(cond_rect, Qt.AlignLeft | Qt.TextWordWrap,
               pt.conditions[:80])

    # Action
    act_font = QFont("Helvetica Neue", max(6, int(mind * 0.010)))
    act_font.setWeight(QFont.Medium)
    p.setFont(act_font)
    ac = QColor(pal["accent"])
    ac.setAlpha(int(200 * a))
    p.setPen(ac)
    act_rect = QRectF(rx + pad + edge_w, ry + mind * 0.085,
                      card_w - 2 * pad - edge_w, mind * 0.04)
    p.drawText(act_rect, Qt.AlignLeft | Qt.TextWordWrap,
               f"→ {pt.action[:70]}")

    # Doctor
    doc_font = QFont("Helvetica Neue", max(6, int(mind * 0.009)))
    doc_font.setWeight(QFont.Light)
    p.setFont(doc_font)
    dc = QColor(pal["text_dim"])
    dc.setAlpha(int(160 * a))
    p.setPen(dc)
    doc_rect = QRectF(rx + pad + edge_w, ry + card_h - mind * 0.028,
                      card_w - 2 * pad - edge_w, mind * 0.025)
    p.drawText(doc_rect, Qt.AlignLeft | Qt.AlignVCenter,
               f"{pt.doctor} · {pt.specialty}")

    # Wireframe mesh face portrait
    _paint_mesh_face(p, rx + card_w - mind * 0.06, ry + card_h * 0.45,
                     mind * 0.04, pt.name, pal, a)


# ---------------------------------------------------------------------------
# $LEDGER token counter — bottom of screen, always visible during processing
# ---------------------------------------------------------------------------

def _paint_token_counter(p: QPainter, cx: float, cy: float, mind: float,
                         t: float, vs: DemoVisualState, scheme: dict,
                         alpha: float) -> None:
    pal = _scheme_palette(scheme)
    R = mind * 0.45

    p.save()
    p.setRenderHint(p.Antialiasing, True)

    # Position: just inside the bottom of the circular area
    counter_y = cy + R * 0.52

    # Animate the counter — rolling number effect
    flash_age = t - vs.token_last_time if vs.token_last_time > 0 else 999
    flash = max(0.0, 1.0 - flash_age / 1.5) if flash_age < 1.5 else 0.0

    # $LEDGER icon + count
    token_font = QFont("Helvetica Neue", max(7, int(mind * 0.012)))
    token_font.setWeight(QFont.Medium)
    token_font.setLetterSpacing(QFont.PercentageSpacing, 130)
    p.setFont(token_font)

    # Glow when tokens are being consumed
    if flash > 0.1:
        glow = QRadialGradient(QPointF(cx, counter_y), mind * 0.08)
        gc = pal["accent"]
        glow.setColorAt(0.0, QColor(gc.red(), gc.green(), gc.blue(),
                                    min(255, int(40 * flash * alpha))))
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(glow))
        p.drawEllipse(QPointF(cx, counter_y), mind * 0.08, mind * 0.04)

    # Token count text
    count_str = f"${vs.tokens_used:,}"
    ac = pal["accent"]
    text_alpha = min(255, int((140 + 80 * flash) * alpha))
    p.setPen(QColor(ac.red(), ac.green(), ac.blue(), text_alpha))
    p.drawText(QRectF(cx - mind * 0.15, counter_y - mind * 0.012,
                      mind * 0.30, mind * 0.025),
               Qt.AlignCenter, f"◆ {count_str} LEDGER")

    # Operation label (fades out)
    if flash > 0.2 and vs.token_last_op:
        op_font = QFont("Helvetica Neue", max(5, int(mind * 0.008)))
        op_font.setLetterSpacing(QFont.PercentageSpacing, 140)
        p.setFont(op_font)
        oc = QColor(pal["text_dim"])
        oc.setAlpha(min(255, int(120 * flash * alpha)))
        p.setPen(oc)
        p.drawText(QRectF(cx - mind * 0.12, counter_y + mind * 0.012,
                          mind * 0.24, mind * 0.015),
                   Qt.AlignCenter, vs.token_last_op.upper())

    p.restore()


# ---------------------------------------------------------------------------
# Q&A — subtle indicator that voice is active
# ---------------------------------------------------------------------------

def _paint_qa(p: QPainter, cx: float, cy: float, mind: float,
              t: float, vs: DemoVisualState, scheme: dict,
              alpha: float) -> None:
    pal = _scheme_palette(scheme)
    R = mind * 0.45

    p.save()
    p.setRenderHint(p.Antialiasing, True)

    # Gentle "Q&A" header
    hdr_font = QFont("Helvetica Neue", max(8, int(mind * 0.014)))
    hdr_font.setWeight(QFont.Medium)
    hdr_font.setLetterSpacing(QFont.PercentageSpacing, 160)
    hdr_col = QColor(pal["accent_dim"])
    pulse = 0.6 + 0.4 * math.sin(t * 1.2)
    hdr_col.setAlpha(int(120 * alpha * pulse))
    _arc_text(p, cx, cy, "ASK  ME  ANYTHING", R * 0.85, -90.0,
              hdr_font, hdr_col, spacing_deg=5.5)

    p.restore()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _paint_mesh_face(p: QPainter, cx: float, cy: float, size: float,
                     name: str, pal: dict, alpha: float) -> None:
    """Draw a procedural wireframe face mesh unique to each patient name."""
    h = sum(ord(c) for c in name)

    jaw_w = 0.7 + (h % 7) * 0.04
    brow_h = 0.28 + (h % 5) * 0.02

    pts = []
    n_pts = 12
    for i in range(n_pts):
        angle = math.pi * 2 * i / n_pts - math.pi / 2
        r = size
        if abs(angle - math.pi / 2) < 0.8:
            r *= jaw_w
        elif abs(angle + math.pi / 2) < 0.6:
            r *= 1.0 + brow_h * 0.15
        pts.append(QPointF(cx + r * math.cos(angle), cy + r * math.sin(angle)))

    wire_col = QColor(pal["accent"])
    wire_col.setAlpha(int(100 * alpha))
    p.setPen(QPen(wire_col, max(0.5, size * 0.02), Qt.SolidLine))
    p.setBrush(Qt.NoBrush)

    for i in range(n_pts):
        p.drawLine(pts[i], pts[(i + 1) % n_pts])

    for i in range(0, n_pts, 2):
        p.drawLine(pts[i], QPointF(cx, cy))

    eye_y = cy - size * 0.18
    eye_sep = size * (0.28 + (h % 3) * 0.04)
    eye_r = size * 0.06
    eye_col = QColor(pal["accent"])
    eye_col.setAlpha(int(160 * alpha))
    p.setPen(QPen(eye_col, max(0.5, size * 0.025)))
    p.drawEllipse(QPointF(cx - eye_sep, eye_y), eye_r, eye_r * 0.7)
    p.drawEllipse(QPointF(cx + eye_sep, eye_y), eye_r, eye_r * 0.7)

    nose_y = cy + size * 0.05
    nose_w = size * 0.08
    p.drawLine(QPointF(cx, cy - size * 0.08),
               QPointF(cx - nose_w, nose_y))
    p.drawLine(QPointF(cx - nose_w, nose_y),
               QPointF(cx + nose_w, nose_y))

    mouth_y = cy + size * 0.28
    mouth_w = size * (0.18 + (h % 4) * 0.03)
    p.drawLine(QPointF(cx - mouth_w, mouth_y),
               QPointF(cx + mouth_w, mouth_y))

    # Cross-mesh lines for wireframe effect
    cross_col = QColor(pal["accent_dim"])
    cross_col.setAlpha(int(50 * alpha))
    p.setPen(QPen(cross_col, max(0.3, size * 0.012)))
    for i in range(0, n_pts, 3):
        j = (i + n_pts // 2) % n_pts
        p.drawLine(pts[i], pts[j])


def _draw_progress_arc(p: QPainter, cx: float, cy: float, radius: float,
                       progress: float, pal: dict, alpha: float) -> None:
    """Thin arc from 6 o'clock clockwise, proportional to progress."""
    if progress <= 0.001:
        return

    p.save()

    # Track (dim)
    track_col = QColor(pal["accent_dim"])
    track_col.setAlpha(int(40 * alpha))
    p.setPen(QPen(track_col, max(1.5, radius * 0.015), Qt.SolidLine, Qt.RoundCap))
    p.setBrush(Qt.NoBrush)
    rect = QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius)
    p.drawArc(rect, -90 * 16, -360 * 16)  # full circle track

    # Fill (bright)
    fill_col = QColor(pal["accent"])
    fill_col.setAlpha(int(200 * alpha))
    p.setPen(QPen(fill_col, max(2.0, radius * 0.02), Qt.SolidLine, Qt.RoundCap))
    span = int(-progress * 360 * 16)
    p.drawArc(rect, 90 * 16, span)

    p.restore()


def _ease_in_out(x: float) -> float:
    if x < 0.5:
        return 2 * x * x
    return 1.0 - (-2 * x + 2) ** 2 / 2

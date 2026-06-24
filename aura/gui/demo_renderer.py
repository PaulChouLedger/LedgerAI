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
    brief_segment_start: float = 0.0
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
    R = mind * 0.45

    p.save()
    p.setRenderHint(p.Antialiasing, True)

    # Progress arc (outer ring)
    _draw_progress_arc(p, cx, cy, R * 0.92, vs.progress, pal, alpha)

    # Section-specific full-screen visual
    seg = vs.brief_segment
    if seg == 0:
        _paint_brief_executive(p, cx, cy, mind, t, vs, pal, alpha)
    elif seg == 1:
        _paint_brief_landscape(p, cx, cy, mind, t, vs, pal, alpha)
    elif seg == 2:
        _paint_brief_risks(p, cx, cy, mind, t, vs, pal, alpha)
    elif seg == 3:
        _paint_brief_recommendations(p, cx, cy, mind, t, vs, pal, alpha)

    # Segment indicator dots at bottom
    if vs.brief_total > 0:
        dot_spacing = mind * 0.035
        total_w = (vs.brief_total - 1) * dot_spacing
        dot_y = cy + R * 0.80
        for i in range(vs.brief_total):
            dx = cx - total_w / 2 + i * dot_spacing
            is_active = i == seg
            dr = mind * (0.008 if is_active else 0.005)
            if is_active:
                pulse = 0.6 + 0.4 * math.sin(t * 3)
                c = QColor(pal["accent"])
                c.setAlpha(int((180 + 70 * pulse) * alpha))
            elif i < seg:
                c = QColor(pal["accent"])
                c.setAlpha(int(140 * alpha))
            else:
                c = QColor(pal["text_dim"])
                c.setAlpha(int(60 * alpha))
            p.setPen(Qt.NoPen)
            p.setBrush(c)
            p.drawEllipse(QPointF(dx, dot_y), dr, dr)

    p.restore()


def _item_fade(elapsed, idx, stagger=12.0, fade_in=1.5):
    """Staggered fade for item idx: appears at idx*stagger seconds."""
    age = elapsed - idx * stagger
    if age < 0:
        return 0.0
    if age < fade_in:
        return _ease_in_out(age / fade_in)
    return 1.0


def _paint_brief_executive(p, cx, cy, mind, t, vs, pal, alpha):
    """Section 1: Giant radial arcs + hero numbers filling the screen."""
    elapsed = t - vs.brief_segment_start
    R = mind * 0.38

    arcs = [
        ("£1.94B", "REVENUE", 0.97, QColor(90, 220, 160), -40, 80),
        ("12,847", "PRACTITIONERS", 0.82, QColor(145, 185, 230), 140, 70),
        ("11.8%", "EBITDA", 0.59, QColor(230, 190, 110), 260, 60),
    ]

    for idx, (val, label, frac, color, start_deg, span_deg) in enumerate(arcs):
        fade = _item_fade(elapsed, idx, stagger=15.0, fade_in=2.0)
        if fade < 0.01:
            continue
        a = alpha * fade

        arc_r = R * (0.85 - idx * 0.18)
        thick = mind * 0.035
        draw_span = span_deg * min(1.0, fade * 1.3)

        # Arc track (dim)
        track_c = QColor(pal["card_bg"])
        track_c.setAlpha(int(80 * a))
        p.setPen(QPen(track_c, thick, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(Qt.NoBrush)
        p.drawArc(QRectF(cx - arc_r, cy - arc_r, arc_r * 2, arc_r * 2),
                  int(start_deg * 16), int(span_deg * 16))

        # Arc fill (bright, animated)
        ac = QColor(color)
        ac.setAlpha(int(220 * a))
        p.setPen(QPen(ac, thick, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(QRectF(cx - arc_r, cy - arc_r, arc_r * 2, arc_r * 2),
                  int(start_deg * 16), int(draw_span * 16))

        # Value at arc midpoint
        mid_deg = math.radians(start_deg + draw_span / 2)
        tx = cx + (arc_r + thick) * math.cos(mid_deg)
        ty = cy - (arc_r + thick) * math.sin(mid_deg)

        vf = QFont("Helvetica Neue", max(22, int(mind * 0.055)))
        vf.setWeight(QFont.Bold)
        p.setFont(vf)
        vc = QColor(color)
        vc.setAlpha(int(250 * a))
        p.setPen(vc)
        p.drawText(QRectF(tx - mind * 0.15, ty - mind * 0.04,
                          mind * 0.30, mind * 0.05),
                   Qt.AlignCenter, val)

        lf = QFont("Helvetica Neue", max(8, int(mind * 0.015)))
        lf.setWeight(QFont.Medium)
        lf.setLetterSpacing(QFont.PercentageSpacing, 200)
        p.setFont(lf)
        lc = QColor(pal["text_dim"])
        lc.setAlpha(int(180 * a))
        p.setPen(lc)
        p.drawText(QRectF(tx - mind * 0.15, ty + mind * 0.02,
                          mind * 0.30, mind * 0.03),
                   Qt.AlignCenter, label)

    # Central "4.2M PATIENTS" appears last
    fade4 = _item_fade(elapsed, 3, stagger=15.0, fade_in=2.0)
    if fade4 > 0.01:
        a4 = alpha * fade4
        cf = QFont("Helvetica Neue", max(36, int(mind * 0.09)))
        cf.setWeight(QFont.Bold)
        p.setFont(cf)
        cc = QColor(pal["text"])
        cc.setAlpha(int(240 * a4))
        p.setPen(cc)
        p.drawText(QRectF(cx - mind * 0.3, cy - mind * 0.06,
                          mind * 0.6, mind * 0.08),
                   Qt.AlignCenter, "4.2M")
        sf = QFont("Helvetica Neue", max(10, int(mind * 0.02)))
        sf.setLetterSpacing(QFont.PercentageSpacing, 250)
        p.setFont(sf)
        sc = QColor(pal["accent"])
        sc.setAlpha(int(180 * a4))
        p.setPen(sc)
        p.drawText(QRectF(cx - mind * 0.2, cy + mind * 0.03,
                          mind * 0.4, mind * 0.03),
                   Qt.AlignCenter, "PATIENTS")


def _paint_brief_landscape(p, cx, cy, mind, t, vs, pal, alpha):
    """Section 2: Full-width animated bar chart with growing fills."""
    elapsed = t - vs.brief_segment_start

    bars = [
        ("GP RETIREMENT", 0.80, "19.6%", QColor(220, 90, 80)),
        ("NURSE VACANCY", 0.44, "10.8%", QColor(230, 160, 70)),
        ("AGENCY SPEND", 0.65, "£94M", QColor(200, 140, 90)),
        ("CDC MARGIN", 0.73, "18%", QColor(90, 210, 150)),
        ("CARR-HILL", 0.15, "-£2.8M", QColor(180, 110, 110)),
    ]

    bar_h = mind * 0.028
    gap = mind * 0.085
    total_h = len(bars) * gap
    start_y = cy - total_h / 2
    bar_w = mind * 0.55
    bar_x = cx - bar_w / 2

    for idx, (label, frac, val_text, color) in enumerate(bars):
        fade = _item_fade(elapsed, idx, stagger=10.0, fade_in=1.5)
        if fade < 0.01:
            continue
        a = alpha * fade
        y = start_y + idx * gap

        # Label left
        lf = QFont("Helvetica Neue", max(8, int(mind * 0.016)))
        lf.setWeight(QFont.DemiBold)
        lf.setLetterSpacing(QFont.PercentageSpacing, 140)
        p.setFont(lf)
        lc = QColor(pal["text"])
        lc.setAlpha(int(200 * a))
        p.setPen(lc)
        p.drawText(QRectF(bar_x, y, bar_w, mind * 0.025),
                   Qt.AlignLeft | Qt.AlignVCenter, label)

        # Value right
        vf = QFont("Helvetica Neue", max(16, int(mind * 0.038)))
        vf.setWeight(QFont.Bold)
        p.setFont(vf)
        vc = QColor(color)
        vc.setAlpha(int(250 * a))
        p.setPen(vc)
        p.drawText(QRectF(bar_x, y, bar_w, mind * 0.025),
                   Qt.AlignRight | Qt.AlignVCenter, val_text)

        # Bar track
        by = y + mind * 0.032
        track = QColor(pal["card_bg"])
        track.setAlpha(int(120 * a))
        p.setPen(Qt.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(bar_x, by, bar_w, bar_h),
                          bar_h / 2, bar_h / 2)

        # Bar fill — animated grow
        grow = min(1.0, fade * 1.4)
        fill_w = bar_w * frac * grow
        if fill_w > 1:
            grad = QLinearGradient(bar_x, by, bar_x + fill_w, by)
            c1 = QColor(color)
            c1.setAlpha(int(100 * a))
            c2 = QColor(color)
            c2.setAlpha(int(220 * a))
            grad.setColorAt(0, c1)
            grad.setColorAt(1, c2)
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(QRectF(bar_x, by, fill_w, bar_h),
                              bar_h / 2, bar_h / 2)


def _paint_brief_risks(p, cx, cy, mind, t, vs, pal, alpha):
    """Section 3: Pulsing concentric warning rings with big numbers."""
    elapsed = t - vs.brief_segment_start

    risks = [
        ("4", "AI INCIDENTS", "UNDER REVIEW", QColor(230, 80, 70)),
        ("£14M", "DIGITAL SPEND", "TRANSFORMATION", QColor(220, 180, 80)),
        ("15%", "MARKET SHARE", "CMA THRESHOLD", QColor(240, 150, 70)),
    ]

    for idx, (value, label, sub, color) in enumerate(risks):
        fade = _item_fade(elapsed, idx, stagger=14.0, fade_in=2.0)
        if fade < 0.01:
            continue
        a = alpha * fade

        ring_r = mind * (0.36 - idx * 0.10)
        thick = mind * (0.025 - idx * 0.005)
        pulse = 0.7 + 0.3 * math.sin(t * 2.0 + idx * 1.5)

        # Pulsing ring
        rc = QColor(color)
        rc.setAlpha(int(180 * a * pulse))
        p.setPen(QPen(rc, thick, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(Qt.NoBrush)
        sweep = min(360, int(360 * min(1.0, fade * 1.5)))
        p.drawArc(QRectF(cx - ring_r, cy - ring_r,
                         ring_r * 2, ring_r * 2),
                  int((90 + idx * 30) * 16), int(-sweep * 16))

        # Glow behind ring
        glow_c = QColor(color)
        glow_c.setAlpha(int(40 * a * pulse))
        p.setPen(QPen(glow_c, thick * 3, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(QRectF(cx - ring_r, cy - ring_r,
                         ring_r * 2, ring_r * 2),
                  int((90 + idx * 30) * 16), int(-sweep * 16))

    # Central big value — show the most dramatic one
    show_idx = min(int(elapsed / 14), len(risks) - 1)
    if show_idx >= 0 and elapsed > 0:
        value, label, sub, color = risks[show_idx]
        inner_fade = _item_fade(elapsed, show_idx, stagger=14.0, fade_in=1.0)
        a = alpha * inner_fade

        vf = QFont("Helvetica Neue", max(40, int(mind * 0.10)))
        vf.setWeight(QFont.Bold)
        p.setFont(vf)
        vc = QColor(color)
        vc.setAlpha(int(250 * a))
        p.setPen(vc)
        p.drawText(QRectF(cx - mind * 0.25, cy - mind * 0.08,
                          mind * 0.5, mind * 0.10),
                   Qt.AlignCenter, value)

        lf = QFont("Helvetica Neue", max(9, int(mind * 0.018)))
        lf.setWeight(QFont.Medium)
        lf.setLetterSpacing(QFont.PercentageSpacing, 180)
        p.setFont(lf)
        lc = QColor(pal["text"])
        lc.setAlpha(int(200 * a))
        p.setPen(lc)
        p.drawText(QRectF(cx - mind * 0.25, cy + mind * 0.04,
                          mind * 0.5, mind * 0.03),
                   Qt.AlignCenter, label)

        sf = QFont("Helvetica Neue", max(7, int(mind * 0.013)))
        sf.setLetterSpacing(QFont.PercentageSpacing, 160)
        p.setFont(sf)
        sc = QColor(pal["text_dim"])
        sc.setAlpha(int(160 * a))
        p.setPen(sc)
        p.drawText(QRectF(cx - mind * 0.25, cy + mind * 0.07,
                          mind * 0.5, mind * 0.025),
                   Qt.AlignCenter, sub)


def _paint_brief_recommendations(p, cx, cy, mind, t, vs, pal, alpha):
    """Section 4: Radial action segments — pie-slice sectors with labels."""
    elapsed = t - vs.brief_segment_start

    actions = [
        ("FAST-TRACK", "GP RETENTION", QColor(90, 220, 150)),
        ("PHASE & SCALE", "CDC EXPANSION", QColor(130, 190, 240)),
        ("SUSPEND", "AI TRIAGE", QColor(230, 110, 90)),
        ("PIVOT TO PARTNER", "MERIDIAN CONNECT", QColor(200, 175, 120)),
    ]

    R = mind * 0.35
    seg_gap = 4

    for idx, (action, label, color) in enumerate(actions):
        fade = _item_fade(elapsed, idx, stagger=12.0, fade_in=2.0)
        if fade < 0.01:
            continue
        a = alpha * fade

        seg_span = 360 / len(actions) - seg_gap
        seg_start = idx * (seg_span + seg_gap) + 90
        draw_span = seg_span * min(1.0, fade * 1.3)

        # Sector arc (thick)
        sc = QColor(color)
        sc.setAlpha(int(200 * a))
        p.setPen(QPen(sc, mind * 0.04, Qt.SolidLine, Qt.FlatCap))
        p.setBrush(Qt.NoBrush)
        p.drawArc(QRectF(cx - R, cy - R, R * 2, R * 2),
                  int(seg_start * 16), int(-draw_span * 16))

        # Inner glow arc
        gc = QColor(color)
        gc.setAlpha(int(60 * a))
        p.setPen(QPen(gc, mind * 0.08, Qt.SolidLine, Qt.FlatCap))
        p.drawArc(QRectF(cx - R, cy - R, R * 2, R * 2),
                  int(seg_start * 16), int(-draw_span * 16))

        # Label at midpoint of arc
        mid_a = math.radians(seg_start + draw_span / 2)
        lx = cx + R * 0.55 * math.cos(-mid_a)
        ly = cy + R * 0.55 * math.sin(-mid_a)

        af = QFont("Helvetica Neue", max(12, int(mind * 0.028)))
        af.setWeight(QFont.Bold)
        p.setFont(af)
        ac = QColor(color)
        ac.setAlpha(int(250 * a))
        p.setPen(ac)
        p.drawText(QRectF(lx - mind * 0.18, ly - mind * 0.025,
                          mind * 0.36, mind * 0.03),
                   Qt.AlignCenter, action)

        lbf = QFont("Helvetica Neue", max(7, int(mind * 0.013)))
        lbf.setLetterSpacing(QFont.PercentageSpacing, 160)
        p.setFont(lbf)
        lbc = QColor(pal["text_dim"])
        lbc.setAlpha(int(170 * a))
        p.setPen(lbc)
        p.drawText(QRectF(lx - mind * 0.18, ly + mind * 0.01,
                          mind * 0.36, mind * 0.025),
                   Qt.AlignCenter, label)


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
    """Full-screen dramatic patient risk display."""
    age = t - pt.appear_time

    if age < 0.8:
        fade = _ease_in_out(age / 0.8)
    elif age > 18.0:
        fade = _ease_in_out(max(0.0, (22.0 - age) / 4.0))
    else:
        fade = 1.0
    a = alpha * fade

    risk_colors = {"Critical": QColor(230, 70, 60), "High": QColor(230, 170, 60),
                   "Moderate-High": QColor(210, 190, 80), "Moderate": QColor(80, 210, 140)}
    rc = risk_colors.get(pt.risk, pal["accent"])

    # Risk-colored radial glow behind everything
    glow = QRadialGradient(QPointF(cx, cy), mind * 0.42)
    gc = QColor(rc)
    gc.setAlpha(int(35 * a))
    glow.setColorAt(0.0, gc)
    glow.setColorAt(0.5, QColor(gc.red(), gc.green(), gc.blue(), int(12 * a)))
    glow.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(glow))
    p.drawEllipse(QPointF(cx, cy), mind * 0.42, mind * 0.42)

    # Wireframe head mesh — large, atmospheric, behind text
    mesh_fade = min(1.0, max(0, age - 0.2) / 1.5)
    if mesh_fade > 0.01:
        _paint_head_mesh(p, cx, cy - mind * 0.02, mind * 0.28,
                         t, pt.name, rc, pal, a * mesh_fade)

    # Pulsing risk ring
    pulse = 0.6 + 0.4 * math.sin(t * 2.0)
    ring_c = QColor(rc)
    ring_c.setAlpha(int(120 * a * pulse))
    ring_r = mind * 0.38
    p.setPen(QPen(ring_c, mind * 0.006, Qt.SolidLine, Qt.RoundCap))
    p.setBrush(Qt.NoBrush)
    sweep = min(360, int(360 * min(1.0, age * 0.8)))
    p.drawArc(QRectF(cx - ring_r, cy - ring_r, ring_r * 2, ring_r * 2),
              90 * 16, -sweep * 16)

    # Giant risk badge at top
    risk_fade = min(1.0, age / 0.5)
    badge_text = pt.risk.upper()
    bf = QFont("Helvetica Neue", max(14, int(mind * 0.028)))
    bf.setWeight(QFont.Bold)
    bf.setLetterSpacing(QFont.PercentageSpacing, 200)
    p.setFont(bf)
    bc = QColor(rc)
    bc.setAlpha(int(240 * a * risk_fade))
    p.setPen(bc)
    p.drawText(QRectF(cx - mind * 0.35, cy - mind * 0.32,
                      mind * 0.70, mind * 0.04),
               Qt.AlignCenter, badge_text)

    # Patient name — huge
    name_fade = min(1.0, max(0, age - 0.3) / 0.6)
    nf = QFont("Helvetica Neue", max(28, int(mind * 0.065)))
    nf.setWeight(QFont.Bold)
    p.setFont(nf)
    nc = QColor(pal["text"])
    nc.setAlpha(int(250 * a * name_fade))
    p.setPen(nc)
    p.drawText(QRectF(cx - mind * 0.40, cy - mind * 0.22,
                      mind * 0.80, mind * 0.08),
               Qt.AlignCenter, pt.name)

    # Age — large, dimmer
    af = QFont("Helvetica Neue", max(16, int(mind * 0.035)))
    af.setWeight(QFont.Light)
    p.setFont(af)
    ac2 = QColor(pal["text_dim"])
    ac2.setAlpha(int(200 * a * name_fade))
    p.setPen(ac2)
    p.drawText(QRectF(cx - mind * 0.30, cy - mind * 0.14,
                      mind * 0.60, mind * 0.05),
               Qt.AlignCenter, f"Age {pt.age}")

    # Conditions — stagger in
    cond_fade = min(1.0, max(0, age - 1.5) / 1.0)
    cf = QFont("Helvetica Neue", max(11, int(mind * 0.022)))
    p.setFont(cf)
    cc = QColor(pal["text"])
    cc.setAlpha(int(210 * a * cond_fade))
    p.setPen(cc)
    p.drawText(QRectF(cx - mind * 0.38, cy - mind * 0.05,
                      mind * 0.76, mind * 0.08),
               Qt.AlignCenter | Qt.TextWordWrap,
               pt.conditions)

    # Divider line
    div_fade = min(1.0, max(0, age - 2.5) / 0.5)
    div_w = mind * 0.30 * div_fade
    dc = QColor(rc)
    dc.setAlpha(int(80 * a))
    p.setPen(QPen(dc, max(1.0, mind * 0.002)))
    p.drawLine(QPointF(cx - div_w / 2, cy + mind * 0.06),
               QPointF(cx + div_w / 2, cy + mind * 0.06))

    # Action — bold, colored
    act_fade = min(1.0, max(0, age - 3.0) / 0.8)
    actf = QFont("Helvetica Neue", max(14, int(mind * 0.028)))
    actf.setWeight(QFont.DemiBold)
    p.setFont(actf)
    actc = QColor(rc)
    actc.setAlpha(int(240 * a * act_fade))
    p.setPen(actc)
    p.drawText(QRectF(cx - mind * 0.38, cy + mind * 0.08,
                      mind * 0.76, mind * 0.05),
               Qt.AlignCenter, pt.action)

    # Doctor + Specialty
    doc_fade = min(1.0, max(0, age - 4.0) / 0.8)
    df = QFont("Helvetica Neue", max(10, int(mind * 0.020)))
    df.setWeight(QFont.Light)
    p.setFont(df)
    dcc = QColor(pal["text_dim"])
    dcc.setAlpha(int(180 * a * doc_fade))
    p.setPen(dcc)
    p.drawText(QRectF(cx - mind * 0.35, cy + mind * 0.15,
                      mind * 0.70, mind * 0.04),
               Qt.AlignCenter,
               f"{pt.doctor}  ·  {pt.specialty}")


# ---------------------------------------------------------------------------
# 3D wireframe head mesh — procedural, unique per patient name
# ---------------------------------------------------------------------------

def _paint_head_mesh(p: QPainter, cx: float, cy: float, size: float,
                     t: float, name: str, accent: QColor, pal: dict,
                     alpha: float) -> None:
    """Large 3D-looking wireframe head built from latitude/longitude lines
    on an ellipsoid, with per-patient shape variation seeded from the name."""
    h = sum(ord(c) * (i + 1) for i, c in enumerate(name))

    jaw_w = 0.75 + (h % 11) * 0.025
    forehead_h = 1.05 + (h % 7) * 0.015
    cheek = 0.88 + (h % 9) * 0.012
    rot = math.sin(t * 0.3 + h) * 0.25

    n_lat = 14
    n_lon = 16

    def _head_r(lat_frac):
        """Radius at a given latitude fraction (0=top, 1=chin)."""
        if lat_frac < 0.3:
            return forehead_h * (0.6 + 0.4 * math.sin(lat_frac / 0.3 * math.pi / 2))
        elif lat_frac < 0.65:
            return cheek
        else:
            t2 = (lat_frac - 0.65) / 0.35
            return cheek * (1.0 - t2 * (1.0 - jaw_w))

    pts = []
    for i in range(n_lat + 1):
        row = []
        lat_f = i / n_lat
        y = cy - size * 0.55 + size * 1.1 * lat_f
        r = _head_r(lat_f) * size * 0.5
        for j in range(n_lon):
            lon = 2 * math.pi * j / n_lon + rot
            x = cx + r * math.sin(lon)
            z = r * math.cos(lon)
            depth = 0.4 + 0.6 * ((z / (size * 0.5)) * 0.5 + 0.5)
            row.append((x, y, depth))
        pts.append(row)

    # Longitude lines (vertical)
    for j in range(n_lon):
        for i in range(n_lat):
            x1, y1, d1 = pts[i][j]
            x2, y2, d2 = pts[i + 1][j]
            d = (d1 + d2) * 0.5
            lc = QColor(accent)
            lc.setAlpha(int(55 * alpha * d))
            p.setPen(QPen(lc, max(0.4, size * 0.003 * d)))
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    # Latitude lines (horizontal rings)
    for i in range(n_lat + 1):
        for j in range(n_lon):
            x1, y1, d1 = pts[i][j]
            x2, y2, d2 = pts[i][(j + 1) % n_lon]
            d = (d1 + d2) * 0.5
            lc = QColor(accent)
            lc.setAlpha(int(45 * alpha * d))
            p.setPen(QPen(lc, max(0.3, size * 0.002 * d)))
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    # Eyes — two small ellipsoids
    eye_lat = 0.38
    eye_y = cy - size * 0.55 + size * 1.1 * eye_lat
    eye_sep = size * (0.14 + (h % 5) * 0.01)
    eye_rx = size * 0.045
    eye_ry = size * 0.025
    ec = QColor(accent)
    ec.setAlpha(int(90 * alpha))
    p.setPen(QPen(ec, max(0.6, size * 0.004)))
    p.setBrush(Qt.NoBrush)
    for sx in (-1, 1):
        ex = cx + sx * eye_sep
        p.drawEllipse(QPointF(ex, eye_y), eye_rx, eye_ry)
        # Iris dot
        ic = QColor(accent)
        ic.setAlpha(int(120 * alpha))
        p.setPen(Qt.NoPen)
        p.setBrush(ic)
        p.drawEllipse(QPointF(ex + sx * size * 0.008, eye_y),
                      size * 0.012, size * 0.012)
        p.setBrush(Qt.NoBrush)

    # Nose — simple ridge
    nose_top = cy - size * 0.55 + size * 1.1 * 0.42
    nose_bot = cy - size * 0.55 + size * 1.1 * 0.58
    nose_w = size * 0.04
    nc = QColor(accent)
    nc.setAlpha(int(60 * alpha))
    p.setPen(QPen(nc, max(0.5, size * 0.003)))
    p.drawLine(QPointF(cx, nose_top), QPointF(cx, nose_bot))
    p.drawLine(QPointF(cx, nose_bot),
               QPointF(cx - nose_w, nose_bot + size * 0.015))
    p.drawLine(QPointF(cx, nose_bot),
               QPointF(cx + nose_w, nose_bot + size * 0.015))

    # Mouth
    mouth_y = cy - size * 0.55 + size * 1.1 * 0.68
    mouth_w = size * (0.08 + (h % 6) * 0.008)
    mc = QColor(accent)
    mc.setAlpha(int(55 * alpha))
    p.setPen(QPen(mc, max(0.5, size * 0.003)))
    path = QPainterPath()
    path.moveTo(cx - mouth_w, mouth_y)
    path.quadTo(cx, mouth_y + size * 0.015, cx + mouth_w, mouth_y)
    p.drawPath(path)


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

"""
gui.complications.domains.education -- Education domain topic.

Carbon atom visualization with animated electron orbitals, nucleus breathing,
proton ring, energy pulse rings, and ambient particle field.
Extracted from carbon_demo.py `_draw_lipizzaner_demo`.
"""

from __future__ import annotations

import math
import subprocess
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt5.QtGui import QPainter

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QBrush, QColor, QFont, QPen, QRadialGradient

from gui.complications.domains.base_domain import BaseDomainComplication
from gui.renderer import clamp


class EducationComplication(BaseDomainComplication):
    name = "Education"
    label = "Education"
    category = "Learning"

    def __init__(self, bus):
        super().__init__(bus)
        self.demo_active = False
        self.demo_start_ts = 0.0
        self._audio_proc = None

    # ------------------------------------------------------------------
    def draw_content(self, p: "QPainter", inner: float, t: float, accent: QColor) -> None:
        # Compact atom icon for the perimeter glyph
        rr = inner * 0.55
        # Orbit ellipses
        pen = QPen(QColor(60, 185, 255, 140))
        pen.setWidthF(max(1.0, inner * 0.025))
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        for i, tilt in enumerate([0.0, 60.0, -60.0]):
            p.save()
            p.rotate(tilt + t * 12.0)
            p.drawEllipse(QRectF(-rr, -rr * 0.35, 2 * rr, rr * 0.70))
            p.restore()

        # Nucleus dot
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(180, 240, 255, 200))
        nr = inner * 0.10
        p.drawEllipse(QPointF(0, 0), nr, nr)

    # ------------------------------------------------------------------
    def draw_overlay(self, p, cx, cy, mind, t, trans):
        """Carbon atom — animated for a 54-second narration experience."""
        a = clamp(float(trans), 0.0, 1.0)
        if a <= 0.002:
            return

        p.save()
        try:
            p.setRenderHint(p.Antialiasing, True)
            local_t = t

            # --- DEEP SPACE BACKDROP ---
            R_bg = mind * 0.414
            bg_grad = QRadialGradient(cx, cy, R_bg)
            bg_grad.setColorAt(0.00, QColor(3, 8, 22, int(248 * a)))
            bg_grad.setColorAt(0.55, QColor(5, 14, 38, int(222 * a)))
            bg_grad.setColorAt(0.88, QColor(2, 5, 16, int(110 * a)))
            bg_grad.setColorAt(1.00, QColor(0, 0, 0, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(bg_grad))
            p.drawEllipse(QPointF(cx, cy), R_bg, R_bg)

            # --- NUCLEUS ---
            R = mind * 0.266
            breathe = 1.0 + 0.07 * math.sin(2.0 * math.pi * local_t * 0.38)
            nuc_r = R * 0.086 * breathe

            # Outer ambient field
            amb = QRadialGradient(cx, cy, R * 0.52)
            amb.setColorAt(0.00, QColor(20, 160, 255, int(38 * a)))
            amb.setColorAt(0.45, QColor(8, 100, 200, int(20 * a)))
            amb.setColorAt(1.00, QColor(0, 40, 120, 0))
            p.setBrush(QBrush(amb))
            p.drawEllipse(QPointF(cx, cy), R * 0.52, R * 0.52)

            # Inner halo — second corona
            corona2_r = nuc_r * 5.2 * (1.0 + 0.12 * math.sin(2.0 * math.pi * local_t * 0.22 + 1.0))
            ihalo = QRadialGradient(cx, cy, corona2_r)
            ihalo.setColorAt(0.00, QColor(100, 222, 255, int(115 * a)))
            ihalo.setColorAt(0.40, QColor(48, 175, 255, int(68 * a)))
            ihalo.setColorAt(1.00, QColor(0, 80, 180, 0))
            p.setBrush(QBrush(ihalo))
            p.drawEllipse(QPointF(cx, cy), corona2_r, corona2_r)

            # Core bright nucleus
            core_g = QRadialGradient(cx, cy - nuc_r * 0.28, nuc_r * 0.5)
            core_g.setColorAt(0.00, QColor(255, 255, 255, int(255 * a)))
            core_g.setColorAt(0.32, QColor(200, 242, 255, int(220 * a)))
            core_g.setColorAt(0.68, QColor(80, 200, 255, int(148 * a)))
            core_g.setColorAt(1.00, QColor(18, 100, 220, int(55 * a)))
            p.setBrush(QBrush(core_g))
            p.drawEllipse(QPointF(cx, cy), nuc_r, nuc_r)

            # Proton ring: 6 dots for Carbon
            n_protons = 6
            p_orb_r = nuc_r * 1.68
            p_ang_off = local_t * 0.26
            for pi2 in range(n_protons):
                pa = 2.0 * math.pi * pi2 / n_protons + p_ang_off
                px2 = cx + p_orb_r * math.cos(pa)
                py2 = cy + p_orb_r * math.sin(pa) * 0.52
                pv = 0.68 + 0.32 * math.sin(pa * 2 + local_t * 1.3)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(180, 240, 255, int(165 * a * pv)))
                p.drawEllipse(QPointF(px2, py2), nuc_r * 0.23, nuc_r * 0.23)

            # --- ENERGY PULSE RINGS ---
            pulse_period = 4.5
            for pulse_offset in [0.0, pulse_period]:
                ph_pulse = (local_t + pulse_offset) % (pulse_period * 2)
                if ph_pulse < pulse_period:
                    pf = ph_pulse / pulse_period
                    pulse_r = R * 0.10 + pf * R * 1.08
                    pulse_a = int(130 * a * (1.0 - pf) ** 2.2)
                    pw = max(0.5, 1.8 * (1.0 - pf * 0.7))
                    if pulse_a > 3:
                        p.setPen(QPen(QColor(60, 185, 255, pulse_a), pw))
                        p.setBrush(Qt.NoBrush)
                        p.drawEllipse(QPointF(cx, cy), pulse_r, pulse_r)

            # --- OUTER DECORATIVE RING ---
            outer_ring_r = R * 1.19
            ring_rot_deg = -(local_t * 5.5) % 360.0
            p.save()
            try:
                p.translate(cx, cy)
                p.rotate(ring_rot_deg)
                p.setPen(QPen(QColor(60, 180, 255, int(52 * a)), 0.8))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(0, 0), outer_ring_r, outer_ring_r)
                for ti in range(24):
                    ang_t = 2.0 * math.pi * ti / 24
                    is_major = (ti % 4 == 0)
                    r_in = outer_ring_r * (0.92 if is_major else 0.96)
                    r_out = outer_ring_r * (1.08 if is_major else 1.04)
                    tick_a = int((78 if is_major else 42) * a)
                    p.setPen(QPen(QColor(80, 200, 255, tick_a), 0.7))
                    p.drawLine(
                        QPointF(r_in * math.cos(ang_t), r_in * math.sin(ang_t)),
                        QPointF(r_out * math.cos(ang_t), r_out * math.sin(ang_t))
                    )
            finally:
                p.restore()

            # --- ORBITAL PLANES + PARTICLES ---
            # Carbon (C-12): 1s² 2s² 2p²  — 6 electrons
            orbits = [
                (0.0, 0.40, 0.13, 2, 1.45),    # 1s²
                (12.0, 0.86, 0.27, 2, 0.63),    # 2s²
                (72.0, 0.92, 0.34, 1, 0.49),    # 2p₁
                (-54.0, 0.89, 0.31, 1, 0.56),   # 2p₂
            ]
            e_colors = [
                (210, 245, 255),   # 1s: near-white cyan
                (40, 190, 255),    # 2s: sky blue
                (80, 222, 255),    # 2p₁: ice cyan
                (18, 158, 240),    # 2p₂: ocean blue
            ]
            TAIL = 16
            TAIL_ARC = 0.44

            sys_rot_deg = local_t * 4.2
            p.save()
            try:
                p.translate(cx, cy)
                p.rotate(sys_rot_deg)
                p.translate(-cx, -cy)

                for idx, (rot_d, rx_u, ry_u, n_e, omega) in enumerate(orbits):
                    rx = R * rx_u
                    ry = R * ry_u
                    cr, cg, cb = e_colors[idx % len(e_colors)]
                    speed_var = 1.0 + 0.28 * math.sin(local_t * 0.18 + idx * 0.9)

                    p.save()
                    try:
                        p.translate(cx, cy)
                        p.rotate(rot_d)

                        # Orbit ring
                        p.setPen(QPen(QColor(cr, cg, cb, int(240 * a)), 3.6,
                                      Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                        p.setBrush(Qt.NoBrush)
                        p.drawEllipse(QRectF(-rx, -ry, 2 * rx, 2 * ry))

                        for ei in range(n_e):
                            base_ang = 2.0 * math.pi * (ei / max(1, n_e))
                            head_ang = base_ang + omega * local_t * speed_var + idx * 1.618

                            for step in range(TAIL + 1):
                                frac2 = step / TAIL
                                trail_ang = head_ang - (1.0 - frac2) * TAIL_ARC
                                ex = rx * math.cos(trail_ang)
                                ey = ry * math.sin(trail_ang)
                                dot_r = 1.1 + frac2 * 5.9

                                p.setPen(Qt.NoPen)
                                if step == TAIL:
                                    # Head: layered concentric glow
                                    p.setBrush(QColor(cr, cg, cb, int(65 * a)))
                                    p.drawEllipse(QPointF(ex, ey), dot_r * 2.4, dot_r * 2.4)
                                    p.setBrush(QColor(cr, cg, cb, int(145 * a)))
                                    p.drawEllipse(QPointF(ex, ey), dot_r * 1.35, dot_r * 1.35)
                                    p.setBrush(QColor(cr, cg, cb, int(215 * a)))
                                    p.drawEllipse(QPointF(ex, ey), dot_r * 0.78, dot_r * 0.78)
                                    p.setBrush(QColor(255, 255, 255, int(245 * a)))
                                    p.drawEllipse(QPointF(ex, ey), dot_r * 0.38, dot_r * 0.38)
                                else:
                                    alpha_f = frac2 * frac2
                                    p.setBrush(QColor(cr, cg, cb, int(88 * a * alpha_f)))
                                    p.drawEllipse(QPointF(ex, ey), dot_r * 0.55, dot_r * 0.55)

                    finally:
                        p.restore()

                # Ambient particle field
                N_PART = 36
                golden_ang = math.pi * (3.0 - math.sqrt(5.0))
                p.setPen(Qt.NoPen)
                for i in range(N_PART):
                    phi_i = i * golden_ang
                    r_i = R * (0.38 + 0.54 * (i / max(1, N_PART - 1)))
                    prec = local_t * (0.018 + 0.012 * math.sin(i * 0.93))
                    px3 = cx + r_i * math.cos(phi_i + prec)
                    py3 = cy + r_i * math.sin(phi_i + prec) * 0.94
                    twinkle = 0.44 + 0.56 * math.sin(
                        local_t * (0.82 + 0.58 * math.cos(i * 0.71)) + i * 0.53
                    )
                    dot_ap = int(52 * a * twinkle)
                    if dot_ap > 3:
                        p.setBrush(QColor(100, 200, 255, dot_ap))
                        p.drawEllipse(QPointF(px3, py3), 1.4, 1.4)

            finally:
                p.restore()

            # --- ELEMENT LABEL ---
            lbl_a = int(108 * a)
            if lbl_a > 4:
                p.setPen(QColor(160, 228, 255, lbl_a))
                f = QFont("Helvetica", max(10, int(mind * 0.017)))
                f.setBold(False)
                f.setLetterSpacing(QFont.AbsoluteSpacing, 3.5)
                p.setFont(f)
                p.drawText(
                    QRectF(cx - R, cy + R * 0.72, 2 * R, R * 0.30),
                    Qt.AlignCenter, "C A R B O N  \u00b7  6"
                )

        finally:
            p.restore()

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def play_audio(self):
        """Start narration audio (non-blocking)."""
        wav = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..",
                           "assets", "atom_narration.wav")
        if os.path.exists(wav):
            try:
                from core.config import ALSA_PLAYBACK_DEVICE
                self._audio_proc = subprocess.Popen(
                    ["aplay", "-D", ALSA_PLAYBACK_DEVICE, wav],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                pass

    def stop_audio(self):
        """Stop narration audio."""
        if self._audio_proc and self._audio_proc.poll() is None:
            self._audio_proc.terminate()
            self._audio_proc = None

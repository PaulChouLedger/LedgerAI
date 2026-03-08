"""
gui.complications.domains.medical -- Medical domain topic.

60-second cinematic cardiac arrhythmia visualization:
wireframe heart with dynamic EKG traces, data readouts,
phase-driven timeline (intro → healthy → arrhythmia → fibrillation → recovery).
Extracted from carbon_demo.py `_draw_heart_demo`.
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
from gui.renderer import clamp, lerp, ease_in_out


class MedicalComplication(BaseDomainComplication):
    name = "Medical"
    label = "Medical"
    category = "Health"

    def __init__(self, bus):
        super().__init__(bus)
        self.demo_active = False
        self.demo_start_ts = 0.0
        self._t0 = 0.0  # reference time for elapsed calc
        self._audio_proc = None

    # ------------------------------------------------------------------
    def draw_content(self, p: "QPainter", inner: float, t: float, accent: QColor) -> None:
        # Compact heart icon for the perimeter glyph
        rr = inner * 0.40
        pen = QPen(QColor(255, 90, 120, 180))
        pen.setWidthF(max(1.4, inner * 0.030))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)

        # Parametric heart outline
        N = 28
        pts = []
        for i in range(N):
            ang = 2.0 * math.pi * i / N
            sa = math.sin(ang)
            ca = math.cos(ang)
            x = rr * 0.75 * (16 * sa * sa * sa) / 16.0
            y = -rr * 0.75 * (13 * ca - 5 * math.cos(2 * ang) - 2 * math.cos(3 * ang) - math.cos(4 * ang)) / 16.0
            pts.append(QPointF(x, y))
        for i in range(N):
            p.drawLine(pts[i], pts[(i + 1) % N])

        # Pulse dot at center
        pulse = 0.5 + 0.5 * math.sin(t * 3.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 90, 120, int(120 + 100 * pulse)))
        dr = inner * 0.06
        p.drawEllipse(QPointF(0, inner * 0.08), dr, dr)

    # ------------------------------------------------------------------
    def draw_overlay(self, p, cx, cy, mind, t, trans):
        """60-second cinematic cardiac arrhythmia visualization."""
        a = clamp(trans, 0.0, 1.0)
        if a < 0.001:
            return

        p.save()
        try:
            p.setRenderHint(p.Antialiasing, True)

            # Elapsed time from demo start
            if self._t0 == 0.0:
                self._t0 = t
            elapsed = max(0.0, t - self._t0)

            R = mind * 0.28

            # ---- Timeline phase ----
            if elapsed < 8.0:
                phase = "intro"
                phase_u = elapsed / 8.0
            elif elapsed < 20.0:
                phase = "healthy"
                phase_u = (elapsed - 8.0) / 12.0
            elif elapsed < 35.0:
                phase = "arrhythmia"
                phase_u = (elapsed - 20.0) / 15.0
            elif elapsed < 50.0:
                phase = "fibrillation"
                phase_u = (elapsed - 35.0) / 15.0
            else:
                phase = "recovery"
                phase_u = min(1.0, (elapsed - 50.0) / 10.0)

            # ---- Scale envelope ----
            scale = ease_in_out(phase_u) if phase == "intro" else 1.0

            # ---- Heartbeat rate ----
            if phase == "healthy":
                bpm = 72.0
                irregularity = 0.0
            elif phase == "arrhythmia":
                bpm = lerp(72.0, 140.0, phase_u)
                irregularity = phase_u * 0.6
            elif phase == "fibrillation":
                bpm = lerp(140.0, 300.0, phase_u * 0.5)
                irregularity = 0.6 + 0.4 * phase_u
            elif phase == "recovery":
                bpm = lerp(160.0, 72.0, ease_in_out(phase_u))
                irregularity = lerp(0.8, 0.0, ease_in_out(phase_u))
            else:
                bpm = 60.0
                irregularity = 0.0

            beat_period = 60.0 / max(30.0, bpm)
            beat_phase = (elapsed % beat_period) / beat_period
            if irregularity > 0.01:
                jitter = irregularity * 0.4 * math.sin(elapsed * 7.3 + math.sin(elapsed * 3.1) * 2.0)
                beat_phase = (beat_phase + jitter) % 1.0

            contraction = max(0.0, 1.0 - beat_phase * 4.0) if beat_phase < 0.25 else 0.0
            if phase == "fibrillation":
                contraction = 0.3 * abs(math.sin(elapsed * 18.0 + math.sin(elapsed * 11.0) * 3.0))

            # ---- Global rotation ----
            rot_speed = 15.0
            if phase == "arrhythmia":
                rot_speed = lerp(15.0, 25.0, phase_u)
            elif phase == "fibrillation":
                rot_speed = lerp(25.0, 8.0, phase_u)
            elif phase == "recovery":
                rot_speed = lerp(8.0, 12.0, phase_u)
            rot_deg = elapsed * rot_speed

            # ---- Deep background ----
            R_bg = mind * 0.44
            bg_grad = QRadialGradient(cx, cy, R_bg)
            bg_grad.setColorAt(0.00, QColor(10, 2, 18, int(240 * a)))
            bg_grad.setColorAt(0.40, QColor(6, 1, 12, int(210 * a)))
            bg_grad.setColorAt(0.80, QColor(2, 0, 6, int(100 * a)))
            bg_grad.setColorAt(1.00, QColor(0, 0, 0, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(bg_grad))
            p.drawEllipse(QPointF(cx, cy), R_bg, R_bg)

            # ---- Warning rings (arrhythmia / fibrillation) ----
            if phase in ("arrhythmia", "fibrillation"):
                warn_u = phase_u if phase == "arrhythmia" else 1.0
                warn_pulse = 0.5 + 0.5 * math.sin(elapsed * (4.0 + 6.0 * warn_u))
                warn_a = int(45 * warn_u * warn_pulse * a)
                if warn_a > 2:
                    warn_pen = QPen(QColor(255, 50, 40, warn_a))
                    warn_pen.setWidthF(max(1.5, mind * 0.004 * (1.0 + warn_u)))
                    p.setPen(warn_pen)
                    p.setBrush(Qt.NoBrush)
                    wr = R * 1.35 * scale
                    p.drawEllipse(QPointF(cx, cy), wr, wr)
                warn_pulse2 = 0.5 + 0.5 * math.sin(elapsed * (5.0 + 4.0 * warn_u) + 1.5)
                warn_a2 = int(25 * warn_u * warn_pulse2 * a)
                if warn_a2 > 2:
                    wp2 = QPen(QColor(255, 30, 30, warn_a2))
                    wp2.setWidthF(max(1.0, mind * 0.003))
                    p.setPen(wp2)
                    p.setBrush(Qt.NoBrush)
                    wr2 = R * 1.50 * scale
                    p.drawEllipse(QPointF(cx, cy), wr2, wr2)

            # ---- Heart wireframe rendering ----
            p.save()
            try:
                p.translate(cx, cy)
                p.rotate(rot_deg)
                p.scale(scale * a, scale * a)

                hr = R * (0.85 + 0.12 * contraction)

                def heart_point(angle):
                    sa = math.sin(angle)
                    ca = math.cos(angle)
                    x = hr * 0.75 * (16 * sa * sa * sa) / 16.0
                    y = -hr * 0.75 * (13 * ca - 5 * math.cos(2 * angle) - 2 * math.cos(3 * angle) - math.cos(4 * angle)) / 16.0
                    return x, y

                N_VERTS = 36
                verts = []
                for i in range(N_VERTS):
                    ang = 2 * math.pi * i / N_VERTS
                    hx, hy = heart_point(ang)
                    if phase == "fibrillation":
                        shake = irregularity * hr * 0.05
                        hx += shake * math.sin(elapsed * 23.0 + i * 0.7)
                        hy += shake * math.cos(elapsed * 19.0 + i * 0.5)
                    verts.append((hx, hy))

                # Internal mesh vertices
                inner_layers = 3
                inner_verts = []
                for layer in range(1, inner_layers + 1):
                    layer_frac = layer / (inner_layers + 1)
                    n_ring = max(6, N_VERTS - layer * 8)
                    for i in range(n_ring):
                        ang = 2 * math.pi * i / n_ring + layer * 0.15
                        hx, hy = heart_point(ang)
                        hx *= layer_frac
                        hy *= layer_frac
                        if phase == "fibrillation":
                            shake = irregularity * hr * 0.03 * layer_frac
                            hx += shake * math.sin(elapsed * 17.0 + i * 1.1 + layer)
                            hy += shake * math.cos(elapsed * 13.0 + i * 0.9 + layer)
                        inner_verts.append((hx, hy))

                all_verts = verts + inner_verts

                # Ambient glow
                glow_r = hr * 1.5
                pulse_bright = 0.4 + 0.6 * contraction
                glow_col = QColor(200, 40, 80) if phase != "fibrillation" else QColor(255, 20, 20)
                glow_grad = QRadialGradient(QPointF(0, 0), glow_r)
                glow_grad.setColorAt(0.0, QColor(glow_col.red(), glow_col.green(), glow_col.blue(),
                                                  int(35 * pulse_bright * a)))
                glow_grad.setColorAt(0.6, QColor(glow_col.red(), glow_col.green(), glow_col.blue(),
                                                  int(10 * pulse_bright * a)))
                glow_grad.setColorAt(1.0, QColor(0, 0, 0, 0))
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(glow_grad))
                p.drawEllipse(QPointF(0, 0), glow_r, glow_r)

                # Wireframe outline edges
                wire_base_a = 180 if phase != "fibrillation" else int(140 + 80 * abs(math.sin(elapsed * 10.0)))
                wire_col = QColor(255, 90, 120, int(wire_base_a * a))
                wire_pen = QPen(wire_col)
                wire_pen.setWidthF(max(1.5, hr * 0.018))
                wire_pen.setCapStyle(Qt.RoundCap)
                p.setPen(wire_pen)
                p.setBrush(Qt.NoBrush)
                for i in range(N_VERTS):
                    n = (i + 1) % N_VERTS
                    p.drawLine(QPointF(verts[i][0], verts[i][1]),
                               QPointF(verts[n][0], verts[n][1]))

                # Internal mesh cross-links
                mesh_a = int(60 * a)
                if phase == "fibrillation":
                    mesh_a = int((40 + 40 * abs(math.sin(elapsed * 8.0))) * a)
                mesh_pen = QPen(QColor(255, 70, 100, mesh_a))
                mesh_pen.setWidthF(max(0.6, hr * 0.008))
                mesh_pen.setCapStyle(Qt.RoundCap)
                p.setPen(mesh_pen)

                # Connect outline verts to nearest inner verts
                for i in range(N_VERTS):
                    ox, oy = verts[i]
                    best_dist = 1e9
                    best_iv = None
                    for iv_idx, (ix, iy) in enumerate(inner_verts):
                        d = math.hypot(ox - ix, oy - iy)
                        if d < best_dist:
                            best_dist = d
                            best_iv = iv_idx
                    if best_iv is not None:
                        ix, iy = inner_verts[best_iv]
                        p.drawLine(QPointF(ox, oy), QPointF(ix, iy))

                # Connect inner verts within each layer ring
                offset = 0
                for layer in range(1, inner_layers + 1):
                    n_ring = max(6, N_VERTS - layer * 8)
                    for i in range(n_ring):
                        n = (i + 1) % n_ring
                        ix0, iy0 = inner_verts[offset + i]
                        ix1, iy1 = inner_verts[offset + n]
                        p.drawLine(QPointF(ix0, iy0), QPointF(ix1, iy1))
                    offset += n_ring

                # Animated cross-struts
                n_struts = 8
                for si in range(n_struts):
                    strut_seed = si * 7.31 + 2.17
                    strut_life = (elapsed * 0.4 + strut_seed) % 3.0
                    if strut_life < 2.0:
                        strut_alpha = min(1.0, strut_life / 0.4) if strut_life < 0.4 else (
                            max(0.0, 1.0 - (strut_life - 1.5) / 0.5) if strut_life > 1.5 else 1.0
                        )
                        idx_a = int((si * 13 + int(elapsed * 0.7)) % len(all_verts))
                        idx_b = int((si * 17 + 5 + int(elapsed * 0.5)) % len(all_verts))
                        if idx_a != idx_b:
                            sa_col = int(50 * strut_alpha * a)
                            sp = QPen(QColor(120, 200, 255, sa_col))
                            sp.setWidthF(max(0.5, hr * 0.006))
                            p.setPen(sp)
                            ax, ay = all_verts[idx_a]
                            bx, by = all_verts[idx_b]
                            p.drawLine(QPointF(ax, ay), QPointF(bx, by))

                # Vertex dots
                for i, (vx, vy) in enumerate(verts):
                    vpulse = 0.6 + 0.4 * math.sin(elapsed * 3.0 + i * 0.5)
                    dot_a = int(200 * vpulse * a)
                    dot_r = max(1.8, hr * 0.025) * (1.0 + 0.3 * contraction)
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(255, 140, 160, dot_a))
                    p.drawEllipse(QPointF(vx, vy), dot_r, dot_r)

                # Inner vertex dots (smaller)
                for i, (vx, vy) in enumerate(inner_verts):
                    vpulse = 0.5 + 0.5 * math.sin(elapsed * 2.5 + i * 0.7 + 1.0)
                    dot_a = int(120 * vpulse * a)
                    dot_r = max(1.2, hr * 0.016)
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(200, 100, 140, dot_a))
                    p.drawEllipse(QPointF(vx, vy), dot_r, dot_r)

                # Contraction pulse rings
                if contraction > 0.05:
                    for i in range(3):
                        pr = hr * (0.5 + 0.5 * (1.0 - contraction) + i * 0.22)
                        pa_val = int(50 * contraction * a * (1.0 - i * 0.3))
                        if pa_val > 2:
                            pp = QPen(QColor(255, 80, 120, pa_val))
                            pp.setWidthF(max(0.8, hr * 0.01))
                            p.setPen(pp)
                            p.setBrush(Qt.NoBrush)
                            p.drawEllipse(QPointF(0, 0), pr, pr)

                # Scanning line
                scan_period = 4.0 if phase != "fibrillation" else 1.5
                scan_u = (elapsed % scan_period) / scan_period
                scan_y = -hr * 0.9 + scan_u * hr * 1.8
                scan_alpha = int(40 * a * (1.0 - abs(scan_u - 0.5) * 2.0))
                if scan_alpha > 3:
                    scan_pen = QPen(QColor(100, 200, 255, scan_alpha))
                    scan_pen.setWidthF(max(0.8, hr * 0.008))
                    p.setPen(scan_pen)
                    p.drawLine(QPointF(-hr * 0.8, scan_y), QPointF(hr * 0.8, scan_y))

            finally:
                p.restore()

            # ---- Dynamic EKG traces ----
            def ekg_wave(u, bpm_val, irreg):
                cycle = (u * bpm_val / 60.0 * 2.5 + elapsed * bpm_val / 60.0) % 1.0
                val = 0.0
                if 0.0 <= cycle < 0.10:
                    val = 0.08 * math.sin(cycle / 0.10 * math.pi)
                elif 0.15 <= cycle < 0.18:
                    val = -0.12
                elif 0.18 <= cycle < 0.22:
                    val = 0.85
                elif 0.22 <= cycle < 0.26:
                    val = -0.20
                elif 0.32 <= cycle < 0.44:
                    val = 0.15 * math.sin((cycle - 0.32) / 0.12 * math.pi)
                if irreg > 0.01:
                    val += irreg * 0.25 * math.sin(u * 47.0 + elapsed * 13.0)
                    val += irreg * 0.15 * math.sin(u * 73.0 + elapsed * 7.0)
                    if irreg > 0.3 and math.sin(u * 5.0 + elapsed * 0.7) > 0.6:
                        val *= 0.1
                return val

            for trace_i in range(4):
                seed = trace_i * 3.77 + 1.23
                trace_cycle = (elapsed * 0.25 + seed) % 5.0
                if trace_cycle > 4.0:
                    continue
                if trace_cycle < 0.5:
                    trace_a = trace_cycle / 0.5
                elif trace_cycle > 3.5:
                    trace_a = max(0.0, (4.0 - trace_cycle) / 0.5)
                else:
                    trace_a = 1.0

                trace_base_y = cy + R * scale * (-0.6 + trace_i * 0.35)
                trace_x_start = cx - R * scale * 0.85
                trace_x_end = cx + R * scale * 0.85
                trace_len = trace_x_end - trace_x_start

                trace_colors = [
                    QColor(80, 255, 180, int(140 * trace_a * a)),
                    QColor(255, 120, 80, int(120 * trace_a * a)),
                    QColor(80, 180, 255, int(130 * trace_a * a)),
                    QColor(255, 200, 80, int(110 * trace_a * a)),
                ]
                ekg_col = trace_colors[trace_i]
                ep = QPen(ekg_col)
                ep.setWidthF(max(1.2, mind * 0.0025))
                ep.setCapStyle(Qt.RoundCap)
                p.setPen(ep)

                n_seg = 60
                prev_pt = None
                for si in range(n_seg):
                    u = si / n_seg
                    sx = trace_x_start + u * trace_len
                    wave = ekg_wave(u + trace_i * 0.3, bpm, irregularity)
                    sy = trace_base_y + wave * R * 0.08 * scale
                    pt = QPointF(sx, sy)
                    if prev_pt is not None:
                        p.drawLine(prev_pt, pt)
                    prev_pt = pt

            # ---- Floating data readouts ----
            data_a = int(70 * a)
            if data_a > 4 and phase != "intro":
                f_data = QFont("Courier", max(7, int(mind * 0.012)))
                f_data.setBold(True)
                p.setFont(f_data)

                bpm_display = int(bpm + 3.0 * math.sin(elapsed * 2.0) * irregularity)
                bpm_col = QColor(80, 255, 180, data_a) if phase in ("healthy", "intro", "recovery") else (
                    QColor(255, 180, 80, data_a) if phase == "arrhythmia" else QColor(255, 60, 60, data_a)
                )
                p.setPen(bpm_col)
                bpm_x = cx - R * scale * 1.25
                bpm_y = cy - R * scale * 0.95
                p.drawText(QPointF(bpm_x, bpm_y), f"HR: {bpm_display}")

                spo2 = 98 if phase in ("healthy", "intro") else (
                    int(lerp(98, 88, phase_u)) if phase == "arrhythmia" else (
                        int(lerp(88, 72, phase_u * 0.5)) if phase == "fibrillation" else int(lerp(78, 97, ease_in_out(phase_u)))
                    )
                )
                spo2_col = QColor(80, 200, 255, data_a) if spo2 > 90 else QColor(255, 100, 60, data_a)
                p.setPen(spo2_col)
                p.drawText(QPointF(bpm_x, bpm_y + mind * 0.025), f"SpO2: {spo2}%")

                rhythm_labels = {
                    "healthy": "NSR",
                    "arrhythmia": "SVT",
                    "fibrillation": "VFIB",
                    "recovery": "CONV",
                }
                rl = rhythm_labels.get(phase, "")
                if rl:
                    rl_flash = 1.0 if phase in ("healthy", "recovery") else (
                        0.5 + 0.5 * math.sin(elapsed * 6.0)
                    )
                    rl_col = QColor(255, 140, 140, int(data_a * rl_flash))
                    p.setPen(rl_col)
                    p.drawText(QPointF(cx + R * scale * 0.65, bpm_y), rl)

            # ---- Particle sparks (during stress phases) ----
            if phase in ("arrhythmia", "fibrillation") and a > 0.1:
                n_sparks = 12 if phase == "fibrillation" else 6
                for si in range(n_sparks):
                    spark_seed = si * 5.13 + elapsed * 1.7
                    spark_life = (spark_seed % 2.0) / 2.0
                    spark_ang = (si * 137.508 + elapsed * 40.0) * math.pi / 180.0
                    spark_r = R * scale * (0.3 + 0.7 * spark_life)
                    sx = cx + spark_r * math.cos(spark_ang)
                    sy = cy + spark_r * math.sin(spark_ang)
                    spark_a = int(100 * (1.0 - spark_life) * a)
                    spark_sz = max(1.0, mind * 0.004 * (1.0 - spark_life * 0.5))
                    if spark_a > 3:
                        p.setPen(Qt.NoPen)
                        p.setBrush(QColor(255, 180, 100, spark_a))
                        p.drawEllipse(QPointF(sx, sy), spark_sz, spark_sz)

            # ---- Outer decorative ring with ticks ----
            outer_r = R * 1.30 * scale
            ring_rot = -(elapsed * 6.0) % 360.0
            p.save()
            try:
                p.translate(cx, cy)
                p.rotate(ring_rot)
                p.setPen(QPen(QColor(255, 80, 120, int(35 * a)), 0.8))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(0, 0), outer_r, outer_r)
                for ti in range(24):
                    ang_t = 2 * math.pi * ti / 24
                    is_major = (ti % 4 == 0)
                    r_in = outer_r * (0.93 if is_major else 0.96)
                    r_out = outer_r * (1.07 if is_major else 1.04)
                    tick_a = int((55 if is_major else 25) * a)
                    p.setPen(QPen(QColor(255, 100, 140, tick_a), 0.7))
                    p.drawLine(
                        QPointF(r_in * math.cos(ang_t), r_in * math.sin(ang_t)),
                        QPointF(r_out * math.cos(ang_t), r_out * math.sin(ang_t))
                    )
            finally:
                p.restore()

            # ---- Phase label ----
            lbl_a = int(90 * a)
            if lbl_a > 4:
                labels = {
                    "intro": "S I N U S   R H Y T H M",
                    "healthy": "N O R M A L   B E A T",
                    "arrhythmia": "A R R H Y T H M I A",
                    "fibrillation": "V - F I B",
                    "recovery": "R E C O V E R Y",
                }
                lbl_col = QColor(255, 160, 180, lbl_a)
                if phase in ("arrhythmia", "fibrillation"):
                    flash = 0.5 + 0.5 * math.sin(elapsed * 5.0)
                    lbl_col = QColor(255, int(80 + 80 * flash), int(80 + 80 * flash),
                                     int((90 + 60 * flash) * a))
                p.setPen(lbl_col)
                f = QFont("Helvetica", max(9, int(mind * 0.016)))
                f.setBold(False)
                f.setLetterSpacing(QFont.AbsoluteSpacing, 2.5)
                p.setFont(f)
                label_r = R * 1.0 * scale
                p.drawText(
                    QRectF(cx - label_r, cy + label_r * 0.85, label_r * 2, label_r * 0.35),
                    Qt.AlignCenter, labels.get(phase, "")
                )

        finally:
            p.restore()

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def play_audio(self):
        """Start cardiac narration audio (non-blocking)."""
        mp3 = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..",
                           "assets", "cardiac_arrest.mp3")
        if os.path.exists(mp3):
            try:
                from core.config import ALSA_PLAYBACK_DEVICE
                ff = subprocess.Popen(
                    ["ffmpeg", "-i", mp3, "-loglevel", "quiet",
                     "-f", "s16le", "-acodec", "pcm_s16le",
                     "-ac", "2", "-ar", "48000", "-"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                )
                self._audio_proc = subprocess.Popen(
                    ["aplay", "-D", ALSA_PLAYBACK_DEVICE,
                     "-f", "S16_LE", "-c", "2", "-r", "48000", "-q"],
                    stdin=ff.stdout,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                ff.stdout.close()
            except FileNotFoundError:
                pass

    def stop_audio(self):
        """Stop narration audio."""
        if self._audio_proc and self._audio_proc.poll() is None:
            self._audio_proc.terminate()
            self._audio_proc = None

    def reset_timeline(self):
        """Reset the demo timeline to start from the beginning."""
        self._t0 = 0.0

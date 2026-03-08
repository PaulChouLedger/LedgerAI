"""
gui.window -- Main QWidget for the Aura round display.

Owns the paintEvent compositing loop and per-frame _tick.
Delegates all drawing to:
  - gui.renderer     (rings, celestial background, bezels, easing)
  - gui.complications (perimeter complications + overlays)
  - gui.touch        (input handling + rotation physics)
"""

from __future__ import annotations

import math
import random
import time
from typing import List, Optional

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QBrush, QColor, QPainter, QRadialGradient

from core.bus import bus
from core.config import SCREEN_W, SCREEN_H, FIXED_ROTATION_DEG
from gui.animations import LoopParams
from gui.complications import registry
from gui.complications.domains.base_domain import BaseDomainComplication
from gui.boot_renderer import (
    BootVisuals, make_falcon_stars, make_phase_bounds, paint_boot_frame,
)
from gui.renderer import (
    BackgroundCache,
    clamp, lerp, ease_in_out,
    make_celestial_stars, make_particles,
    draw_celestial, draw_chapter_ticks, draw_center_ring,
    draw_mist, draw_rings, draw_mute_wash,
)
from gui.touch import (
    RotationState,
    tick_rotation, maybe_tick_detent,
    on_drag_start, on_drag_move, on_drag_end,
    hit_rim_drag_zone, hit_center, hit_complication,
    hit_domain_glyph, glyph_layout,
    play_click, rotate_point,
)


# ---------------------------------------------------------------------------
# Default loop geometry (4 harmonic rings)
# ---------------------------------------------------------------------------

DEFAULT_LOOPS: List[LoopParams] = [
    LoopParams(0.23, 0.20, 3.0, 2.0, 0.2, 1.1, 0.07, 0),
    LoopParams(0.21, 0.24, 2.0, 3.0, 2.1, 0.6, 0.06, 10),
    LoopParams(0.19, 0.18, 5.0, 3.0, 1.4, 2.7, 0.09, -8),
    LoopParams(0.17, 0.22, 4.0, 5.0, 2.8, 1.9, 0.08, 6),
]


class AuraWindow(QWidget):
    """Main Aura display widget — full compositing + physics."""

    # Tuning
    PERIM_MARGIN_FRAC = 0.075
    DISPLAY_DIAM_MM = 70.0

    def __init__(self, parent=None, boot_mode: bool = False):
        super().__init__(parent)
        self.t0 = time.time()

        # Window setup (frameless fullscreen on round display)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background-color: black;")
        self.resize(SCREEN_W, SCREEN_H)

        # Boot mode state
        self._boot_mode = boot_mode
        self._boot_transitioning = False   # True during crossfade
        self._boot_crossfade = 0.0         # 0..1 (0=boot, 1=normal)
        self._boot_crossfade_speed = 0.35  # per second (~3s total)

        # Boot visuals (falcon animation)
        if boot_mode:
            self._boot_vis = BootVisuals(
                stars=make_falcon_stars(),
                t0=time.time(),
                phase_bounds=make_phase_bounds(6),
            )
        else:
            self._boot_vis = None

        # Rotation physics
        self.rs = RotationState()

        # Normal visual data (deferred in boot mode until transition)
        if boot_mode:
            self._stars = None
            self._particles = None
            self._bg = None
            self._loops = None
        else:
            self._stars = make_celestial_stars()
            self._particles = make_particles(180)
            self._bg = BackgroundCache()
            self._loops = list(DEFAULT_LOOPS)

        # State flags
        self.speaking = False
        self.muted = False

        # Overlay transitions (0..1)
        self.trans = 0.0          # volume
        self.trans_dir = 0
        self.bal_trans = 0.0      # balance
        self.bal_trans_dir = 0
        self.set_trans = 0.0      # settings
        self.set_trans_dir = 0
        self.settings_open = False
        self.mode = "home"
        self.mode_balance = False

        # Focus animation (shrink/fade non-focused complications)
        self.focus_comp: Optional[str] = None
        self._focus_anim = 0.0
        self._focus_scale_others = 0.45
        self._tour_highlight: Optional[str] = None  # tour mode highlight

        # Domain glyph cross-fade
        self.active_glyph: Optional[str] = None
        self._glyph_target: Optional[str] = None
        self._glyph_rings_alpha = 1.0
        self._glyph_content_alpha = 0.0
        self._glyph_fade_speed = 3.5
        self._glyph_names: List[str] = []

        # Demo dynamics
        self._pulse_until = 0.0
        self._vol_num_fade = 0.0

        # Demo views
        self.horse_demo = False
        self.horse_demo_trans = 0.0
        self.heart_demo = False
        self.heart_demo_trans = 0.0
        self._atom_audio_proc = None
        self._heart_audio_proc = None

        # Alerts demo state
        self._alert_spike_until = 0.0
        self._alert_spike_level = 0.0

        # Fade-in
        self._fade_in_alpha = 255
        self._fade_in_speed = 6

        # Render loop (~60 fps)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)
        self._last_tick = time.perf_counter()

        # Bus subscriptions
        bus.on("mute.toggled", self._on_mute)
        bus.on("volume.changed", self._on_volume)
        bus.on("tour.highlight", self._on_tour_highlight)
        if boot_mode:
            bus.on("boot.phase", self._on_boot_phase)

    def show(self):
        if not self._boot_mode:
            # Populate domain glyph names from registry
            self._glyph_names = [
                c.name for c in registry.get_all()
                if isinstance(c, BaseDomainComplication)
            ]
        super().showFullScreen()

    # ------------------------------------------------------------------
    # Bus handlers
    # ------------------------------------------------------------------

    def _on_mute(self, muted: bool = False, **_kw):
        self.muted = muted

    def _on_volume(self, level: int = 50, **_kw):
        self._vol_num_fade = 1.0

    def _on_tour_highlight(self, comp_name: str = None, **_kw):
        """Highlight a specific complication during the guided tour."""
        self._tour_highlight = comp_name
        if comp_name:
            self.focus_comp = comp_name
        else:
            self.focus_comp = None

    def _on_boot_phase(self, phase: str = "", progress: float = 0.0,
                       text: str = "", **_kw):
        """Update boot visuals from orchestrator events."""
        if self._boot_vis is not None:
            self._boot_vis.progress = progress
            self._boot_vis.phase_text = text

    # ------------------------------------------------------------------
    # Boot → normal transition
    # ------------------------------------------------------------------

    def transition_to_normal(self) -> None:
        """Begin crossfade from boot animation to normal GUI.

        Called by aura.py when boot.complete fires. Initializes normal
        visual data (stars, particles, background cache, loops) and
        starts the crossfade animation.
        """
        if not self._boot_mode:
            return
        print("[window] Starting boot → normal transition")

        # Initialize normal visual data (deferred from __init__)
        if self._stars is None:
            self._stars = make_celestial_stars()
        if self._particles is None:
            self._particles = make_particles(180)
        if self._bg is None:
            self._bg = BackgroundCache()
        if self._loops is None:
            self._loops = list(DEFAULT_LOOPS)

        # Populate domain glyphs from registry
        self._glyph_names = [
            c.name for c in registry.get_all()
            if isinstance(c, BaseDomainComplication)
        ]

        self._boot_transitioning = True
        self._boot_crossfade = 0.0

    # ------------------------------------------------------------------
    # Complication / glyph labels (from registry)
    # ------------------------------------------------------------------

    @property
    def labels(self) -> List[str]:
        return [c.name for c in registry.get_docked()]

    # ------------------------------------------------------------------
    # Frame tick
    # ------------------------------------------------------------------

    def _tick(self):
        now = time.perf_counter()
        dt = max(1e-3, min(now - self._last_tick, 0.033))
        self._last_tick = now

        # Boot crossfade transition
        if self._boot_transitioning:
            self._boot_crossfade = min(1.0, self._boot_crossfade + self._boot_crossfade_speed * dt)
            if self._boot_crossfade >= 1.0:
                self._boot_transitioning = False
                self._boot_mode = False
                self._boot_vis = None
                bus.off("boot.phase", self._on_boot_phase)
                print("[window] Boot transition complete")

        # In pure boot mode, only update timer for repaint
        if self._boot_mode and not self._boot_transitioning:
            self.update()
            return

        # Rotation physics
        tick_rotation(self.rs, dt)
        maybe_tick_detent(self.rs, play_click)

        # Overlay transitions
        self._tick_overlay_trans(dt)

        # Domain glyph cross-fade
        self._tick_glyph_fade(dt)

        # Demo dynamics (random-walk values for complication visuals)
        self._tick_demo_dynamics(dt)

        # Focus fade
        self._tick_focus_anim(dt)

        # Demo view transitions
        self._tick_demo_views(dt)

        # Fade-in
        if self._fade_in_alpha > 0:
            self._fade_in_alpha = max(0, self._fade_in_alpha - self._fade_in_speed)

        # Volume number fade
        self._vol_num_fade = max(0.0, self._vol_num_fade - 0.06)

        # Tick all docked complications
        for comp in registry.get_docked():
            comp.tick(dt)

        # Tick domain glyphs (for overlay_trans animation)
        for dname in self._glyph_names:
            dcomp = registry.get(dname)
            if dcomp:
                dcomp.tick(dt)

        self.update()

    # ----- Overlay transitions -----

    def _tick_overlay_trans(self, dt: float):
        # Volume
        if self.trans_dir != 0:
            self.trans += self.trans_dir * 0.06
            if self.trans >= 1.0:
                self.trans = 1.0; self.trans_dir = 0; self.mode = "volume"
            elif self.trans <= 0.0:
                self.trans = 0.0; self.trans_dir = 0; self.mode = "home"

        # Balance
        if self.bal_trans_dir != 0:
            self.bal_trans += self.bal_trans_dir * 0.06
            if self.bal_trans >= 1.0:
                self.bal_trans = 1.0; self.bal_trans_dir = 0; self.mode_balance = True
            elif self.bal_trans <= 0.0:
                self.bal_trans = 0.0; self.bal_trans_dir = 0; self.mode_balance = False
                if self.focus_comp == "Ledger Balance":
                    self.focus_comp = None

        # Settings
        if self.set_trans_dir != 0:
            self.set_trans += self.set_trans_dir * 0.06
            if self.set_trans >= 1.0:
                self.set_trans = 1.0; self.set_trans_dir = 0; self.settings_open = True
            elif self.set_trans <= 0.0:
                self.set_trans = 0.0; self.set_trans_dir = 0; self.settings_open = False
                if self.focus_comp in ("Settings", "Alerts"):
                    self.focus_comp = None

    # ----- Glyph cross-fade -----

    def _tick_glyph_fade(self, dt: float):
        spd = self._glyph_fade_speed * dt
        if self._glyph_target is not None:
            self._glyph_rings_alpha = max(0.0, self._glyph_rings_alpha - spd)
            if self._glyph_rings_alpha <= 0.01:
                self.active_glyph = self._glyph_target
                self._glyph_content_alpha = min(1.0, self._glyph_content_alpha + spd)
        else:
            self._glyph_content_alpha = max(0.0, self._glyph_content_alpha - spd)
            if self._glyph_content_alpha <= 0.01:
                self.active_glyph = None
                self._glyph_rings_alpha = min(1.0, self._glyph_rings_alpha + spd)

    # ----- Demo dynamics -----

    def _tick_demo_dynamics(self, dt: float):
        # Ledger Balance random walk
        bal = registry.get("Ledger Balance")
        if bal:
            v = bal._value
            v += (random.random() - 0.5) * 0.010
            v += (0.55 - v) * 0.002
            bal._value = clamp(v, 0.0, 1.0)

        # Alerts: severity walk + spikes + pulse
        alerts = registry.get("Alerts")
        if alerts:
            sev = alerts.severity
            sev += (random.random() - 0.5) * 0.020
            sev += (0.18 - sev) * 0.004
            sev = clamp(sev, 0.0, 1.0)

            now_wall = time.time()
            if now_wall > self._alert_spike_until and random.random() < 0.020:
                level = 0.55 + 0.45 * random.random()
                self._alert_spike_level = level
                self._alert_spike_until = now_wall + (0.45 + 0.90 * random.random())
                alerts.count = int(clamp(alerts.count + random.randint(1, 3), 0, 12))
                alerts._alert_pulse = 1.0

            if now_wall < self._alert_spike_until:
                sev = max(sev, self._alert_spike_level)

            alerts.severity = sev

            ap = alerts._alert_pulse
            ap = max(0.0, ap - 0.06)
            if random.random() < 0.06:
                ap = clamp(ap + 0.10 * random.random(), 0.0, 1.0)
            alerts._alert_pulse = ap

            if random.random() < 0.010:
                alerts.count = max(0, alerts.count - 1)

        # Settings: three telemetry needles
        settings = registry.get("Settings")
        if settings:
            settings._cfg_heat = clamp(settings._cfg_heat + (random.random() - 0.5) * 0.020, 0.0, 1.0)
            settings._net_flux = clamp(settings._net_flux + (random.random() - 0.5) * 0.028, 0.0, 1.0)
            settings._sys_load = clamp(settings._sys_load + (random.random() - 0.5) * 0.016, 0.0, 1.0)

    # ----- Focus animation -----

    def _tick_focus_anim(self, dt: float):
        overlay_open = (
            self.trans > 0.02 or self.bal_trans > 0.02 or self.set_trans > 0.02
            or self.settings_open or self.mode != "home" or self.mode_balance
        )
        # Tour highlight also drives the focus animation
        target = 1.0 if (overlay_open or self._tour_highlight) else 0.0
        k = 1.0 - math.exp(-dt * 6.0)
        self._focus_anim += (target - self._focus_anim) * k

    # ----- Demo view transitions -----

    def _tick_demo_views(self, dt: float):
        # Auto-dismiss domain overlays when audio finishes
        for dname in self._glyph_names:
            dcomp = registry.get(dname)
            if dcomp and dcomp.overlay_open and hasattr(dcomp, '_audio_proc'):
                proc = dcomp._audio_proc
                if proc is not None and proc.poll() is not None:
                    dcomp.close_overlay()
                    dcomp._audio_proc = None

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, _event):
        W, H = self.width(), self.height()
        cx, cy = W * 0.5, H * 0.5
        mind = min(W, H)
        t = time.time() - self.t0

        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing)

            # Route to boot / transition / normal painting
            if self._boot_mode and not self._boot_transitioning:
                self._paint_boot(p, W, H, cx, cy, mind, t)
            elif self._boot_transitioning:
                self._paint_transition(p, W, H, cx, cy, mind, t)
            else:
                self._paint_normal(p, W, H, cx, cy, mind, t)

        finally:
            p.end()

    def _paint_boot(self, p: QPainter, W: int, H: int,
                    cx: float, cy: float, mind: float, t: float) -> None:
        """Render the falcon boot animation."""
        # Apply fixed display rotation
        angle_deg = float(FIXED_ROTATION_DEG or 0.0)
        if angle_deg != 0.0:
            p.save()
            p.translate(cx, cy)
            p.rotate(-angle_deg)
            p.translate(-cx, -cy)

        paint_boot_frame(p, W, H, t, self._boot_vis)

        if angle_deg != 0.0:
            p.restore()

    def _paint_transition(self, p: QPainter, W: int, H: int,
                          cx: float, cy: float, mind: float, t: float) -> None:
        """Crossfade from boot to normal: draw both, blend via alpha."""
        cf = self._boot_crossfade  # 0 = full boot, 1 = full normal

        # Draw boot frame with fading alpha
        if cf < 1.0 and self._boot_vis is not None:
            p.save()
            p.setOpacity(1.0 - cf)
            angle_deg = float(FIXED_ROTATION_DEG or 0.0)
            if angle_deg != 0.0:
                p.translate(cx, cy)
                p.rotate(-angle_deg)
                p.translate(-cx, -cy)
            paint_boot_frame(p, W, H, t, self._boot_vis)
            p.restore()

        # Draw normal frame with increasing alpha
        if cf > 0.0:
            p.save()
            p.setOpacity(cf)
            self._paint_normal(p, W, H, cx, cy, mind, t)
            p.restore()

    def _paint_normal(self, p: QPainter, W: int, H: int,
                      cx: float, cy: float, mind: float, t: float) -> None:
        """Standard GUI rendering (all normal layers)."""
        loop_scale = ((2.1 * 1.25) * 1.33) * 0.92

        # Overlay blending
        overlay_trans = max(self.trans, self.bal_trans)
        pixelate = ease_in_out(overlay_trans)
        rings_alpha = 1.0 - overlay_trans

        # Global dial rotation wrapper
        p.save()
        p.translate(cx, cy)
        p.rotate(self.rs.rot_deg)
        p.translate(-cx, -cy)

        # --- Layer 1: Background (cached dial plate) ---
        if self._bg is not None:
            if self.muted:
                bg = self._bg.get_red(W, H, mind)
            else:
                bg = self._bg.get_blue(W, H, mind)
            if bg is not None:
                p.drawPixmap(0, 0, bg)
            else:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(10, 18, 38))
                p.drawRect(0, 0, W, H)
        else:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(10, 18, 38))
            p.drawRect(0, 0, W, H)

        # --- Layer 2: Celestial starfield ---
        if self._stars is not None:
            draw_celestial(p, cx, cy, mind, t, self._stars)

        # --- Layer 3: Chapter ticks ---
        draw_chapter_ticks(p, cx, cy, mind, t, alpha=0.85)

        # --- Layer 4: Inner rings (hero element) ---
        glyph_ra = self._glyph_rings_alpha
        domain_overlay_t = self._max_domain_overlay_trans()
        combined_rings = rings_alpha * glyph_ra * (1.0 - domain_overlay_t)
        base_speed = 0.32 if self.speaking else 0.20

        if combined_rings > 0.01 and self._loops is not None:
            draw_rings(
                p, cx, cy, mind, t, self._loops,
                base_speed=base_speed, loop_scale=loop_scale,
                alpha_scale=combined_rings, pixelate=pixelate,
                speaking=self.speaking, muted=self.muted,
            )

        # --- Layer 5: (center ring removed) ---

        # --- Layer 6: Mist / gold dust ---
        if self._particles is not None:
            draw_mist(p, cx, cy, mind, self._particles)

        # --- Layer 7: Domain glyph content (replaces rings) ---
        # TODO: draw active glyph content when glyph_content_alpha > 0

        # --- Layer 8: Domain overlays (Education atom, Medical heart, etc.) ---
        for dname in self._glyph_names:
            dcomp = registry.get(dname)
            if dcomp and dcomp.overlay_trans > 0.002:
                tr = dcomp.overlay_trans
                eased = ease_in_out(tr)
                # Dim the center
                dim_a = int(180 * eased)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(0, 0, 0, dim_a))
                p.drawEllipse(QPointF(cx, cy), mind * 0.50, mind * 0.50)
                # Draw the domain visualization (pass eased value for smooth fade)
                dcomp.draw_overlay(p, cx, cy, mind, t, eased)

        # --- Layer 9-11: Overlay compositing ---
        for comp in registry.get_docked():
            if comp.name == "Volume" and self.trans > 0.0:
                comp.draw_overlay(p, cx, cy, mind, t, self.trans)
            elif comp.name == "Settings" and self.set_trans > 0.0:
                comp.draw_overlay(p, cx, cy, mind, t, self.set_trans)
            elif comp.name == "Ledger Balance" and self.bal_trans > 0.0:
                comp.draw_overlay(p, cx, cy, mind, t, self.bal_trans)

        # --- Layer 12: Perimeter complications ---
        self._draw_perimeter_complications(p, cx, cy, mind, t)

        # --- Layer 13: Domain glyphs (between complications) ---
        self._draw_domain_glyphs(p, cx, cy, mind, t)

        # --- Layer 14: Mute wash ---
        if self.muted:
            draw_mute_wash(p, cx, cy, mind, W, H)

        # End rotation wrapper
        p.restore()

        # --- Layer 15: Fade-in overlay (not rotated) ---
        if self._fade_in_alpha > 0:
            p.save()
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, self._fade_in_alpha))
            p.drawRect(0, 0, W, H)
            p.restore()

    # ------------------------------------------------------------------
    # Perimeter drawing helpers
    # ------------------------------------------------------------------

    def _max_domain_overlay_trans(self) -> float:
        """Return the max overlay_trans across all domain glyphs."""
        mx = 0.0
        for dname in self._glyph_names:
            dcomp = registry.get(dname)
            if dcomp:
                mx = max(mx, dcomp.overlay_trans)
        return mx

    def _perimeter_geometry(self, mind: float):
        """Compute shared perimeter ring geometry."""
        base = mind * 0.085
        comp_size = base * 2.35
        perim_margin = mind * self.PERIM_MARGIN_FRAC
        outer_margin = perim_margin * 0.35
        max_radius = comp_size * 0.5
        rim_r = (mind * 0.5) - perim_margin - outer_margin - max_radius
        px_per_mm = mind / max(1.0, self.DISPLAY_DIAM_MM)
        rim_r += 3.0 * px_per_mm
        # Push 30% closer to perimeter edge
        gap_to_edge = (mind * 0.5) - rim_r
        rim_r += gap_to_edge * 0.30
        return comp_size, rim_r

    def _draw_perimeter_complications(self, p: QPainter, cx: float, cy: float,
                                      mind: float, t: float):
        comp_size, rim_r = self._perimeter_geometry(mind)
        labels = self.labels
        n = max(1, len(labels))
        docked = registry.get_docked()

        # Motion trail params
        v = self.rs.vel_dps
        vabs = abs(v)
        blur_on = self.rs.blur_enabled
        max_trail = self.rs.blur_max_deg
        ref_speed = self.rs.blur_speed_ref
        u = 0.0 if ref_speed <= 1e-6 else min(1.0, vabs / ref_speed)
        trail_deg = (u * max_trail) * (1.0 if v >= 0 else -1.0)
        ghost_alpha = 0.22 * u

        # Demo dim factor
        demo_t = self._max_domain_overlay_trans()

        pulse = time.time() < self._pulse_until

        overlay_open = (
            self.trans > 0.02 or self.bal_trans > 0.02 or self.set_trans > 0.02
            or self.settings_open or self.mode != "home" or self.mode_balance
        )
        # Tour highlight takes precedence (works even with no overlay open)
        focus = self.focus_comp if (overlay_open or self._tour_highlight) else None
        a = clamp(self._focus_anim, 0.0, 1.0)

        for i, comp in enumerate(docked):
            theta = -math.pi / 2 + i * (2 * math.pi / n)
            x = cx + rim_r * math.cos(theta)
            y = cy + rim_r * math.sin(theta)
            rot = math.degrees(theta) + 90

            # Focus scaling — highlighted comp pops, others shrink + dim
            if focus and comp.name != focus:
                local_size = comp_size * lerp(1.0, self._focus_scale_others, a)
                local_opacity = lerp(1.0, 0.25, a)
            elif focus and comp.name == focus:
                local_size = comp_size * lerp(1.0, 1.15, a)
                local_opacity = 1.0
            else:
                local_size = comp_size
                local_opacity = 1.0

            # Demo dimming
            if demo_t > 0.01:
                local_opacity *= lerp(1.0, 0.30, demo_t)
                local_size *= lerp(1.0, 0.55, demo_t)

            # Ghost trail pass (motion blur)
            if blur_on and u > 0.06:
                p.save()
                p.translate(x, y)
                p.rotate(rot - trail_deg)
                p.setOpacity(max(0.0, min(1.0, ghost_alpha * local_opacity)))
                comp.draw_glyph(p, local_size, t)
                p.restore()

            # Normal pass
            p.save()
            p.translate(x, y)
            p.rotate(rot)
            p.setOpacity(max(0.0, min(1.0, local_opacity)))
            comp.draw_glyph(p, local_size, t)
            p.restore()

    def _draw_domain_glyphs(self, p: QPainter, cx: float, cy: float,
                            mind: float, t: float):
        if not self._glyph_names:
            return
        comp_size, rim_r = self._perimeter_geometry(mind)
        glyph_size = comp_size * 0.54
        labels = self.labels
        demo_t = self._max_domain_overlay_trans()

        # Tour/focus dimming for domain glyphs
        overlay_open = (
            self.trans > 0.02 or self.bal_trans > 0.02 or self.set_trans > 0.02
            or self.settings_open or self.mode != "home" or self.mode_balance
        )
        focus = self.focus_comp if (overlay_open or self._tour_highlight) else None
        fa = clamp(self._focus_anim, 0.0, 1.0)

        for gname, gtheta in glyph_layout(labels, self._glyph_names):
            gx = cx + rim_r * math.cos(gtheta)
            gy = cy + rim_r * math.sin(gtheta)
            grot = math.degrees(gtheta) + 90

            is_active = (self.active_glyph == gname or self._glyph_target == gname)
            g_opacity = 1.0 if is_active else 0.85
            local_glyph_size = glyph_size

            # Tour/focus: highlight matching glyph, dim everything else
            if focus and fa > 0.01:
                if gname == focus:
                    # This glyph is highlighted — pop it up
                    local_glyph_size *= lerp(1.0, 1.30, fa)
                    g_opacity = 1.0
                else:
                    # Dim non-focused glyphs
                    g_opacity *= lerp(1.0, 0.20, fa)
                    local_glyph_size *= lerp(1.0, 0.50, fa)

            if demo_t > 0.01:
                g_opacity *= lerp(1.0, 0.30, demo_t)
                local_glyph_size *= lerp(1.0, 0.55, demo_t)

            p.save()
            p.translate(gx, gy)
            p.rotate(grot)
            p.setOpacity(g_opacity)
            domain_comp = registry.get(gname)
            if domain_comp and isinstance(domain_comp, BaseDomainComplication):
                domain_comp.draw_glyph(p, local_glyph_size, t)
            p.restore()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def mousePressEvent(self, ev):
        # During boot: tap to skip enrollment
        if self._boot_mode:
            bus.emit("boot.skip")
            return

        x_raw, y_raw = ev.x(), ev.y()
        cx, cy = self.width() * 0.5, self.height() * 0.5
        mind = min(self.width(), self.height())

        # Un-rotate click coordinates to match static glyph/complication positions
        # (paintEvent applies rs.rot_deg to drawing; invert it for hit-testing)
        x, y = rotate_point(x_raw, y_raw, cx, cy, -self.rs.rot_deg)

        # Check complication hits
        for comp in registry.get_docked():
            if hit_complication(comp.name, x, y, self.labels, cx, cy, mind):
                if comp.on_tap():
                    return

        # Check domain glyph hits
        if self._glyph_names:
            gname = hit_domain_glyph(x, y, cx, cy, mind, self.labels, self._glyph_names)
            if gname:
                domain_comp = registry.get(gname)
                if domain_comp and isinstance(domain_comp, BaseDomainComplication):
                    if domain_comp.overlay_open:
                        # Close: stop audio, dismiss overlay
                        domain_comp.close_overlay()
                        if hasattr(domain_comp, 'stop_audio'):
                            domain_comp.stop_audio()
                    else:
                        # Close any other open domain overlays first
                        for other_name in self._glyph_names:
                            if other_name != gname:
                                other = registry.get(other_name)
                                if other and other.overlay_open:
                                    other.close_overlay()
                                    if hasattr(other, 'stop_audio'):
                                        other.stop_audio()
                        # Open this one
                        domain_comp.open_overlay()
                        if hasattr(domain_comp, 'play_audio'):
                            domain_comp.play_audio()
                return

        # Check center tap
        if hit_center(x, y, cx, cy, mind):
            bus.emit("center.tap")
            return

        # Check rim drag (use raw coords — drag is in screen space)
        if hit_rim_drag_zone(x_raw, y_raw, cx, cy, mind):
            on_drag_start(self.rs, x_raw, y_raw, cx, cy)

    def mouseMoveEvent(self, ev):
        if self.rs.dragging:
            cx, cy = self.width() * 0.5, self.height() * 0.5
            on_drag_move(self.rs, ev.x(), ev.y(), cx, cy)

    def mouseReleaseEvent(self, ev):
        on_drag_end(self.rs)

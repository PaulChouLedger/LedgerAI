"""
gui.touch -- Touch/mouse input handling and rotation physics.

Extracted from carbon_demo.py's mouse*Event methods and elastic rotation.
All functions are stateless — they read/write a RotationState dataclass
that the window owns.
"""

from __future__ import annotations

import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

from gui.renderer import clamp


# ---------------------------------------------------------------------------
# Rotation state (owned by window, passed into physics functions)
# ---------------------------------------------------------------------------

@dataclass
class RotationState:
    rot_deg: float = 0.0
    vel_dps: float = 0.0
    target_deg: float = 0.0

    dragging: bool = False
    inertia: bool = False

    # Deferred tap/drag — press records position, move decides
    press_pending: bool = False
    press_x: float = 0.0
    press_y: float = 0.0
    press_lx: float = 0.0   # logical (rotation-corrected) coords
    press_ly: float = 0.0
    drag_threshold: float = 12.0  # pixels of movement before drag starts

    # Drag reference frame
    drag_start_rot: float = 0.0
    drag_ref_rot: float = 0.0
    drag_start_ang: float = 0.0
    last_move_t: Optional[float] = None
    last_target: Optional[float] = None
    last_drag_t: Optional[float] = None
    last_drag_ang: Optional[float] = None

    # Tunings
    spring_k: float = 125.0
    damping: float = 34.0
    friction: float = 3.2
    snap_enabled: bool = False
    snap_step: float = 30.0
    snap_k: float = 60.0
    snap_damp: float = 20.0
    snap_vel_thresh: float = 15.0
    gain_boost: float = 1.35
    vel_ref_dps: float = 220.0

    # Detent click
    detent_step: float = 0.0
    detent_last_idx: Optional[int] = None
    detent_last_t: float = 0.0
    detent_cooldown: float = 0.03

    # Blur trail
    blur_enabled: bool = True
    blur_max_deg: float = 3.0
    blur_speed_ref: float = 900.0


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def rotate_point(x: float, y: float, cx: float, cy: float, deg: float):
    """Rotate (x,y) around (cx,cy) by *deg* degrees."""
    a = math.radians(deg)
    s = math.sin(a)
    c = math.cos(a)
    dx = x - cx
    dy = y - cy
    return cx + dx * c - dy * s, cy + dx * s + dy * c


def ang_diff(a: float, b: float) -> float:
    return (a - b + 540.0) % 360.0 - 180.0


def rubber(x: float) -> float:
    """Soft nonlinearity for iPhone-like elastic feel."""
    return 180.0 * (x / 180.0) / (1.0 + abs(x / 180.0))


# ---------------------------------------------------------------------------
# Rotation physics (called every tick)
# ---------------------------------------------------------------------------

def tick_rotation(rs: RotationState, dt: float) -> None:
    """Update rotation based on spring/damping physics."""
    cur = rs.rot_deg

    if rs.dragging:
        # Direct follow: rotation tracks the finger target immediately
        # (velocity is estimated separately for inertia handoff)
        rs.rot_deg = rs.target_deg

    elif rs.inertia:
        cur = (cur + rs.vel_dps * dt) % 360.0
        rs.vel_dps *= math.exp(-rs.friction * dt)
        if abs(rs.vel_dps) < 2.0:
            rs.vel_dps = 0.0
            rs.inertia = False
        rs.rot_deg = cur

    # Snap-to-detent
    if rs.snap_enabled and (not rs.dragging) and (not rs.inertia):
        if abs(rs.vel_dps) < rs.snap_vel_thresh:
            step = rs.snap_step
            target = (round(cur / step) * step) % 360.0
            e = rubber(ang_diff(target, cur))
            if abs(e) > 0.25:
                a = rs.snap_k * e - rs.snap_damp * rs.vel_dps
                rs.vel_dps += a * dt
                cur = (cur + rs.vel_dps * dt) % 360.0
                rs.rot_deg = cur
            else:
                rs.rot_deg = target
                rs.vel_dps = 0.0


def maybe_tick_detent(rs: RotationState, click_fn) -> None:
    """Haptic detent click during rotation."""
    if not (rs.dragging or rs.inertia):
        return
    if rs.detent_step <= 0.0:
        return

    now = time.time()
    if now - rs.detent_last_t < rs.detent_cooldown:
        return

    ang = rs.rot_deg % 360.0
    idx = int(round(ang / rs.detent_step))
    if rs.detent_last_idx is None:
        rs.detent_last_idx = idx
        return

    if idx != rs.detent_last_idx:
        rs.detent_last_idx = idx
        rs.detent_last_t = now
        click_fn()


# ---------------------------------------------------------------------------
# Drag event handlers
# ---------------------------------------------------------------------------

def on_drag_start(rs: RotationState, x0: float, y0: float, cx: float, cy: float) -> None:
    """Begin a rotation drag."""
    rs.dragging = True
    rs.inertia = False
    rs.vel_dps = 0.0
    rs.target_deg = rs.rot_deg
    rs.last_move_t = None
    rs.last_target = rs.rot_deg
    rs.drag_start_rot = rs.rot_deg
    rs.drag_ref_rot = rs.rot_deg
    rs.last_drag_t = None
    rs.last_drag_ang = None

    x_ref, y_ref = rotate_point(x0, y0, cx, cy, -rs.drag_ref_rot)
    rs.drag_start_ang = math.degrees(math.atan2(y_ref - cy, x_ref - cx))


def on_drag_move(rs: RotationState, x0: float, y0: float, cx: float, cy: float) -> None:
    """Continue a rotation drag."""
    if not rs.dragging:
        return

    x_ref, y_ref = rotate_point(x0, y0, cx, cy, -rs.drag_ref_rot)
    ang = math.degrees(math.atan2(y_ref - cy, x_ref - cx))
    delta = ang - rs.drag_start_ang

    # Velocity-scaled gain
    now = time.time()
    dt = 0.016 if rs.last_drag_t is None else max(1e-3, now - rs.last_drag_t)
    last_ang = rs.last_drag_ang if rs.last_drag_ang is not None else ang
    d_ang = ang - last_ang
    if d_ang > 180.0: d_ang -= 360.0
    elif d_ang < -180.0: d_ang += 360.0
    vel_dps = abs(d_ang) / dt
    u = 1.0 - math.exp(-vel_dps / rs.vel_ref_dps)
    gain = 1.0 + rs.gain_boost * u
    delta *= gain
    rs.last_drag_t = now
    rs.last_drag_ang = ang

    if delta > 180.0: delta -= 360.0
    elif delta < -180.0: delta += 360.0

    target = (rs.drag_start_rot + delta) % 360.0

    # Low-pass + deadband (high beta for direct feel)
    prev_tgt = rs.target_deg
    d_tgt = (target - prev_tgt + 540.0) % 360.0 - 180.0
    if abs(d_tgt) < 0.15:
        d_tgt = 0.0
    beta = 0.70
    smoothed = (prev_tgt + beta * d_tgt) % 360.0
    rs.target_deg = smoothed
    rs.inertia = False

    # Velocity estimate for inertia handoff
    t_now = time.time()
    if rs.last_move_t is not None and rs.last_target is not None:
        vdt = t_now - rs.last_move_t
        if vdt < 0.016: vdt = 0.016
        elif vdt > 0.050: vdt = 0.050
        d = (smoothed - rs.last_target + 540.0) % 360.0 - 180.0
        if abs(d) < 0.35: d = 0.0
        v = 1.4 * (d / vdt)
        v = max(-540.0, min(540.0, v))
        rs.vel_dps = 0.6 * rs.vel_dps + 0.4 * v

    rs.last_move_t = t_now
    rs.last_target = smoothed


def on_drag_end(rs: RotationState) -> None:
    """End rotation drag; transfer momentum to inertia."""
    if not rs.dragging:
        return
    rs.dragging = False

    v = rs.vel_dps
    if abs(v) > 12.0:
        v = max(-1080.0, min(1080.0, v))
        rs.vel_dps = v
        rs.inertia = True
    else:
        rs.vel_dps = 0.0
        rs.inertia = False


# ---------------------------------------------------------------------------
# Hit testing
# ---------------------------------------------------------------------------

def hit_rim_drag_zone(x: float, y: float, cx: float, cy: float, mind: float) -> bool:
    r = math.hypot(x - cx, y - cy)
    r_outer = mind * 0.5
    return (r_outer * 0.60) <= r <= (r_outer * 0.98)


def hit_center(x: float, y: float, cx: float, cy: float, mind: float) -> bool:
    r = mind * 0.12
    dx, dy = x - cx, y - cy
    return (dx * dx + dy * dy) <= (r * r)


def hit_complication(
    name: str, x: float, y: float,
    labels: list, cx: float, cy: float, mind: float,
    perim_margin_frac: float = 0.038,
) -> bool:
    """Test if (x,y) hits the named complication on the perimeter ring."""
    if name not in labels:
        return False

    base = mind * 0.085
    comp_size = base * 2.35
    perim_margin = mind * perim_margin_frac
    outer_margin = perim_margin * 0.35
    max_radius = comp_size * 0.5
    rim_r = (mind * 0.5) - perim_margin - outer_margin - max_radius
    px_per_mm = mind / max(1.0, 70.0)
    rim_r += 3.0 * px_per_mm
    # Push 30% closer to perimeter edge (must match window._perimeter_geometry)
    gap_to_edge = (mind * 0.5) - rim_r
    rim_r += gap_to_edge * 0.30

    idx = labels.index(name)
    n = max(1, len(labels))
    theta = -math.pi / 2 + idx * (2 * math.pi / n)
    gx = cx + rim_r * math.cos(theta)
    gy = cy + rim_r * math.sin(theta)

    hit_r = max_radius * 1.35
    dx = x - gx
    dy = y - gy
    return (dx * dx + dy * dy) <= (hit_r * hit_r)


def glyph_layout(labels: list, glyph_names: list):
    """Return list of (name, theta) for domain glyphs between complications."""
    n_comp = len(labels)
    out = []
    for i, gname in enumerate(glyph_names):
        theta_a = -math.pi / 2 + i * (2 * math.pi / n_comp)
        theta_b = -math.pi / 2 + ((i + 1) % n_comp) * (2 * math.pi / n_comp)
        if theta_b <= theta_a:
            theta_b += 2 * math.pi
        theta_mid = (theta_a + theta_b) * 0.5
        out.append((gname, theta_mid))
    return out


def hit_domain_glyph(
    x: float, y: float, cx: float, cy: float, mind: float,
    labels: list, glyph_names: list,
    perim_margin_frac: float = 0.038,
) -> Optional[str]:
    """Return the name of the hit glyph, or None."""
    base = mind * 0.085
    comp_size = base * 2.35
    glyph_size = comp_size * 0.54

    perim_margin = mind * perim_margin_frac
    outer_margin = perim_margin * 0.35
    max_radius = comp_size * 0.5
    rim_r = (mind * 0.5) - perim_margin - outer_margin - max_radius
    px_per_mm = mind / max(1.0, 70.0)
    rim_r += 3.0 * px_per_mm
    # Push 30% closer to perimeter edge (must match window._perimeter_geometry)
    gap_to_edge = (mind * 0.5) - rim_r
    rim_r += gap_to_edge * 0.30

    for gname, gtheta in glyph_layout(labels, glyph_names):
        gx = cx + rim_r * math.cos(gtheta)
        gy = cy + rim_r * math.sin(gtheta)
        hit_r = glyph_size * 0.5 * 1.5
        dx = x - gx
        dy = y - gy
        if (dx * dx + dy * dy) <= (hit_r * hit_r):
            return gname
    return None


# ---------------------------------------------------------------------------
# Volume arc helpers
# ---------------------------------------------------------------------------

def vol_arc_angles():
    return 225.0, 270.0


def hit_volume_arc(x: float, y: float, cx: float, cy: float, mind: float) -> bool:
    R = mind * 0.34 * 0.75
    dx, dy = x - cx, y - cy
    d = math.sqrt(dx * dx + dy * dy)
    return (R * 0.72) <= d <= (R * 1.10)


def vol_xy_to_value(x: float, y: float, cx: float, cy: float) -> int:
    dx, dy = x - cx, y - cy
    ang = math.degrees(math.atan2(dy, dx))
    if ang < 0:
        ang += 360.0

    start_deg, sweep_deg = vol_arc_angles()

    def dist_cw(a, b):
        d = b - a
        if d < 0: d += 360.0
        return d

    d = dist_cw(start_deg, ang)

    if d > sweep_deg:
        def circ_dist(a, b):
            d1 = dist_cw(a, b)
            d2 = dist_cw(b, a)
            return min(d1, d2)
        if circ_dist(ang, start_deg) <= circ_dist(ang, (start_deg + sweep_deg) % 360.0):
            d = 0.0
        else:
            d = sweep_deg

    v = (d / sweep_deg) * 100.0
    return int(round(clamp(v, 0.0, 100.0)))


# ---------------------------------------------------------------------------
# Click sound helper
# ---------------------------------------------------------------------------

_CLICK_WAV = "/home/ledger/LedgerAI/assets/click.wav"
_last_click_ts = 0.0


def play_click() -> None:
    global _last_click_ts
    now = time.time()
    if now - _last_click_ts < 0.040:
        return
    _last_click_ts = now
    try:
        from core.config import ALSA_PLAYBACK_DEVICE
        subprocess.Popen(
            ["aplay", "-D", ALSA_PLAYBACK_DEVICE, "-q", _CLICK_WAV],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass

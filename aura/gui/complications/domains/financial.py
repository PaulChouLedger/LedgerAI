"""
gui.complications.domains.financial -- Financial domain complication.

Joke portfolio overlay: four meme tickers ($PENGUIN, $LOBSTAR, $ALIENS,
$PSYOPANIME) all pump for 10 seconds, then crash wildly to zero over
120 seconds total.  gone.mp3 plays immediately on open (140s clip).
The overlay persists for 140s so the audio plays out 20s past the crash.
"""

from __future__ import annotations

import math
import os
import subprocess
import random
from typing import TYPE_CHECKING, List, Tuple, Dict

if TYPE_CHECKING:
    from PyQt5.QtGui import QPainter

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QBrush, QColor, QFont, QPen, QRadialGradient, QPainterPath

from gui.complications.domains.base_domain import BaseDomainComplication
from gui.renderer import clamp


# ---------------------------------------------------------------------------
# Ticker config
# ---------------------------------------------------------------------------

_DURATION = 140.0       # total overlay time (matches audio clip)
_N_TICKS = 280          # one tick every 0.5s for the full 140s
_ZERO_TICK = 240        # tick at which all stocks hit zero (= 120s = 12/14 of 140)
_BULL_TICKS = 20        # first 20 ticks (10s) are the bull pump

_PORTFOLIO_START = 1_000_000.00   # $1M starting portfolio value

_TICKERS: List[Dict] = [
    {"sym": "$PENGUIN",      "start": 4.20,   "peak_mult": 3.2,  "seed": 101,
     "color": (85, 210, 235),   "color_hi": (140, 235, 255)},
    {"sym": "$LOBSTAR",      "start": 0.85,   "peak_mult": 3.8,  "seed": 202,
     "color": (255, 120, 80),   "color_hi": (255, 175, 140)},
    {"sym": "$ALIENS",       "start": 47.00,  "peak_mult": 2.6,  "seed": 303,
     "color": (120, 255, 140),  "color_hi": (180, 255, 200)},
    {"sym": "$PSYOPANIME",   "start": 12.50,  "peak_mult": 3.0,  "seed": 404,
     "color": (200, 140, 255),  "color_hi": (225, 185, 255)},
]


def _generate_ticker(start: float, peak_mult: float, seed: int) -> List[float]:
    """Generate one ticker: pump for 10s, crash to zero by 120s, stay at zero to 140s."""
    rng = random.Random(seed)
    punch_rng = random.Random(seed + 1000)
    p1 = punch_rng.uniform(0.15, 0.30)
    p2 = punch_rng.uniform(0.42, 0.58)
    p3 = punch_rng.uniform(0.72, 0.85)
    b1 = punch_rng.uniform(0.25, 0.40)
    b2 = punch_rng.uniform(0.55, 0.70)

    vals = [start]
    v = start
    peak = start * peak_mult

    for i in range(1, _N_TICKS):
        if i >= _ZERO_TICK:
            # --- DEAD ZONE: stay at zero for the last 20s ---
            vals.append(0.00)
            continue

        bull_progress = i / max(1, _BULL_TICKS)

        if i <= _BULL_TICKS:
            # --- BULL PHASE: pump it up ---
            target = start + (peak - start) * (bull_progress ** 0.7)
            noise = rng.gauss(0, 0.04) * v
            v = v * 0.85 + target * 0.15 + noise
            v = max(start * 0.5, v)
        else:
            # --- CRASH PHASE: chaotic decline toward zero at _ZERO_TICK ---
            crash_progress = (i - _BULL_TICKS) / (_ZERO_TICK - _BULL_TICKS - 1)

            envelope = peak * max(0.0, (1.0 - crash_progress ** 1.2) ** 2.5)

            chaos = 1.0 - abs(crash_progress - 0.4) * 1.3
            chaos = max(0.15, min(1.0, chaos))
            noise = rng.gauss(0, 0.25 * chaos) * v

            gut = 1.0
            for pt in [p1, p2, p3]:
                if abs(crash_progress - pt) < 0.02:
                    gut = rng.uniform(0.30, 0.55)

            bounce = 1.0
            if abs(crash_progress - b1) < 0.03:
                bounce = rng.uniform(1.10, 1.25)
            elif abs(crash_progress - b2) < 0.02:
                bounce = rng.uniform(1.05, 1.15)

            v = v * gut * bounce + noise
            v = v * 0.92 + envelope * 0.08
            v = max(0.001, v)

        vals.append(v)

    # Force the last 8 active ticks (before the dead zone) into the dirt
    for i in range(_ZERO_TICK - 8, _ZERO_TICK):
        if 0 <= i < len(vals):
            vals[i] = max(0.001, vals[i] * 0.12)
    # Ensure exact zero at the boundary
    for i in range(_ZERO_TICK, _N_TICKS):
        if i < len(vals):
            vals[i] = 0.00

    return vals


# Pre-bake all trajectories
_TRAJECTORIES: List[List[float]] = []
_PEAKS: List[float] = []
for _tk in _TICKERS:
    _traj = _generate_ticker(_tk["start"], _tk["peak_mult"], _tk["seed"])
    _TRAJECTORIES.append(_traj)
    _PEAKS.append(max(_traj))


class FinancialComplication(BaseDomainComplication):
    name = "Financial"
    label = "Financial"
    category = "Topics"

    def __init__(self, bus):
        super().__init__(bus)
        self._t0: float = 0.0
        self._audio_proc = None
        self._audio_started = False

    # ------------------------------------------------------------------
    def draw_content(self, p: "QPainter", inner: float, t: float, accent: QColor) -> None:
        pass  # Icon drawn by BaseDomainComplication.draw_glyph

    # ------------------------------------------------------------------
    def play_audio(self):
        """Play gone.mp3 immediately when overlay opens."""
        self._t0 = 0.0
        self._audio_started = False

    def stop_audio(self):
        if self._audio_proc and self._audio_proc.poll() is None:
            self._audio_proc.terminate()
        self._audio_proc = None
        self._audio_started = False
        self._t0 = 0.0

    def _start_audio(self):
        """Play gone.mp3 at overlay open."""
        if self._audio_started:
            return
        self._audio_started = True
        mp3 = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..",
                           "assets", "gone.mp3")
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

    # ------------------------------------------------------------------
    def draw_overlay(self, p, cx, cy, mind, t, trans):
        """Live meme-stock portfolio crash over 140 seconds."""
        a = clamp(float(trans), 0.0, 1.0)
        if a < 0.002:
            return

        # Track elapsed time from first draw
        if self._t0 == 0.0:
            self._t0 = t
            self._start_audio()
        elapsed = max(0.0, t - self._t0)

        if elapsed > _DURATION:
            elapsed = _DURATION

        # Single unified timeline: 0→1 over 140s, one tick per 0.5s
        progress = clamp(elapsed / _DURATION, 0.0, 1.0)
        ticks_shown = int(progress * (_N_TICKS - 1)) + 1
        ticks_shown = min(ticks_shown, _N_TICKS)

        # All dead once we reach the zero tick (120s = tick 240)
        all_dead = ticks_shown >= _ZERO_TICK
        post_crash_secs = max(0, int(elapsed - (_ZERO_TICK * _DURATION / _N_TICKS)))

        p.save()
        try:
            p.setRenderHint(p.Antialiasing, True)

            R = mind * 0.36

            # --- Portfolio value ($1M at start, proportional to aggregate) ---
            total_start_raw = sum(tk["start"] for tk in _TICKERS)
            total_now_raw = sum(_TRAJECTORIES[ti][ticks_shown - 1] for ti in range(4))
            ratio = total_now_raw / total_start_raw if total_start_raw > 0 else 0
            portfolio_val = _PORTFOLIO_START * ratio
            portfolio_chg = (ratio - 1.0) * 100.0

            doom = clamp(1.0 - ratio, 0.0, 1.0)

            # --- Backdrop (reddens as doom rises) ---
            bg = QRadialGradient(cx, cy, R)
            bg_r = int(6 + 45 * doom)
            bg_g = int(14 - 8 * doom)
            bg_b = int(32 - 18 * doom)
            bg.setColorAt(0.00, QColor(bg_r, bg_g, bg_b, int(248 * a)))
            bg.setColorAt(0.65, QColor(int(4 + 22 * doom), 8, 22, int(235 * a)))
            bg.setColorAt(1.00, QColor(0, 0, 0, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(bg))
            p.drawEllipse(QPointF(cx, cy), R, R)

            clip = QPainterPath()
            clip.addEllipse(QPointF(cx, cy), R * 0.96, R * 0.96)
            p.setClipPath(clip)

            # --- Portfolio value line (top) ---
            pv_h = R * 0.10
            pv_top = cy - R * 0.88
            if not all_dead:
                f_pv = QFont("Helvetica", max(6, int(R * 0.055)))
                f_pv.setBold(True)
                p.setFont(f_pv)

                if portfolio_chg >= 0:
                    pv_col = QColor(80, 220, 155, int(220 * a))
                else:
                    ri = clamp(abs(portfolio_chg) / 100.0, 0.0, 1.0)
                    pv_col = QColor(int(80 + 175 * ri), int(220 - 170 * ri),
                                    int(155 - 130 * ri), int(220 * a))
                p.setPen(pv_col)
                sign = "+" if portfolio_chg >= 0 else ""
                if portfolio_val >= 1000:
                    pv_str = f"${portfolio_val:,.0f}"
                elif portfolio_val >= 1:
                    pv_str = f"${portfolio_val:,.2f}"
                else:
                    pv_str = f"${portfolio_val:.4f}"
                p.drawText(
                    QRectF(cx - R * 0.85, pv_top, R * 1.70, pv_h),
                    Qt.AlignCenter, f"{pv_str}  ({sign}{portfolio_chg:.1f}%)"
                )

            # --- 2x2 chart grid (centered vertically in the circle) ---
            gap = R * 0.03
            cell_w = R * 0.74
            cell_h = R * 0.46
            grid_total_h = 2 * cell_h + gap
            grid_total_w = 2 * cell_w + gap
            grid_x0 = cx - grid_total_w * 0.5
            grid_y0 = cy - grid_total_h * 0.5 + R * 0.04  # nudge down slightly for portfolio text
            grid_positions = [
                (grid_x0,              grid_y0),
                (grid_x0 + cell_w + gap, grid_y0),
                (grid_x0,              grid_y0 + cell_h + gap),
                (grid_x0 + cell_w + gap, grid_y0 + cell_h + gap),
            ]

            for ti in range(4):
                tk = _TICKERS[ti]
                traj = _TRAJECTORIES[ti]
                peak = _PEAKS[ti]
                gx, gy = grid_positions[ti]
                cr, cg, cb = tk["color"]

                cur_val = traj[ticks_shown - 1]
                tk_doom = clamp(1.0 - (cur_val / max(0.001, peak)), 0.0, 1.0)

                # Cell background
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(12, 16, 30, int(50 * a)))
                p.drawRoundedRect(QRectF(gx, gy, cell_w, cell_h),
                                  R * 0.012, R * 0.012)

                # Cell border
                alive_a = int((20 + 22 * (1.0 - tk_doom)) * a)
                border_pen = QPen(QColor(cr, cg, cb, alive_a))
                border_pen.setWidthF(max(0.4, R * 0.003))
                p.setPen(border_pen)
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(QRectF(gx, gy, cell_w, cell_h),
                                  R * 0.012, R * 0.012)

                # Ticker label (small, top-left inside chart)
                f_sym = QFont("Helvetica", max(4, int(R * 0.034)))
                f_sym.setBold(True)
                p.setFont(f_sym)
                p.setPen(QColor(cr, cg, cb, int(200 * a)))
                p.drawText(
                    QRectF(gx + cell_w * 0.04, gy + cell_h * 0.04,
                           cell_w * 0.92, cell_h * 0.16),
                    Qt.AlignVCenter | Qt.AlignLeft, tk["sym"]
                )

                # Chart area (fills most of the cell)
                ch_l = gx + cell_w * 0.04
                ch_r_x = gx + cell_w * 0.96
                ch_top = gy + cell_h * 0.20
                ch_bot = gy + cell_h * 0.96
                ch_w = ch_r_x - ch_l
                ch_h = ch_bot - ch_top

                v_max = peak * 1.08
                if v_max < 0.01:
                    v_max = 1.0

                def _v2y(v, _vmax=v_max, _top=ch_top, _h=ch_h):
                    vn = clamp(v / _vmax, 0.0, 1.0)
                    return _top + (1.0 - math.sqrt(vn)) * _h

                # Subtle grid
                grid_pen = QPen(QColor(80, 110, 150, int(8 * a)))
                grid_pen.setWidthF(max(0.3, R * 0.001))
                p.setPen(grid_pen)
                for gi in range(4):
                    gy_l = ch_top + gi * (ch_h / 3)
                    p.drawLine(QPointF(ch_l, gy_l), QPointF(ch_r_x, gy_l))

                # Price line
                if ticks_shown >= 2:
                    step_x = ch_w / (_N_TICKS - 1)
                    line_path = QPainterPath()
                    for i in range(ticks_shown):
                        px = ch_l + i * step_x
                        py = _v2y(traj[i])
                        if i == 0:
                            line_path.moveTo(px, py)
                        else:
                            line_path.lineTo(px, py)

                    # Fill under line
                    fill_path = QPainterPath(line_path)
                    last_x = ch_l + (ticks_shown - 1) * step_x
                    fill_path.lineTo(last_x, ch_bot)
                    fill_path.lineTo(ch_l, ch_bot)
                    fill_path.closeSubpath()

                    fr = int(cr * (1.0 - 0.5 * tk_doom) + 200 * 0.5 * tk_doom)
                    fg = int(cg * (1.0 - 0.7 * tk_doom) + 60 * 0.7 * tk_doom)
                    fb = int(cb * (1.0 - 0.6 * tk_doom) + 50 * 0.6 * tk_doom)
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(fr, fg, fb, int(20 * a)))
                    p.drawPath(fill_path)

                    # Line stroke
                    lr = int(cr * (1.0 - 0.4 * tk_doom) + 230 * 0.4 * tk_doom)
                    lg_ = int(cg * (1.0 - 0.6 * tk_doom) + 70 * 0.6 * tk_doom)
                    lb = int(cb * (1.0 - 0.5 * tk_doom) + 55 * 0.5 * tk_doom)
                    line_pen = QPen(QColor(lr, lg_, lb, int(190 * a)))
                    line_pen.setWidthF(max(0.7, R * 0.005))
                    line_pen.setCapStyle(Qt.RoundCap)
                    line_pen.setJoinStyle(Qt.RoundJoin)
                    p.setPen(line_pen)
                    p.setBrush(Qt.NoBrush)
                    p.drawPath(line_path)

                    # Live dot
                    dot_x = ch_l + (ticks_shown - 1) * step_x
                    dot_y = _v2y(cur_val)
                    pulse = 0.5 + 0.5 * math.sin(t * 5.0 + ti * 1.5)
                    dot_sz = R * (0.007 + 0.003 * pulse)
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(255, 255, 255, int(200 * a)))
                    p.drawEllipse(QPointF(dot_x, dot_y), dot_sz, dot_sz)
                    p.setBrush(QColor(lr, lg_, lb, int(35 * a * pulse)))
                    p.drawEllipse(QPointF(dot_x, dot_y), dot_sz * 2.5, dot_sz * 2.5)

                # Red X when zeroed
                if cur_val <= 0.001:
                    x_pen = QPen(QColor(235, 70, 60, int(110 * a)))
                    x_pen.setWidthF(max(1.2, R * 0.010))
                    x_pen.setCapStyle(Qt.RoundCap)
                    p.setPen(x_pen)
                    mx = ch_h * 0.12
                    p.drawLine(QPointF(ch_l + mx, ch_top + mx),
                               QPointF(ch_r_x - mx, ch_bot - mx))
                    p.drawLine(QPointF(ch_r_x - mx, ch_top + mx),
                               QPointF(ch_l + mx, ch_bot - mx))

            # --- "And... it's gone." when all dead ---
            if all_dead:
                # Dark overlay dims the charts
                p.setClipPath(clip)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(6, 3, 3, int(180 * a)))
                p.drawEllipse(QPointF(cx, cy), R, R)

                # Pulsing red vignette on top
                vig_pulse = 0.5 + 0.5 * math.sin(t * 2.0)
                vig = QRadialGradient(cx, cy, R * 0.90)
                vig.setColorAt(0.0, QColor(0, 0, 0, 0))
                vig.setColorAt(0.6, QColor(100, 15, 10, int(50 * a * vig_pulse)))
                vig.setColorAt(1.0, QColor(160, 25, 15, int(100 * a * vig_pulse)))
                p.setBrush(QBrush(vig))
                p.drawEllipse(QPointF(cx, cy), R, R)

                blink = 0.5 + 0.5 * math.sin(t * 3.0)

                # Portfolio value: $0.00 at top
                f_zero = QFont("Helvetica", max(7, int(R * 0.060)))
                f_zero.setBold(True)
                p.setFont(f_zero)
                p.setPen(QColor(235, 70, 60, int(220 * a * blink)))
                p.drawText(
                    QRectF(cx - R * 0.80, pv_top, R * 1.60, pv_h),
                    Qt.AlignCenter, "$0.00  (-100.0%)"
                )

                # Red glow behind main text
                glow = QRadialGradient(cx, cy, R * 0.40)
                glow.setColorAt(0.0, QColor(255, 50, 30, int(45 * a * blink)))
                glow.setColorAt(1.0, QColor(0, 0, 0, 0))
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(glow))
                p.drawEllipse(QPointF(cx, cy), R * 0.55, R * 0.25)

                # "And... it's gone." — large, dead center
                f_gone = QFont("Helvetica", max(14, int(R * 0.14)))
                f_gone.setBold(True)
                p.setFont(f_gone)

                gone_rect = QRectF(cx - R * 0.85, cy - R * 0.12, R * 1.70, R * 0.24)

                # Shadow
                shd = R * 0.008
                p.setPen(QColor(0, 0, 0, int(200 * a)))
                p.drawText(gone_rect.adjusted(shd, shd, shd, shd),
                           Qt.AlignCenter, "And... it's gone.")

                # Main text
                gone_a = int((210 + 45 * blink) * a)
                p.setPen(QColor(245, 70, 50, gone_a))
                p.drawText(gone_rect, Qt.AlignCenter, "And... it's gone.")

                # Subtitle below
                f_sub = QFont("Helvetica", max(5, int(R * 0.038)))
                p.setFont(f_sub)
                p.setPen(QColor(180, 120, 110, int(140 * a * blink)))
                secs_since = post_crash_secs
                p.drawText(
                    QRectF(cx - R * 0.70, cy + R * 0.14, R * 1.40, R * 0.10),
                    Qt.AlignCenter,
                    f"All positions liquidated  \u00b7  {secs_since}s ago"
                )

        finally:
            p.restore()

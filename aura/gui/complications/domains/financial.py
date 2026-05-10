"""
gui.complications.domains.financial -- Daily Brief overlay (solar system).

The most-important topic from the active briefing sits at the center as
the sun. The next 4-5 most-important topics orbit it as planets. A
"focus" cycles around the system: at any moment exactly one body (sun
OR a planet) is highlighted, Aura narrates the relevant sentence from
the briefing, then the focus moves to the next body. Other bodies stay
visible but dim so the user can see what's coming up.

Class name + `name = "Financial"` are kept for registry compatibility;
only the `label` is renamed to "Daily Brief".
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from PyQt5.QtGui import QPainter

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QFontMetricsF, QLinearGradient, QPainterPath,
    QPen, QRadialGradient,
)

from gui.complications.domains.base_domain import BaseDomainComplication
from gui.renderer import clamp


# ─────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────

_BRIEFINGS_DIR = Path(__file__).resolve().parents[4] / "data" / "briefings"

_STOPWORDS = frozenset({
    "the","a","an","and","or","but","is","are","was","were","be","been",
    "being","have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","must","of","in","on","at","to",
    "for","with","from","by","as","into","about","after","before","up",
    "down","out","off","over","under","again","further","then","once",
    "here","there","when","where","why","how","all","any","both","each",
    "few","more","most","other","some","such","no","nor","not","only",
    "own","same","so","than","too","very","i","me","my","myself","we",
    "our","ours","ourselves","you","your","yours","he","him","his","she",
    "her","it","its","they","them","their","theirs","this","that","these",
    "those","am","im","ive","youre","thats","whats","lets","dont","didnt",
    "wont","wouldnt","if","yes","no","ok","okay","well","right","good",
    "morning","today","just","also","still","really","going","got","get",
    "you","us","let","now","one","two","lot","way","back","around",
    "through","things","thing","lots","much","make","made","see","seen",
    "saying","say","said","look","looks","looking","seems","seem","seemed",
    "while","across","front",
})

_WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z\-']{2,}\b")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_recent_briefings(limit: int = 10) -> List[dict]:
    if not _BRIEFINGS_DIR.is_dir():
        return []
    files = sorted(
        _BRIEFINGS_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]
    return [b for b in (_read_json(p) for p in files) if b]


def _top_topics(text: str, top_n: int = 6) -> List[str]:
    counts: dict[str, int] = {}
    for w in _WORD_RE.findall(text or ""):
        wl = w.lower()
        if wl in _STOPWORDS or len(wl) < 4:
            continue
        counts[wl] = counts.get(wl, 0) + 1
    pairs = sorted(counts.items(), key=lambda kv: -kv[1])
    return [w for w, _ in pairs[:top_n]]


def _sentence_for_topic(topic: str, text: str) -> str:
    if not text:
        return ""
    needle = topic.lower()
    for s in _SENT_SPLIT.split(text):
        if needle in s.lower():
            cleaned = s.strip().strip("\"'`*_").strip()
            if len(cleaned) > 220:
                cleaned = cleaned[:217].rstrip() + "…"
            return cleaned
    return ""


# ─────────────────────────────────────────────────────────────────────────
# HUD palette
# ─────────────────────────────────────────────────────────────────────────

_HUD_BG_DEEP   = QColor(4, 10, 22)
_HUD_BG_MID    = QColor(8, 18, 38)
_HUD_GRID      = QColor(60, 200, 230)
_HUD_PRIMARY   = QColor(120, 230, 255)        # cyan — for active body
_HUD_SUN       = QColor(255, 195, 90)          # warm gold — for the sun
_HUD_DIM       = QColor(110, 145, 180)         # for inactive planets
_HUD_TEXT      = QColor(220, 240, 255)
_HUD_TEXT_DIM  = QColor(140, 175, 210)


# ─────────────────────────────────────────────────────────────────────────
# Backdrop helpers
# ─────────────────────────────────────────────────────────────────────────

def _draw_starfield(p, cx, cy, R, t, alpha):
    """A handful of tiny static stars, very translucent — they're set
    dressing, never competing with the text in front of them."""
    n = 22
    for i in range(n):
        sa = (i * 2.39996323) % (2.0 * math.pi)
        sr = math.sqrt((i + 0.5) / n) * R * 0.95
        sx = cx + sr * math.cos(sa)
        sy = cy + sr * math.sin(sa)
        pulse = max(0.0, 0.4 + 0.6 * math.sin(t * (0.6 + (i % 7) * 0.15) + i * 1.3))
        a = int(alpha * pulse ** 1.6)
        if a < 4:
            continue
        sz = max(0.4, R * (0.0008 + (i % 5) * 0.00025))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 252, 235, a))
        p.drawEllipse(QPointF(sx, sy), sz, sz)


def _draw_scan_line(p, cx, cy, R, t, trans):
    period = 9.0
    phase = (t % period) / period
    y = (cy - R) + phase * (2 * R)
    h_band = R * 0.13
    grad = QLinearGradient(0, y - h_band, 0, y + h_band)
    grad.setColorAt(0.0, QColor(0, 0, 0, 0))
    grad.setColorAt(0.5, QColor(_HUD_PRIMARY.red(), _HUD_PRIMARY.green(),
                                  _HUD_PRIMARY.blue(), int(35 * trans)))
    grad.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(grad))
    p.drawRect(QRectF(cx - R, y - h_band, 2 * R, 2 * h_band))


def _draw_arc_text(p, cx, cy, text, radius, center_deg, font, color,
                   spacing_deg=6.5, flip=False):
    chars = list(text)
    if flip:
        chars = chars[::-1]
    n = len(chars)
    if n == 0:
        return
    span = (n - 1) * spacing_deg
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


def _wrap_text(text: str, font, max_w: float, max_lines: int) -> List[str]:
    fm = QFontMetricsF(font)
    words = (text or "").split()
    lines: List[str] = []
    current = ""
    for w in words:
        trial = (current + " " + w).strip()
        if fm.horizontalAdvance(trial) <= max_w:
            current = trial
        else:
            if current:
                lines.append(current)
            current = w
            if len(lines) >= max_lines:
                break
    if current and len(lines) < max_lines:
        lines.append(current)
    if lines and len(lines) == max_lines:
        last = lines[-1]
        if fm.horizontalAdvance(last) > max_w:
            while last and fm.horizontalAdvance(last + "…") > max_w:
                last = last[:-1]
            lines[-1] = last.rstrip() + "…"
    return lines


# ─────────────────────────────────────────────────────────────────────────
# Complication
# ─────────────────────────────────────────────────────────────────────────

# Total brief = four minutes. Per-body window = 240s / number of bodies.
# Hard cut-off so the tour never loops on the user.
_TOTAL_BRIEF_S   = 240.0
_BODY_FADE_S     = 0.8


class FinancialComplication(BaseDomainComplication):
    name = "Financial"
    label = "Daily Brief"
    category = "Topics"

    def __init__(self, bus):
        super().__init__(bus)
        self._t0: float = 0.0
        # Cache
        self._cache_ts: float = 0.0
        self._date_label: str = ""
        self._user_label: str = ""
        # bodies: list of dicts. Index 0 is the sun (primary topic),
        # 1..N are planets. Each dict:
        #   { "word": str, "sentence": str (one-line fallback),
        #     "orbit_r": float (0 for sun, fraction-of-R for planets),
        #     "orbit_phase": float (radians, fixed at extract time),
        #     "size_mult": float (relative size) }
        self._bodies: List[dict] = []
        # Per-body LLM-generated long-form analyses (≈ 80-120 words
        # each). Filled asynchronously by _generate_analyses(); index
        # mirrors self._bodies. None means "not ready yet — fall back
        # to the one-line sentence in body['sentence']".
        self._analyses: List[Optional[str]] = []
        # Threads
        self._narration_thread: Optional[threading.Thread] = None
        self._analyses_thread: Optional[threading.Thread] = None
        self._narration_stop = threading.Event()

    # ── Lifecycle ─────────────────────────────────────────────────────

    def play_audio(self):
        print("[daily_brief] play_audio() called")
        self._narration_stop.clear()
        self._refresh()
        print(f"[daily_brief] {len(self._bodies)} bodies queued for narration")
        if not self._bodies:
            print("[daily_brief] no bodies — aborting narration")
            return
        self._t0 = 0.0
        # Reset analyses so we re-generate against the current briefing.
        self._analyses = [None] * len(self._bodies)

        # Async LLM analysis generation in the background — narration
        # uses each one as it becomes ready, otherwise falls back to
        # the one-line sentence so the user never hits dead air.
        if not (self._analyses_thread and self._analyses_thread.is_alive()):
            self._analyses_thread = threading.Thread(
                target=self._generate_analyses, daemon=True,
                name="dailybrief-analyse",
            )
            self._analyses_thread.start()

        if self._narration_thread and self._narration_thread.is_alive():
            print("[daily_brief] narration thread already alive — not re-spawning")
            return
        self._narration_thread = threading.Thread(
            target=self._narrate, daemon=True, name="dailybrief-narrate",
        )
        self._narration_thread.start()
        print("[daily_brief] narration thread started")

    def stop_audio(self):
        self._narration_stop.set()
        try:
            from voice.speaker import speaker
            speaker.interrupt()
        except Exception:
            pass

    def draw_content(self, p, inner, t, accent):
        pass

    # ── Narration thread ──────────────────────────────────────────────

    def _narrate(self):
        # The speaker singleton is constructed in aura.py and is NOT
        # exposed at module scope, so importing it directly fails. The
        # speaker IS subscribed to the "llm.sentence" bus event though,
        # so we route through that — same path the LLM-driven voice
        # replies use.
        from core.bus import bus
        print("[daily_brief] beginning narration via bus.emit('llm.sentence')")

        n = max(1, len(self._bodies))
        body_dur = _TOTAL_BRIEF_S / n
        for i, body in enumerate(self._bodies):
            if self._narration_stop.is_set():
                return
            word = body["word"].title()

            # Wait briefly (up to ~6 s) for the LLM analysis to be ready.
            # If still not ready, fall back to the one-line sentence so
            # the user never hits silence.
            wait_deadline = time.time() + 6.0
            while (i < len(self._analyses) and self._analyses[i] is None
                   and time.time() < wait_deadline):
                if self._narration_stop.is_set():
                    return
                time.sleep(0.25)
            text = (self._analyses[i] if i < len(self._analyses)
                    else None) or body["sentence"] or f"No detail on {word}."

            if i == 0:
                lead = f"The headline today is {word}."
            else:
                lead = f"Next, {word}."
            spoken = f"{lead} {text}"
            kind = "LLM" if (i < len(self._analyses) and self._analyses[i]) else "fallback"
            print(f"[daily_brief] narrating ({i+1}/{n}, {body_dur:.1f}s, {kind}): "
                  f"{spoken[:90]}")
            try:
                bus.emit("llm.sentence", text=spoken, style="warm")
            except Exception as e:
                print(f"[daily_brief] bus emit failed for {word}: {e}")

            # Hold this topic's window so on-screen highlight stays in
            # sync with the spoken sentence.
            elapsed = 0.0
            while elapsed < body_dur:
                if self._narration_stop.is_set():
                    return
                time.sleep(0.2)
                elapsed += 0.2
        print("[daily_brief] narration complete (4-minute brief done)")

    def _generate_analyses(self) -> None:
        """Asynchronously fill self._analyses with longer LLM-generated
        per-topic write-ups. Each runs serially through llm_engine
        (which holds gpu_lock internally), populating the list as each
        finishes so _narrate can pick them up as soon as they're ready.
        """
        try:
            from voice.llm_engine import llm_engine
        except Exception as e:
            print(f"[daily_brief] llm_engine import failed: {e}")
            return
        if not getattr(llm_engine, "loaded", False):
            print("[daily_brief] llm_engine not loaded — skipping analyses")
            return

        from core.state import state
        primary = state.pending_briefing
        insight = (primary or {}).get("insight") or ""

        sys_prompt = (
            "You are Aura giving a deep, thorough analysis as part of a "
            "spoken daily brief. For the requested topic, generate a "
            "conversational paragraph between 80 and 120 words that goes "
            "well beyond the surface — what does this mean, why it "
            "matters right now, what to watch for next, and (where "
            "relevant) what action makes sense. Use the briefing context "
            "for grounding. Output ONLY the paragraph: no headings, no "
            "bullet points, no preamble, no quotes."
        )

        for i, body in enumerate(self._bodies):
            if self._narration_stop.is_set():
                return
            topic = body["word"].title()
            user_prompt = (
                f"Briefing context:\n{insight[:1500]}\n\n"
                f"One-line lead-in already known:\n{body['sentence']}\n\n"
                f"Now expand into a deep analysis of: {topic}"
            )
            try:
                text = llm_engine.chat_direct(
                    sys_prompt, user_prompt,
                    max_tokens=240, temperature=0.7,
                )
                if text:
                    text = text.strip().strip('"').strip("'")
                    if text:
                        self._analyses[i] = text
                        print(f"[daily_brief] analysis ready for {topic} "
                              f"({len(text)} chars)")
            except Exception as e:
                print(f"[daily_brief] analysis gen failed for {topic}: {e}")

    # ── Cache refresh ─────────────────────────────────────────────────

    def _refresh(self) -> None:
        from core.state import state

        recent = _load_recent_briefings(limit=10)
        primary = state.pending_briefing or (recent[0] if recent else None)

        if primary:
            ts = primary.get("generated_at")
            if ts:
                self._date_label = time.strftime("%b %d  %H:%M",
                                                 time.localtime(ts)).upper()
            else:
                self._date_label = (primary.get("date") or "").upper()
            self._user_label = (primary.get("user_name") or "").upper()
            insight = primary.get("insight") or ""
        else:
            self._date_label = time.strftime("%b %d  %H:%M").upper()
            self._user_label = ""
            insight = ""

        all_text = " ".join(
            [insight] + [r.get("insight") or "" for r in recent]
        )
        topics = _top_topics(all_text, top_n=6)

        # Build bodies — sun first, then planets.
        bodies: List[dict] = []
        used_sentences: set[str] = set()
        for i, topic in enumerate(topics):
            sentence = _sentence_for_topic(topic, insight)
            if not sentence:
                for r in recent:
                    s = _sentence_for_topic(topic, r.get("insight") or "")
                    if s:
                        sentence = s
                        break
            if not sentence or sentence in used_sentences:
                continue
            used_sentences.add(sentence)
            if i == 0:
                bodies.append({
                    "word": topic.upper(),
                    "sentence": sentence,
                    "orbit_r": 0.0,        # sun is at center
                    "orbit_phase": 0.0,
                    "size_mult": 1.0,
                })
            else:
                # Planets at varied orbital radii so they don't all sit on
                # the same circle. Phase is evenly spaced — they'll all
                # rotate together at a common angular speed in the draw.
                planet_count_so_far = len(bodies) - 1   # exclude sun
                bodies.append({
                    "word": topic.upper(),
                    "sentence": sentence,
                    "orbit_r": [0.30, 0.45, 0.55, 0.65, 0.72][planet_count_so_far % 5],
                    "orbit_phase": planet_count_so_far * (2 * math.pi / 5),
                    "size_mult": [0.85, 0.78, 0.72, 0.68, 0.62][planet_count_so_far % 5],
                })

        if not bodies:
            bodies = [{
                "word": "STAND BY",
                "sentence": "No briefing is queued yet — Aura will fill this out at the next rumination cycle.",
                "orbit_r": 0.0, "orbit_phase": 0.0, "size_mult": 1.0,
            }]
        self._bodies = bodies
        self._cache_ts = time.time()

    # ── Layout helpers ────────────────────────────────────────────────

    def _body_position(self, i: int, R: float, t: float) -> Tuple[float, float]:
        """Where is body i right now? Sun is locked at origin; planets
        rotate slowly so the system feels alive without the eye having
        to chase anything."""
        body = self._bodies[i]
        if body["orbit_r"] == 0.0:
            return 0.0, 0.0
        ang = body["orbit_phase"] + t * 0.06   # ~one revolution per ~100 s
        rr = R * body["orbit_r"]
        return rr * math.cos(ang), rr * math.sin(ang)

    # ── Main draw ─────────────────────────────────────────────────────

    def draw_overlay(self, p, cx, cy, mind, t, trans):
        a = clamp(float(trans), 0.0, 1.0)
        if a < 0.002:
            return
        if self._t0 == 0.0:
            self._t0 = t
        elapsed = max(0.0, t - self._t0)

        if not self._bodies or (time.time() - self._cache_ts) > 30.0:
            self._refresh()

        n = max(1, len(self._bodies))
        body_dur = _TOTAL_BRIEF_S / n
        # Hard cut-off at 2 minutes — stick on the last body afterward
        # so the screen still has something readable, but everything
        # stops moving and the narration thread has already exited.
        brief_complete = elapsed >= _TOTAL_BRIEF_S
        if brief_complete:
            cur_idx = n - 1
            cur_phase = 1.0
            seg_alpha = 1.0
        else:
            cur_idx = int(elapsed // body_dur) % n
            cur_phase = (elapsed % body_dur) / body_dur
            fade_frac = _BODY_FADE_S / body_dur
            if cur_phase < fade_frac:
                seg_alpha = cur_phase / fade_frac
            elif cur_phase > 1.0 - fade_frac:
                seg_alpha = (1.0 - cur_phase) / fade_frac
            else:
                seg_alpha = 1.0
            seg_alpha = clamp(seg_alpha, 0.0, 1.0)

        R = mind * 0.36
        # The system lives in a small zone near the top of the dial,
        # so the BIG TEXT can own the visual center. The planets are
        # context, not the main event.
        sys_cx = cx
        sys_cy = cy - R * 0.50
        sys_R  = R * 0.36                  # tighter orbits, smaller system

        p.save()
        try:
            p.setRenderHint(p.Antialiasing, True)

            # Clip
            clip = QPainterPath()
            clip.addEllipse(QPointF(cx, cy), R, R)
            p.setClipPath(clip)

            # Backdrop
            bg = QRadialGradient(QPointF(cx, cy), R)
            bg.setColorAt(0.0, _HUD_BG_MID)
            bg.setColorAt(0.7, _HUD_BG_DEEP)
            bg.setColorAt(1.0, QColor(1, 3, 8))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(bg))
            p.drawEllipse(QPointF(cx, cy), R, R)

            # Stars are background dressing only — knocked way down so
            # they never compete with the text in front.
            _draw_starfield(p, cx, cy, R, t, int(70 * a))
            _draw_scan_line(p, cx, cy, R, t, a * 0.6)

            # Bezel
            bz = QPen(QColor(_HUD_PRIMARY.red(), _HUD_PRIMARY.green(),
                              _HUD_PRIMARY.blue(), int(160 * a)))
            bz.setWidthF(max(1.6, mind * 0.0028))
            p.setPen(bz)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx, cy), R * 0.985, R * 0.985)
            inner_pen = QPen(QColor(_HUD_PRIMARY.red(), _HUD_PRIMARY.green(),
                                     _HUD_PRIMARY.blue(), int(70 * a)))
            inner_pen.setWidthF(max(1.0, mind * 0.0014))
            p.setPen(inner_pen)
            p.drawEllipse(QPointF(cx, cy), R * 0.94, R * 0.94)

            # Top arc
            hdr_font = QFont("Helvetica Neue", max(11, int(mind * 0.020)))
            hdr_font.setWeight(QFont.DemiBold)
            hdr_font.setLetterSpacing(QFont.PercentageSpacing, 175)
            _draw_arc_text(p, cx, cy, "DAILY  BRIEF", R * 0.86, -90.0,
                           hdr_font,
                           QColor(_HUD_PRIMARY.red(), _HUD_PRIMARY.green(),
                                   _HUD_PRIMARY.blue(), int(245 * a)),
                           spacing_deg=6.5)
            sub_font = QFont("Helvetica Neue", max(8, int(mind * 0.012)))
            sub_font.setWeight(QFont.Medium)
            sub_font.setLetterSpacing(QFont.PercentageSpacing, 165)
            sub_text = self._date_label
            if self._user_label:
                sub_text = f"{self._date_label}  ·  {self._user_label}"
            _draw_arc_text(p, cx, cy, sub_text, R * 0.78, -90.0,
                           sub_font,
                           QColor(_HUD_TEXT_DIM.red(), _HUD_TEXT_DIM.green(),
                                   _HUD_TEXT_DIM.blue(), int(220 * a)),
                           spacing_deg=4.5)

            # ── Faint orbit guides (one per planet, centered on sun) ─
            for i, body in enumerate(self._bodies):
                if body["orbit_r"] == 0.0:
                    continue
                rr = sys_R * body["orbit_r"]
                guide_pen = QPen(QColor(_HUD_DIM.red(), _HUD_DIM.green(),
                                          _HUD_DIM.blue(), int(40 * a)))
                guide_pen.setWidthF(max(0.4, mind * 0.0008))
                guide_pen.setStyle(Qt.DotLine)
                p.setPen(guide_pen)
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(sys_cx, sys_cy), rr, rr)

            # ── Spotlight beam from sun to active planet ─────────
            if 0 < cur_idx < n:
                px, py = self._body_position(cur_idx, sys_R, t)
                bx = sys_cx + px
                by = sys_cy + py
                beam_pen = QPen(QColor(_HUD_PRIMARY.red(), _HUD_PRIMARY.green(),
                                        _HUD_PRIMARY.blue(),
                                        int(110 * seg_alpha * a)))
                beam_pen.setWidthF(max(1.2, mind * 0.0024))
                beam_pen.setCapStyle(Qt.RoundCap)
                p.setPen(beam_pen)
                p.drawLine(QPointF(sys_cx, sys_cy), QPointF(bx, by))

            # ── Draw planets first, sun on top ───────────────────
            # (so the sun's glow always reads, even if a planet
            #  overlaps it during transitions)
            for i, body in enumerate(self._bodies):
                if body["orbit_r"] == 0.0:
                    continue
                px, py = self._body_position(i, sys_R, t)
                bx = sys_cx + px
                by = sys_cy + py
                is_active = (i == cur_idx)
                base_size = sys_R * 0.06 * body["size_mult"]
                if is_active:
                    size = base_size * 1.6
                    halo_col = QColor(_HUD_PRIMARY.red(), _HUD_PRIMARY.green(),
                                        _HUD_PRIMARY.blue(),
                                        int(180 * seg_alpha * a))
                    body_inner = QColor(255, 252, 240, int(245 * seg_alpha * a))
                    body_mid = QColor(_HUD_PRIMARY.red(), _HUD_PRIMARY.green(),
                                        _HUD_PRIMARY.blue(),
                                        int(230 * seg_alpha * a))
                    body_outer = QColor(40, 90, 130, int(190 * seg_alpha * a))
                else:
                    size = base_size
                    halo_col = QColor(_HUD_DIM.red(), _HUD_DIM.green(),
                                        _HUD_DIM.blue(), int(40 * a))
                    body_inner = QColor(190, 210, 230, int(200 * a))
                    body_mid = QColor(_HUD_DIM.red(), _HUD_DIM.green(),
                                        _HUD_DIM.blue(), int(180 * a))
                    body_outer = QColor(35, 60, 90, int(150 * a))

                # Halo
                halo = QRadialGradient(QPointF(bx, by), size * 4)
                halo.setColorAt(0.0, halo_col)
                halo.setColorAt(1.0, QColor(0, 0, 0, 0))
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(halo))
                p.drawEllipse(QPointF(bx, by), size * 4, size * 4)
                # Body
                bg2 = QRadialGradient(QPointF(bx - size * 0.3, by - size * 0.3),
                                       size * 1.4)
                bg2.setColorAt(0.0, body_inner)
                bg2.setColorAt(0.7, body_mid)
                bg2.setColorAt(1.0, body_outer)
                p.setBrush(QBrush(bg2))
                p.drawEllipse(QPointF(bx, by), size, size)

            # Sun at center of the system
            sun_active = (cur_idx == 0)
            sun_size = sys_R * 0.13
            # Wide corona — always visible since sun is the anchor
            corona = QRadialGradient(QPointF(sys_cx, sys_cy), sun_size * 5)
            corona_alpha = int((180 if sun_active else 100) * a *
                                (seg_alpha if sun_active else 1.0))
            corona.setColorAt(0.0, QColor(_HUD_SUN.red(), _HUD_SUN.green(),
                                            _HUD_SUN.blue(), corona_alpha))
            corona.setColorAt(0.45, QColor(_HUD_SUN.red(), _HUD_SUN.green(),
                                             _HUD_SUN.blue(),
                                             int(corona_alpha * 0.4)))
            corona.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(corona))
            p.drawEllipse(QPointF(sys_cx, sys_cy), sun_size * 5, sun_size * 5)
            # Sun body
            sun_grad = QRadialGradient(
                QPointF(sys_cx - sun_size * 0.25, sys_cy - sun_size * 0.30),
                sun_size * 1.6)
            sun_grad.setColorAt(0.0, QColor(255, 252, 230, int(255 * a)))
            sun_grad.setColorAt(0.5, QColor(255, 215, 130, int(245 * a)))
            sun_grad.setColorAt(1.0, QColor(195, 130, 60, int(220 * a)))
            p.setBrush(QBrush(sun_grad))
            p.drawEllipse(QPointF(sys_cx, sys_cy), sun_size, sun_size)
            # Catchlight
            p.setBrush(QColor(255, 255, 255, int(170 * a)))
            p.drawEllipse(QPointF(sys_cx - sun_size * 0.3, sys_cy - sun_size * 0.32),
                          sun_size * 0.32, sun_size * 0.22)

            # ── Active body label + sentence (description card) ──
            cur = self._bodies[cur_idx]
            word = cur["word"]
            sentence = cur["sentence"]

            # Description card — must stay strictly inside the circle.
            # At y = cy + R*0.30, max chord half-width = R*sqrt(1-0.09) ≈ R*0.95
            # At y = cy + R*0.55, max chord half-width = R*sqrt(1-0.30) ≈ R*0.84
            # Pick conservative widths well inside those bounds.
            word_max_w = R * 1.55       # 0.775*R either side, well inside chord
            body_max_w = R * 1.30       # 0.65*R either side

            # Auto-fit the big word: shrink font until it fits.
            word_size = max(16, int(mind * 0.050))
            word_font = QFont("Helvetica Neue", word_size, QFont.DemiBold)
            word_font.setLetterSpacing(QFont.PercentageSpacing, 130)
            fm = QFontMetricsF(word_font)
            while fm.horizontalAdvance(word) > word_max_w and word_size > 10:
                word_size -= 1
                word_font = QFont("Helvetica Neue", word_size, QFont.DemiBold)
                word_font.setLetterSpacing(QFont.PercentageSpacing, 130)
                fm = QFontMetricsF(word_font)
            tw = fm.horizontalAdvance(word)
            th = fm.height()
            wcx = cx
            wcy = cy + R * 0.05         # near the visual center
            # Glow behind the word (also clamped to in-bounds size)
            glow_r = min(max(tw, th) * 0.85, R * 0.55)
            glow = QRadialGradient(QPointF(wcx, wcy), glow_r)
            glow.setColorAt(0.0, QColor(_HUD_PRIMARY.red(), _HUD_PRIMARY.green(),
                                          _HUD_PRIMARY.blue(),
                                          int(70 * seg_alpha * a)))
            glow.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(glow))
            p.drawEllipse(QPointF(wcx, wcy), glow_r, glow_r * 0.7)
            p.setFont(word_font)
            p.setPen(QColor(_HUD_TEXT.red(), _HUD_TEXT.green(),
                             _HUD_TEXT.blue(), int(255 * seg_alpha * a)))
            word_rect = QRectF(cx - word_max_w / 2, wcy - th * 0.6,
                                word_max_w, th * 1.2)
            p.drawText(word_rect, Qt.AlignCenter, word)

            # Description sentence (wrapped, 2 lines max)
            body_size = max(10, int(mind * 0.017))
            body_font = QFont("Helvetica Neue", body_size)
            body_font.setWeight(QFont.Light)
            body_font.setLetterSpacing(QFont.PercentageSpacing, 110)
            p.setFont(body_font)
            body_lines = _wrap_text(sentence, body_font, body_max_w, max_lines=2)
            body_top = cy + R * 0.27
            line_h = body_size * 1.32
            p.setPen(QColor(_HUD_TEXT.red(), _HUD_TEXT.green(),
                             _HUD_TEXT.blue(), int(230 * seg_alpha * a)))
            for li, line in enumerate(body_lines):
                rect = QRectF(cx - body_max_w / 2, body_top + li * line_h,
                               body_max_w, line_h)
                p.drawText(rect, Qt.AlignHCenter | Qt.AlignVCenter, line)

            # If the brief is over, replace the dot row with a clear
            # "BRIEF COMPLETE" indicator so the user knows it's safe to
            # dismiss.
            if brief_complete:
                done_font = QFont("Helvetica Neue", max(8, int(mind * 0.013)),
                                   QFont.Medium)
                done_font.setLetterSpacing(QFont.PercentageSpacing, 175)
                p.setFont(done_font)
                p.setPen(QColor(_HUD_PRIMARY.red(), _HUD_PRIMARY.green(),
                                 _HUD_PRIMARY.blue(), int(220 * a)))
                p.drawText(QRectF(cx - R * 0.6, cy + R * 0.72,
                                   R * 1.2, R * 0.10),
                           Qt.AlignCenter, "BRIEF  COMPLETE")

            # Bottom arc — call to action
            cta_font = QFont("Helvetica Neue", max(8, int(mind * 0.012)))
            cta_font.setWeight(QFont.Medium)
            cta_font.setLetterSpacing(QFont.PercentageSpacing, 175)
            _draw_arc_text(p, cx, cy, "TAP  TO  DISMISS",
                           R * 0.92, 90.0, cta_font,
                           QColor(_HUD_PRIMARY.red(), _HUD_PRIMARY.green(),
                                   _HUD_PRIMARY.blue(), int(180 * a)),
                           spacing_deg=4.8, flip=True)
        finally:
            p.restore()

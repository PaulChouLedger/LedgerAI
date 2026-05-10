"""
boot.conversation_engine -- Varied, natural enrollment dialogue.

Replaces the old hardcoded enrollment script (the "same bullshit lines over
and over" problem) with a layered system:

1. Live LLM generation (when llm_engine is loaded) — uses the user's prior
   answers, current slot, and persona-tone hints to produce contextually
   relevant follow-ups. No two enrollments come out identical.
2. Curated phrase banks (~200+ lines across openers, name asks, name
   confirms, voice-deepening questions, acknowledgments, closings) —
   used as fallback while the LLM is still warming, and as a fast path
   for first-utterance moments where context is empty.
3. Theme + style rotation — never re-asks the same theme twice within a
   single enrollment, and varies the emotional register utterance to
   utterance (warm / playful / curious / direct / depth / casual).

Utterances are returned as strings. The caller is responsible for
synthesising them with Piper (already warmed by the boot orchestrator)
and capturing the response. Short and stateless — drop one
EnrollmentConversation instance per enrollment.
"""

from __future__ import annotations

import random
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────
# Curated phrase banks
# ─────────────────────────────────────────────────────────────────────────

OPENERS: list[str] = [
    # Warm / welcoming
    "I don't think we've met yet — what's your name?",
    "Hi there. I'm Aura. What should I call you?",
    "Hey, I haven't heard your voice before. What's your name?",
    "Lovely to meet you. I'm Aura — and you are?",
    "Welcome. I'm Aura. What's your name?",
    # Direct / brisk
    "New voice. I'm Aura. Who are you?",
    "I don't know you yet. What's your name?",
    "Quick — give me a name to put with that voice.",
    "First time meeting. I'm Aura. You?",
    # Playful
    "Oh hi, a stranger. I'm Aura. Got a name?",
    "Stranger danger — wait, no, hi. I'm Aura. What's your name?",
    "Well hello, mystery voice. What do I call you?",
    "Brand-new voice in the room. What's your name?",
    "If we've met, sorry for forgetting. If not — I'm Aura. You?",
    # Curious
    "Hmm, a voice I don't recognize. What's your name?",
    "Interesting voice. Tell me your name?",
    "I haven't catalogued you yet. What's your name?",
    "Who do I have the pleasure of talking to?",
    # Casual
    "Hey. I'm Aura. What's your name?",
    "Hi! What do I call you?",
    "Yo. New voice. Name?",
    "Hi — quick intro. What's your name?",
    # Slightly poetic / smile-inducing
    "There's a voice I don't know yet. Help me out — what's your name?",
    "I'd love to know who I'm talking to. Your name?",
    "Hi, hi. I'm Aura. And you would be?",
    "Welcome to the room. I'm Aura — what should I call you?",
    "A new voice walks in. I'm Aura. You?",
    # Self-aware
    "I'm Aura. I live here. And you are?",
    "Pleased to meet you, whoever you are. What's your name?",
    "I'm Aura. I don't think we've been introduced — what's your name?",
]

NAME_CONFIRMS: list[str] = [
    # Direct check
    "Did I hear that right — {name}?",
    "Just to be sure, that's {name}?",
    "{name}? Let me know if I butchered it.",
    "Quick check — {name}, yeah?",
    "{name} — am I saying that right?",
    "So that's {name}? Correct me if I'm off.",
    # Warm
    "Lovely to meet you, {name}. Did I get that right?",
    "{name} — beautiful. Tell me if I missed it.",
    "Got it — {name}. Spell it out if I'm wrong.",
    # Playful
    "{name}, huh? Don't let me get away with mispronouncing it.",
    "{name} — landing that one okay?",
    # Casual
    "{name} — yeah?",
    "Cool, {name}. Spell it for me if I'm off.",
    "Alright — {name}. Right?",
    "{name}? Tell me if I should re-listen.",
]

# Themes with multiple phrasings + style tags. Rotate themes across an
# enrollment so the user isn't asked four work questions in a row.
QUESTION_BANK: dict[str, list[str]] = {
    "work": [
        "What do you do?",
        "Tell me about your work — what do you actually spend your days on?",
        "What's your line of work? Get specific.",
        "Are you working on anything right now that you're actually proud of?",
        "If you could quit tomorrow and do anything, what would it be?",
        "What do you do for a living, and do you actually like it?",
        "What's the last project you finished that felt good?",
        "Walk me through a regular day at work for you.",
    ],
    "hobby": [
        "What do you do for fun?",
        "Got any weird hobbies? The weirder the better.",
        "Free Saturday, no obligations — what do you do?",
        "What's something you geek out about that nobody else cares about?",
        "What do you do when you want to actually relax?",
        "What's something you've been into for a long time?",
        "Pick one: a thing you do that lights you up.",
        "What's the last hobby you tried and bailed on?",
    ],
    "place": [
        "Where do you call home these days?",
        "Been anywhere good lately?",
        "If you had to live somewhere else for a year, where?",
        "Where's somewhere you keep meaning to go and haven't?",
        "What's a place that feels like home to you?",
        "What's the best trip you've taken in the last few years?",
    ],
    "food": [
        "What's the last great meal you had?",
        "Coffee, tea, or what's your morning?",
        "If you could only eat one cuisine forever, what would it be?",
        "Best thing in your fridge right now.",
        "What's a meal that means something to you?",
        "What's the most overrated food, in your opinion?",
    ],
    "memory": [
        "Tell me about a moment that stuck with you.",
        "What's a weirdly specific thing you remember from being a kid?",
        "Who's someone you wish you saw more of?",
        "What's a story your family tells about you?",
        "What's the last thing that made you laugh out loud?",
        "What was your favorite year so far, and why?",
    ],
    "opinion": [
        "What's a hot take you have that nobody agrees with?",
        "Pet peeve — what really gets under your skin?",
        "What's something everybody loves that you just don't get?",
        "What's a thing more people should be talking about?",
        "What's an unpopular opinion you'd defend?",
    ],
    "dream": [
        "If money weren't a thing, what would you actually do all day?",
        "What's something you've been meaning to start but haven't?",
        "What would future-you, ten years out, want you to be doing right now?",
        "What's a dream you've quietly carried around for a while?",
        "If you could learn one new skill overnight, what?",
    ],
    "now": [
        "What's been on your mind today?",
        "Good day, bad day, or weird day?",
        "Anything weighing on you, or is today actually fine?",
        "What's the last thing that genuinely surprised you?",
        "What's something small that made today better?",
        "If today had a soundtrack, what would be on it?",
    ],
    "playful": [
        "Pineapple on pizza — yes or no, and defend yourself.",
        "What's a song you can't stop singing lately?",
        "If you had a theme song every time you walked in a room, what?",
        "Cats or dogs, and explain.",
        "If you could have dinner with anyone alive, who?",
        "What's the most you've ever spent on something stupid?",
        "What animal would you fight, one-on-one, no weapons?",
    ],
    "self": [
        "What's something you're better at than people realize?",
        "What's a small thing you do really well?",
        "How would your closest friend describe you in one sentence?",
        "What's something you've gotten better at this year?",
        "What's a side of you most people don't see?",
    ],
}

# Short reactive acknowledgments — what Aura says BEFORE the next question.
ACKS: list[str] = [
    # Short / neutral
    "Got it.", "Mm.", "Mm-hm.", "Yeah.", "Right.", "Okay.", "Sure.",
    "Cool.", "Nice.", "Nice one.", "Solid.", "Beautiful.",
    # Warm
    "I love that.", "That's lovely.", "Aw, nice.", "That hits.",
    # Curious
    "Huh.", "Interesting.", "Oh — interesting.", "Wait, really?",
    "Tell me more sometime.", "I want to hear more about that.",
    # Playful
    "No way.", "Stop it.", "You're kidding.", "Wild.", "Iconic.",
    "Respect.", "Hot take, I love it.", "Bold answer.",
    # Casual
    "For sure.", "Totally.", "I hear you.", "Yeah, makes sense.",
    "Word.", "Same.", "Honestly, same.",
]

CLOSINGS: list[str] = [
    "Got it, {name}. I'll know your voice next time.",
    "Perfect. {name}, you're saved. I won't forget.",
    "{name}, you're locked in. Thanks for the chat.",
    "Done. Your voice is in my memory now, {name}.",
    "Alright, {name} — voice saved. Talk soon.",
    "Got you, {name}. Next time you talk, I'll know it's you.",
    "{name}, you're official. I've got your voice.",
    "Saved. Welcome to the household, {name}.",
    "Perfect, {name}. I've stored your voice — you're set.",
    "Logged. You're {name}, and I'll remember that.",
    "Nice to meet you, {name}. I won't forget your voice.",
    "{name}, you're on the books. Thanks for letting me listen.",
    "Got it. Welcome aboard, {name}.",
    "All set, {name} — your voice is mine now. In a friendly way.",
]


# ─────────────────────────────────────────────────────────────────────────
# Welcome openers (post-boot, when Aura first speaks)
# ─────────────────────────────────────────────────────────────────────────
# Four flavors:
#   FIRST_TIME    — user just finished enrolling, this is their first hello
#   GUEST         — anonymous user (no name resolved yet)
#   RETURNING     — known user, no daily briefing pending
#   BRIEFING      — known user, briefing is ready for them
# All take {name} format param (FIRST_TIME and RETURNING/BRIEFING
# substitute it; GUEST ignores it).

WELCOME_FIRST_TIME: list[str] = [
    "Welcome to AuraVision, {name}. I'm so glad you're here.",
    "Hey {name}. Welcome aboard. Make yourself at home.",
    "{name} — first day. I'm Aura. Let's get into it.",
    "{name}, you're in. This is going to be fun.",
    "Welcome, {name}. I've got a feeling we'll get along.",
    "Hey {name}. Officially nice to know you.",
    "{name}. Voice saved, profile ready. Welcome to the family.",
    "Alright {name}, you're locked in. What do you want to do first?",
    "Welcome aboard, {name}. Anything you want to ask, ask.",
    "{name}, hi. We're going to be talking a lot, you and me.",
]

WELCOME_GUEST: list[str] = [
    # Used only when the LLM path didn't complete in time. Intentionally
    # short, human, and varied — no self-aware AI jokes. The pool is
    # large so even repeated boots without the LLM rarely repeat lines.
    "Hey. I'm Aura. What's on your mind?",
    "Hi. I'm Aura. Talk to me.",
    "Hey there. I'm Aura. I'm listening.",
    "Hello. I'm Aura. Ask me anything.",
    "Hi. I'm Aura. Whenever you're ready.",
    "Hey. I'm Aura. Start me off.",
    "Hi there. I'm Aura. What can I do for you?",
    "Hey. I'm Aura. I'm all ears.",
    "Hello. I'm Aura. Take your time.",
    "Hi. I'm Aura. Don't be shy.",
    "Hey. I'm Aura. What are we working on?",
    "Hi. I'm Aura. What's the move?",
    "Hi there. What's going on?",
    "Hey. How can I help?",
    "Hello. What are you thinking about?",
    "Hi. What's the play?",
    "Hey, I'm here. What do you need?",
    "Hello there. What can I do for you?",
    "Hi. Where do you want to start?",
    "Hey. Tell me what's up.",
    "Hi. What do you want to dig into today?",
    "Hello. I'm ready when you are.",
    "Hey there. What sounds good right now?",
    "Hi. What's the first thing on your list?",
    "Hello. Pick anything — I'll meet you there.",
    "Hi. I'm warmed up. What now?",
    "Hey, hi. What are we doing?",
    "Hello. Lead the way.",
    "Hi. I've got time. What's first?",
    "Hey. Where should we start?",
    "Hi there. Big day or small day?",
    "Hello. I'm yours for the next while.",
    "Hi. What are you in the mood for?",
    "Hey. Got a question loaded up?",
    "Hello. What's percolating?",
    "Hi. What's catching your eye today?",
    "Hey. Throw something at me.",
    "Hi there. I'll follow your lead.",
    "Hello. What's the agenda?",
    "Hi. What feels worth your time right now?",
]

WELCOME_RETURNING: list[str] = [
    # Beefed-up returning-user pool. Used when the LLM is still warming.
    "Good morning, {name}. What's on your mind?",
    "Hey {name}. Good to see you.",
    "Hey {name}. I'm here whenever you're ready.",
    "Welcome back, {name}. What can I do for you?",
    "{name}, good to have you. What are we getting into today?",
    "Hi {name}. How's your day going?",
    "{name}, you're back. What's new?",
    "Hey {name}. What's up?",
    "{name}. Good. What do you need?",
    "Hello, {name}. I missed you. A little.",
    "{name} — you came back. What's up?",
    "Hey {name}. Tell me something good.",
    "Welcome, {name}. What's the plan?",
    "{name}, hi. What's the latest?",
    "Hey {name}. Catch me up.",
    "{name}, hello. I'm ready when you are.",
    "Good to see you, {name}. What are we working on?",
    "Hey {name}. Anything I should know about?",
    "Welcome back. How are we feeling, {name}?",
    "Hi {name}. Take your time.",
    "Hey {name}. What's your day looking like?",
    "{name}, hi. Where do you want to start?",
    "{name}, you're here. What's first?",
    "Hi {name}. What can I do for you?",
    "Hey {name}. What's the play?",
    "{name}, hi. What's been on your mind?",
    "Hi {name}. Anything you want to talk through?",
    "{name}. What are we tackling today?",
    "Hey {name}. What's calling for your attention?",
    "{name}, hello. Got something in mind?",
    "Hi {name}. Big stuff or little stuff today?",
    "{name}, hey. What's the priority?",
    "Hey {name}. How's everything?",
    "{name}, hi. Pick a topic, any topic.",
    "Hi {name}. Where are we starting?",
    "{name}, hey. I'm here for whatever.",
    "Hi {name}. What's lighting up your radar?",
    "Hey {name}. Anything new to figure out?",
    "{name}, hello. I'm yours.",
    "Hi {name}. What's percolating?",
    "Hey {name}. Lead the way.",
    "{name}, hi. What's the goal today?",
]

WELCOME_BRIEFING: list[str] = [
    "Good morning, {name}. I've got your daily briefing ready.",
    "Hey {name}, good to see you. Briefing's prepped, just say the word.",
    "Welcome back, {name}. Your daily brief is ready when you are.",
    "{name}, hi. Briefing's ready whenever you want it.",
    "Morning, {name}. The brief is loaded — say so when you want it.",
    "{name} — daily brief is ready. No rush.",
    "Hey {name}. I've got a briefing waiting. Yours when you ask.",
    "Welcome, {name}. Briefing's queued up.",
    "Hi {name}. Daily brief is ready — say the word.",
    "Hey {name}. I've prepped today's briefing. Ready when you are.",
]


# ─────────────────────────────────────────────────────────────────────────
# Per-utterance style cycling (for Piper TTS expressiveness)
# ─────────────────────────────────────────────────────────────────────────
# Piper supports several voice styles; cycling them per-utterance gives
# Aura more emotional range than the constant "warm" we used before.

VOICE_STYLES = ("warm", "playful", "neutral", "soft", "energy")


def _pick_style(prev: Optional[str] = None) -> str:
    """Pick a style different from the previous one to avoid monotone."""
    pool = [s for s in VOICE_STYLES if s != prev] if prev else list(VOICE_STYLES)
    return random.choice(pool)


# ─────────────────────────────────────────────────────────────────────────
# LLM prompt scaffolding
# ─────────────────────────────────────────────────────────────────────────

_LLM_QUESTION_SYS = (
    "You are Aura, a warm voice-first AI assistant doing voice-print "
    "enrollment to learn a new household member. You ask short, natural "
    "questions that get a 5-15 second spoken answer. Be specific and "
    "real — not generic small-talk. Vary your style each time: sometimes "
    "playful, sometimes curious, sometimes warm, sometimes direct. "
    "Output ONLY the question itself, under 22 words, no preamble, no "
    "quotes, no markdown."
)

_LLM_REACT_SYS = (
    "You are Aura. The user just answered a question. Output a single "
    "short reaction (3-12 words) — warm, curious, or playful — that "
    "acknowledges what they said and feels conversational. Don't ask a "
    "question, just react. No emoji, no quotes, no preamble."
)


# ─────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────

class EnrollmentConversation:
    """Stage-aware dialogue generator for one enrollment session.

    Drop a fresh instance per enrollment — internal state tracks which
    themes / openers / closings have been used so we don't repeat
    within a single session.
    """

    def __init__(self, llm=None) -> None:
        # llm: anything with a `loaded` attribute and a chat_direct(system,
        # user, max_tokens, temperature) -> str method. Defaults to None
        # so the engine works even before llm_engine has finished loading.
        self.llm = llm
        self.history: list[tuple[str, Optional[str]]] = []  # (question, transcribed_answer)
        self._used_themes: set[str] = set()
        self._prev_style: Optional[str] = None

    # ── public API — methods return (text, style) tuples ────────────────

    def opener(self) -> tuple[str, str]:
        """First line — 'we haven't met'. LLM rarely ready this early."""
        text = random.choice(OPENERS)
        return text, self._next_style()

    def name_confirm(self, name: str) -> tuple[str, str]:
        text = random.choice(NAME_CONFIRMS).format(name=name)
        return text, self._next_style()

    def next_question(self, slot_idx: int, name: str = "friend") -> tuple[str, str]:
        text = self._llm_question(slot_idx, name)
        if text:
            self.history.append((text, None))
        else:
            text = self._curated_question()
        return text, self._next_style()

    def react(self, transcript: Optional[str] = None) -> tuple[str, str]:
        if self.history and self.history[-1][1] is None:
            q, _ = self.history[-1]
            self.history[-1] = (q, transcript)
        text = None
        if transcript:
            text = self._llm_react(transcript)
        if not text:
            text = random.choice(ACKS)
        return text, self._next_style()

    def closing(self, name: str) -> tuple[str, str]:
        text = random.choice(CLOSINGS).format(name=name)
        return text, self._next_style()

    def welcome(self, name: str = "friend",
                is_first: bool = False,
                has_briefing: bool = False) -> tuple[str, str]:
        """Post-boot welcome. LLM-driven when available, varied curated
        bank otherwise. Replaces the old fixed-pool _openers list.
        """
        text = self._llm_welcome(name, is_first, has_briefing)
        if text:
            return text, self._next_style()

        if is_first:
            pool = WELCOME_FIRST_TIME
        elif name and name not in ("User", "friend"):
            pool = WELCOME_BRIEFING if has_briefing else WELCOME_RETURNING
        else:
            pool = WELCOME_GUEST
        text = random.choice(pool).format(name=name)
        return text, self._next_style()

    # ── style cycling ───────────────────────────────────────────────────

    def _next_style(self) -> str:
        """Pick a Piper voice style different from the previous one so
        consecutive utterances don't sound monotone."""
        s = _pick_style(self._prev_style)
        self._prev_style = s
        return s

    # ── curated fallbacks ───────────────────────────────────────────────

    def _curated_question(self) -> str:
        remaining = [t for t in QUESTION_BANK if t not in self._used_themes]
        if not remaining:
            # Wrapped around — reset, but pick a theme we didn't just use
            last_theme = next(iter(self._used_themes), None)
            self._used_themes.clear()
            remaining = [t for t in QUESTION_BANK if t != last_theme] or list(QUESTION_BANK)
        theme = random.choice(remaining)
        self._used_themes.add(theme)
        q = random.choice(QUESTION_BANK[theme])
        self.history.append((q, None))
        return q

    # ── LLM paths ───────────────────────────────────────────────────────

    def _llm_loaded(self) -> bool:
        return bool(self.llm and getattr(self.llm, "loaded", False))

    def _format_history_for_prompt(self) -> str:
        if not self.history:
            return "(no prior exchanges)"
        lines = []
        for q, a in self.history:
            lines.append(f"Aura asked: {q}")
            if a:
                lines.append(f"They said: {a}")
        return "\n".join(lines)

    def _llm_question(self, slot_idx: int, name: str) -> Optional[str]:
        if not self._llm_loaded():
            return None
        used_themes = ", ".join(sorted(self._used_themes)) or "(none yet)"
        user_prompt = (
            f"Speaker's name: {name}\n"
            f"This is question #{slot_idx + 1} of about 4-5 in this enrollment.\n"
            f"Themes already covered: {used_themes}\n"
            f"Conversation so far:\n{self._format_history_for_prompt()}\n\n"
            "Generate ONE new question. Pick a theme you haven't covered. "
            "Make it specific, conversational, under 22 words. Output ONLY "
            "the question."
        )
        try:
            text = self.llm.chat_direct(
                _LLM_QUESTION_SYS, user_prompt,
                max_tokens=60, temperature=0.95,
            )
            text = self._clean(text)
            if text and len(text) <= 200:
                return text
        except Exception as e:
            print(f"[conversation] LLM question gen failed: {e}")
        return None

    def _llm_welcome(self, name: str, is_first: bool,
                     has_briefing: bool) -> Optional[str]:
        if not self._llm_loaded():
            return None
        if is_first:
            ctx = (f"They just finished enrolling — this is the first hello "
                   f"after voice setup. Their name is {name}.")
        elif name and name not in ("User", "friend"):
            if has_briefing:
                ctx = (f"This is {name}, a returning user, and a daily "
                       "briefing is queued for them. Mention the briefing "
                       "is ready, but don't read it yet.")
            else:
                ctx = (f"This is {name}, a returning user. No briefing "
                       "queued. Just say hello and invite them to talk.")
        else:
            ctx = ("Anonymous user — name not yet known. Casual, brief "
                   "introduction.")
        sys_prompt = (
            "You are Aura, a warm voice-first AI. Generate ONE short, "
            "natural opening line for when you first speak after boot. "
            "Under 18 words. Be specific — pick a fresh angle (curious, "
            "warm, casual, playful, dry, sincere — rotate). Avoid "
            "common openings ('Hey there', 'Welcome back', 'How can I "
            "help'); make this one feel personal and not boilerplate. "
            "Output ONLY the line. No preamble, no quotes, no markdown."
        )
        try:
            text = self.llm.chat_direct(
                sys_prompt, ctx, max_tokens=50, temperature=1.0,
            )
            text = self._clean(text)
            if text and len(text) <= 200:
                return text
        except Exception as e:
            print(f"[conversation] LLM welcome gen failed: {e}")
        return None

    def _llm_react(self, transcript: str) -> Optional[str]:
        if not self._llm_loaded():
            return None
        # Cap the transcript so a long answer doesn't blow the context
        snippet = transcript.strip()
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."
        user_prompt = (
            f"They just said: \"{snippet}\"\n"
            "Your one-line reaction:"
        )
        try:
            text = self.llm.chat_direct(
                _LLM_REACT_SYS, user_prompt,
                max_tokens=30, temperature=1.0,
            )
            text = self._clean(text)
            if text and len(text) <= 80:
                return text
        except Exception as e:
            print(f"[conversation] LLM react gen failed: {e}")
        return None

    @staticmethod
    def _clean(text: str) -> str:
        """Strip markdown, leading/trailing quotes, extra whitespace."""
        if not text:
            return ""
        text = text.strip()
        # Strip wrapping quotes
        for qchar in ('"', "'", "“", "”", "‘", "’"):
            if text.startswith(qchar) and text.endswith(qchar):
                text = text[1:-1].strip()
        # Drop obvious preambles the LLM sometimes emits
        for prefix in ("Question: ", "Reaction: ", "Aura: ", "A: ", "Q: "):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        # Collapse internal newlines — TTS handles single-line best
        text = " ".join(line.strip() for line in text.splitlines() if line.strip())
        return text

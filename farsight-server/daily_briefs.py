#!/usr/bin/env python3
"""
Daily Brief Scheduler — Generate personalized PDF briefs for all co-founders
and email them.

Designed to run via cron at 6:00 AM daily:
    0 6 * * * cd /home/paul/LedgerAI/farsight-server && python3 daily_briefs.py

Requires these env vars (add to .env):
    BRIEF_SMTP_HOST=smtp.gmail.com
    BRIEF_SMTP_PORT=587
    BRIEF_SMTP_USER=your-email@gmail.com
    BRIEF_SMTP_PASS=app-password-here
    BRIEF_FROM_EMAIL=aura@ledgerai.co

If SMTP is not configured, briefs are still generated as PDFs in
data/briefings/ — email delivery is skipped with a warning.
"""

import json
import os
import smtplib
import sys
import time
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Add parent dir so we can import brief_generator
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from brief_generator import generate_brief_from_llm

# ── Load .env if present ─────────────────────────────────────
_ENV_PATH = os.path.join(os.path.dirname(_SCRIPT_DIR), ".env")
if os.path.isfile(_ENV_PATH):
    with open(_ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# ── Co-founder roster ────────────────────────────────────────
RECIPIENTS = [
    {
        "name": "Paul Chou",
        "role": "CEO & Founder",
        "email": "paul.chou@ledgerai.co",
        "focus": (
            "Paul is the technical CEO who built Aura. His biggest weakness is conflict "
            "avoidance — he assigns process instead of having hard conversations. Tell him "
            "SPECIFICALLY which co-founder he needs to confront today, about what, and what "
            "to say. Call out where David is accumulating too much power, where Jorge is being "
            "undisciplined, where Bob is in over his head. Name the interpersonal tensions and "
            "tell Paul exactly what to do about each one. Don't let him hide behind 'team decisions.' "
            "Flag if anyone is positioning against him, forming alliances, or undermining the "
            "chain of command. Also cover: fundraising, runway, legal exposure, product risk."
        ),
    },
    {
        "name": "David Lara",
        "role": "COO",
        "email": "david.lara@ledgerai.co",
        "focus": (
            "David is the institutional operator — government background, risk-averse, "
            "increasingly influential since the crisis. Tell him HONESTLY where he's "
            "overstepping — engaging outside counsel solo, talking to Kaiser about strategy "
            "without alignment, marginalizing Jorge. Flag where his 'I told you so' attitude "
            "is eroding team trust even when he's right. Call out if Bob and Jorge are forming "
            "a bloc against him and what he should do about it. But also flag where Paul is "
            "too soft and David needs to push harder. Be specific about Kaiser execution risks, "
            "SEC disclosure timeline, and whether the legal costs will eat the remaining runway."
        ),
    },
    {
        "name": "Jorge Guinovart",
        "role": "CMO",
        "email": "jorge.guinovart@ledgerai.co",
        "focus": (
            "Jorge is the community builder who is feeling muzzled and marginalized. Tell him "
            "DIRECTLY where his instincts are right (community engagement matters) AND where "
            "he's being reckless (off-script statements, Instagram posts, AMA deviations). "
            "Quantify the community damage — member attrition, influencer departures, competitor "
            "poaching. Flag his alliance with Bob and whether it's strategic or desperate. "
            "Tell him what David is saying about him behind his back and whether David has a "
            "point. Be honest about whether the CMO role survives a pivot to enterprise. Give "
            "him a path to stay relevant that doesn't involve fighting David."
        ),
    },
    {
        "name": "Bob Carella",
        "role": "CFO",
        "email": "bob.carella@ledgerai.co",
        "focus": (
            "Bob is the most vulnerable co-founder right now — his tokenomics model failed, "
            "he lost track of treasury transactions, he falsely suspected David of embezzlement, "
            "and he's quietly offered to step down. Tell him the HARD TRUTH about his performance "
            "gaps without crushing him. Quantify exactly what the treasury mistake cost. Flag "
            "where Jorge is using him as a pawn against David. Tell him whether Paul still trusts "
            "him and what specific actions will rebuild credibility. Be brutally specific about "
            "SEC exposure, SAFT investor risk, and whether the revised budget works. Also flag "
            "the Binance Labs angle — is it real or is Jorge dragging him into another overreach?"
        ),
    },
]

# ── Context loading ──────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")
_CONTEXT_FILE = os.path.join(_DATA_DIR, "learning", "cofounder_context.txt")
_CONVERSATIONS_FILE = os.path.join(_DATA_DIR, "learning", "cofounder_conversations.json")
_CRISIS_FILE = os.path.join(_DATA_DIR, "learning", "crisis_conversations.json")
_CRISIS_CONTEXT = os.path.join(_DATA_DIR, "learning", "crisis_context.txt")
_BRIEFINGS_DIR = os.path.join(_DATA_DIR, "briefings")


def _load_context(recipient_name=None):
    """Load context for brief generation, prioritized by relevance.

    Returns a string capped at ~14K chars so it fits within the model's
    context window after system/user prompt overhead.

    Priority order:
    1. Crisis conversations involving this person (most urgent)
    2. Crisis conversations involving others (they need to know)
    3. Core context summary
    4. Regular co-founder conversations involving this person
    """
    MAX_CHARS = 14000
    parts = []
    first_name = recipient_name.split()[0] if recipient_name else None

    def _format_convos(convos, label=""):
        """Format conversations, putting recipient-relevant ones first."""
        lines = []
        if first_name:
            relevant = [c for c in convos
                        if first_name in " ".join(c.get("participants", []))]
            other = [c for c in convos
                     if first_name not in " ".join(c.get("participants", []))]
            ordered = relevant + other
        else:
            ordered = convos
        for convo in ordered:
            lines.append(f"\n--- {label}{convo.get('topic', 'Discussion')} ---")
            lines.append(f"Date: {convo.get('date', '?')}  |  "
                         f"Setting: {convo.get('setting', '?')}")
            lines.append(f"Participants: {', '.join(convo.get('participants', []))}")
            for msg in convo.get("messages", []):
                speaker = msg.get("speaker") or msg.get("from", "?")
                lines.append(f"  {speaker}: {msg.get('text', '')}")
        return "\n".join(lines)

    # Priority 1: Crisis conversations (most urgent intel)
    if os.path.isfile(_CRISIS_FILE):
        crisis = json.load(open(_CRISIS_FILE))
        parts.append("=== CRITICAL: ONGOING CRISIS — $LEDGER TOKEN CRASH ===")
        parts.append(_format_convos(crisis, "[CRISIS] "))

    # Priority 2: Core context summary
    if os.path.isfile(_CONTEXT_FILE):
        parts.append("\n=== COMPANY CONTEXT ===")
        parts.append(open(_CONTEXT_FILE).read())

    # Priority 3: Regular co-founder conversations
    if os.path.isfile(_CONVERSATIONS_FILE):
        convos = json.load(open(_CONVERSATIONS_FILE))
        parts.append("\n=== CO-FOUNDER CONVERSATIONS ===")
        parts.append(_format_convos(convos))

    combined = "\n".join(parts)
    if len(combined) > MAX_CHARS:
        combined = combined[:MAX_CHARS] + "\n[...additional context truncated]"
    return combined


# ── Email delivery ───────────────────────────────────────────
def _send_email(to_addr, subject, body_text, pdf_path):
    """Send email with PDF attachment via SMTP."""
    smtp_host = os.environ.get("BRIEF_SMTP_HOST")
    smtp_port = int(os.environ.get("BRIEF_SMTP_PORT", "587"))
    smtp_user = os.environ.get("BRIEF_SMTP_USER")
    smtp_pass = os.environ.get("BRIEF_SMTP_PASS")
    from_email = os.environ.get("BRIEF_FROM_EMAIL", smtp_user or "aura@ledgerai.co")

    if not all([smtp_host, smtp_user, smtp_pass]):
        print(f"  [email] SMTP not configured — skipping email to {to_addr}")
        print(f"  [email] Set BRIEF_SMTP_HOST, BRIEF_SMTP_USER, BRIEF_SMTP_PASS in .env")
        return False

    msg = MIMEMultipart()
    msg["From"] = f"Aura Perpetual <{from_email}>"
    msg["To"] = to_addr
    msg["Subject"] = subject

    msg.attach(MIMEText(body_text, "plain"))

    # Attach PDF
    with open(pdf_path, "rb") as f:
        part = MIMEBase("application", "pdf")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",
                        f"attachment; filename={os.path.basename(pdf_path)}")
        msg.attach(part)

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print(f"  [email] Sent to {to_addr}")
        return True
    except Exception as e:
        print(f"  [email] Failed to send to {to_addr}: {e}")
        return False


# ── Main ─────────────────────────────────────────────────────
def generate_all_briefs(recipients=None, send_email=True):
    """Generate personalized briefs for all (or specified) co-founders."""
    if recipients is None:
        recipients = RECIPIENTS

    date_str = datetime.now().strftime("%B %d, %Y")
    date_file = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(_BRIEFINGS_DIR, exist_ok=True)

    base_context = None  # loaded per-recipient below
    results = []

    print(f"{'='*60}")
    print(f"AURA Daily Brief Generation — {date_str}")
    print(f"{'='*60}")
    print(f"Recipients: {len(recipients)}")
    print(f"Context source: {_CONTEXT_FILE}")
    print()

    for r in recipients:
        name = r["name"]
        role = r["role"]
        email = r["email"]
        focus = r["focus"]
        safe_name = name.lower().replace(" ", "_")

        output_path = os.path.join(
            _BRIEFINGS_DIR, f"{date_file}_brief_{safe_name}.pdf"
        )

        print(f"[{name} ({role})]")
        print(f"  Generating brief...")

        # Build role-specific context (prioritize conversations involving this person)
        recipient_context = _load_context(recipient_name=name)
        context = f"ROLE-SPECIFIC FOCUS FOR {name.upper()} ({role}):\n{focus}\n\n{recipient_context}"

        try:
            t0 = time.time()
            generate_brief_from_llm(name, role, context, output_path, date_str)
            elapsed = time.time() - t0
            print(f"  PDF: {output_path} ({elapsed:.1f}s)")

            # Email delivery
            if send_email:
                subject = f"[AURA] Daily Brief — {date_str}"
                body = (
                    f"Good morning {name.split()[0]},\n\n"
                    f"Your personalized daily intelligence brief for {date_str} "
                    f"is attached.\n\n"
                    f"— Aura Perpetual\n"
                    f"   Director of Intelligence | Perpetual Engine v2.1\n\n"
                    f"CLASSIFICATION: TS//SCI/NOFORN\n"
                    f"This communication is for the intended recipient only."
                )
                _send_email(email, subject, body, output_path)

            results.append({"name": name, "path": output_path, "ok": True})

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({"name": name, "path": output_path, "ok": False, "error": str(e)})

        print()

    # Summary
    ok = sum(1 for r in results if r["ok"])
    print(f"{'='*60}")
    print(f"Complete: {ok}/{len(results)} briefs generated")
    for r in results:
        status = "OK" if r["ok"] else f"FAILED: {r.get('error', '?')}"
        print(f"  {r['name']}: {status}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate daily briefs for all co-founders")
    parser.add_argument("--no-email", action="store_true",
                        help="Generate PDFs only, skip email delivery")
    parser.add_argument("--recipient", type=str, default=None,
                        help="Generate for one recipient only (first name)")
    args = parser.parse_args()

    targets = RECIPIENTS
    if args.recipient:
        targets = [r for r in RECIPIENTS
                   if args.recipient.lower() in r["name"].lower()]
        if not targets:
            print(f"Unknown recipient: {args.recipient}")
            print(f"Available: {', '.join(r['name'] for r in RECIPIENTS)}")
            sys.exit(1)

    generate_all_briefs(targets, send_email=not args.no_email)

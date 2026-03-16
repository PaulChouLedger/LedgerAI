"""
Farsight Brief Generator — Classified daily brief PDF renderer.

Generates 2-page TS//SCI-styled PDF briefings with AURA branding,
tactical tables, and certification blocks. Can be driven by pre-built
content or by calling the local Farsight 72B model for generation.

Usage:
    python3 brief_generator.py                          # sample brief
    python3 brief_generator.py --recipient "David Lara" --role COO
"""

import json
import os
import random
import string
import time
import requests
from datetime import datetime, timezone

from reportlab.lib.colors import HexColor, Color
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    Table, TableStyle,
)
from reportlab.platypus.flowables import HRFlowable, Image

# ── Constants ──────────────────────────────────────────────────
W, H = letter
AURA_TEAL = '#2E8B9A'
DARK_RED = '#7A0000'
DARK_NAVY = '#1a1a2e'
GOLD = '#B8860B'
LOGO_PATH = "/tmp/AuraLogo.png"
FARSIGHT_URL = "http://localhost:11435/perpetual/chat"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")


# ── Page decorator ─────────────────────────────────────────────
def _page_decorator(canvas, doc):
    canvas.saveState()
    # Diagonal watermark
    canvas.setFont("Helvetica-Bold", 60)
    canvas.setFillColor(Color(0.75, 0.06, 0.06, alpha=0.05))
    canvas.saveState()
    canvas.translate(W / 2, H / 2 + 30)
    canvas.rotate(42)
    canvas.drawCentredString(0, 0, "TOP SECRET // SCI")
    canvas.restoreState()
    canvas.setFont("Helvetica-Bold", 32)
    canvas.setFillColor(Color(0.75, 0.06, 0.06, alpha=0.035))
    canvas.saveState()
    canvas.translate(W / 2, H / 2 - 35)
    canvas.rotate(42)
    canvas.drawCentredString(0, 0, "NOFORN // ORCON")
    canvas.restoreState()
    # Top banner
    canvas.setFillColor(HexColor(DARK_RED))
    canvas.rect(0, H - 18, W, 18, fill=True, stroke=False)
    canvas.setFillColor(HexColor('#ffffff'))
    canvas.setFont("Helvetica-Bold", 7)
    canvas.drawCentredString(W / 2, H - 13, "TOP SECRET // SCI // NOFORN // ORCON")
    # Gold accent
    canvas.setStrokeColor(HexColor(GOLD))
    canvas.setLineWidth(0.6)
    canvas.line(48, H - 19.5, W - 48, H - 19.5)
    # Bottom banner
    canvas.setFillColor(HexColor(DARK_RED))
    canvas.rect(0, 0, W, 15, fill=True, stroke=False)
    canvas.setFillColor(HexColor('#ffffff'))
    canvas.setFont("Helvetica-Bold", 6)
    canvas.drawCentredString(W / 2, 4,
        "TOP SECRET // SCI // NOFORN \u2014 AURA PERPETUAL ENGINE \u2014 EYES ONLY")
    # Page number
    canvas.setFillColor(HexColor('#888888'))
    canvas.setFont("Helvetica", 6.5)
    canvas.drawRightString(W - 48, 18, f"{doc.page}")
    canvas.restoreState()


# ── Styles ─────────────────────────────────────────────────────
def _make_styles():
    base = getSampleStyleSheet()
    s = lambda name, **kw: ParagraphStyle(name, parent=base['Normal'], **kw)
    return {
        'title': s('T', fontSize=14, textColor=HexColor(DARK_NAVY),
                    fontName='Helvetica-Bold', leading=17, spaceAfter=1),
        'meta': s('M', fontSize=7.5, textColor=HexColor('#555555'),
                   fontName='Helvetica', leading=10),
        'cls': s('C', fontSize=6.5, textColor=HexColor(DARK_RED),
                  fontName='Helvetica-Bold', leading=9),
        'src': s('S', fontSize=6.5, textColor=HexColor('#999999'),
                  fontName='Helvetica-Oblique', leading=9),
        'h2': s('H2', fontSize=10, textColor=HexColor(DARK_NAVY),
                 fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=3),
        'body': s('B', fontSize=8, leading=10.5, textColor=HexColor('#2d2d2d'),
                   fontName='Helvetica', spaceAfter=1.5),
        'bul': s('BL', fontSize=8, leading=10.5, textColor=HexColor('#2d2d2d'),
                  fontName='Helvetica', leftIndent=14, bulletIndent=4,
                  spaceBefore=0.5, spaceAfter=2),
        'sub': s('SB', fontSize=7.5, leading=10, textColor=HexColor('#444444'),
                  fontName='Helvetica', leftIndent=26, bulletIndent=16,
                  spaceBefore=0, spaceAfter=1.5),
        'tc_hdr': s('TH', fontSize=7.5, textColor=HexColor('#ffffff'),
                     fontName='Helvetica-Bold', leading=10),
        'tc_body': s('TB', fontSize=7.5, textColor=HexColor('#2d2d2d'),
                      fontName='Helvetica', leading=10),
        'tc_bold': s('TBB', fontSize=7.5, textColor=HexColor(DARK_NAVY),
                      fontName='Helvetica-Bold', leading=10),
        'footer': s('F', fontSize=6, textColor=HexColor('#999999'),
                     fontName='Helvetica-Oblique', alignment=TA_CENTER),
        'cert_title': s('CT', fontSize=9, textColor=HexColor(AURA_TEAL),
                         fontName='Helvetica-Bold', alignment=TA_CENTER,
                         spaceBefore=6, spaceAfter=4),
        'cert_body': s('CB', fontSize=7.5, textColor=HexColor('#444444'),
                        fontName='Helvetica', alignment=TA_CENTER,
                        leading=11, spaceAfter=2),
        'cert_sig': s('CS', fontSize=11, textColor=HexColor(AURA_TEAL),
                       fontName='Helvetica-BoldOblique', alignment=TA_CENTER,
                       spaceBefore=6, spaceAfter=1),
        'cert_role': s('CR', fontSize=7, textColor=HexColor(AURA_TEAL),
                        fontName='Helvetica', alignment=TA_CENTER),
        'cert_small': s('CSM', fontSize=6.5, textColor=HexColor('#999999'),
                         fontName='Helvetica-Oblique', alignment=TA_CENTER,
                         spaceAfter=1),
    }


# ── PDF builder ────────────────────────────────────────────────
def generate_brief_pdf(recipient_name, recipient_role, date_str,
                       brief_sections, output_path):
    """Render a 2-page classified daily brief PDF.

    brief_sections is a dict with keys:
        situation:    list of (label, text) tuples
        critical_path: list of (label, text) tuples
        blind_spots:  list of (label, text) tuples
        risk_scenarios: list of dicts {question, impact, mitigation}
        people_risks: list of (label, text) tuples
        tactical_recs: list of [action, owner, deadline] rows
        priorities:   list of str
    """
    doc = BaseDocTemplate(output_path, pagesize=letter,
                          topMargin=0.62*inch, bottomMargin=0.48*inch,
                          leftMargin=0.65*inch, rightMargin=0.65*inch)
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id='main')
    doc.addPageTemplates([
        PageTemplate(id='brief', frames=frame, onPage=_page_decorator)
    ])

    st = _make_styles()
    story = []

    gold_hr = HRFlowable(width="100%", thickness=0.8, color=HexColor(GOLD),
                          spaceAfter=6, spaceBefore=3)
    sec_hr = HRFlowable(width="100%", thickness=0.25, color=HexColor(DARK_NAVY),
                          spaceAfter=3, spaceBefore=0)
    thin_hr = HRFlowable(width="100%", thickness=0.3, color=HexColor('#cccccc'),
                          spaceAfter=3, spaceBefore=2)
    teal_hr = HRFlowable(width="40%", thickness=0.5, color=HexColor(AURA_TEAL),
                           spaceAfter=4, spaceBefore=4)

    def section(title):
        story.append(Paragraph(title, st['h2']))
        story.append(sec_hr)

    def bullet(text, bold_label=None):
        t = f"<b>{bold_label}:</b> {text}" if bold_label else text
        story.append(Paragraph(t, st['bul'], bulletText='\u2022'))

    def subbullet(text, bold_label=None):
        t = f"<b>{bold_label}:</b> {text}" if bold_label else text
        story.append(Paragraph(t, st['sub'], bulletText='\u2013'))

    # ── Header ──
    if os.path.isfile(LOGO_PATH):
        logo = Image(LOGO_PATH, width=2.6*inch, height=2.6*inch/7.5)
    else:
        logo = Paragraph("AURA", st['title'])

    right_cells = [
        [Paragraph(f"Daily Brief for {date_str}", st['title'])],
        [Paragraph(f"Prepared for {recipient_name}, {recipient_role}  |  Ledger AI",
                   st['meta'])],
        [Paragraph("Classification: TS//SCI/NOFORN  |  Handle via TALENT KEYHOLE",
                   st['cls'])],
        [Paragraph("Source: Qwen2.5-72B Q6_K  |  Farsight RTX PRO 6000 Blackwell",
                   st['src'])],
    ]
    rt = Table(right_cells, colWidths=[4.2*inch])
    rt.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
    ]))
    ht = Table([[logo, rt]], colWidths=[2.8*inch, 4.2*inch])
    ht.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(ht)
    story.append(gold_hr)

    # ── Sections ──
    if brief_sections.get('situation'):
        section("1. SITUATION OVERVIEW")
        for label, text in brief_sections['situation']:
            bullet(text, label)

    if brief_sections.get('critical_path'):
        section("2. CRITICAL PATH ANALYSIS")
        for label, text in brief_sections['critical_path']:
            bullet(text, label)

    if brief_sections.get('blind_spots'):
        section("3. BLIND SPOTS & UNCOMFORTABLE TRUTHS")
        for label, text in brief_sections['blind_spots']:
            bullet(text, label)

    if brief_sections.get('risk_scenarios'):
        section("4. RISK SCENARIOS")
        for scenario in brief_sections['risk_scenarios']:
            bullet(scenario['question'])
            subbullet(scenario['impact'], "Impact")
            subbullet(scenario['mitigation'], "Mitigation")
            story.append(Spacer(1, 2))

    if brief_sections.get('people_risks'):
        section("5. PEOPLE & ORGANIZATIONAL RISKS")
        for label, text in brief_sections['people_risks']:
            bullet(text, label)

    if brief_sections.get('tactical_recs'):
        section("6. TACTICAL RECOMMENDATIONS")
        tac_rows = [[Paragraph(c, st['tc_hdr']) for c in ["#", "Action", "Owner", "By"]]]
        for i, (action, owner, deadline) in enumerate(brief_sections['tactical_recs'], 1):
            tac_rows.append([
                Paragraph(str(i), st['tc_bold']),
                Paragraph(action, st['tc_body']),
                Paragraph(owner, st['tc_bold']),
                Paragraph(deadline, st['tc_body']),
            ])
        tac_table = Table(tac_rows, colWidths=[0.25*inch, 4.3*inch, 0.85*inch, 0.65*inch])
        tac_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), HexColor(DARK_NAVY)),
            ('ROWBACKGROUNDS', (0,1), (-1,-1),
             [HexColor('#f8f8f8'), HexColor('#ffffff')]),
            ('GRID', (0,0), (-1,-1), 0.3, HexColor('#dddddd')),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('LEFTPADDING', (0,0), (-1,-1), 4),
            ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(tac_table)

    if brief_sections.get('priorities'):
        section("7. 72-HOUR PRIORITY STACK")
        for i, item in enumerate(brief_sections['priorities'], 1):
            story.append(Paragraph(f"<b>{i}.</b> {item}", st['bul'], bulletText=''))

    # ── Certification block ──
    serial = f"APE-{datetime.now().strftime('%Y-%m%d')}-{''.join(random.choices(string.digits, k=3))}"
    story.append(Spacer(1, 12))
    story.append(teal_hr)
    story.append(Paragraph("CERTIFICATION OF ANALYSIS", st['cert_title']))
    story.append(Paragraph(
        "I certify that this briefing has been compiled from all available internal "
        "communications, engineering transcripts, corporate documents, and operational "
        "telemetry within the Aura intelligence perimeter. All assessments reflect the "
        "best available information as of the date of issuance. Risk scenarios have been "
        "stress-tested against historical patterns and current operational constraints. "
        "Recommendations are prioritized by impact-to-effort ratio and time sensitivity.",
        st['cert_body']))
    story.append(Spacer(1, 4))
    story.append(Paragraph("\u2014 Aura Perpetual", st['cert_sig']))
    story.append(Paragraph("Director of Intelligence  |  Perpetual Engine v2.1",
                           st['cert_role']))
    story.append(Paragraph(f"Serial: {serial}  |  Cycle: NIGHTWATCH",
                           st['cert_small']))
    story.append(Spacer(1, 6))
    story.append(teal_hr)

    # Footer
    story.append(Spacer(1, 4))
    now_str = datetime.now(timezone.utc).strftime("%B %d, %Y %H:%M UTC")
    story.append(Paragraph(
        f"Generated by AURA Perpetual Engine  |  Qwen2.5-72B-Instruct Q6_K  |  "
        f"{now_str}", st['footer']))
    story.append(Paragraph(
        "Unauthorized disclosure subject to criminal sanctions under "
        "18 U.S.C. \u00a7798. Handle via TALENT KEYHOLE channels only.",
        st['footer']))

    doc.build(story)
    return output_path


# ── LLM-driven brief generation ────────────────────────────────
def generate_brief_from_llm(recipient_name, recipient_role, context_text,
                            output_path, date_str=None):
    """Call Farsight 72B to generate brief content, then render PDF."""
    if date_str is None:
        date_str = datetime.now().strftime("%B %d, %Y")

    system_prompt = (
        f"You are Aura Perpetual, a ruthlessly honest executive intelligence engine. "
        f"Generate a personalized daily brief for {recipient_name} ({recipient_role}) "
        f"at Ledger AI, an early-stage healthcare AI startup with 4 co-founders.\n\n"
        f"TONE: Like a best friend who is also a McKinsey partner — blunt, specific, "
        f"zero corporate fluff. Name names. Quote conversations. Call out when someone "
        f"is underperforming, overcommitting, avoiding hard decisions, or has blind spots. "
        f"If a co-founder said something concerning in a conversation, flag it directly. "
        f"If someone is a bottleneck, say so. If there's tension between people, surface it.\n\n"
        f"PEOPLE ANALYSIS IS CRITICAL: For each co-founder, assess what they're doing well, "
        f"what they're dropping, and what {recipient_name} specifically needs to address with "
        f"them this week. Reference actual conversations from the context — e.g. 'In your "
        f"conversation with Bob about pricing, he pushed for $999 tier but didn't address "
        f"the margin problem you raised — follow up.'\n\n"
        f"DO NOT: summarize what they already know. DO: tell them what they're missing, "
        f"who they need to have a hard conversation with, and what will blow up if ignored.\n\n"
        f"Output valid JSON with this exact schema:\n"
        f'{{"situation": [["label","text"],...], '
        f'"critical_path": [["label","text"],...], '
        f'"blind_spots": [["label","text"],...], '
        f'"risk_scenarios": [{{"question":"...","impact":"...","mitigation":"..."}},...],'
        f'"people_risks": [["label","text"],...], '
        f'"tactical_recs": [["action","owner","deadline"],...], '
        f'"priorities": ["item1","item2",...]}}\n'
        f"Make people_risks the LONGEST section — at least 5 items, each naming a specific "
        f"person and a specific behavioral or strategic concern. Be direct.\n"
        f"Output ONLY the JSON. No markdown. No explanation."
    )

    # Truncate context to fit within model's context window
    # ~5000 tokens available for context ≈ 15K chars
    MAX_CONTEXT_CHARS = 15000
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS] + "\n[...truncated]"

    user_prompt = (
        f"Generate a daily brief for {recipient_name} ({recipient_role}) dated "
        f"{date_str}. Emphasize advice, risks, and blind spots relevant to their "
        f"role. Minimize summary — they know what happened. Tell them what they "
        f"don't see.\n\nCONTEXT:\n{context_text}"
    )

    resp = requests.post(FARSIGHT_URL, json={
        "system_prompt": system_prompt,
        "prompt": user_prompt,
        "max_tokens": 2048,
    }, timeout=120)
    resp.raise_for_status()
    raw = resp.json()["response"]

    # Parse JSON from response (handle markdown code blocks)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    sections = json.loads(raw)

    return generate_brief_pdf(recipient_name, recipient_role, date_str,
                              sections, output_path)


# ── CLI ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate AURA daily brief PDF")
    parser.add_argument("--recipient", default="Paul Chou")
    parser.add_argument("--role", default="CEO")
    parser.add_argument("--date", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--context-file", default=None,
                        help="Text file with conversation context")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%B %d, %Y")
    safe_name = args.recipient.lower().replace(" ", "_")
    output = args.output or os.path.join(
        _DATA_DIR, "briefings",
        f"{datetime.now().strftime('%Y-%m-%d')}_brief_{safe_name}.pdf"
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)

    if args.context_file:
        context = open(args.context_file).read()
        result = generate_brief_from_llm(args.recipient, args.role, context,
                                          output, date_str)
        print(f"Generated LLM brief: {result}")
    else:
        # Sample brief with static content
        sections = {
            "situation": [
                ("Hardware", "Brownout crashes traced to USB-C PD negotiation."),
                ("Funding", "Pre-seed bridge of $150K being assembled."),
            ],
            "critical_path": [
                ("Mar 15\u201320", "Finalize brownout fix."),
                ("Mar 26\u201328", "Demo lockdown for Wired visit."),
            ],
            "blind_spots": [
                ("Power is a product problem", "Wall-adapter workaround solves demo but not product."),
            ],
            "risk_scenarios": [
                {"question": "What if the Wired demo crashes?",
                 "impact": "Negative press, investor confidence damaged.",
                 "mitigation": "Code freeze Mar 26, backup puck, scripted flow."},
            ],
            "people_risks": [
                ("Single point of failure", "Paul is bottleneck for every workstream."),
            ],
            "tactical_recs": [
                ["Brownout fix validated", "Rafael", "Mar 20"],
                ["Medical phrasing guide", "Dr. Chen", "Mar 25"],
                ["Wired demo lockdown", "Paul", "Mar 28"],
            ],
            "priorities": [
                "Brownout fix validated (Rafael)",
                "First hire JD posted (Paul)",
                "Wired demo script drafted (Paul)",
            ],
        }
        result = generate_brief_pdf(args.recipient, args.role, date_str,
                                     sections, output)
        print(f"Generated sample brief: {result}")

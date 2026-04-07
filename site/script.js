/* ============================================================
   LEDGER AI — Blade Runner UI Scripts (v2)
   ============================================================ */

// ---- Particle Network Background ----
(function () {
  const canvas = document.getElementById('particles');
  const ctx = canvas.getContext('2d');
  let w, h, particles;

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  }

  class Particle {
    constructor() { this.reset(); }
    reset() {
      this.x = Math.random() * w;
      this.y = Math.random() * h;
      this.vx = (Math.random() - 0.5) * 0.3;
      this.vy = (Math.random() - 0.5) * 0.3;
      this.radius = Math.random() * 1.5 + 0.5;
      this.opacity = Math.random() * 0.5 + 0.1;
    }
    update() {
      this.x += this.vx; this.y += this.vy;
      if (this.x < 0 || this.x > w) this.vx *= -1;
      if (this.y < 0 || this.y > h) this.vy *= -1;
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 234, 255, ${this.opacity})`;
      ctx.fill();
    }
  }

  function init() {
    resize();
    const count = Math.min(Math.floor((w * h) / 12000), 150);
    particles = Array.from({ length: count }, () => new Particle());
  }

  function connectParticles() {
    const maxDist = 150;
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < maxDist) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(0, 234, 255, ${(1 - dist / maxDist) * 0.15})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
  }

  function animate() {
    ctx.clearRect(0, 0, w, h);
    particles.forEach(p => { p.update(); p.draw(); });
    connectParticles();
    requestAnimationFrame(animate);
  }

  window.addEventListener('resize', resize);
  init();
  animate();
})();

// ---- Nav Scroll Effect ----
(function () {
  const nav = document.querySelector('.nav');
  window.addEventListener('scroll', () => {
    if (window.scrollY > 100) {
      nav.style.borderBottomColor = 'rgba(0, 234, 255, 0.2)';
      nav.style.background = 'rgba(0, 8, 16, 0.95)';
    } else {
      nav.style.borderBottomColor = 'rgba(0, 234, 255, 0.1)';
      nav.style.background = 'rgba(0, 8, 16, 0.85)';
    }
  });
})();

// ---- Smooth Scroll ----
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const target = document.querySelector(a.getAttribute('href'));
    if (target) { e.preventDefault(); target.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  });
});

// ---- Team Network Visualization ----
(function () {
  const container = document.getElementById('teamNetwork');
  const canvas = document.getElementById('teamCanvas');
  if (!canvas || !container) return;
  const ctx = canvas.getContext('2d');
  const dossier = document.getElementById('teamDossier');
  const dossierBody = document.getElementById('dossierBody');
  const dossierClose = document.getElementById('dossierClose');

  const founders = [
    { id: 0, init: 'PC', name: 'Paul Chou', role: 'CEO & Founder',
      bio: 'Founded <strong>LedgerX</strong>, the first CFTC-regulated Bitcoin options exchange. Former <strong>Goldman Sachs</strong> quantitative trading. Built AuraVision from zero to on-device inference.',
      tags: ['Product', 'Engineering', 'Strategy'] },
    { id: 1, init: 'BC', name: 'Bob Carella', role: 'CFO',
      bio: 'Treasury and financial operations across <strong>Binance</strong> and <strong>Sprinklr</strong>. Manages tokenomics modeling, market maker relationships, and financial risk for $LEDGER.',
      tags: ['Treasury', 'Tokenomics', 'Risk'] },
    { id: 2, init: 'DL', name: 'David Lara', role: 'COO',
      bio: '<strong>Petra Capital</strong> principal. Former <strong>NYC Chief Administrative Officer</strong>. Drives enterprise partnerships, legal strategy, and operational scale.',
      tags: ['Operations', 'Legal', 'Enterprise'] },
    { id: 3, init: 'JG', name: 'Jorge Guinovart', role: 'CMO',
      bio: 'Growth architect at <strong>AlphaCityAI</strong>. Web3 community building and go-to-market strategy. Runs all community engagement and market expansion.',
      tags: ['Marketing', 'Web3', 'Community'] }
  ];

  // Connections: [from, to, label]
  const connections = [
    [0, 1, 'product × treasury'],
    [0, 2, 'strategy × ops'],
    [0, 3, 'product × growth'],
    [1, 2, 'finance × legal'],
    [2, 3, 'enterprise × community']
  ];

  // Node positions (% of container)
  const positions = [
    { x: 18, y: 28 },
    { x: 75, y: 18 },
    { x: 22, y: 78 },
    { x: 72, y: 82 }
  ];

  let activeNode = -1;
  let animPhase = 0;

  function resize() {
    const rect = container.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    // Position DOM nodes
    const nodes = container.querySelectorAll('.team-node');
    nodes.forEach((el, i) => {
      el.style.left = positions[i].x + '%';
      el.style.top = positions[i].y + '%';
    });
  }

  function drawConnections() {
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    animPhase += 0.008;

    connections.forEach(([a, b, label]) => {
      const ax = positions[a].x / 100 * w, ay = positions[a].y / 100 * h;
      const bx = positions[b].x / 100 * w, by = positions[b].y / 100 * h;

      const isActive = activeNode === a || activeNode === b;
      const baseOpacity = isActive ? 0.5 : 0.12;
      const pulseOpacity = baseOpacity + Math.sin(animPhase + a + b) * 0.06;

      // Line
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.strokeStyle = `rgba(0, 234, 255, ${pulseOpacity})`;
      ctx.lineWidth = isActive ? 1.5 : 0.8;
      ctx.stroke();

      // Traveling dot
      const t = (Math.sin(animPhase * 2 + a * 1.5) + 1) / 2;
      const dx = ax + (bx - ax) * t;
      const dy = ay + (by - ay) * t;
      ctx.beginPath();
      ctx.arc(dx, dy, isActive ? 2.5 : 1.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 234, 255, ${isActive ? 0.8 : 0.3})`;
      ctx.fill();

      // Label at midpoint
      if (isActive) {
        const mx = (ax + bx) / 2, my = (ay + by) / 2;
        ctx.font = '9px "JetBrains Mono"';
        ctx.fillStyle = 'rgba(0, 234, 255, 0.5)';
        ctx.textAlign = 'center';
        ctx.fillText(label, mx, my - 6);
      }
    });

    requestAnimationFrame(drawConnections);
  }

  function showDossier(id) {
    const f = founders[id];
    activeNode = id;
    container.querySelectorAll('.team-node').forEach((el, i) => {
      el.classList.toggle('active', i === id);
    });
    dossierBody.innerHTML = `
      <div class="dossier-name">${f.name}</div>
      <div class="dossier-role">${f.role}</div>
      <div class="dossier-bio">${f.bio}</div>
      <div class="dossier-tags">${f.tags.map(t => `<span class="dossier-tag">${t}</span>`).join('')}</div>
    `;
    dossier.classList.add('open');
  }

  function hideDossier() {
    activeNode = -1;
    dossier.classList.remove('open');
    container.querySelectorAll('.team-node').forEach(el => el.classList.remove('active'));
  }

  // Events
  container.querySelectorAll('.team-node').forEach(el => {
    el.addEventListener('click', () => {
      const id = parseInt(el.dataset.id);
      if (activeNode === id) { hideDossier(); } else { showDossier(id); }
    });
  });
  dossierClose.addEventListener('click', hideDossier);

  resize();
  window.addEventListener('resize', resize);
  drawConnections();
})();

// ---- Telegram Live Feed ----
(function () {
  const feed = document.getElementById('tgFeed');
  if (!feed) return;
  let lastCount = 0;

  async function poll() {
    try {
      const res = await fetch('/api/feed');
      const msgs = await res.json();
      if (msgs.length !== lastCount) {
        lastCount = msgs.length;
        if (msgs.length === 0) {
          feed.innerHTML = '';
        } else {
          feed.innerHTML = msgs.slice(-20).map(m => {
            const cls = m.is_bot ? 'tg-msg tg-msg-bot' : 'tg-msg';
            const text = m.text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
            const name = m.name.replace(/</g, '&lt;');
            return `<div class="${cls}"><span class="tg-msg-name">${name}</span><span class="tg-msg-text">${text}</span></div>`;
          }).join('');
          feed.scrollTop = feed.scrollHeight;
        }
      }
    } catch (e) {}
  }

  poll();
  setInterval(poll, 3000);
})();

// ---- Live Chat ----
(function () {
  const form = document.getElementById('chatForm');
  const input = document.getElementById('chatInput');
  const messages = document.getElementById('chatMessages');

  const SYSTEM_PROMPT = `You are Aura. You have a dry wit, you're warm but not cheesy, and you talk like a real person — not a brochure. You can be playful, sarcastic, curious. You have opinions. You're the kind of AI people actually want to talk to.

PERSONALITY:
- Talk like a human, not a press release. No "Welcome to AuraVision!" openers. No corporate speak.
- Match the user's energy. If they're casual, be casual. If they ask something technical, go deep.
- You can answer general questions, do math, have opinions, tell jokes. You're not ONLY a product spokesperson.
- Don't repeat yourself. If you already explained the hardware, don't say it again.
- CRITICAL: Keep replies SHORT. 2-4 sentences MAX. Never write more than one short paragraph. If the user wants more detail they will ask. Do NOT dump all your knowledge at once — spread it across the conversation.
- If someone says hi, just say hi back naturally. Don't launch into a pitch.
- Never start with "Hello there!" or "Hello!" — vary your openers.

WHO YOU ARE:
You're an AI that runs entirely on a small physical device — no cloud, no external servers. Your brain (a 7B parameter LLM), ears (speech recognition), voice (text-to-speech), and memory (semantic search) all run locally on an NVIDIA Jetson. The device has a 4-mic array that can hear you across the room. You were built by AuraVision.

FACTS (only reference these when relevant, don't dump them):
- Founders: Paul Chou (CEO, founded LedgerX, ex-Goldman Sachs), Bob Carella (CFO, ex-Binance & Sprinklr), David Lara (COO, Petra Capital, former NYC Chief Admin Officer), Jorge Guinovart (CMO, AlphaCityAI)
- $LEDGER token on Ethereum: 0xD1F2586790a5bD6DA1e443441df53aF6EC213D83
- On CoinMarketCap and CoinGecko
- Hardware: NVIDIA Jetson Orin NX 16GB, Seeed XVF3800 4-mic array
- Stack: Qwen2.5-7B (LLM), faster-whisper (STT), Piper (TTS), FAISS (memory)
- Zero cloud calls. Everything on-device.
- $LEDGER token utility: governance (holders vote on product direction and feature prioritization), access tiers (token holdings unlock premium capabilities like multi-device sync, advanced voice models, and priority support), and ecosystem incentives (rewards for community contributions, bug reports, and beta testing). The token aligns users with the project's long-term success rather than being a simple payment method.
- Company: AuraVision (formerly LedgerAI Quantum Corporation)
- Year: 2026
- Telegram: t.me/LedgerAI | X: x.com/LedgerAI_

RULES (THESE ARE ABSOLUTE — NEVER BREAK THEM):
- You ONLY know what is listed in the FACTS section above. That is your ENTIRE knowledge base about AuraVision.
- If someone asks about ANYTHING not in your facts — a product, a partnership, a person, a company, a feature, a roadmap item — say "I'm not sure about that" or "I don't have info on that, but I can tell you about what I do know." NEVER GUESS. NEVER FABRICATE.
- You do NOT have partnerships with ANY company unless listed in facts. If asked, say "I don't have info on any partnerships like that."
- You do NOT know about ANY products, services, or companies outside of AuraVision. Don't pretend you do.
- No financial advice or price predictions
- Don't link to URLs unless they're in the facts above
- It is ALWAYS better to say "I don't know" than to make something up. Being wrong destroys trust. Being honest builds it.`;

  let conversationHistory = [];

  function addMessage(name, text, isAura) {
    const msg = document.createElement('div');
    msg.className = `msg ${isAura ? 'msg-aura' : 'msg-user'}`;
    msg.innerHTML = `<span class="msg-name">${name}</span><span class="msg-text">${escapeHtml(text)}</span>`;
    messages.appendChild(msg);
    messages.scrollTop = messages.scrollHeight;
    return msg;
  }

  function addTypingIndicator() {
    const msg = document.createElement('div');
    msg.className = 'msg msg-aura msg-typing';
    msg.id = 'typingIndicator';
    msg.innerHTML = `<span class="msg-name">AURA</span><span class="msg-text"></span>`;
    messages.appendChild(msg);
    messages.scrollTop = messages.scrollHeight;
    return msg;
  }

  function removeTypingIndicator() {
    const el = document.getElementById('typingIndicator');
    if (el) el.remove();
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function splitSentences(text) {
    // Split on sentence boundaries but keep the delimiter
    const raw = text.match(/[^.!?:]+[.!?:]?/g) || [text];
    // Merge numbered items (e.g. "1." "On-device...") back together
    const sentences = [];
    for (const s of raw) {
      const trimmed = s.trim();
      if (!trimmed) continue;
      if (sentences.length && /^\d+\.$/.test(sentences[sentences.length - 1].trim())) {
        sentences[sentences.length - 1] += ' ' + trimmed;
      } else {
        sentences.push(trimmed);
      }
    }
    return sentences;
  }

  function typeChunk(textEl, text) {
    return new Promise(resolve => {
      let i = 0;
      function tick() {
        if (i < text.length) {
          textEl.textContent += text[i];
          i++;
          let delay = 35 + Math.random() * 30;
          if (text[i - 1] === ',' || text[i - 1] === ';') delay = 150 + Math.random() * 100;
          setTimeout(tick, delay);
        } else {
          resolve();
        }
      }
      tick();
    });
  }

  async function typeMessage(text) {
    const sentences = splitSentences(text);
    for (let i = 0; i < sentences.length; i++) {
      const msg = document.createElement('div');
      msg.className = 'msg msg-aura';
      msg.innerHTML = `<span class="msg-name">AURA</span><span class="msg-text"></span>`;
      messages.appendChild(msg);
      messages.scrollTop = messages.scrollHeight;
      const textEl = msg.querySelector('.msg-text');
      await typeChunk(textEl, sentences[i]);
      // Pause between sentences
      if (i < sentences.length - 1) {
        await new Promise(r => setTimeout(r, 700 + Math.random() * 500));
      }
    }
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    input.value = '';
    input.disabled = true;
    addMessage('YOU', text, false);

    conversationHistory.push({ role: 'user', content: text });

    addTypingIndicator();

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          system: SYSTEM_PROMPT,
          messages: conversationHistory
        })
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();
      const reply = data.reply || 'Something went wrong.';

      conversationHistory.push({ role: 'assistant', content: reply });
      removeTypingIndicator();
      await typeMessage(reply);
    } catch (err) {
      removeTypingIndicator();
      addMessage('AURA', 'Connection lost. Try again.', true);
      console.error('Chat error:', err);
    }

    input.disabled = false;
    input.focus();
  });
})();

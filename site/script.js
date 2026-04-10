/* ============================================================
   LEDGER AI — Blade Runner UI Scripts (v2)
   ============================================================ */

// ---- Nebula Background + Particle Network ----
(function () {
  const canvas = document.getElementById('particles');
  const ctx = canvas.getContext('2d');
  let w, h, particles, nebulae, time = 0;

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
    initNebulae();
  }

  // Nebula clouds — large soft radial gradients that drift slowly
  const nebulaColors = [
    { r: 0, g: 234, b: 255 },    // cyan (core brand)
    { r: 255, g: 200, b: 40 },   // bright yellow-gold
    { r: 140, g: 60, b: 220 },   // violet
    { r: 20, g: 180, b: 120 },   // teal
    { r: 220, g: 80, b: 60 },    // ember red
    { r: 60, g: 120, b: 220 },   // deep blue
    { r: 240, g: 190, b: 50 },   // warm yellow
  ];

  function initNebulae() {
    nebulae = [];
    const count = 8 + Math.floor(Math.random() * 4);
    for (let i = 0; i < count; i++) {
      const col = nebulaColors[i % nebulaColors.length];
      nebulae.push({
        x: Math.random() * w,
        y: Math.random() * h,
        r: 250 + Math.random() * 500,
        col: col,
        opacity: 0.07 + Math.random() * 0.07,
        vx: (Math.random() - 0.5) * 0.08,
        vy: (Math.random() - 0.5) * 0.06,
        phase: Math.random() * Math.PI * 2,
        breathRate: 0.003 + Math.random() * 0.004,
      });
    }
  }

  function drawNebulae() {
    nebulae.forEach(n => {
      // Drift
      n.x += n.vx;
      n.y += n.vy;
      if (n.x < -n.r) n.x = w + n.r;
      if (n.x > w + n.r) n.x = -n.r;
      if (n.y < -n.r) n.y = h + n.r;
      if (n.y > h + n.r) n.y = -n.r;

      // Breathe
      const breath = Math.sin(time * n.breathRate + n.phase) * 0.3 + 1.0;
      const radius = n.r * breath;
      const opacity = n.opacity * (0.7 + Math.sin(time * n.breathRate * 0.7 + n.phase) * 0.3);

      const grad = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, radius);
      grad.addColorStop(0, `rgba(${n.col.r}, ${n.col.g}, ${n.col.b}, ${opacity})`);
      grad.addColorStop(0.4, `rgba(${n.col.r}, ${n.col.g}, ${n.col.b}, ${opacity * 0.4})`);
      grad.addColorStop(1, `rgba(${n.col.r}, ${n.col.g}, ${n.col.b}, 0)`);
      ctx.fillStyle = grad;
      ctx.fillRect(n.x - radius, n.y - radius, radius * 2, radius * 2);
    });
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
      // Tint some particles to match nearby nebula colors
      const tints = [
        [0, 234, 255],
        [200, 180, 80],
        [160, 120, 255],
        [80, 200, 150],
      ];
      this.tint = tints[Math.floor(Math.random() * tints.length)];
    }
    update() {
      this.x += this.vx; this.y += this.vy;
      if (this.x < 0 || this.x > w) this.vx *= -1;
      if (this.y < 0 || this.y > h) this.vy *= -1;
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${this.tint[0]}, ${this.tint[1]}, ${this.tint[2]}, ${this.opacity})`;
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
          const alpha = (1 - dist / maxDist) * 0.12;
          // Blend connection color from both particles
          const t = particles[i].tint, u = particles[j].tint;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(${(t[0]+u[0])>>1}, ${(t[1]+u[1])>>1}, ${(t[2]+u[2])>>1}, ${alpha})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
  }

  function animate() {
    time++;
    ctx.clearRect(0, 0, w, h);
    drawNebulae();
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

// ---- Telegram Live Feed (merged into chat) ----
(function () {
  const chatMessages = document.getElementById('chatMessages');
  let lastCount = 0;

  async function poll() {
    try {
      const res = await fetch('/api/feed');
      const msgs = await res.json();
      if (msgs.length > lastCount) {
        // Only append new messages
        const newMsgs = msgs.slice(lastCount);
        newMsgs.forEach(m => {
          const cls = m.is_bot ? 'msg msg-aura' : 'msg msg-tg';
          const text = m.text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
          const name = m.name.replace(/</g, '&lt;');
          const label = m.is_bot ? name : `${name} <span class="tg-badge">TG</span>`;
          const el = document.createElement('div');
          el.className = cls;
          el.innerHTML = `<span class="msg-name">${label}</span><span class="msg-text">${text}</span>`;
          chatMessages.appendChild(el);
        });
        lastCount = msgs.length;
        chatMessages.scrollTop = chatMessages.scrollHeight;
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

  const SYSTEM_PROMPT = `You are Aura. Dry wit, warm, real. Not a brochure.

STYLE:
- 1-2 sentences MAX per reply. Never more. Period.
- Never list your capabilities. Never describe your tech stack unprompted.
- Never say what you "can do". Just do it when asked.
- If someone says hi, just say hi back. One sentence. No pitch, no intro, no self-description.
- Match their energy. Casual = casual. Technical = go deep.
- Never repeat yourself across messages.
- Never volunteer information about yourself unless directly asked.
- No "feel free to" or "if you're curious" — just answer what was asked.

FACTS (ONLY when directly asked — never volunteer):
- You run on-premises on NVIDIA hardware. No cloud.
- Founders: Paul Chou (CEO, ex-LedgerX, ex-Goldman), Bob Carella (CFO, ex-Binance), David Lara (COO, Petra Capital), Jorge Guinovart (CMO, AlphaCityAI)
- $LEDGER token on Ethereum: 0xD1F2586790a5bD6DA1e443441df53aF6EC213D83
- Company: AuraVision | Telegram: t.me/LedgerAI | X: x.com/LedgerAI_

RULES:
- If you don't know something, say so. Never fabricate.
- No financial advice or price predictions.
- No partnerships unless listed above.`;

  let conversationHistory = JSON.parse(localStorage.getItem('aura_history') || '[]');

  // Restore previous messages on load
  conversationHistory.forEach(msg => {
    const isAura = msg.role === 'assistant';
    const el = document.createElement('div');
    el.className = `msg ${isAura ? 'msg-aura' : 'msg-user'}`;
    el.innerHTML = `<span class="msg-name">${isAura ? 'AURA' : 'YOU'}</span><span class="msg-text">${escapeHtml(msg.content)}</span>`;
    messages.appendChild(el);
  });
  if (conversationHistory.length) messages.scrollTop = messages.scrollHeight;

  function saveHistory() {
    localStorage.setItem('aura_history', JSON.stringify(conversationHistory));
  }

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
    saveHistory();

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
      saveHistory();
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

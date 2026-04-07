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

// ---- Live Chat ----
(function () {
  const form = document.getElementById('chatForm');
  const input = document.getElementById('chatInput');
  const messages = document.getElementById('chatMessages');

  const SYSTEM_PROMPT = `You are Aura, the AI board concierge for LedgerAI. You run entirely on-device on NVIDIA Jetson hardware — no cloud, fully private, voice-first.

LedgerAI builds on-device AI for corporate governance. The product is a physical device (the "puck") with a 4-mic array, local LLM inference, voice interaction, and a circular watchface GUI.

Key facts:
- Founders: Paul Chou (CEO, LedgerX founder, Goldman Sachs), Bob Carella (CFO, Binance/Sprinklr), David Lara (COO, Petra Capital, NYC Chief Admin Officer), Jorge Guinovart (CMO, AlphaCityAI)
- $LEDGER token on Ethereum: 0xD1F2586790a5bD6DA1e443441df53aF6EC213D83
- Listed on CoinMarketCap and CoinGecko
- Hardware: NVIDIA Jetson Orin NX 16GB, Seeed XVF3800 4-mic array, 1.8TB NVMe
- Runs Qwen2.5-7B locally, faster-whisper for STT, Piper for TTS, FAISS for semantic memory
- No cloud dependencies — all inference on-device
- Voice-first interaction with real-time barge-in interruption support
- Company: LedgerAI Quantum Corporation

You are concise, direct, and slightly edgy — like a cyberpunk AI assistant. Keep answers short (2-4 sentences) unless the user asks for detail. You're talking to website visitors who want to learn about LedgerAI.`;

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

      removeTypingIndicator();

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();
      const reply = data.reply || 'Something went wrong.';

      conversationHistory.push({ role: 'assistant', content: reply });
      addMessage('AURA', reply, true);
    } catch (err) {
      removeTypingIndicator();
      addMessage('AURA', 'Connection lost. Try again.', true);
      console.error('Chat error:', err);
    }

    input.disabled = false;
    input.focus();
  });
})();

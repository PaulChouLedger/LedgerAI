/* ============================================================
   LEDGER AI — Blade Runner UI Scripts (v2)
   ============================================================ */

// ---- Deep Space: Stars, Nebulae, Shooting Stars, Supernovae ----
(function () {
  const canvas = document.getElementById('particles');
  const ctx = canvas.getContext('2d');
  let w, h, time = 0;

  // --- Stars (static background field, ~400 stars) ---
  let stars = [];
  function initStars() {
    stars = [];
    const count = Math.min(Math.floor((w * h) / 4000), 500);
    for (let i = 0; i < count; i++) {
      const tints = [[210,230,255],[255,230,180],[180,200,255],[255,200,160],[160,220,255],[255,180,220],[200,255,220]];
      stars.push({
        x: Math.random() * w, y: Math.random() * h,
        r: Math.random() * 1.4 + 0.3,
        baseOpacity: Math.random() * 0.6 + 0.15,
        twinkleRate: 0.01 + Math.random() * 0.03,
        twinklePhase: Math.random() * Math.PI * 2,
        tint: tints[Math.floor(Math.random() * tints.length)],
      });
    }
  }

  function drawStars() {
    stars.forEach(s => {
      const twinkle = 0.5 + 0.5 * Math.sin(time * s.twinkleRate + s.twinklePhase);
      const op = s.baseOpacity * (0.4 + twinkle * 0.6);
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${s.tint[0]},${s.tint[1]},${s.tint[2]},${op})`;
      ctx.fill();
      // Bright stars get a subtle cross-flare
      if (s.r > 1.2 && twinkle > 0.7) {
        const fl = op * 0.3;
        const len = s.r * 4;
        ctx.strokeStyle = `rgba(${s.tint[0]},${s.tint[1]},${s.tint[2]},${fl})`;
        ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(s.x - len, s.y); ctx.lineTo(s.x + len, s.y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(s.x, s.y - len); ctx.lineTo(s.x, s.y + len); ctx.stroke();
      }
    });
  }

  // --- Nebulae (drifting gas clouds) ---
  const nebulaColors = [
    { r: 0, g: 240, b: 255 },    // cyan
    { r: 255, g: 195, b: 30 },   // gold
    { r: 155, g: 50, b: 240 },   // violet
    { r: 20, g: 200, b: 130 },   // teal
    { r: 235, g: 75, b: 55 },    // ember
    { r: 50, g: 110, b: 240 },   // deep blue
    { r: 245, g: 185, b: 40 },   // warm yellow
    { r: 200, g: 35, b: 175 },   // magenta
    { r: 255, g: 120, b: 180 },  // rose
    { r: 100, g: 220, b: 255 },  // ice blue
  ];
  let nebulae = [];

  function initNebulae() {
    nebulae = [];
    const count = 10 + Math.floor(Math.random() * 4);
    for (let i = 0; i < count; i++) {
      const col = nebulaColors[i % nebulaColors.length];
      nebulae.push({
        x: Math.random() * w, y: Math.random() * h,
        r: 200 + Math.random() * 500,
        col, opacity: 0.08 + Math.random() * 0.08,
        vx: (Math.random() - 0.5) * 0.07, vy: (Math.random() - 0.5) * 0.05,
        phase: Math.random() * Math.PI * 2, breathRate: 0.002 + Math.random() * 0.003,
      });
    }
  }

  function drawNebulae() {
    nebulae.forEach(n => {
      n.x += n.vx; n.y += n.vy;
      if (n.x < -n.r) n.x = w + n.r; if (n.x > w + n.r) n.x = -n.r;
      if (n.y < -n.r) n.y = h + n.r; if (n.y > h + n.r) n.y = -n.r;
      const breath = Math.sin(time * n.breathRate + n.phase) * 0.3 + 1.0;
      const radius = n.r * breath;
      const op = n.opacity * (0.7 + Math.sin(time * n.breathRate * 0.7 + n.phase) * 0.3);
      const grad = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, radius);
      grad.addColorStop(0, `rgba(${n.col.r},${n.col.g},${n.col.b},${op})`);
      grad.addColorStop(0.3, `rgba(${n.col.r},${n.col.g},${n.col.b},${op * 0.5})`);
      grad.addColorStop(0.7, `rgba(${n.col.r},${n.col.g},${n.col.b},${op * 0.15})`);
      grad.addColorStop(1, `rgba(${n.col.r},${n.col.g},${n.col.b},0)`);
      ctx.fillStyle = grad;
      ctx.fillRect(n.x - radius, n.y - radius, radius * 2, radius * 2);
    });
  }

  // --- Cosmic dust lanes (faint wispy streaks) ---
  let dustLanes = [];
  function initDust() {
    dustLanes = [];
    for (let i = 0; i < 3; i++) {
      dustLanes.push({
        y: h * 0.2 + Math.random() * h * 0.6,
        amplitude: 30 + Math.random() * 60,
        freq: 0.002 + Math.random() * 0.003,
        opacity: 0.02 + Math.random() * 0.025,
        thickness: 40 + Math.random() * 80,
        speed: 0.0003 + Math.random() * 0.0005,
        color: nebulaColors[Math.floor(Math.random() * nebulaColors.length)],
      });
    }
  }

  function drawDust() {
    dustLanes.forEach(d => {
      ctx.beginPath();
      for (let x = 0; x < w; x += 4) {
        const y = d.y + Math.sin(x * d.freq + time * d.speed) * d.amplitude
                      + Math.sin(x * d.freq * 2.3 + time * d.speed * 1.7) * d.amplitude * 0.3;
        if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = `rgba(${d.color.r},${d.color.g},${d.color.b},${d.opacity})`;
      ctx.lineWidth = d.thickness;
      ctx.lineCap = 'round';
      ctx.stroke();
    });
  }

  // --- Shooting stars (occasional streaks) ---
  let shootingStars = [];
  function spawnShootingStar() {
    const colors = [[0,240,255],[255,210,80],[190,160,255],[255,140,70],[255,120,180],[100,255,200]];
    const col = colors[Math.floor(Math.random() * colors.length)];
    shootingStars.push({
      x: Math.random() * w * 0.8, y: Math.random() * h * 0.4,
      vx: 4 + Math.random() * 6, vy: 2 + Math.random() * 4,
      life: 1.0, decay: 0.008 + Math.random() * 0.012,
      len: 60 + Math.random() * 100, col,
    });
  }

  function drawShootingStars() {
    shootingStars = shootingStars.filter(s => s.life > 0);
    shootingStars.forEach(s => {
      s.x += s.vx; s.y += s.vy; s.life -= s.decay;
      const tailX = s.x - s.vx * s.len * 0.15;
      const tailY = s.y - s.vy * s.len * 0.15;
      const grad = ctx.createLinearGradient(tailX, tailY, s.x, s.y);
      grad.addColorStop(0, `rgba(${s.col[0]},${s.col[1]},${s.col[2]},0)`);
      grad.addColorStop(1, `rgba(${s.col[0]},${s.col[1]},${s.col[2]},${s.life * 0.8})`);
      ctx.beginPath(); ctx.moveTo(tailX, tailY); ctx.lineTo(s.x, s.y);
      ctx.strokeStyle = grad; ctx.lineWidth = 1.5 * s.life; ctx.stroke();
      // Head glow
      ctx.beginPath(); ctx.arc(s.x, s.y, 2 * s.life, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255,255,255,${s.life * 0.6})`; ctx.fill();
    });
  }

  // --- Supernovae (rare, dramatic flash + expanding ring) ---
  let supernovae = [];
  function spawnSupernova() {
    const colors = [[255,240,200],[0,234,255],[255,180,60],[200,140,255]];
    const col = colors[Math.floor(Math.random() * colors.length)];
    supernovae.push({
      x: w * 0.1 + Math.random() * w * 0.8,
      y: h * 0.1 + Math.random() * h * 0.8,
      age: 0, maxAge: 300 + Math.random() * 200,
      col, maxR: 150 + Math.random() * 200,
    });
  }

  function drawSupernovae() {
    supernovae = supernovae.filter(s => s.age < s.maxAge);
    supernovae.forEach(s => {
      s.age++;
      const t = s.age / s.maxAge;
      // Phase 1: bright flash (0-0.1)
      if (t < 0.1) {
        const flashI = Math.sin(t / 0.1 * Math.PI);
        const r = 5 + flashI * 40;
        const grad = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, r);
        grad.addColorStop(0, `rgba(255,255,255,${flashI * 0.9})`);
        grad.addColorStop(0.3, `rgba(${s.col[0]},${s.col[1]},${s.col[2]},${flashI * 0.7})`);
        grad.addColorStop(1, `rgba(${s.col[0]},${s.col[1]},${s.col[2]},0)`);
        ctx.fillStyle = grad;
        ctx.fillRect(s.x - r, s.y - r, r * 2, r * 2);
        // Lens flare cross
        const fl = flashI * 0.6;
        ctx.strokeStyle = `rgba(255,255,255,${fl})`;
        ctx.lineWidth = 1.5;
        const fLen = r * 3;
        ctx.beginPath(); ctx.moveTo(s.x - fLen, s.y); ctx.lineTo(s.x + fLen, s.y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(s.x, s.y - fLen); ctx.lineTo(s.x, s.y + fLen); ctx.stroke();
      }
      // Phase 2: expanding ring + fading glow (0.1-1.0)
      if (t >= 0.05) {
        const ringT = (t - 0.05) / 0.95;
        const ringR = ringT * s.maxR;
        const ringOp = (1 - ringT) * 0.4;
        ctx.beginPath(); ctx.arc(s.x, s.y, ringR, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(${s.col[0]},${s.col[1]},${s.col[2]},${ringOp})`;
        ctx.lineWidth = 2 * (1 - ringT) + 0.5;
        ctx.stroke();
        // Inner glow fade
        if (ringT < 0.5) {
          const glowOp = (1 - ringT * 2) * 0.15;
          const glowR = 20 + ringT * 60;
          const grad = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, glowR);
          grad.addColorStop(0, `rgba(${s.col[0]},${s.col[1]},${s.col[2]},${glowOp})`);
          grad.addColorStop(1, `rgba(${s.col[0]},${s.col[1]},${s.col[2]},0)`);
          ctx.fillStyle = grad;
          ctx.fillRect(s.x - glowR, s.y - glowR, glowR * 2, glowR * 2);
        }
        // Scatter particles from explosion
        if (ringT < 0.3) {
          const numDebris = 5;
          for (let i = 0; i < numDebris; i++) {
            const angle = (i / numDebris) * Math.PI * 2 + time * 0.02;
            const dist = ringR * (0.7 + Math.sin(angle * 3 + time * 0.05) * 0.3);
            const dx = s.x + Math.cos(angle) * dist;
            const dy = s.y + Math.sin(angle) * dist;
            const debrisOp = (1 - ringT / 0.3) * 0.5;
            ctx.beginPath(); ctx.arc(dx, dy, 1.2, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255,255,255,${debrisOp})`; ctx.fill();
          }
        }
      }
    });
  }

  // --- Particle network (connections between nearby stars) ---
  let particles = [];
  function initParticles() {
    const count = Math.min(Math.floor((w * h) / 15000), 100);
    const tints = [[0,234,255],[200,180,80],[160,120,255],[80,200,150],[255,180,100]];
    particles = Array.from({ length: count }, () => ({
      x: Math.random() * w, y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.25, vy: (Math.random() - 0.5) * 0.25,
      r: Math.random() * 1.5 + 0.5, opacity: Math.random() * 0.4 + 0.15,
      tint: tints[Math.floor(Math.random() * tints.length)],
    }));
  }

  function drawParticles() {
    particles.forEach(p => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h) p.vy *= -1;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.tint[0]},${p.tint[1]},${p.tint[2]},${p.opacity})`;
      ctx.fill();
    });
    // Connections
    const maxDist = 130;
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x, dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < maxDist) {
          const a = (1 - dist / maxDist) * 0.1;
          const t = particles[i].tint, u = particles[j].tint;
          ctx.beginPath(); ctx.moveTo(particles[i].x, particles[i].y); ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(${(t[0]+u[0])>>1},${(t[1]+u[1])>>1},${(t[2]+u[2])>>1},${a})`;
          ctx.lineWidth = 0.4; ctx.stroke();
        }
      }
    }
  }

  // --- Init & Loop ---
  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
    initStars(); initNebulae(); initDust(); initParticles();
  }

  function animate() {
    time++;
    ctx.clearRect(0, 0, w, h);

    // Layer order: dust → nebulae → stars → particles → shooting stars → supernovae
    drawDust();
    drawNebulae();
    drawStars();
    drawParticles();
    drawShootingStars();
    drawSupernovae();

    // Spawn shooting stars (avg every ~3s at 60fps)
    if (Math.random() < 0.006) spawnShootingStar();
    // Spawn supernovae (avg every ~45s)
    if (Math.random() < 0.00037) spawnSupernova();

    requestAnimationFrame(animate);
  }

  window.addEventListener('resize', resize);
  resize();
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

// ---- Seamless Logo Marquee + Pulse ----
(function () {
  const track = document.querySelector('.marquee-track');
  if (!track) return;
  const speed = 0.41; // px per frame (was 0.5, slowed 18%)
  let offset = 0;
  let setWidth = 0; // width of one full set of logos (7 images + gaps)

  // Clone logos until we have enough to fill 3x viewport for seamless wrap
  function ensureClones() {
    const origImgs = track.querySelectorAll('img');
    // We have 14 in HTML (7+7). Add more clones to guarantee no gap.
    const first7 = [];
    for (let i = 0; i < 7 && i < origImgs.length; i++) first7.push(origImgs[i]);
    // Add another full set (third copy) for safety
    first7.forEach(img => {
      const clone = img.cloneNode(true);
      track.appendChild(clone);
    });
  }

  function measureSet() {
    const imgs = track.querySelectorAll('img');
    if (imgs.length < 7) return;
    // Measure exact width of first 7 logos including gaps via bounding rects
    const firstRect = imgs[0].getBoundingClientRect();
    const eighthRect = imgs[7].getBoundingClientRect();
    setWidth = eighthRect.left - firstRect.left;
  }

  // Subtle random pulse on individual logos
  function pulse() {
    const imgs = track.querySelectorAll('img');
    imgs.forEach(img => {
      if (Math.random() < 0.008) {
        const bright = 0.55 + Math.random() * 0.35;
        img.style.opacity = bright;
        img.style.filter = `brightness(0) invert(1) drop-shadow(0 0 8px rgba(0,234,255,${bright * 0.4}))`;
        setTimeout(() => {
          img.style.opacity = '';
          img.style.filter = '';
        }, 1500 + Math.random() * 1500);
      }
    });
  }

  function animate() {
    offset -= speed;
    // When we've scrolled one full set, jump back seamlessly
    if (setWidth > 0 && Math.abs(offset) >= setWidth) {
      offset += setWidth;
    }
    track.style.transform = `translateX(${offset}px)`;
    pulse();
    requestAnimationFrame(animate);
  }

  ensureClones();

  // Wait for all images to load before measuring
  const allImgs = track.querySelectorAll('img');
  let loaded = 0;
  const total = allImgs.length;
  function onReady() { measureSet(); }
  allImgs.forEach(img => {
    if (img.complete) { loaded++; } else { img.onload = () => { loaded++; if (loaded >= total) onReady(); }; }
  });
  if (loaded >= total) onReady();
  setTimeout(onReady, 1000); // fallback

  animate();
})();

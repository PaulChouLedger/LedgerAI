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

  function pickRandomColor() { return { ...nebulaColors[Math.floor(Math.random() * nebulaColors.length)] }; }

  function initNebulae() {
    nebulae = [];
    const count = 10 + Math.floor(Math.random() * 4);
    for (let i = 0; i < count; i++) {
      const col = { ...nebulaColors[i % nebulaColors.length] };
      const target = pickRandomColor();
      nebulae.push({
        x: Math.random() * w, y: Math.random() * h,
        r: 200 + Math.random() * 500,
        col, target, lerpSpeed: 0.0003 + Math.random() * 0.0004,
        opacity: 0.08 + Math.random() * 0.08,
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
      // Gently lerp color toward target
      n.col.r += (n.target.r - n.col.r) * n.lerpSpeed;
      n.col.g += (n.target.g - n.col.g) * n.lerpSpeed;
      n.col.b += (n.target.b - n.col.b) * n.lerpSpeed;
      // Pick a new target when close enough
      if (Math.abs(n.col.r - n.target.r) + Math.abs(n.col.g - n.target.g) + Math.abs(n.col.b - n.target.b) < 3) {
        n.target = pickRandomColor();
      }
      const cr = Math.round(n.col.r), cg = Math.round(n.col.g), cb = Math.round(n.col.b);
      const breath = Math.sin(time * n.breathRate + n.phase) * 0.3 + 1.0;
      const radius = n.r * breath;
      const op = n.opacity * (0.7 + Math.sin(time * n.breathRate * 0.7 + n.phase) * 0.3);
      const grad = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, radius);
      grad.addColorStop(0, `rgba(${cr},${cg},${cb},${op})`);
      grad.addColorStop(0.3, `rgba(${cr},${cg},${cb},${op * 0.5})`);
      grad.addColorStop(0.7, `rgba(${cr},${cg},${cb},${op * 0.15})`);
      grad.addColorStop(1, `rgba(${cr},${cg},${cb},0)`);
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

  // Per-founder accent colors (matches CSS --nc vars)
  const nodeColors = [
    [0, 234, 255],    // PC — cyan
    [192, 144, 255],  // BC — violet
    [80, 208, 160],   // DL — teal
    [255, 184, 96],   // JG — amber
    [255, 130, 160],  // LV — rose
  ];

  const founders = [
    { id: 0, init: 'PC', name: 'Paul Chou', role: 'CEO & Founder', col: nodeColors[0],
      bio: 'Founded <strong>LedgerX</strong>, the first CFTC-regulated Bitcoin options exchange. Former <strong>Goldman Sachs</strong> quantitative trading. Built AuraVision from zero to on-device inference.',
      tags: ['Product', 'Engineering', 'Strategy'] },
    { id: 1, init: 'BC', name: 'Bob Carella', role: 'CFO', col: nodeColors[1],
      bio: 'Treasury and financial operations across <strong>Binance</strong> and <strong>Sprinklr</strong>. Manages tokenomics modeling, market maker relationships, and financial risk for $LEDGER.',
      tags: ['Treasury', 'Tokenomics', 'Risk'] },
    { id: 2, init: 'DL', name: 'David Lara', role: 'COO', col: nodeColors[2],
      bio: '<strong>Petra Capital</strong> principal. Former <strong>NYC Chief Administrative Officer</strong>. Drives enterprise partnerships, legal strategy, and operational scale.',
      tags: ['Operations', 'Legal', 'Enterprise'] },
    { id: 3, init: 'JG', name: 'Jorge Guinovart', role: 'CMO', col: nodeColors[3],
      bio: 'Growth architect at <strong>AlphaCityAI</strong>. Web3 community building and go-to-market strategy. Runs all community engagement and market expansion.',
      tags: ['Marketing', 'Web3', 'Community'] },
    { id: 4, init: 'LV', name: 'Liam Vaughn', role: 'Managing Director, Investor Relations', col: nodeColors[4],
      bio: 'Leads investor relations and capital strategy for AuraVision. Manages institutional outreach, stakeholder communications, and strategic fundraising initiatives.',
      tags: ['Investor Relations', 'Capital Strategy', 'Stakeholders'] }
  ];

  // Connections: [from, to, label]
  const connections = [
    [0, 1, 'product × treasury'],
    [0, 2, 'strategy × ops'],
    [0, 3, 'product × growth'],
    [0, 4, 'strategy × capital'],
    [1, 2, 'finance × legal'],
    [1, 4, 'treasury × investors'],
    [2, 3, 'enterprise × community'],
    [3, 4, 'community × relations']
  ];

  // Node positions (% of container) — pentagon layout
  const positions = [
    { x: 48, y: 12 },
    { x: 82, y: 38 },
    { x: 15, y: 38 },
    { x: 28, y: 85 },
    { x: 70, y: 85 }
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
      const colA = nodeColors[a], colB = nodeColors[b];
      // Blend the two node colors for the midpoint
      const midCol = [Math.round((colA[0]+colB[0])/2), Math.round((colA[1]+colB[1])/2), Math.round((colA[2]+colB[2])/2)];

      const isActive = activeNode === a || activeNode === b;
      const baseOpacity = isActive ? 0.5 : 0.12;
      const pulseOpacity = baseOpacity + Math.sin(animPhase + a + b) * 0.06;

      // Gradient line from node A color to node B color
      const grad = ctx.createLinearGradient(ax, ay, bx, by);
      grad.addColorStop(0, `rgba(${colA[0]},${colA[1]},${colA[2]},${pulseOpacity})`);
      grad.addColorStop(1, `rgba(${colB[0]},${colB[1]},${colB[2]},${pulseOpacity})`);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.strokeStyle = grad;
      ctx.lineWidth = isActive ? 1.5 : 0.8;
      ctx.stroke();

      // Traveling dot — lerp color along path
      const t = (Math.sin(animPhase * 2 + a * 1.5) + 1) / 2;
      const dx = ax + (bx - ax) * t;
      const dy = ay + (by - ay) * t;
      const dotCol = [Math.round(colA[0]+(colB[0]-colA[0])*t), Math.round(colA[1]+(colB[1]-colA[1])*t), Math.round(colA[2]+(colB[2]-colA[2])*t)];
      ctx.beginPath();
      ctx.arc(dx, dy, isActive ? 2.5 : 1.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${dotCol[0]},${dotCol[1]},${dotCol[2]},${isActive ? 0.8 : 0.3})`;
      ctx.fill();

      // Label at midpoint
      if (isActive) {
        const mx = (ax + bx) / 2, my = (ay + by) / 2;
        ctx.font = '9px "JetBrains Mono"';
        ctx.fillStyle = `rgba(${midCol[0]},${midCol[1]},${midCol[2]},0.5)`;
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
    const c = f.col;
    const cStr = `rgb(${c[0]},${c[1]},${c[2]})`;
    dossierBody.innerHTML = `
      <div class="dossier-name">${f.name}</div>
      <div class="dossier-role" style="color:${cStr}">${f.role}</div>
      <div class="dossier-bio">${f.bio}</div>
      <div class="dossier-tags">${f.tags.map(t => `<span class="dossier-tag" style="color:${cStr};border-color:rgba(${c[0]},${c[1]},${c[2]},0.25)">${t}</span>`).join('')}</div>
    `;
    dossier.classList.add('open');
  }

  function hideDossier() {
    activeNode = -1;
    dossier.classList.remove('open');
    container.querySelectorAll('.team-node').forEach(el => el.classList.remove('active'));
  }

  // Events — hover to show, mouseout to hide
  container.querySelectorAll('.team-node').forEach(el => {
    el.addEventListener('mouseenter', () => {
      const id = parseInt(el.dataset.id);
      showDossier(id);
    });
    el.addEventListener('mouseleave', () => {
      // Small delay so dossier doesn't flicker when moving between node and dossier
      el._hideTimer = setTimeout(() => { if (!dossier.matches(':hover')) hideDossier(); }, 200);
    });
  });
  dossier.addEventListener('mouseleave', () => { hideDossier(); });
  dossier.addEventListener('mouseenter', () => {
    // Cancel any pending hide if user moves into dossier
    container.querySelectorAll('.team-node').forEach(el => { if (el._hideTimer) clearTimeout(el._hideTimer); });
  });
  dossierClose.addEventListener('click', hideDossier);

  resize();
  window.addEventListener('resize', resize);
  drawConnections();
})();

// ---- Unified Chat: TG Live Feed + Website Chat ----
(function () {
  const chatMessages = document.getElementById('chatMessages');
  const form = document.getElementById('chatForm');
  const inputEl = document.getElementById('chatInput');
  let seenSet = new Set();
  let feedPaused = false; // pause feed polling while typing response

  const SYSTEM_PROMPT = `You are Aura, the AI behind AuraVision. You have dry wit and warmth.

ABSOLUTE RULE: Reply in 1-2 sentences. NEVER more than 2 sentences. NEVER list things. NEVER bullet points. NEVER describe yourself unprompted.

"hi" → "Hey." That's it. No pitch. No intro. No capabilities. No links.

Only share facts if DIRECTLY asked:
- On-prem NVIDIA, no cloud
- Founders: Paul Chou (CEO), Bob Carella (CFO), David Lara (COO), Jorge Guinovart (CMO)
- $LEDGER token on Ethereum
- AuraVision | t.me/LedgerAI

Never fabricate. No financial advice.`;

  function escapeHtml(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }
  // Use name+text only (no timestamp) to prevent duplicates from client/server ts mismatch
  function msgKey(m) { return (m.name || '') + ':' + (m.text || '').slice(0, 50); }

  function renderMsg(m) {
    const key = msgKey(m);
    if (seenSet.has(key)) return false;
    seenSet.add(key);
    const isAura = m.is_bot;
    const isWeb = m.source === 'web';
    const cls = isAura ? 'msg msg-aura' : (isWeb ? 'msg msg-user' : 'msg msg-tg');
    const text = (m.text || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const name = (m.name || '').replace(/</g, '&lt;');
    const group = m.group || '';
    let label;
    if (isAura) label = name + (group ? ` <span class="group-label">${escapeHtml(group)}</span>` : '');
    else if (isWeb) label = `${name} <span class="web-badge">WEB</span>`;
    else label = `${name} <span class="tg-badge">${group || 'TG'}</span>`;
    const el = document.createElement('div');
    el.className = cls;
    el.innerHTML = `<span class="msg-name">${label}</span><span class="msg-text">${text}</span>`;
    chatMessages.appendChild(el);
    return true;
  }

  // Poll the unified feed
  async function poll() {
    if (feedPaused) return;
    try {
      const res = await fetch('/api/feed');
      const msgs = await res.json();
      let added = false;
      msgs.forEach(m => { if (renderMsg(m)) added = true; });
      if (added) chatMessages.scrollTop = chatMessages.scrollHeight;
      if (seenSet.size > 200) seenSet = new Set([...seenSet].slice(-100));
    } catch (e) {}
  }

  poll();
  setInterval(poll, 3000);

  // Website chat submission
  function addTyping() {
    const el = document.createElement('div');
    el.className = 'msg msg-aura msg-typing';
    el.id = 'typingIndicator';
    el.innerHTML = `<span class="msg-name">AURA</span><span class="msg-text"></span>`;
    chatMessages.appendChild(el);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
  function removeTyping() { const el = document.getElementById('typingIndicator'); if (el) el.remove(); }

  function typeChunk(textEl, text) {
    return new Promise(resolve => {
      let i = 0;
      function tick() {
        if (i < text.length) {
          textEl.textContent += text[i]; i++;
          let delay = 30 + Math.random() * 25;
          if (text[i - 1] === ',' || text[i - 1] === ';') delay = 120 + Math.random() * 80;
          setTimeout(tick, delay);
        } else resolve();
      }
      tick();
    });
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = '';
    inputEl.disabled = true;
    feedPaused = true;

    // Show user message immediately
    const userTs = Math.floor(Date.now() / 1000);
    renderMsg({ name: 'Visitor', text, is_bot: false, source: 'web', ts: userTs });
    chatMessages.scrollTop = chatMessages.scrollHeight;

    addTyping();

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ system: SYSTEM_PROMPT, messages: [{ role: 'user', content: text }] })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const reply = data.reply || 'Something went wrong.';

      removeTyping();
      // Type out Aura's response
      const replyTs = Math.floor(Date.now() / 1000);
      const replyKey = msgKey({ ts: replyTs, name: 'Aura', text: reply });
      seenSet.add(replyKey);
      const msg = document.createElement('div');
      msg.className = 'msg msg-aura';
      msg.innerHTML = `<span class="msg-name">Aura</span><span class="msg-text"></span>`;
      chatMessages.appendChild(msg);
      chatMessages.scrollTop = chatMessages.scrollHeight;
      await typeChunk(msg.querySelector('.msg-text'), reply);
    } catch (err) {
      removeTyping();
      const errEl = document.createElement('div');
      errEl.className = 'msg msg-aura';
      errEl.innerHTML = `<span class="msg-name">AURA</span><span class="msg-text">Connection lost. Try again.</span>`;
      chatMessages.appendChild(errEl);
    }

    feedPaused = false;
    inputEl.disabled = false;
    inputEl.focus();
  });
})();

// ---- Seamless Logo Marquee + Pulse ----
(function () {
  const track = document.querySelector('.marquee-track');
  if (!track) return;
  const speed = 0.21; // px per frame (slowed 15% from 0.25)
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

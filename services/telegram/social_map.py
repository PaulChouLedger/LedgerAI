#!/usr/bin/env python3
"""
social_map -- Real-time social graph visualization for Aura.

Lightweight Flask server that renders an interactive force-directed graph
showing Aura's connections: users, groups, relationship depth, reputation.

Run: python3 social_map.py
Open: http://localhost:8899
"""

import json
import time
from pathlib import Path
from flask import Flask, jsonify, Response

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "telegram"

app = Flask(__name__)


def _load(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _build_graph() -> dict:
    profiles = _load(DATA_DIR / "profiles.json")
    social = _load(DATA_DIR / "social_graph.json").get("users", {})
    reputation = _load(DATA_DIR / "reputation.json")
    temperatures = _load(DATA_DIR / "engagement_temp.json")

    nodes = []
    edges = []
    group_ids_seen = set()

    # Aura — center node
    nodes.append({
        "id": "aura",
        "label": "Aura",
        "type": "aura",
        "size": 40,
        "color": "#00d4ff",
    })

    # Users
    for uid, profile in profiles.items():
        name = profile.get("preferred_name") or profile.get("display_name") or f"User {uid}"
        username = profile.get("username", "")
        msg_count = profile.get("message_count", 0)
        sg = social.get(uid, {})
        depth = sg.get("relationship_depth", "stranger")
        groups = sg.get("groups_seen_in", [])
        is_connector = sg.get("is_connector", False)
        influence = sg.get("influence_score", 0.0)
        dm_count = sg.get("dm_count", 0)
        group_interactions = sg.get("group_interactions", 0)

        # Node size based on message count
        size = min(10 + msg_count * 0.8, 35)

        # Color by relationship depth
        depth_colors = {
            "stranger": "#666666",
            "acquaintance": "#88aacc",
            "familiar": "#44cc88",
            "advocate": "#ffaa00",
        }
        color = depth_colors.get(depth, "#666666")

        nodes.append({
            "id": f"user_{uid}",
            "label": name,
            "type": "user",
            "size": round(size, 1),
            "color": color,
            "username": username,
            "depth": depth,
            "messages": msg_count,
            "dms": dm_count,
            "group_interactions": group_interactions,
            "is_connector": is_connector,
            "influence": round(influence, 3),
        })

        # Edge to Aura — thickness based on interaction count
        total = dm_count + group_interactions
        if total > 0:
            width = min(1 + total * 0.3, 8)
            edge_color = depth_colors.get(depth, "#444444")
            edges.append({
                "from": "aura",
                "to": f"user_{uid}",
                "width": round(width, 1),
                "color": edge_color,
                "label": f"{total} msgs",
            })

        # Track which groups this user is in
        for gid in groups:
            group_ids_seen.add(gid)

    # Groups
    for gid_str, rep in reputation.items():
        gid = int(gid_str)
        group_ids_seen.add(gid)
        gname = rep.get("group_name", f"Group {gid}")
        warmth = rep.get("warmth_level", "new")
        temp_data = temperatures.get(gid_str, {})
        temperature = temp_data.get("temperature", 0.5)
        total_responses = rep.get("total_responses", 0)
        replies = rep.get("replies_to_aura", 0)
        rep_score = rep.get("reputation_score", 0.0)

        warmth_colors = {
            "new": "#555555",
            "warming": "#cc8844",
            "established": "#44aa44",
            "trusted": "#ffcc00",
        }

        nodes.append({
            "id": f"group_{gid}",
            "label": gname,
            "type": "group",
            "size": min(20 + total_responses * 2, 50),
            "color": warmth_colors.get(warmth, "#555555"),
            "shape": "diamond",
            "warmth": warmth,
            "temperature": round(temperature, 2),
            "reputation": round(rep_score, 3),
            "responses": total_responses,
            "replies": replies,
        })

        # Edge from Aura to group
        if total_responses > 0:
            edges.append({
                "from": "aura",
                "to": f"group_{gid}",
                "width": min(2 + total_responses * 0.5, 10),
                "color": warmth_colors.get(warmth, "#555555"),
                "dashes": True,
                "label": f"{total_responses} resp",
            })

    # User-to-group edges
    for uid, sg in social.items():
        groups = sg.get("groups_seen_in", [])
        for gid in groups:
            edges.append({
                "from": f"user_{uid}",
                "to": f"group_{gid}",
                "width": 1,
                "color": "#333333",
                "dashes": [2, 4],
            })

    return {"nodes": nodes, "edges": edges}


def _build_sitrep() -> dict:
    """Compute real-time performance metrics for the SITREP panel."""
    profiles = _load(DATA_DIR / "profiles.json")
    social = _load(DATA_DIR / "social_graph.json").get("users", {})
    reputation = _load(DATA_DIR / "reputation.json")
    temperatures = _load(DATA_DIR / "engagement_temp.json")
    growth = _load(DATA_DIR / "growth_log.json")
    dm_eligible = _load(DATA_DIR / "dm_eligible.json")
    socialite_state = _load(DATA_DIR / "socialite_state.json")

    now = time.time()

    # --- Force disposition ---
    total_users = len(profiles)
    total_messages = sum(p.get("message_count", 0) for p in profiles.values())

    # Relationship depth breakdown
    depth_counts = {"stranger": 0, "acquaintance": 0, "familiar": 0, "advocate": 0}
    for u in social.values():
        d = u.get("relationship_depth", "stranger")
        depth_counts[d] = depth_counts.get(d, 0) + 1

    connectors = sum(1 for u in social.values() if u.get("is_connector"))
    dm_eligible_count = len(dm_eligible)

    # --- Theater status (groups) ---
    active_groups = []
    total_responses = 0
    total_replies = 0
    total_ignored = 0
    for gid_str, rep in reputation.items():
        if rep.get("kicked"):
            continue
        responses = rep.get("total_responses", 0)
        replies = rep.get("replies_to_aura", 0)
        ignored = rep.get("ignored_responses", 0)
        temp_data = temperatures.get(gid_str, {})
        temperature = temp_data.get("temperature", 0.5)

        total_responses += responses
        total_replies += replies
        total_ignored += ignored

        active_groups.append({
            "name": rep.get("group_name", gid_str)[:20],
            "warmth": rep.get("warmth_level", "new"),
            "temperature": round(temperature, 2),
            "reputation": round(rep.get("reputation_score", 0.0), 2),
            "responses": responses,
            "replies": replies,
        })

    # --- Engagement metrics ---
    engagement_rate = round(total_replies / max(total_responses, 1) * 100, 1)
    ignore_rate = round(total_ignored / max(total_responses, 1) * 100, 1)
    avg_temp = 0.0
    if temperatures:
        temps = [v.get("temperature", 0.5) for v in temperatures.values()]
        avg_temp = round(sum(temps) / len(temps), 2)

    # --- Growth stats ---
    growth_stats = growth.get("stats", {})
    events = growth.get("events", [])
    joins_7d = sum(1 for e in events if e.get("type") == "joined" and now - e.get("ts", 0) < 604800)
    kicks_7d = sum(1 for e in events if e.get("type") == "kicked" and now - e.get("ts", 0) < 604800)

    # --- Socialite ops ---
    pending_followups = len(socialite_state.get("pending_followups", []))
    daily_dms_sent = socialite_state.get("daily_dm_count", 0)
    daily_dm_max = 3

    # --- Advocacy pipeline (top 5 closest to advocate) ---
    pipeline = []
    for uid, u in social.items():
        depth = u.get("relationship_depth", "stranger")
        if depth == "advocate":
            continue
        total = u.get("dm_count", 0) + u.get("group_interactions", 0)
        groups = len(u.get("groups_seen_in", []))
        progress = min(total / 50.0, 1.0)
        if groups >= 2:
            progress = min(progress + 0.15, 1.0)
        if u.get("dm_count", 0) >= 3:
            progress = min(progress + 0.10, 1.0)
        name = profiles.get(uid, {}).get("preferred_name") or profiles.get(uid, {}).get("display_name") or f"User {uid}"
        pipeline.append({
            "name": name[:15],
            "depth": depth,
            "progress": round(progress * 100),
            "interactions": total,
        })
    pipeline.sort(key=lambda x: x["progress"], reverse=True)
    pipeline = pipeline[:5]

    # --- Overall readiness grade ---
    # A-F based on: engagement rate, group count, avg temp, advocate count
    score = 0
    score += min(engagement_rate / 10, 3)       # up to 3 pts for engagement
    score += min(len(active_groups) * 0.5, 2)   # up to 2 pts for groups
    score += min(avg_temp * 2, 1.5)             # up to 1.5 pts for temperature
    score += min(depth_counts.get("advocate", 0) * 0.5, 1.5)  # up to 1.5 pts for advocates
    score += min(depth_counts.get("familiar", 0) * 0.3, 1)    # up to 1 pt for familiars
    score += 0.5 if connectors > 0 else 0       # 0.5 pts for having connectors

    if score >= 8:
        grade = "A"
        grade_color = "#44ff44"
    elif score >= 6:
        grade = "B"
        grade_color = "#88cc44"
    elif score >= 4:
        grade = "C"
        grade_color = "#ccaa44"
    elif score >= 2:
        grade = "D"
        grade_color = "#cc6644"
    else:
        grade = "F"
        grade_color = "#cc4444"

    return {
        "grade": grade,
        "grade_color": grade_color,
        "score": round(score, 1),
        "force": {
            "total_users": total_users,
            "total_messages": total_messages,
            "depth": depth_counts,
            "connectors": connectors,
            "dm_eligible": dm_eligible_count,
        },
        "theaters": active_groups,
        "engagement": {
            "total_responses": total_responses,
            "engagement_rate": engagement_rate,
            "ignore_rate": ignore_rate,
            "avg_temperature": avg_temp,
        },
        "growth": {
            "active_groups": growth_stats.get("active_groups", len(active_groups)),
            "total_joined": growth_stats.get("total_groups_joined", 0),
            "total_kicked": growth_stats.get("total_groups_kicked", 0),
            "net_7d": joins_7d - kicks_7d,
        },
        "ops": {
            "pending_followups": pending_followups,
            "daily_dms": f"{daily_dms_sent}/{daily_dm_max}",
        },
        "pipeline": pipeline,
    }


@app.route("/api/sitrep")
def api_sitrep():
    return jsonify(_build_sitrep())


@app.route("/api/graph")
def api_graph():
    return jsonify(_build_graph())


@app.route("/api/analytics")
def api_analytics():
    """Analytics dashboard data — growth metrics, funnel tracking, engagement."""
    from analytics import analytics
    from social_graph import social_graph
    from reputation import reputation_tracker
    from growth import growth_engine
    from memory import profile_cache

    return jsonify(analytics.get_dashboard_data(
        social_graph, reputation_tracker, growth_engine, profile_cache,
    ))


@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Aura Social Map</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0a0a0f;
    color: #ccc;
    font-family: 'Helvetica Neue', sans-serif;
    overflow: hidden;
    display: flex;
  }
  #graph { flex: 1; height: 100vh; }

  /* ── SITREP panel ─────────────────────────────────── */
  #sitrep {
    width: 300px;
    min-width: 300px;
    height: 100vh;
    overflow-y: auto;
    background: #0c0c14;
    border-left: 1px solid #1a1a2e;
    padding: 16px 14px;
    font-size: 11px;
    line-height: 1.6;
    z-index: 20;
  }
  #sitrep::-webkit-scrollbar { width: 4px; }
  #sitrep::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }

  #sitrep .header {
    text-align: center;
    border-bottom: 1px solid #1a1a2e;
    padding-bottom: 10px;
    margin-bottom: 12px;
  }
  #sitrep .header h2 {
    font-size: 13px;
    letter-spacing: 3px;
    color: #00d4ff;
    font-weight: 400;
    margin-bottom: 4px;
  }
  #sitrep .header .grade-box {
    display: inline-block;
    font-size: 32px;
    font-weight: 700;
    padding: 4px 16px;
    border: 2px solid;
    border-radius: 4px;
    margin: 6px 0;
    letter-spacing: 0;
  }
  #sitrep .header .score-label {
    font-size: 10px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  #sitrep .section {
    margin-bottom: 14px;
  }
  #sitrep .section-title {
    font-size: 10px;
    letter-spacing: 2px;
    color: #00d4ff;
    text-transform: uppercase;
    border-bottom: 1px solid #1a1a2e;
    padding-bottom: 4px;
    margin-bottom: 8px;
    font-weight: 500;
  }

  #sitrep .metric {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 2px 0;
  }
  #sitrep .metric .m-label { color: #666; }
  #sitrep .metric .m-val { color: #ddd; font-weight: 600; font-variant-numeric: tabular-nums; }

  #sitrep .bar-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 3px 0;
  }
  #sitrep .bar-name {
    width: 60px;
    color: #888;
    font-size: 10px;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  #sitrep .bar-track {
    flex: 1;
    height: 6px;
    background: #1a1a2e;
    border-radius: 3px;
    overflow: hidden;
  }
  #sitrep .bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
  }
  #sitrep .bar-val {
    width: 30px;
    font-size: 10px;
    color: #888;
    font-variant-numeric: tabular-nums;
  }

  #sitrep .theater-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 0;
    border-bottom: 1px solid #111;
  }
  #sitrep .theater-name {
    color: #aaa;
    font-size: 11px;
    max-width: 120px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  #sitrep .theater-badges { display: flex; gap: 4px; align-items: center; }
  #sitrep .badge {
    font-size: 9px;
    padding: 1px 5px;
    border-radius: 3px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  #sitrep .badge-new { background: #222; color: #666; }
  #sitrep .badge-warming { background: #332211; color: #cc8844; }
  #sitrep .badge-established { background: #112211; color: #44aa44; }
  #sitrep .badge-trusted { background: #332200; color: #ffcc00; }

  #sitrep .temp-indicator {
    width: 40px;
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-size: 11px;
    font-weight: 600;
  }

  #sitrep .pipeline-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 0;
  }
  #sitrep .pipeline-name { color: #aaa; font-size: 11px; }
  #sitrep .pipeline-pct {
    font-size: 11px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }

  #sitrep .pulse {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #44ff44;
    margin-right: 6px;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  /* ── Existing panels ───────────────────────────────── */
  #info {
    position: fixed;
    top: 16px;
    right: 316px;
    background: rgba(10,10,15,0.92);
    border: 1px solid #222;
    border-radius: 8px;
    padding: 16px;
    min-width: 240px;
    max-width: 320px;
    font-size: 13px;
    line-height: 1.5;
    display: none;
    z-index: 10;
  }
  #info h3 { color: #00d4ff; margin-bottom: 8px; font-size: 15px; }
  #info .row { display: flex; justify-content: space-between; }
  #info .label { color: #888; }
  #info .val { color: #eee; font-weight: 500; }
  #title {
    position: fixed;
    top: 16px;
    left: 16px;
    z-index: 10;
  }
  #title h1 {
    font-size: 20px;
    color: #00d4ff;
    font-weight: 300;
    letter-spacing: 2px;
  }
  #title p { font-size: 11px; color: #555; margin-top: 4px; }
  #legend {
    position: fixed;
    bottom: 16px;
    left: 16px;
    font-size: 11px;
    z-index: 10;
  }
  #legend .item { display: flex; align-items: center; gap: 6px; margin: 4px 0; }
  #legend .dot {
    width: 10px; height: 10px; border-radius: 50%; display: inline-block;
  }
  #legend .diamond {
    width: 10px; height: 10px; transform: rotate(45deg); display: inline-block;
  }
</style>
</head>
<body>
<div id="title">
  <h1>AURA SOCIAL MAP</h1>
  <p id="status">Loading...</p>
</div>
<div id="graph"></div>
<div id="info"></div>
<div id="legend">
  <div class="item"><span class="dot" style="background:#00d4ff"></span> Aura</div>
  <div class="item"><span class="dot" style="background:#666"></span> Stranger</div>
  <div class="item"><span class="dot" style="background:#88aacc"></span> Acquaintance</div>
  <div class="item"><span class="dot" style="background:#44cc88"></span> Familiar</div>
  <div class="item"><span class="dot" style="background:#ffaa00"></span> Advocate</div>
  <div class="item"><span class="diamond" style="background:#555"></span> Group (new)</div>
  <div class="item"><span class="diamond" style="background:#cc8844"></span> Group (warming)</div>
  <div class="item"><span class="diamond" style="background:#44aa44"></span> Group (established)</div>
</div>

<!-- SITREP panel -->
<div id="sitrep">
  <div class="header">
    <h2>SITREP</h2>
    <div class="grade-box" id="grade">-</div>
    <div class="score-label"><span class="pulse"></span>READINESS SCORE</div>
  </div>

  <div class="section">
    <div class="section-title">Force Disposition</div>
    <div class="metric"><span class="m-label">Personnel tracked</span><span class="m-val" id="s-users">-</span></div>
    <div class="metric"><span class="m-label">Total intercepts</span><span class="m-val" id="s-messages">-</span></div>
    <div class="metric"><span class="m-label">DM-eligible</span><span class="m-val" id="s-dm-eligible">-</span></div>
    <div class="metric"><span class="m-label">Connectors</span><span class="m-val" id="s-connectors">-</span></div>
    <div id="depth-bars"></div>
  </div>

  <div class="section">
    <div class="section-title">Theater Status</div>
    <div id="theater-list"></div>
  </div>

  <div class="section">
    <div class="section-title">Engagement Metrics</div>
    <div class="metric"><span class="m-label">Total responses</span><span class="m-val" id="s-responses">-</span></div>
    <div class="metric"><span class="m-label">Engagement rate</span><span class="m-val" id="s-engage-rate">-</span></div>
    <div class="metric"><span class="m-label">Ignore rate</span><span class="m-val" id="s-ignore-rate">-</span></div>
    <div class="metric"><span class="m-label">Avg temperature</span><span class="m-val" id="s-avg-temp">-</span></div>
  </div>

  <div class="section">
    <div class="section-title">Growth Intel</div>
    <div class="metric"><span class="m-label">Active theaters</span><span class="m-val" id="s-active-groups">-</span></div>
    <div class="metric"><span class="m-label">Total joins</span><span class="m-val" id="s-total-joins">-</span></div>
    <div class="metric"><span class="m-label">Total kicks</span><span class="m-val" id="s-total-kicks">-</span></div>
    <div class="metric"><span class="m-label">Net 7d</span><span class="m-val" id="s-net-7d">-</span></div>
  </div>

  <div class="section">
    <div class="section-title">Socialite Ops</div>
    <div class="metric"><span class="m-label">Pending followups</span><span class="m-val" id="s-pending">-</span></div>
    <div class="metric"><span class="m-label">Daily DMs sent</span><span class="m-val" id="s-daily-dms">-</span></div>
  </div>

  <div class="section">
    <div class="section-title">Advocacy Pipeline</div>
    <div id="pipeline-list"></div>
  </div>
</div>

<script>
const container = document.getElementById('graph');
const infoPanel = document.getElementById('info');
const statusEl = document.getElementById('status');
let network = null;

function buildVis(data) {
  const nodes = data.nodes.map(n => ({
    id: n.id,
    label: n.label,
    size: n.size,
    color: {
      background: n.color,
      border: n.color,
      highlight: { background: '#fff', border: n.color },
    },
    shape: n.shape || (n.type === 'aura' ? 'dot' : 'dot'),
    font: {
      color: '#aaa',
      size: n.type === 'aura' ? 16 : n.type === 'group' ? 13 : 11,
      face: 'Helvetica Neue',
    },
    borderWidth: n.type === 'aura' ? 3 : 1,
    shadow: n.type === 'aura',
    _data: n,
  }));

  const edges = data.edges.map(e => ({
    from: e.from,
    to: e.to,
    width: e.width || 1,
    color: { color: e.color || '#333', opacity: 0.6 },
    dashes: e.dashes || false,
    smooth: { type: 'continuous' },
    font: { color: '#444', size: 9, strokeWidth: 0 },
  }));

  const options = {
    physics: {
      solver: 'forceAtlas2Based',
      forceAtlas2Based: {
        gravitationalConstant: -80,
        centralGravity: 0.008,
        springLength: 150,
        springConstant: 0.03,
        damping: 0.5,
      },
      stabilization: { iterations: 200 },
    },
    interaction: {
      hover: true,
      tooltipDelay: 100,
    },
    nodes: {
      borderWidth: 1,
      shadow: false,
    },
    edges: {
      smooth: { type: 'continuous' },
    },
  };

  if (network) {
    network.setData({ nodes, edges });
  } else {
    network = new vis.Network(container, { nodes, edges }, options);
    network.on('click', function(params) {
      if (params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        const node = data.nodes.find(n => n.id === nodeId);
        if (node) showInfo(node);
      } else {
        infoPanel.style.display = 'none';
      }
    });
  }
}

function showInfo(node) {
  let html = `<h3>${node.label}</h3>`;
  if (node.type === 'user') {
    if (node.username) html += row('Username', '@' + node.username);
    html += row('Relationship', node.depth);
    html += row('Messages', node.messages);
    html += row('DMs', node.dms);
    html += row('Group msgs', node.group_interactions);
    html += row('Influence', node.influence);
    if (node.is_connector) html += row('Status', 'Connector');
  } else if (node.type === 'group') {
    html += row('Warmth', node.warmth);
    html += row('Temperature', node.temperature);
    html += row('Reputation', node.reputation);
    html += row('Responses', node.responses);
    html += row('Replies', node.replies);
  } else {
    html += row('Type', 'Core Node');
  }
  infoPanel.innerHTML = html;
  infoPanel.style.display = 'block';
}

function row(label, val) {
  return `<div class="row"><span class="label">${label}</span><span class="val">${val}</span></div>`;
}

/* ── SITREP rendering ─────────────────────────────── */

function tempColor(t) {
  if (t >= 0.7) return '#44ff44';
  if (t >= 0.5) return '#88cc44';
  if (t >= 0.3) return '#ccaa44';
  return '#cc4444';
}

function depthColor(d) {
  const c = {stranger:'#666',acquaintance:'#88aacc',familiar:'#44cc88',advocate:'#ffaa00'};
  return c[d] || '#666';
}

function renderSitrep(s) {
  // Grade
  const gradeEl = document.getElementById('grade');
  gradeEl.textContent = s.grade;
  gradeEl.style.color = s.grade_color;
  gradeEl.style.borderColor = s.grade_color;

  // Force
  document.getElementById('s-users').textContent = s.force.total_users;
  document.getElementById('s-messages').textContent = s.force.total_messages.toLocaleString();
  document.getElementById('s-dm-eligible').textContent = s.force.dm_eligible;
  document.getElementById('s-connectors').textContent = s.force.connectors;

  // Depth bars
  const depthDiv = document.getElementById('depth-bars');
  const maxDepth = Math.max(1, ...Object.values(s.force.depth));
  let depthHtml = '';
  for (const [name, count] of Object.entries(s.force.depth)) {
    const pct = Math.round(count / maxDepth * 100);
    depthHtml += `<div class="bar-row">
      <span class="bar-name">${name}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${depthColor(name)}"></div></div>
      <span class="bar-val">${count}</span>
    </div>`;
  }
  depthDiv.innerHTML = depthHtml;

  // Theaters
  const theaterDiv = document.getElementById('theater-list');
  if (s.theaters.length === 0) {
    theaterDiv.innerHTML = '<div style="color:#444;padding:4px 0">No active theaters</div>';
  } else {
    let tHtml = '';
    for (const t of s.theaters) {
      const badgeClass = 'badge badge-' + t.warmth;
      tHtml += `<div class="theater-row">
        <span class="theater-name">${t.name}</span>
        <span class="theater-badges">
          <span class="${badgeClass}">${t.warmth}</span>
          <span class="temp-indicator" style="color:${tempColor(t.temperature)}">${t.temperature}</span>
        </span>
      </div>`;
    }
    theaterDiv.innerHTML = tHtml;
  }

  // Engagement
  document.getElementById('s-responses').textContent = s.engagement.total_responses;
  document.getElementById('s-engage-rate').textContent = s.engagement.engagement_rate + '%';
  document.getElementById('s-ignore-rate').textContent = s.engagement.ignore_rate + '%';
  const avgT = s.engagement.avg_temperature;
  const avgTEl = document.getElementById('s-avg-temp');
  avgTEl.textContent = avgT;
  avgTEl.style.color = tempColor(avgT);

  // Growth
  document.getElementById('s-active-groups').textContent = s.growth.active_groups;
  document.getElementById('s-total-joins').textContent = s.growth.total_joined;
  document.getElementById('s-total-kicks').textContent = s.growth.total_kicked;
  const net7d = s.growth.net_7d;
  const net7dEl = document.getElementById('s-net-7d');
  net7dEl.textContent = (net7d >= 0 ? '+' : '') + net7d;
  net7dEl.style.color = net7d > 0 ? '#44ff44' : net7d < 0 ? '#cc4444' : '#888';

  // Ops
  document.getElementById('s-pending').textContent = s.ops.pending_followups;
  document.getElementById('s-daily-dms').textContent = s.ops.daily_dms;

  // Pipeline
  const pipeDiv = document.getElementById('pipeline-list');
  if (s.pipeline.length === 0) {
    pipeDiv.innerHTML = '<div style="color:#444;padding:4px 0">No candidates</div>';
  } else {
    let pHtml = '';
    for (const p of s.pipeline) {
      const pctColor = p.progress >= 75 ? '#44ff44' : p.progress >= 50 ? '#ccaa44' : '#666';
      pHtml += `<div class="bar-row">
        <span class="bar-name" title="${p.name}">${p.name}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${p.progress}%;background:${pctColor}"></div></div>
        <span class="bar-val" style="color:${pctColor}">${p.progress}%</span>
      </div>`;
    }
    pipeDiv.innerHTML = pHtml;
  }
}

/* ── Refresh loops ────────────────────────────────── */

async function refresh() {
  try {
    const resp = await fetch('/api/graph');
    const data = await resp.json();
    buildVis(data);
    const userCount = data.nodes.filter(n => n.type === 'user').length;
    const groupCount = data.nodes.filter(n => n.type === 'group').length;
    statusEl.textContent = `${userCount} users \u00b7 ${groupCount} groups \u00b7 Live`;
  } catch (e) {
    statusEl.textContent = 'Connection error';
  }
}

async function refreshSitrep() {
  try {
    const resp = await fetch('/api/sitrep');
    const data = await resp.json();
    renderSitrep(data);
  } catch (e) {
    console.warn('SITREP fetch error', e);
  }
}

refresh();
refreshSitrep();
setInterval(refresh, 10000);
setInterval(refreshSitrep, 10000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("Aura Social Map → http://localhost:8899")
    app.run(host="0.0.0.0", port=8899, debug=False)

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


@app.route("/api/graph")
def api_graph():
    return jsonify(_build_graph())


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
  }
  #graph { width: 100vw; height: 100vh; }
  #info {
    position: fixed;
    top: 16px;
    right: 16px;
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
    if (node.is_connector) html += row('Status', '🔗 Connector');
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

async function refresh() {
  try {
    const resp = await fetch('/api/graph');
    const data = await resp.json();
    buildVis(data);
    const userCount = data.nodes.filter(n => n.type === 'user').length;
    const groupCount = data.nodes.filter(n => n.type === 'group').length;
    statusEl.textContent = `${userCount} users · ${groupCount} groups · Live`;
  } catch (e) {
    statusEl.textContent = 'Connection error';
  }
}

refresh();
setInterval(refresh, 10000);  // refresh every 10s
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("Aura Social Map → http://localhost:8899")
    app.run(host="0.0.0.0", port=8899, debug=False)

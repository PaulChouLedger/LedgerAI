"""
rag -- RAG (Retrieval-Augmented Generation) for TG bot.

Thin wrapper around the shared RAG client, configured for RTX paths.
Searches local FAISS index and returns context for LLM prompt injection.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# Base data dir — on RTX this is the repo's data/ directory
_BASE_DIR = str(Path(__file__).resolve().parent.parent.parent / "data")

_client = None
_init_failed = False


def _import_rag_module():
    """Import containers/llm/rag package without sys.path pollution."""
    rag_pkg = Path(__file__).resolve().parent.parent.parent / "containers" / "llm" / "rag"
    # Load __init__.py from the rag package
    spec = importlib.util.spec_from_file_location(
        "llm_rag", rag_pkg / "__init__.py",
        submodule_search_locations=[str(rag_pkg)],
    )
    mod = importlib.util.module_from_spec(spec)
    # Temporarily register so sub-imports (rag_client, cpu_faiss_auto_ingest) resolve
    import sys
    sys.modules["llm_rag"] = mod
    # Also register as "rag" subpackage for internal relative imports
    old_rag = sys.modules.get("rag")
    sys.modules["rag"] = mod
    spec.loader.exec_module(mod)
    # Restore original rag module reference (this file)
    if old_rag is not None:
        sys.modules["rag"] = old_rag
    else:
        # Keep the containers/llm/rag in sys.modules as "rag" since
        # the rag_client module's internal imports need it
        pass
    return mod


def _get_client():
    """Lazy-init RAG client singleton."""
    global _client, _init_failed
    if _client is not None:
        return _client
    if _init_failed:
        return None
    try:
        os.environ.setdefault("RAG_MODE", "CPU")
        rag_mod = _import_rag_module()
        _client = rag_mod.get_rag_client(use_gpu=False, base_dir=_BASE_DIR)
        log.info("RAG client initialized (base_dir=%s, chunks=%d)",
                 _BASE_DIR, len(getattr(_client, '_cpu_chunks', [])))
        return _client
    except Exception as e:
        log.error("RAG init failed: %s", e, exc_info=True)
        _init_failed = True
        return None


def search(query: str, k: int = 3, threshold: float = 0.15) -> list[dict]:
    """Search RAG index. Returns list of {text, score, metadata} dicts."""
    client = _get_client()
    if not client:
        return []
    try:
        return client.search(query=query, k=k, threshold=threshold)
    except Exception as e:
        log.warning("RAG search failed: %s", e)
        return []


def format_context(results: list[dict], max_chars: int = 2000) -> str:
    """Format RAG results into a context block for system prompt injection."""
    if not results:
        return ""
    lines = []
    total = 0
    for r in results:
        text = r.get("text", "").strip()
        source = r.get("metadata", {}).get("document_name", "")
        if not text:
            continue
        entry = f"[{source}] {text}" if source else text
        room = max_chars - total
        if room < 200:
            break
        # An oversize chunk is truncated, not dropped — the old `break`
        # returned an EMPTY context whenever the best chunk alone exceeded
        # max_chars, which starved exactly the queries with the richest
        # answers (found 2026-09-04).
        if len(entry) > room:
            entry = entry[:room]
        lines.append(entry)
        total += len(entry)
    if not lines:
        return ""
    return "RELEVANT KNOWLEDGE:\n" + "\n---\n".join(lines)


def rag_context_for(query: str, k: int = 5, max_chars: int = 3000,
                    threshold: float = 0.15,
                    exclude_prefixes: tuple = (),
                    pin_docs: tuple = ()) -> str:
    """One-call convenience: search + format. Returns empty string if nothing found.

    exclude_prefixes drops documents by name prefix — the project-question
    path uses it to keep chat exports (tg_*) and the news firehose (news_*)
    out of answers that should come from the curated project docs.

    pin_docs always includes those documents' chunks, ahead of scored
    results — the owner's DM briefing is a designated preferential
    channel ("that way i can preferentially share news to her"), so it
    rides along on every project answer instead of having to win a
    similarity contest against polished marketing copy.
    """
    results = search(query, k=k, threshold=threshold)
    if exclude_prefixes:
        results = [r for r in results
                   if not str(r.get("metadata", {}).get("document_name", ""))
                   .startswith(exclude_prefixes)]
    if pin_docs:
        #: the pin reads the FILE, not the index — always current (no
        #: ingest latency, no stale chunk generations), and the file is
        #: newest-first so a budget cut costs April's news, not this
        #: week's. Capped to a third of the budget: the pin gets a seat,
        #: not the whole table.
        parts = []
        for name in pin_docs:
            for ext in (".txt", ".md"):
                p = Path(_BASE_DIR) / "input" / f"{name}{ext}"
                if p.exists():
                    body = "\n".join(
                        ln for ln in p.read_text().splitlines()
                        if ln.strip() and not ln.startswith("#"))
                    if body:
                        parts.append(f"[{name}] {body}")
                    break
        pin_block = ""
        if parts:
            pin_block = ("RELEVANT KNOWLEDGE:\n"
                         + "\n---\n".join(parts)[: max_chars // 3])
        scored = [
            r for r in results
            if r.get("metadata", {}).get("document_name") not in pin_docs]
        rest = format_context(scored, max_chars=max_chars - len(pin_block))
        if pin_block and rest:
            return (pin_block + "\n---\n"
                    + rest[len("RELEVANT KNOWLEDGE:\n"):])
        return pin_block or rest
    return format_context(results, max_chars=max_chars)


# ---------------------------------------------------------------------------
# Periodic feed → RAG sync
# ---------------------------------------------------------------------------

# Group chat IDs we export to RAG (no DMs — privacy)
_GROUP_NAMES: dict[int, str] = {
    -1002111119265: "LedgerAi_Official_LEDGER",
    -1002322513545: "LedgerAI_Ambassadors",
    -1001408551359: "CryptoKids",
    -1001876350591: "Alpha_Meta",
    -1002903110439: "Sleyman_Crew",
    -1003025733750: "Area31",
    -1002753949117: "LedgerAI_Raiders",
    -1003836185794: "NetSol_Technologies",
}


def _sync_owner_dm(base: Path, by_dm: dict[int, list[dict]]) -> int:
    """Export the OWNER's DM thread with Aura → data/input/owner_briefing.txt.

    DMs are excluded from RAG by design (privacy) with exactly one owner-
    requested exception (2026-09-04: "any DM between me and Aura will
    automatically be RAG ingested, that way i can preferentially share
    news to her"). Scope is hard-limited to config.OWNER_USER_IDS, and
    only the owner's own words are exported — the bot's replies stay out
    so her paraphrases never become sources. The file deliberately has NO
    tg_ prefix: the project-question retrieval path excludes tg_*, and
    this file exists precisely to be found there. Consequence the owner
    accepted: anything he DMs her is fair game for public group answers.
    """
    from datetime import datetime

    try:
        from config import OWNER_USER_IDS
    except ImportError:
        return 0

    msgs = []
    for cid in OWNER_USER_IDS:
        msgs.extend(m for m in by_dm.get(cid, []) if not m.get("is_bot"))
    #: NEWEST FIRST — chunking and every truncation downstream keep the
    #: head and eat the tail, so the head must be this week's news, not
    #: April's (measured: the newest fact was the one that kept falling
    #: off the end).
    msgs.sort(key=lambda m: m.get("ts", 0), reverse=True)
    if not msgs:
        return 0

    lines = [
        "# Owner briefing — facts, news and updates Paul (LedgerAI founder)",
        "# has shared directly with Aura. Authoritative and current.",
        "# Newest first.",
        "",
    ]
    for m in msgs:
        ts = m.get("ts", 0)
        when = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else "?"
        text = (m.get("text") or "").strip()
        #: chatter filter — the briefing indexed as ONE chunk, and "status
        #: check" / "hello?" / "lol" diluted the embedding until a real
        #: fact in the same chunk scored 0.46 (measured). Facts are long;
        #: phatic lines are short.
        if not text or text.startswith("/") or len(text) < 25:
            continue
        lines.append(f"[{when}] {text}")

    out = base / "input" / "owner_briefing.txt"
    content = "\n".join(lines)
    if out.exists() and out.read_text() == content:
        return 0
    out.write_text(content)
    log.info("RAG owner-DM sync: owner_briefing.txt updated (%d lines)",
             len(lines) - 3)
    return 1


def sync_feed_to_rag() -> int:
    """Re-export tg_feed.jsonl → per-group text files in data/input/.

    Exports group chats plus exactly one DM thread: the owner's (see
    _sync_owner_dm). Returns number of files updated. The FAISS
    auto-ingest watchdog will detect changed files and re-index.
    """
    import json
    from datetime import datetime

    base = Path(_BASE_DIR)
    feed_path = base / "tg_feed.jsonl"
    input_dir = base / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    if not feed_path.exists():
        log.warning("tg_feed.jsonl not found at %s", feed_path)
        return 0

    # Load reputation for group name fallbacks
    rep_path = base / "telegram" / "reputation.json"
    rep_names: dict[str, str] = {}
    if rep_path.exists():
        try:
            with open(rep_path) as f:
                for gid, data in json.load(f).items():
                    if data.get("group_name"):
                        rep_names[gid] = data["group_name"]
        except Exception:
            pass

    # Bucket messages by group chat ID; DMs are collected separately and
    # used ONLY for the owner's briefing thread (see _sync_owner_dm).
    by_chat: dict[int, list[dict]] = {}
    by_dm: dict[int, list[dict]] = {}
    with open(feed_path) as f:
        for line in f:
            try:
                msg = json.loads(line)
                cid = msg.get("chat_id", 0)
                if cid >= 0:
                    by_dm.setdefault(cid, []).append(msg)
                    continue  # DMs never join the group export
                by_chat.setdefault(cid, []).append(msg)
            except Exception:
                pass

    updated = 0
    for cid, msgs in by_chat.items():
        name = _GROUP_NAMES.get(cid) or rep_names.get(str(cid)) or f"chat_{cid}"
        safe_name = name.replace(" ", "_").replace("/", "_").replace("|", "").replace("$", "")

        msgs.sort(key=lambda m: m.get("ts", 0))

        lines = [
            f"# Telegram Chat History: {name}",
            f"# Chat ID: {cid}",
            f"# Messages: {len(msgs)}",
        ]
        first_ts = msgs[0].get("ts", 0)
        last_ts = msgs[-1].get("ts", 0)
        if first_ts:
            lines.append(f"# First message: {datetime.fromtimestamp(first_ts).isoformat()}")
        if last_ts:
            lines.append(f"# Last message: {datetime.fromtimestamp(last_ts).isoformat()}")
        lines.append("")

        for msg in msgs:
            ts = msg.get("ts", 0)
            speaker = msg.get("name", "Unknown")
            text = msg.get("text", "").strip()
            if not text:
                continue
            time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "?"
            bot_tag = " [BOT]" if msg.get("is_bot") else ""
            lines.append(f"[{time_str}] {speaker}{bot_tag}: {text}")

        out_path = input_dir / f"tg_{safe_name}.txt"
        content = "\n".join(lines)

        # Only write if content changed (avoid unnecessary re-index)
        if out_path.exists() and out_path.read_text() == content:
            continue

        out_path.write_text(content)
        updated += 1

    updated += _sync_owner_dm(base, by_dm)

    if updated:
        log.info("RAG feed sync: updated %d group file(s)", updated)
        # Trigger re-index if client is initialized
        client = _get_client()
        if client and hasattr(client, '_auto_ingest') and client._auto_ingest:
            try:
                result = client._auto_ingest.scan_and_process()
                if result.get("processed", 0) > 0:
                    client._auto_ingest.load_existing_embeddings()
                    client._cpu_chunks = client._auto_ingest.chunks
                    client._cpu_metadata = client._auto_ingest.metadata
                    client._rebuild_cpu_index()
                    log.info("RAG re-indexed: %d chunks", len(client._cpu_chunks))
            except Exception as e:
                log.warning("RAG re-index failed: %s", e)

    return updated

#!/usr/bin/env python3
"""
news_poller.py -- Fetch current events from RSS feeds and write to RAG input directory.

The FAISS auto-ingest watchdog detects changed files in data/input/ and re-indexes
automatically. This script is meant to run on a schedule (cron or systemd timer).

Usage:
    python3 news_poller.py              # one-shot fetch
    python3 news_poller.py --loop 900   # fetch every 15 minutes
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import feedparser

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [news_poller] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "input"

# RSS feeds by category
FEEDS: dict[str, list[tuple[str, str]]] = {
    "world": [
        ("Reuters World", "https://feeds.reuters.com/Reuters/worldNews"),
        ("AP Top News", "https://rsshub.app/apnews/topics/apf-topnews"),
        ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
        ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
    ],
    "us_politics": [
        ("Reuters US", "https://feeds.reuters.com/Reuters/domesticNews"),
        ("AP Politics", "https://rsshub.app/apnews/topics/apf-politics"),
        ("BBC US/Canada", "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml"),
    ],
    "business": [
        ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
        ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ],
    "tech": [
        ("Reuters Tech", "https://feeds.reuters.com/reuters/technologyNews"),
        ("TechCrunch", "https://techcrunch.com/feed/"),
        ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
    ],
    "crypto": [
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("CoinTelegraph", "https://cointelegraph.com/rss"),
    ],
    "sports": [
        ("ESPN Top", "https://www.espn.com/espn/rss/news"),
        ("BBC Sport", "https://feeds.bbci.co.uk/sport/rss.xml"),
    ],
    "science": [
        ("Reuters Science", "https://feeds.reuters.com/reuters/scienceNews"),
        ("BBC Science", "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"),
    ],
}

# Max articles per category (keep files manageable for RAG chunking)
MAX_ARTICLES_PER_CATEGORY = 30

# Max age of articles to include (hours)
MAX_AGE_HOURS = 48


# ---------------------------------------------------------------------------
# Fetch + write
# ---------------------------------------------------------------------------

def fetch_category(category: str, feeds: list[tuple[str, str]]) -> list[dict]:
    """Fetch all feeds for a category, deduplicate, sort by date."""
    articles: list[dict] = []
    seen_titles: set[str] = set()

    for source_name, url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                title = (entry.get("title") or "").strip()
                if not title:
                    continue

                # Deduplicate by title hash
                key = hashlib.md5(title.lower().encode()).hexdigest()
                if key in seen_titles:
                    continue
                seen_titles.add(key)

                # Parse published date
                published = None
                for date_field in ("published_parsed", "updated_parsed"):
                    tp = entry.get(date_field)
                    if tp:
                        try:
                            published = datetime(*tp[:6], tzinfo=timezone.utc)
                        except Exception:
                            pass
                        break

                # Skip old articles
                if published:
                    age_hours = (datetime.now(timezone.utc) - published).total_seconds() / 3600
                    if age_hours > MAX_AGE_HOURS:
                        continue

                # Extract summary
                summary = (entry.get("summary") or entry.get("description") or "").strip()
                # Strip HTML tags
                import re
                summary = re.sub(r"<[^>]+>", "", summary)
                summary = re.sub(r"\s+", " ", summary).strip()

                articles.append({
                    "title": title,
                    "source": source_name,
                    "published": published,
                    "summary": summary[:500],
                    "link": entry.get("link", ""),
                })

        except Exception as e:
            log.warning("Failed to fetch %s (%s): %s", source_name, url, e)

    # Sort by date (newest first)
    articles.sort(key=lambda a: a.get("published") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return articles[:MAX_ARTICLES_PER_CATEGORY]


def write_category(category: str, articles: list[dict]) -> bool:
    """Write articles to a single text file per category in the RAG input directory.

    Articles are separated by paragraph breaks so the RAG chunker treats
    each article as its own chunk for better search relevance.

    Returns True if the file changed (triggering re-index).
    """
    if not articles:
        return False

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INPUT_DIR / f"news_{category}.txt"

    lines = []

    for art in articles:
        ts = art["published"].strftime("%Y-%m-%d %H:%M UTC") if art.get("published") else "Unknown date"
        # Each article is a self-contained paragraph with enough context
        # for the RAG chunker to treat it as a separate chunk
        article_block = f"NEWS ({category.replace('_', ' ').upper()}): {art['title']}. "
        article_block += f"Source: {art['source']}. Date: {ts}. "
        if art["summary"]:
            article_block += art["summary"]
        lines.append(article_block)
        lines.append("")  # blank line = paragraph break for chunker

    content = "\n".join(lines)

    # Only write if content meaningfully changed
    if out_path.exists():
        if out_path.read_text() == content:
            return False

    out_path.write_text(content)
    return True


def poll_all() -> int:
    """Fetch all categories and write to disk. Returns number of files updated."""
    updated = 0
    total_articles = 0

    for category, feeds in FEEDS.items():
        articles = fetch_category(category, feeds)
        total_articles += len(articles)
        if write_category(category, articles):
            updated += 1
            log.info("Updated %s: %d articles", category, len(articles))
        else:
            log.debug("No changes for %s (%d articles)", category, len(articles))

    log.info("Poll complete: %d articles across %d categories, %d files updated",
             total_articles, len(FEEDS), updated)
    return updated


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch news RSS feeds for RAG ingestion")
    parser.add_argument("--loop", type=int, default=0,
                        help="Poll interval in seconds (0 = one-shot)")
    args = parser.parse_args()

    if args.loop > 0:
        log.info("Starting news poller (interval: %ds)", args.loop)
        while True:
            try:
                poll_all()
            except Exception as e:
                log.error("Poll error: %s", e)
            time.sleep(args.loop)
    else:
        poll_all()


if __name__ == "__main__":
    main()

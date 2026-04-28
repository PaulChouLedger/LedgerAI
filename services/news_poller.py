#!/usr/bin/env python3
"""
news_poller.py -- Fetch current events from RSS feeds and write to RAG input directory.

The FAISS auto-ingest watchdog detects changed files in data/input/ and re-indexes
automatically. This script is meant to run on a schedule (cron or systemd timer).

Usage:
    python3 news_poller.py              # one-shot fetch
    python3 news_poller.py --loop 300   # fetch every 5 minutes
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import feedparser

# Set a browser-like User-Agent to avoid HTTP 403 from picky feeds
feedparser.USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"

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
        ("Reuters World", "https://www.reutersagency.com/feed/?best-topics=political-general&post_type=best"),
        ("AP Top News", "https://feedx.net/rss/ap.xml"),
        ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
        ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
        ("NPR World", "https://feeds.npr.org/1004/rss.xml"),
        ("France24", "https://www.france24.com/en/rss"),
        ("DW News", "https://rss.dw.com/rdf/rss-en-all"),
        ("Guardian World", "https://www.theguardian.com/world/rss"),
        ("NYT World", "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"),
    ],
    "us_politics": [
        ("Reuters US", "https://www.reutersagency.com/feed/?best-topics=political-general&post_type=best"),
        ("AP Politics", "https://feedx.net/rss/ap.xml"),
        ("BBC US/Canada", "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml"),
        ("NPR Politics", "https://feeds.npr.org/1014/rss.xml"),
        ("Politico", "https://www.politico.com/rss/politicopicks.xml"),
        ("The Hill", "https://thehill.com/feed/"),
        ("NYT Politics", "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml"),
        ("Guardian US", "https://www.theguardian.com/us-news/rss"),
    ],
    "business": [
        ("Reuters Business", "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best"),
        ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
        ("CNBC Top", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
        ("Bloomberg", "https://feeds.bloomberg.com/markets/news.rss"),
        ("Financial Times", "https://www.ft.com/rss/home"),
        ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories/"),
        ("NYT Business", "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml"),
        ("WSJ Markets", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
    ],
    "markets": [
        ("CNBC Markets", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
        ("Investing.com", "https://www.investing.com/rss/news.rss"),
        ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
        ("WSJ Markets", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
        ("Reuters Markets", "https://www.reutersagency.com/feed/?best-topics=markets&post_type=best"),
    ],
    "tech": [
        ("Reuters Tech", "https://www.reutersagency.com/feed/?best-topics=tech&post_type=best"),
        ("TechCrunch", "https://techcrunch.com/feed/"),
        ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
        ("The Verge", "https://www.theverge.com/rss/index.xml"),
        ("Wired", "https://www.wired.com/feed/rss"),
        ("Hacker News", "https://hnrss.org/frontpage"),
        ("MIT Tech Review", "https://www.technologyreview.com/feed/"),
        ("Engadget", "https://www.engadget.com/rss.xml"),
        ("ZDNet", "https://www.zdnet.com/news/rss.xml"),
    ],
    "ai": [
        ("MIT AI News", "https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml"),
        ("VentureBeat AI", "https://venturebeat.com/category/ai/feed/"),
        ("The Decoder", "https://the-decoder.com/feed/"),
        ("AI News", "https://www.artificialintelligence-news.com/feed/"),
        ("Google AI Blog", "https://blog.google/technology/ai/rss/"),
        ("OpenAI Blog", "https://openai.com/blog/rss.xml"),
        ("Hugging Face Blog", "https://huggingface.co/blog/feed.xml"),
    ],
    "crypto": [
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("CoinTelegraph", "https://cointelegraph.com/rss"),
        ("The Block", "https://www.theblock.co/rss.xml"),
        ("Decrypt", "https://decrypt.co/feed"),
        ("Bitcoin Magazine", "https://bitcoinmagazine.com/.rss/full/"),
        ("CryptoSlate", "https://cryptoslate.com/feed/"),
        ("DeFi Pulse", "https://defipulse.com/blog/feed/"),
        ("Blockworks", "https://blockworks.co/feed"),
    ],
    "sports": [
        ("ESPN Top", "https://www.espn.com/espn/rss/news"),
        ("BBC Sport", "https://feeds.bbci.co.uk/sport/rss.xml"),
        ("BBC Football", "https://feeds.bbci.co.uk/sport/football/rss.xml"),
        ("Sky Sports", "https://www.skysports.com/rss/12040"),
        ("Yahoo Sports", "https://sports.yahoo.com/rss/"),
        ("CBS Sports", "https://www.cbssports.com/rss/headlines/"),
        ("Guardian Football", "https://www.theguardian.com/football/rss"),
        ("ESPN Soccer", "https://www.espn.com/espn/rss/soccer/news"),
        ("ESPN NFL", "https://www.espn.com/espn/rss/nfl/news"),
        ("ESPN NBA", "https://www.espn.com/espn/rss/nba/news"),
        ("ESPN MLB", "https://www.espn.com/espn/rss/mlb/news"),
        ("Bleacher Report", "https://bleacherreport.com/articles/feed"),
    ],
    "science": [
        ("Reuters Science", "https://www.reutersagency.com/feed/?best-topics=science&post_type=best"),
        ("BBC Science", "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"),
        ("Nature News", "https://www.nature.com/nature.rss"),
        ("Science Daily", "https://www.sciencedaily.com/rss/all.xml"),
        ("New Scientist", "https://www.newscientist.com/feed/home/"),
        ("Scientific American", "https://rss.sciam.com/ScientificAmerican-Global"),
        ("Space.com", "https://www.space.com/feeds/all"),
        ("Phys.org", "https://phys.org/rss-feed/"),
    ],
    "health": [
        ("Reuters Health", "https://www.reutersagency.com/feed/?best-topics=health&post_type=best"),
        ("BBC Health", "https://feeds.bbci.co.uk/news/health/rss.xml"),
        ("NPR Health", "https://feeds.npr.org/1128/rss.xml"),
        ("WHO News", "https://www.who.int/feeds/entity/mediacentre/news/en/rss.xml"),
        ("Medical News Today", "https://www.medicalnewstoday.com/newsfeeds/rss"),
        ("WebMD Health", "https://rssfeeds.webmd.com/rss/rss.aspx?RSSSource=RSS_PUBLIC"),
    ],
    "entertainment": [
        ("BBC Entertainment", "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml"),
        ("Variety", "https://variety.com/feed/"),
        ("Hollywood Reporter", "https://www.hollywoodreporter.com/feed/"),
        ("Rolling Stone", "https://www.rollingstone.com/feed/"),
        ("Pitchfork", "https://pitchfork.com/feed/feed-news/rss"),
        ("Guardian Film", "https://www.theguardian.com/film/rss"),
    ],
    "environment": [
        ("BBC Environment", "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"),
        ("Guardian Environment", "https://www.theguardian.com/environment/rss"),
        ("Carbon Brief", "https://www.carbonbrief.org/feed/"),
        ("Climate Home", "https://www.climatechangenews.com/feed/"),
        ("Mongabay", "https://news.mongabay.com/feed/"),
    ],
    "middle_east": [
        ("Al Jazeera ME", "https://www.aljazeera.com/xml/rss/all.xml"),
        ("BBC Middle East", "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml"),
        ("Times of Israel", "https://www.timesofisrael.com/feed/"),
        ("Middle East Eye", "https://www.middleeasteye.net/rss"),
        ("Arab News", "https://www.arabnews.com/rss.xml"),
        ("Reuters ME", "https://www.reutersagency.com/feed/?best-topics=political-general&post_type=best"),
    ],
    "asia": [
        ("BBC Asia", "https://feeds.bbci.co.uk/news/world/asia/rss.xml"),
        ("South China Morning Post", "https://www.scmp.com/rss/91/feed"),
        ("Japan Times", "https://www.japantimes.co.jp/feed/"),
        ("Nikkei Asia", "https://asia.nikkei.com/rss"),
        ("Channel News Asia", "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml"),
    ],
    "europe": [
        ("BBC Europe", "https://feeds.bbci.co.uk/news/world/europe/rss.xml"),
        ("Euronews", "https://www.euronews.com/rss"),
        ("Guardian Europe", "https://www.theguardian.com/world/europe-news/rss"),
        ("DW Europe", "https://rss.dw.com/rdf/rss-en-eu"),
        ("Politico EU", "https://www.politico.eu/feed/"),
    ],
    "africa": [
        ("BBC Africa", "https://feeds.bbci.co.uk/news/world/africa/rss.xml"),
        ("Al Jazeera Africa", "https://www.aljazeera.com/xml/rss/all.xml"),
        ("Guardian Africa", "https://www.theguardian.com/world/africa/rss"),
        ("AllAfrica", "https://allafrica.com/tools/headlines/rdf/latest/headlines.rdf"),
    ],
    "latin_america": [
        ("BBC Latin America", "https://feeds.bbci.co.uk/news/world/latin_america/rss.xml"),
        ("Guardian Americas", "https://www.theguardian.com/world/americas/rss"),
        ("Reuters Americas", "https://www.reutersagency.com/feed/?best-topics=political-general&post_type=best"),
    ],
}

# Max articles per feed entry fetch
MAX_ENTRIES_PER_FEED = 50

# Max articles per category after dedup
MAX_ARTICLES_PER_CATEGORY = 100

# Max age of articles to include (hours)
MAX_AGE_HOURS = 72

# Force rewrite if file is older than this (hours) — prevents indefinite staleness
FORCE_REWRITE_HOURS = 6


# ---------------------------------------------------------------------------
# Fetch + write
# ---------------------------------------------------------------------------

def fetch_category(category: str, feeds: list[tuple[str, str]]) -> list[dict]:
    """Fetch all feeds for a category, deduplicate, sort by date."""
    articles: list[dict] = []
    seen_titles: set[str] = set()

    for source_name, url in feeds:
        try:
            feed = feedparser.parse(url, request_headers={"User-Agent": feedparser.USER_AGENT})
            for entry in feed.entries[:MAX_ENTRIES_PER_FEED]:
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
                summary = re.sub(r"<[^>]+>", "", summary)
                summary = re.sub(r"\s+", " ", summary).strip()

                articles.append({
                    "title": title,
                    "source": source_name,
                    "published": published,
                    "summary": summary[:800],
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

    # Only write if content changed OR file is stale (prevents indefinite staleness)
    if out_path.exists():
        file_age_hours = (time.time() - out_path.stat().st_mtime) / 3600
        if out_path.read_text() == content and file_age_hours < FORCE_REWRITE_HOURS:
            return False

    out_path.write_text(content)
    return True


def poll_all() -> int:
    """Fetch all categories and write to disk. Returns number of files updated."""
    updated = 0
    total_articles = 0
    empty_categories = []

    for category, feeds in FEEDS.items():
        articles = fetch_category(category, feeds)
        total_articles += len(articles)
        if not articles:
            empty_categories.append(category)
        if write_category(category, articles):
            updated += 1
            log.info("Updated %s: %d articles", category, len(articles))
        else:
            log.debug("No changes for %s (%d articles)", category, len(articles))

    log.info("Poll complete: %d articles across %d categories, %d files updated",
             total_articles, len(FEEDS), updated)
    if empty_categories:
        log.warning("Empty categories (all feeds failed): %s", ", ".join(empty_categories))
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

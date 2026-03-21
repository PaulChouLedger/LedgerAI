"""
analytics -- Growth metrics, funnel tracking, and engagement correlation.

Provides:
  - Invite funnel: which interactions preceded an invite?
  - Engagement correlation: which behaviors drive the most engagement per group?
  - User journey tracking: time from first_seen to each relationship depth
  - Advocacy pipeline: who is closest to becoming an advocate?
  - Daily summary logging
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from config import ANALYTICS_FILE, DATA_DIR

log = logging.getLogger(__name__)


def _load_json(path: Path, default) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return default


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


class Analytics:
    """Growth and engagement analytics tracker."""

    def __init__(self) -> None:
        self._data: dict = _load_json(ANALYTICS_FILE, {
            "engagement_events": [],  # [{type, chat_id, user_id, ts, details}]
            "user_journeys": {},      # {user_id: [{depth, ts}]}
            "daily_summaries": [],    # [{date, metrics}]
        })

    def _save(self) -> None:
        _save_json(ANALYTICS_FILE, self._data)

    # -- event tracking -----------------------------------------------------

    def track_event(
        self,
        event_type: str,
        chat_id: Optional[int] = None,
        user_id: Optional[int] = None,
        details: str = "",
    ) -> None:
        """Track an engagement event for correlation analysis."""
        events = self._data.setdefault("engagement_events", [])
        events.append({
            "type": event_type,
            "chat_id": chat_id,
            "user_id": user_id,
            "ts": time.time(),
            "details": details,
        })
        # Keep last 1000 events
        self._data["engagement_events"] = events[-1000:]
        self._save()

    # -- user journey tracking ----------------------------------------------

    def record_depth_change(self, user_id: int, new_depth: str) -> None:
        """Record when a user's relationship depth changes."""
        journeys = self._data.setdefault("user_journeys", {})
        key = str(user_id)
        journey = journeys.setdefault(key, [])

        # Don't record duplicate consecutive depths
        if journey and journey[-1].get("depth") == new_depth:
            return

        journey.append({"depth": new_depth, "ts": time.time()})
        # Keep last 20 transitions per user
        journeys[key] = journey[-20:]
        self._save()

    # -- advocacy pipeline --------------------------------------------------

    def get_advocacy_pipeline(self, social_graph, profile_cache) -> list[dict]:
        """Rank users by proximity to advocate status.

        Returns sorted list of users with their progress metrics.
        """
        pipeline = []

        for uid_str, user in social_graph._data.get("users", {}).items():
            depth = user.get("relationship_depth", "stranger")
            if depth == "advocate":
                continue  # Already there

            total = user.get("dm_count", 0) + user.get("group_interactions", 0)
            groups = len(user.get("groups_seen_in", []))
            name = profile_cache.get_name(int(uid_str)) or f"User {uid_str}"

            # Score: how close to advocate?
            # Advocate = 50 interactions OR invited Aura
            progress = min(total / 50.0, 1.0)

            # Bonus for connectors (multi-group users)
            if groups >= 2:
                progress = min(progress + 0.15, 1.0)

            # Bonus for DM engagement
            if user.get("dm_count", 0) >= 3:
                progress = min(progress + 0.10, 1.0)

            pipeline.append({
                "user_id": uid_str,
                "name": name,
                "depth": depth,
                "progress": round(progress, 2),
                "total_interactions": total,
                "dm_count": user.get("dm_count", 0),
                "groups": groups,
                "is_connector": user.get("is_connector", False),
            })

        pipeline.sort(key=lambda x: x["progress"], reverse=True)
        return pipeline[:20]

    # -- engagement correlation ---------------------------------------------

    def get_engagement_stats(self) -> dict:
        """Compute engagement statistics by event type."""
        events = self._data.get("engagement_events", [])
        if not events:
            return {}

        # Count by type
        type_counts: dict[str, int] = {}
        for e in events:
            t = e.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        # Recent trend (last 24h vs previous 24h)
        now = time.time()
        recent = [e for e in events if now - e.get("ts", 0) < 86400]
        previous = [e for e in events if 86400 <= now - e.get("ts", 0) < 172800]

        return {
            "total_events": len(events),
            "by_type": type_counts,
            "last_24h": len(recent),
            "previous_24h": len(previous),
            "trend": "up" if len(recent) > len(previous) else "down" if len(recent) < len(previous) else "flat",
        }

    # -- daily summary ------------------------------------------------------

    def generate_daily_summary(self, social_graph, reputation_tracker, growth_engine, profile_cache) -> dict:
        """Generate a daily summary of all metrics."""
        growth_stats = growth_engine.get_stats()
        engagement = self.get_engagement_stats()
        pipeline = self.get_advocacy_pipeline(social_graph, profile_cache)

        # Group summaries
        group_summaries = []
        for gid_str, rep in reputation_tracker._data.items():
            if rep.get("kicked"):
                continue
            group_summaries.append({
                "group": rep.get("group_name", gid_str),
                "warmth": rep.get("warmth_level", "new"),
                "score": rep.get("reputation_score", 0),
                "responses": rep.get("total_responses", 0),
            })

        summary = {
            "date": time.strftime("%Y-%m-%d"),
            "ts": time.time(),
            "growth": growth_stats,
            "engagement": engagement,
            "top_pipeline": pipeline[:5],
            "groups": group_summaries,
            "total_users": len(profile_cache._profiles),
        }

        self._data.setdefault("daily_summaries", []).append(summary)
        # Keep last 90 days
        self._data["daily_summaries"] = self._data["daily_summaries"][-90:]
        self._save()

        log.info("Daily summary: %d users, %d groups, trend=%s",
                 summary["total_users"], len(group_summaries),
                 engagement.get("trend", "?"))

        return summary

    # -- API endpoint data --------------------------------------------------

    def get_dashboard_data(self, social_graph, reputation_tracker, growth_engine, profile_cache) -> dict:
        """Return all analytics data for the social map dashboard."""
        return {
            "growth": growth_engine.get_stats(),
            "engagement": self.get_engagement_stats(),
            "pipeline": self.get_advocacy_pipeline(social_graph, profile_cache),
            "user_journeys": dict(self._data.get("user_journeys", {})),
            "daily_summaries": self._data.get("daily_summaries", [])[-7:],
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
analytics = Analytics()

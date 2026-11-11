"""Group events by IDE session and split at lifecycle events or inactivity gaps.

Defaults are a gap over 15 minutes and at least 10 events per coding session.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

from pipeline.preprocessing.canonical_event import CanonicalEvent, CanonicalEventType


# Configuration

# Default inactivity threshold: 15 minutes of no events = new session
DEFAULT_INACTIVITY_THRESHOLD = timedelta(minutes=15)

# Minimum number of events for a session to be considered meaningful
DEFAULT_MIN_SESSION_EVENTS = 10

# If an event has no timestamp, use this epoch as fallback
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


# Segmenter class

class SessionSegmenter:
    """Split canonical IDE events into coding sessions."""

    def __init__(
        self,
        inactivity_threshold_min: float = 15.0,
        min_session_events: int = DEFAULT_MIN_SESSION_EVENTS,
    ) -> None:
        """
        Args:
            inactivity_threshold_min: Gap in minutes that defines a session boundary.
            min_session_events: Sessions with fewer events are discarded.
        """
        self.inactivity_threshold = timedelta(minutes=inactivity_threshold_min)
        self.min_session_events = min_session_events

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(
        self,
        events: list[CanonicalEvent],
    ) -> list[list[CanonicalEvent]]:
        """Return coding sessions from events sorted by session ID and timestamp."""
        # Pass 1: Group by IDE session ID
        groups = self._group_by_session_id(events)

        # Pass 2: Split each group at inactivity boundaries
        all_sessions: list[list[CanonicalEvent]] = []
        for session_id, group in groups.items():
            sub_sessions = self._split_by_inactivity(group)
            all_sessions.extend(sub_sessions)

        # Pass 3: Discard sessions that are too short
        all_sessions = [
            s for s in all_sessions
            if len(s) >= self.min_session_events
        ]

        # Sort sessions by their start time
        all_sessions.sort(key=lambda s: s[0].timestamp)

        return all_sessions

    @staticmethod
    def sort_events(events: list[CanonicalEvent]) -> list[CanonicalEvent]:
        """Sort by session ID and timestamp before segmentation."""
        return sorted(events, key=lambda e: (e.session_id, e.timestamp))

    def get_session_stats(
        self,
        sessions: list[list[CanonicalEvent]],
    ) -> dict[str, object]:
        """Summarize retained sessions. Short sessions are filtered during segmentation."""
        import statistics

        if not sessions:
            return {
                "total_sessions": 0,
                "total_events": 0,
                "avg_events_per_session": 0.0,
                "median_events_per_session": 0.0,
                "avg_duration_minutes": 0.0,
                "median_duration_minutes": 0.0,
            }

        event_counts = [len(s) for s in sessions]
        durations_min = [
            (s[-1].timestamp - s[0].timestamp).total_seconds() / 60.0
            for s in sessions
            if len(s) >= 2
        ]

        return {
            "total_sessions": len(sessions),
            "total_events": sum(event_counts),
            "avg_events_per_session": statistics.mean(event_counts),
            "median_events_per_session": statistics.median(event_counts),
            "avg_duration_minutes": statistics.mean(durations_min) if durations_min else 0.0,
            "median_duration_minutes": statistics.median(durations_min) if durations_min else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal: grouping and splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_session_id(
        events: list[CanonicalEvent],
    ) -> dict[str, list[CanonicalEvent]]:
        """
        Group events by their session_id field.

        Events are assumed to be pre-sorted by timestamp within each group.
        """
        groups: dict[str, list[CanonicalEvent]] = {}
        for event in events:
            sid = event.session_id
            if sid not in groups:
                groups[sid] = []
            groups[sid].append(event)
        return groups

    def _split_by_inactivity(
        self,
        events: list[CanonicalEvent],
    ) -> list[list[CanonicalEvent]]:
        """Split at lifecycle events and gaps above the inactivity threshold."""
        if not events:
            return []

        sessions: list[list[CanonicalEvent]] = []
        current_session: list[CanonicalEvent] = []

        for i, event in enumerate(events):
            # SESSION_START always begins a new session
            if event.canonical_type == CanonicalEventType.SESSION_START:
                if current_session:
                    sessions.append(current_session)
                current_session = [event]
                continue

            # Check for inactivity gap with the previous event
            if current_session:
                prev_ts = current_session[-1].timestamp
                curr_ts = event.timestamp

                # Use UTC timestamps; fall back to epoch if either is missing
                prev_dt = prev_ts if prev_ts else _EPOCH
                curr_dt = curr_ts if curr_ts else _EPOCH

                gap = curr_dt - prev_dt
                if gap > self.inactivity_threshold:
                    # Gap too large -> start a new session
                    sessions.append(current_session)
                    current_session = []

            current_session.append(event)

            # SESSION_END marks the end of the current session
            if event.canonical_type == CanonicalEventType.SESSION_END:
                sessions.append(current_session)
                current_session = []

        # Don't forget the last session
        if current_session:
            sessions.append(current_session)

        return sessions


# Convenience function for one-shot usage

def segment_events(
    events: list[CanonicalEvent],
    inactivity_min: float = 15.0,
    min_events: int = DEFAULT_MIN_SESSION_EVENTS,
) -> list[list[CanonicalEvent]]:
    """Sort and segment events with the supplied inactivity and size limits."""
    segmenter = SessionSegmenter(
        inactivity_threshold_min=inactivity_min,
        min_session_events=min_events,
    )
    sorted_events = segmenter.sort_events(events)
    return segmenter.segment(sorted_events)


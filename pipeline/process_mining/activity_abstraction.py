"""Event abstraction from raw canonical events to activity bursts."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from pipeline.process_mining.config import ProcessMiningConfig


ACTION_TO_ACTIVITY: dict[str, str] = {
    "editing": "EDIT_BURST",
    "navigation": "NAVIGATION_BURST",
    "file_context": "NAVIGATION_BURST",
    "search": "SEARCH_SESSION",
    "completion": "CODE_COMPLETION",
    "build": "BUILD",
    "test": "TEST_RUN",
    "debug": "DEBUG_SESSION",
    "vcs": "VCS_ACTIVITY",
    "command_other": "IDE_COMMAND_ACTIVITY",
    "inactivity": "INACTIVITY",
    "system": "SYSTEM_ACTIVITY",
    "lifecycle": "LIFECYCLE",
}

NON_BURST_ACTIVITIES = {
    "BUILD",
    "TEST_RUN",
    "VCS_ACTIVITY",
    "LIFECYCLE",
    "INACTIVITY",
}


def infer_activity_label(event_row: pd.Series) -> str:
    """Map one canonical event row to the activity vocabulary used for mining."""
    activity = ACTION_TO_ACTIVITY.get(str(event_row.get("action_type")), "OTHER_ACTIVITY")

    # Split VCS into commit/sync because it often behaves like a task boundary.
    if activity == "VCS_ACTIVITY":
        vcs_action = str(event_row.get("vcs_action") or "").lower()
        if vcs_action == "commit":
            return "VCS_COMMIT"
        if vcs_action in {"push", "pull", "merge", "branch", "stash", "revert"}:
            return "VCS_SYNC"
    return activity


def _seconds_between(left: pd.Timestamp, right: pd.Timestamp) -> float:
    return max(0.0, (right - left).total_seconds())


def _dominant(values: list[str]) -> str:
    values = [value for value in values if value]
    if not values:
        return ""
    return Counter(values).most_common(1)[0][0]


def _field_counts(events: list[dict[str, Any]], field: str) -> dict[str, int]:
    values: list[str] = []
    for event in events:
        value = event.get(field)
        if value is None or value == "":
            continue
        values.append(str(value))
    return dict(Counter(values))


def _build_activity_record(
    session_index: int,
    activity_index: int,
    label: str,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create one activity-burst row and preserve source event provenance."""
    start_time = min(event["timestamp"] for event in events)
    end_time = max(event["timestamp"] for event in events)
    event_types = Counter(str(event["canonical_type"]) for event in events)
    action_types = Counter(str(event["action_type"]) for event in events)

    return {
        "activity_id": f"s{session_index}_a{activity_index}",
        "session_index": session_index,
        "session_id": str(events[0]["session_id"]),
        "activity_index": activity_index,
        "activity": label,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": _seconds_between(start_time, end_time),
        "event_count": len(events),
        "source_event_types": dict(event_types),
        "source_action_types": dict(action_types),
        "dominant_file": _dominant([str(event.get("active_file", "")) for event in events]),
        "file_count": len({str(event.get("active_file", "")) for event in events if event.get("active_file")}),
        "dominant_symbol": _dominant([str(event.get("active_symbol", "")) for event in events]),
        "file_extension_counts": _field_counts(events, "file_extension"),
        "file_role_counts": _field_counts(events, "file_role"),
        "command_family_counts": _field_counts(events, "command_family"),
        "command_name_counts": _field_counts(events, "command_name"),
        "command_category_counts": _field_counts(events, "command_category"),
        "symbol_kind_counts": _field_counts(events, "symbol_kind"),
        "debug_action_counts": _field_counts(events, "debug_action"),
        "vcs_action_counts": _field_counts(events, "vcs_action"),
        "navigation_kind_counts": _field_counts(events, "navigation_kind"),
        "change_kind_counts": _field_counts(events, "change_kind"),
        "build_success_counts": _field_counts(events, "build_success"),
        "active_window_counts": _field_counts(events, "active_window"),
    }


def build_activity_bursts(events: pd.DataFrame, config: ProcessMiningConfig) -> pd.DataFrame:
    """Group consecutive compatible events into activity bursts."""
    if events.empty:
        return pd.DataFrame()

    event_records = events.to_dict("records")
    bursts: list[dict[str, Any]] = []
    current_label: str | None = None
    current_events: list[dict[str, Any]] = []
    activity_index = 0
    session_index = int(events["session_index"].iloc[0])

    def flush() -> None:
        nonlocal activity_index, current_label, current_events
        if current_label and current_events:
            bursts.append(
                _build_activity_record(
                    session_index=session_index,
                    activity_index=activity_index,
                    label=current_label,
                    events=current_events,
                )
            )
            activity_index += 1
        current_label = None
        current_events = []

    for row in event_records:
        label = infer_activity_label(pd.Series(row))
        timestamp = row["timestamp"]

        if current_label is None:
            current_label = label
            current_events = [row]
            continue

        previous_time = current_events[-1]["timestamp"]
        gap_seconds = _seconds_between(previous_time, timestamp)
        same_label = label == current_label
        can_burst = current_label not in NON_BURST_ACTIVITIES
        within_gap = gap_seconds <= config.burst_gap_seconds
        within_size = len(current_events) < config.max_burst_events

        if same_label and can_burst and within_gap and within_size:
            current_events.append(row)
        else:
            flush()
            current_label = label
            current_events = [row]

    flush()
    return pd.DataFrame(bursts)


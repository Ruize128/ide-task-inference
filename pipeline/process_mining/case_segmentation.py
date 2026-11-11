"""Pseudo-case segmentation for IDE streams without natural case ids."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from pipeline.process_mining.config import ProcessMiningConfig


ACTIVITY_FAMILY: dict[str, str] = {
    "EDIT_BURST": "editing",
    "CODE_COMPLETION": "editing",
    "NAVIGATION_BURST": "navigation",
    "SEARCH_SESSION": "navigation",
    "BUILD": "runtime_feedback",
    "TEST_RUN": "runtime_feedback",
    "DEBUG_SESSION": "runtime_feedback",
    "VCS_COMMIT": "vcs",
    "VCS_SYNC": "vcs",
    "VCS_ACTIVITY": "vcs",
    "IDE_COMMAND_ACTIVITY": "workspace",
    "SYSTEM_ACTIVITY": "system",
    "LIFECYCLE": "system",
    "INACTIVITY": "system",
}


def activity_family(activity: str) -> str:
    return ACTIVITY_FAMILY.get(activity, "other")


def _gap_seconds(previous: dict[str, Any], current: dict[str, Any]) -> float:
    return max(0.0, (current["start_time"] - previous["end_time"]).total_seconds())


def _new_case_record(
    case_index: int,
    session_index: int,
    activities: list[dict[str, Any]],
    start_reason: str,
    close_reason: str,
) -> dict[str, Any]:
    """Create one pseudo-case while preserving activity and event provenance."""
    activity_ids = [str(activity["activity_id"]) for activity in activities]
    activity_start_times = [activity["start_time"] for activity in activities]
    activity_end_times = [activity["end_time"] for activity in activities]
    activity_event_counts = [int(activity["event_count"]) for activity in activities]
    source_event_ids: list[str] = []
    for activity in activities:
        source_event_ids.extend(activity.get("source_event_ids", []))

    labels = [str(activity["activity"]) for activity in activities]
    families = [activity_family(label) for label in labels]
    start_time = min(activity["start_time"] for activity in activities)
    end_time = max(activity["end_time"] for activity in activities)
    family_counts = Counter(families)

    return {
        "case_id": f"s{session_index}_c{case_index}",
        "session_index": session_index,
        "session_id": str(activities[0]["session_id"]),
        "case_index": case_index,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": max(0.0, (end_time - start_time).total_seconds()),
        "activity_count": len(activities),
        "event_count": sum(int(activity["event_count"]) for activity in activities),
        "activities": labels,
        "activity_ids": activity_ids,
        "activity_start_times": activity_start_times,
        "activity_end_times": activity_end_times,
        "activity_event_counts": activity_event_counts,
        "source_event_ids": source_event_ids,
        "dominant_family": family_counts.most_common(1)[0][0] if family_counts else "unknown",
        "family_counts": dict(family_counts),
        "start_reason": start_reason,
        "close_reason": close_reason,
    }


def _should_start_new_case(
    current_case: list[dict[str, Any]],
    next_activity: dict[str, Any],
    config: ProcessMiningConfig,
) -> tuple[bool, str]:
    """Return whether the next activity should begin a new pseudo-case."""
    if not current_case:
        return False, ""

    previous = current_case[-1]
    gap = _gap_seconds(previous, next_activity)
    if gap >= config.inactivity_gap_seconds:
        return True, f"inactivity_gap>={int(config.inactivity_gap_seconds)}s"

    if len(current_case) >= config.max_case_activities:
        return True, f"max_case_activities={config.max_case_activities}"

    # VCS commit/sync is often an end-of-work signal. Close the current case
    # after it rather than mixing post-commit navigation into the same trace.
    if (
        str(previous["activity"]) in config.close_case_after_activities
        and len(current_case) >= config.min_case_activities
    ):
        return True, f"after_{previous['activity']}"

    previous_family = activity_family(str(previous["activity"]))
    next_family = activity_family(str(next_activity["activity"]))
    if (
        previous_family != next_family
        and previous_family not in {"system", "other"}
        and next_family not in {"system", "other"}
        and gap >= config.context_shift_gap_seconds
        and len(current_case) >= config.min_case_activities
    ):
        return True, f"context_shift:{previous_family}->{next_family}"

    return False, ""


def segment_pseudo_cases(
    activities: pd.DataFrame,
    config: ProcessMiningConfig,
) -> list[dict[str, Any]]:
    """Split one session's activity stream into task-like pseudo-cases."""
    if activities.empty:
        return []

    records = activities.sort_values(["start_time", "activity_index"]).to_dict("records")
    cases: list[dict[str, Any]] = []
    current_case: list[dict[str, Any]] = []
    start_reason = "session_start"
    session_index = int(records[0]["session_index"])
    case_index = 0

    for activity in records:
        # System/lifecycle events rarely help task pattern discovery. They still
        # remain available through raw provenance if later analysis needs them.
        if activity_family(str(activity["activity"])) == "system":
            continue

        split, reason = _should_start_new_case(current_case, activity, config)
        if split and len(current_case) >= config.min_case_activities:
            cases.append(
                _new_case_record(
                    case_index=case_index,
                    session_index=session_index,
                    activities=current_case,
                    start_reason=start_reason,
                    close_reason=reason,
                )
            )
            case_index += 1
            current_case = []
            start_reason = reason

        current_case.append(activity)

    if len(current_case) >= config.min_case_activities:
        cases.append(
            _new_case_record(
                case_index=case_index,
                session_index=session_index,
                activities=current_case,
                start_reason=start_reason,
                close_reason="session_end",
            )
        )

    return cases

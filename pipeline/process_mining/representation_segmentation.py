"""Data-driven segmentation using reduced activity representations."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pipeline.process_mining.case_segmentation import activity_family

# Export context in trace order for prefix feature extraction.
_CONTEXT_TRACE_FIELDS = (
    ("file_role", "file_role_counts"),
    ("navigation_kind", "navigation_kind_counts"),
    ("debug_action", "debug_action_counts"),
    ("vcs_action", "vcs_action_counts"),
    ("command_category", "command_category_counts"),
)


@dataclass(slots=True)
class RepresentationSegmentationConfig:
    """Parameters for representation-shift segmentation."""

    min_segment_activities: int = 3
    max_segment_activities: int = 60
    context_radius: int = 6
    boundary_quantile: float = 0.94
    min_boundary_distance: float = 0.25
    idle_gap_seconds: float = 30 * 60.0
    close_after_activities: tuple[str, ...] = ("VCS_COMMIT", "VCS_SYNC")
    max_boundary_records: int = 200_000
    # At the length limit, split at the strongest internal behavior change.
    split_capped_segments_at_internal_shift: bool = True


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    similarity = float(np.dot(left, right) / (left_norm * right_norm))
    return float(1.0 - np.clip(similarity, -1.0, 1.0))


def _gap_seconds(previous: dict[str, Any], current: dict[str, Any]) -> float:
    return max(0.0, (pd.Timestamp(current["start_time"]) - pd.Timestamp(previous["end_time"])).total_seconds())


def _update_counter(counter: Counter[str], value: Any) -> None:
    if not isinstance(value, dict):
        return
    for key, count in value.items():
        if key is None or key == "":
            continue
        try:
            numeric = float(count)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            counter[str(key)] += numeric


def _dominant_context_value(counts: Any) -> str:
    """Return the highest-count key of an activity context counter, or ''."""
    if not isinstance(counts, dict):
        return ""
    candidates: list[tuple[float, str]] = []
    for name, count in counts.items():
        if name is None or name == "":
            continue
        try:
            numeric = float(count)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            candidates.append((-numeric, str(name)))
    if not candidates:
        return ""
    return min(candidates)[1]


def _entropy_from_counts(counts: Counter[str]) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    probs = np.array([float(value) / total for value in counts.values() if value > 0], dtype=float)
    return float(-(probs * np.log2(probs)).sum())


def compute_boundary_distances(
    activities: pd.DataFrame,
    embeddings: np.ndarray,
    config: RepresentationSegmentationConfig,
) -> pd.DataFrame:
    """Return boundary distances as a diagnostic table.

    Full-data segmentation uses an array to reduce memory use.
    """
    if activities.empty or len(embeddings) != len(activities):
        return pd.DataFrame()

    sorted_activities = activities.reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for _, group in sorted_activities.groupby("session_index", sort=False):
        indices = group.index.to_list()
        for local_pos in range(1, len(indices)):
            current_index = indices[local_pos]
            previous_index = indices[local_pos - 1]
            left_start = max(0, local_pos - config.context_radius)
            right_end = min(len(indices), local_pos + config.context_radius)
            left_indices = indices[left_start:local_pos]
            right_indices = indices[local_pos:right_end]
            if not left_indices or not right_indices:
                continue
            left_mean = embeddings[left_indices].mean(axis=0)
            right_mean = embeddings[right_indices].mean(axis=0)
            distance = cosine_distance(left_mean, right_mean)
            previous = sorted_activities.loc[previous_index].to_dict()
            current = sorted_activities.loc[current_index].to_dict()
            rows.append(
                {
                    "session_index": int(current["session_index"]),
                    "before_activity_id": str(previous["activity_id"]),
                    "after_activity_id": str(current["activity_id"]),
                    "before_global_index": int(previous_index),
                    "after_global_index": int(current_index),
                    "representation_distance": float(distance),
                    "gap_seconds": float(_gap_seconds(previous, current)),
                    "previous_activity": str(previous["activity"]),
                    "current_activity": str(current["activity"]),
                    "previous_family": activity_family(str(previous["activity"])),
                    "current_family": activity_family(str(current["activity"])),
                }
            )
    return pd.DataFrame(rows)


def boundary_threshold(boundaries: pd.DataFrame, config: RepresentationSegmentationConfig) -> float:
    if boundaries.empty:
        return float("inf")
    quantile_value = float(boundaries["representation_distance"].quantile(config.boundary_quantile))
    return max(config.min_boundary_distance, quantile_value)


def _segment_record(
    session_index: int,
    segment_index: int,
    activities: list[dict[str, Any]],
    start_reason: str,
    close_reason: str,
    boundary_distance: float | None,
) -> dict[str, Any]:
    labels = [str(activity["activity"]) for activity in activities]
    families = [activity_family(label) for label in labels]
    start_time = min(pd.Timestamp(activity["start_time"]) for activity in activities)
    end_time = max(pd.Timestamp(activity["end_time"]) for activity in activities)
    activity_counts = Counter(labels)
    family_counts = Counter(families)

    file_extensions: Counter[str] = Counter()
    file_roles: Counter[str] = Counter()
    command_families: Counter[str] = Counter()
    command_names: Counter[str] = Counter()
    command_categories: Counter[str] = Counter()
    symbol_kinds: Counter[str] = Counter()
    debug_actions: Counter[str] = Counter()
    vcs_actions: Counter[str] = Counter()
    navigation_kinds: Counter[str] = Counter()
    source_events: Counter[str] = Counter()
    source_actions: Counter[str] = Counter()
    files: list[str] = []
    symbols: list[str] = []

    activity_contexts: list[dict[str, str]] = []
    for activity in activities:
        for counter, key in [
            (file_extensions, "file_extension_counts"),
            (file_roles, "file_role_counts"),
            (command_families, "command_family_counts"),
            (command_names, "command_name_counts"),
            (command_categories, "command_category_counts"),
            (symbol_kinds, "symbol_kind_counts"),
            (debug_actions, "debug_action_counts"),
            (vcs_actions, "vcs_action_counts"),
            (navigation_kinds, "navigation_kind_counts"),
            (source_events, "source_event_types"),
            (source_actions, "source_action_types"),
        ]:
            _update_counter(counter, activity.get(key))
        file_name = str(activity.get("dominant_file") or "")
        symbol = str(activity.get("dominant_symbol") or "")
        if file_name:
            files.append(file_name)
        if symbol:
            symbols.append(symbol)
        context = {
            field: _dominant_context_value(activity.get(source_key))
            for field, source_key in _CONTEXT_TRACE_FIELDS
        }
        activity_contexts.append({field: value for field, value in context.items() if value})

    return {
        "segment_id": f"s{session_index}_seg{segment_index}",
        "session_index": int(session_index),
        "session_id": str(activities[0]["session_id"]),
        "segment_index": int(segment_index),
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": max(0.0, (end_time - start_time).total_seconds()),
        "activity_count": len(activities),
        "event_count": int(sum(int(activity["event_count"]) for activity in activities)),
        "activities": labels,
        "activity_ids": [str(activity["activity_id"]) for activity in activities],
        "activity_counts": dict(activity_counts),
        "family_counts": dict(family_counts),
        "dominant_family": family_counts.most_common(1)[0][0] if family_counts else "unknown",
        "dominant_activity": activity_counts.most_common(1)[0][0] if activity_counts else "unknown",
        "file_count": len(set(files)),
        "file_extension_counts": dict(file_extensions),
        "file_role_counts": dict(file_roles),
        "command_family_counts": dict(command_families),
        "command_name_counts": dict(command_names),
        "command_category_counts": dict(command_categories),
        "symbol_kind_counts": dict(symbol_kinds),
        "debug_action_counts": dict(debug_actions),
        "vcs_action_counts": dict(vcs_actions),
        "navigation_kind_counts": dict(navigation_kinds),
        "source_event_types": dict(source_events),
        "source_action_types": dict(source_actions),
        "unique_file_count": len(set(files)),
        "unique_symbol_count": len(set(symbols)),
        "file_switch_count": sum(1 for left, right in zip(files, files[1:], strict=False) if left != right),
        "symbol_switch_count": sum(1 for left, right in zip(symbols, symbols[1:], strict=False) if left != right),
        "file_context_entropy": _entropy_from_counts(Counter(files)),
        "symbol_context_entropy": _entropy_from_counts(Counter(symbols)),
        "command_context_entropy": _entropy_from_counts(command_families),
        "start_reason": start_reason,
        "close_reason": close_reason,
        "close_boundary_distance": boundary_distance,
        "trace": " -> ".join(labels),
        # Ordered per-activity context, aligned with the trace tokens.
        "context_trace": json.dumps(activity_contexts, separators=(",", ":")),
    }


def _merge_short_session_segments(
    entries: list[dict[str, Any]],
    min_segment_activities: int,
) -> list[dict[str, Any]]:
    """Merge very short segment fragments into adjacent session-local segments."""
    if len(entries) <= 1:
        return entries

    merged: list[dict[str, Any]] = []
    for entry in entries:
        if len(entry["activities"]) < min_segment_activities and merged:
            merged[-1]["activities"].extend(entry["activities"])
            merged[-1]["close_reason"] = entry["close_reason"]
            merged[-1]["boundary_distance"] = entry["boundary_distance"]
        else:
            merged.append(entry)

    if len(merged) > 1 and len(merged[0]["activities"]) < min_segment_activities:
        first = merged.pop(0)
        merged[0]["activities"] = first["activities"] + merged[0]["activities"]
        merged[0]["start_reason"] = first["start_reason"]

    return merged


def segment_activity_streams_by_representation(
    activities: pd.DataFrame,
    embeddings: np.ndarray,
    config: RepresentationSegmentationConfig | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame, float]:
    """Split activity streams where local reduced representations shift."""
    config = config or RepresentationSegmentationConfig()
    if activities.empty:
        return [], pd.DataFrame(), float("inf")

    sorted_activities = activities.sort_values(["session_index", "start_time", "activity_index"]).reset_index(drop=True)
    if len(embeddings) != len(sorted_activities):
        raise ValueError(f"Embedding/activity row mismatch: {len(embeddings):,} != {len(sorted_activities):,}")

    distance_by_after_index = np.zeros(len(sorted_activities), dtype=np.float32)
    distance_values: list[float] = []
    candidate_count = 0

    for _, group in sorted_activities.groupby("session_index", sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        for local_pos in range(1, len(indices)):
            current_index = int(indices[local_pos])
            left_start = max(0, local_pos - config.context_radius)
            right_end = min(len(indices), local_pos + config.context_radius)
            left_indices = indices[left_start:local_pos]
            right_indices = indices[local_pos:right_end]
            if len(left_indices) == 0 or len(right_indices) == 0:
                continue
            distance = cosine_distance(embeddings[left_indices].mean(axis=0), embeddings[right_indices].mean(axis=0))
            distance_by_after_index[current_index] = distance
            distance_values.append(distance)
            candidate_count += 1

    threshold = max(config.min_boundary_distance, float(np.quantile(np.asarray(distance_values, dtype=np.float32), config.boundary_quantile))) if distance_values else float("inf")
    del distance_values

    diagnostic_interval = max(1, candidate_count // max(1, config.max_boundary_records))
    boundary_rows: list[dict[str, Any]] = []
    accepted_count = 0
    boundary_ordinal = 0
    segments: list[dict[str, Any]] = []

    for _, group in sorted_activities.groupby("session_index", sort=False):
        records = group.to_dict("records")
        index_values = group.index.to_list()
        if not records:
            continue
        session_index = int(records[0]["session_index"])
        current_segment: list[dict[str, Any]] = []
        # Boundary distance observed just before each activity in the current
        # segment; index 0 corresponds to the segment-opening activity.
        current_distances: list[float] = []
        session_entries: list[dict[str, Any]] = []
        start_reason = "session_start"

        for local_pos, activity in enumerate(records):
            if not current_segment:
                current_segment.append(activity)
                current_distances.append(0.0)
                continue

            previous = current_segment[-1]
            distance = float(distance_by_after_index[int(index_values[local_pos])])
            gap = _gap_seconds(previous, activity)
            split = False
            internal_split = False
            reason = ""

            if gap >= config.idle_gap_seconds and len(current_segment) >= config.min_segment_activities:
                split = True
                reason = f"long_idle_guard>={int(config.idle_gap_seconds)}s"
            elif len(current_segment) >= config.max_segment_activities:
                split = True
                internal_split = config.split_capped_segments_at_internal_shift
                reason = f"max_segment_activities={config.max_segment_activities}"
            elif str(previous["activity"]) in config.close_after_activities and len(current_segment) >= config.min_segment_activities:
                split = True
                reason = f"after_{previous['activity']}"
            elif distance >= threshold and len(current_segment) >= config.min_segment_activities:
                split = True
                reason = f"representation_shift>=q{config.boundary_quantile:.2f}"

            if split:
                accepted_count += 1

            keep_boundary_row = split or (boundary_ordinal % diagnostic_interval == 0 and len(boundary_rows) < config.max_boundary_records)
            if keep_boundary_row:
                boundary_rows.append(
                    {
                        "session_index": session_index,
                        "before_activity_id": str(previous["activity_id"]),
                        "after_activity_id": str(activity["activity_id"]),
                        "representation_distance": float(distance),
                        "threshold": float(threshold),
                        "gap_seconds": float(gap),
                        "split": bool(split),
                        "reason": reason,
                        "previous_activity": str(previous["activity"]),
                        "current_activity": str(activity["activity"]),
                        "previous_family": activity_family(str(previous["activity"])),
                        "current_family": activity_family(str(activity["activity"])),
                    }
                )
            boundary_ordinal += 1

            if split:
                if internal_split and len(current_segment) > config.min_segment_activities:
                    # Close at the strongest internal representation shift so
                    # the cut lands on a behavioral boundary; the tail stays
                    # open and keeps accumulating activities.
                    eligible = current_distances[config.min_segment_activities:]
                    cut = int(np.argmax(np.asarray(eligible, dtype=np.float32))) + config.min_segment_activities
                    session_entries.append(
                        {
                            "activities": current_segment[:cut],
                            "start_reason": start_reason,
                            "close_reason": reason,
                            "boundary_distance": float(current_distances[cut]),
                        }
                    )
                    current_segment = current_segment[cut:]
                    current_distances = current_distances[cut:]
                else:
                    session_entries.append({"activities": current_segment, "start_reason": start_reason, "close_reason": reason, "boundary_distance": distance})
                    current_segment = []
                    current_distances = []
                start_reason = reason

            current_segment.append(activity)
            current_distances.append(distance)

        if current_segment:
            session_entries.append({"activities": current_segment, "start_reason": start_reason, "close_reason": "session_end", "boundary_distance": None})

        session_entries = _merge_short_session_segments(session_entries, config.min_segment_activities)
        for segment_index, entry in enumerate(session_entries):
            segments.append(
                _segment_record(
                    session_index=session_index,
                    segment_index=segment_index,
                    activities=entry["activities"],
                    start_reason=entry["start_reason"],
                    close_reason=entry["close_reason"],
                    boundary_distance=entry["boundary_distance"],
                )
            )

    boundary_table = pd.DataFrame(boundary_rows)
    boundary_table.attrs["candidate_count"] = int(candidate_count)
    boundary_table.attrs["accepted_count"] = int(accepted_count)
    boundary_table.attrs["diagnostic_interval"] = int(diagnostic_interval)
    return segments, boundary_table, threshold


def segment_summary_table(segments: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for segment in segments:
        rows.append(
            {
                "segment_id": segment["segment_id"],
                "session_index": segment["session_index"],
                "segment_index": segment["segment_index"],
                "activity_count": segment["activity_count"],
                "event_count": segment["event_count"],
                "duration_seconds": segment["duration_seconds"],
                "dominant_family": segment["dominant_family"],
                "dominant_activity": segment["dominant_activity"],
                "file_count": segment.get("file_count", 0),
                "unique_file_count": segment.get("unique_file_count", 0),
                "unique_symbol_count": segment.get("unique_symbol_count", 0),
                "file_context_entropy": segment.get("file_context_entropy", 0.0),
                "command_context_entropy": segment.get("command_context_entropy", 0.0),
                "start_reason": segment["start_reason"],
                "close_reason": segment["close_reason"],
                "close_boundary_distance": segment.get("close_boundary_distance"),
                "trace": segment["trace"],
                "context_trace": segment.get("context_trace", ""),
            }
        )
    return pd.DataFrame(rows)

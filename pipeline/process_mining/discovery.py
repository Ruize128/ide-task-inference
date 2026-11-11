"""Directly-follows and local pattern discovery."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from pipeline.process_mining.config import ProcessMiningConfig


# These heuristics describe process maps. Training uses the manual labels in task_taxonomy.py.
TASK_RULES = [
    (
        "Runtime Feedback",
        "Debugging / Runtime Investigation",
        {"DEBUG_SESSION"},
    ),
    (
        "Runtime Feedback",
        "Build Feedback / Diagnostics",
        {"TEST_RUN", "BUILD"},
    ),
    (
        "Code Modification",
        "Code Editing",
        {"EDIT_BURST"},
    ),
    (
        "Code Understanding / Investigation",
        "Search / Query",
        {"SEARCH_SESSION"},
    ),
    (
        "Code Understanding / Investigation",
        "Code Reading / Navigation",
        {"NAVIGATION_BURST"},
    ),
    (
        "Project Maintenance / Version Control",
        "Version Control",
        {"VCS_COMMIT", "VCS_SYNC"},
    ),
    (
        "Workspace / Context Management",
        "Workspace / Context Management",
        {"IDE_COMMAND_ACTIVITY"},
    ),
]


def discover_dfg(
    cases: list[dict[str, Any]],
    min_edge_count: int = 1,
) -> dict[str, Any]:
    """Discover a directly-follows graph from pseudo-case traces."""
    node_counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()
    start_counts: Counter[str] = Counter()
    end_counts: Counter[str] = Counter()

    for case in cases:
        activities = list(case["activities"])
        if not activities:
            continue
        node_counts.update(activities)
        start_counts[activities[0]] += 1
        end_counts[activities[-1]] += 1
        edge_counts.update(zip(activities, activities[1:], strict=False))

    edges = [
        {"source": src, "target": dst, "count": int(count)}
        for (src, dst), count in edge_counts.items()
        if count >= min_edge_count
    ]
    edges.sort(key=lambda row: (-row["count"], row["source"], row["target"]))

    return {
        "nodes": [{"activity": name, "count": int(count)} for name, count in node_counts.most_common()],
        "edges": edges,
        "starts": dict(start_counts),
        "ends": dict(end_counts),
        "case_count": len(cases),
    }


def frequent_traces(cases: list[dict[str, Any]], top_n: int = 25) -> list[dict[str, Any]]:
    """Return the most common complete pseudo-case traces."""
    counter = Counter(" -> ".join(case["activities"]) for case in cases)
    return [{"trace": trace, "count": int(count)} for trace, count in counter.most_common(top_n)]


def mine_local_patterns(
    cases: list[dict[str, Any]],
    min_length: int = 2,
    max_length: int = 4,
    top_n: int = 40,
) -> list[dict[str, Any]]:
    """Mine frequent contiguous activity n-grams as local process patterns."""
    counts: Counter[tuple[str, ...]] = Counter()
    for case in cases:
        activities = list(case["activities"])
        for length in range(min_length, max_length + 1):
            for idx in range(0, max(0, len(activities) - length + 1)):
                counts[tuple(activities[idx : idx + length])] += 1

    return [
        {"pattern": " -> ".join(pattern), "length": len(pattern), "count": int(count)}
        for pattern, count in counts.most_common(top_n)
    ]


def interpret_task(activity_counts: Counter[str]) -> tuple[str, str, float, list[str]]:
    """Suggest a task description for the process-mining summary."""
    total = max(1, sum(activity_counts.values()))
    evidence: list[str] = []
    best: tuple[str, str, float] | None = None

    for family, subtype, signals in TASK_RULES:
        signal_count = sum(activity_counts.get(signal, 0) for signal in signals)
        if signal_count <= 0:
            continue
        score = signal_count / total
        evidence.append(f"{subtype}: {signal_count}/{total} signal activities")
        if best is None or score > best[2]:
            best = (family, subtype, score)

    if best is None:
        return (
            "Workspace / Context Management",
            "Workspace / Context Management",
            0.25,
            ["No dominant task-specific activity found"],
        )

    confidence = min(0.95, 0.45 + best[2])
    return best[0], best[1], confidence, evidence


def summarize_case_group(
    cases: list[dict[str, Any]],
    config: ProcessMiningConfig,
    group_id: str,
) -> dict[str, Any]:
    """Create a process-pattern summary for one case group or cluster."""
    activity_counts: Counter[str] = Counter()
    for case in cases:
        activity_counts.update(case["activities"])

    family, subtype, confidence, evidence = interpret_task(activity_counts)
    dfg = discover_dfg(cases, min_edge_count=config.min_edge_count)

    return {
        "pattern_id": group_id,
        "case_count": len(cases),
        "activity_count": int(sum(activity_counts.values())),
        "task_family": family,
        "task_subtype": subtype,
        "confidence": round(float(confidence), 4),
        "evidence": evidence,
        "top_activities": [
            {"activity": activity, "count": int(count)}
            for activity, count in activity_counts.most_common(10)
        ],
        "frequent_traces": frequent_traces(cases, top_n=config.top_n_traces),
        "local_patterns": mine_local_patterns(cases, top_n=config.top_n_patterns),
        "dfg": dfg,
    }


def summarize_clusters(
    cases: list[dict[str, Any]],
    clustered_trace_table: pd.DataFrame,
    config: ProcessMiningConfig,
) -> list[dict[str, Any]]:
    """Summarize each trace cluster as a candidate developer task pattern."""
    if clustered_trace_table.empty or "cluster_id" not in clustered_trace_table:
        return [summarize_case_group(cases, config, "all")]

    case_by_id = {case["case_id"]: case for case in cases}
    summaries: list[dict[str, Any]] = [summarize_case_group(cases, config, "all")]

    for cluster_id in sorted(clustered_trace_table["cluster_id"].unique()):
        case_ids = clustered_trace_table.loc[
            clustered_trace_table["cluster_id"] == cluster_id, "case_id"
        ].tolist()
        cluster_cases = [case_by_id[case_id] for case_id in case_ids if case_id in case_by_id]
        if cluster_cases:
            summaries.append(
                summarize_case_group(cluster_cases, config, f"cluster_{int(cluster_id)}")
            )

    return summaries

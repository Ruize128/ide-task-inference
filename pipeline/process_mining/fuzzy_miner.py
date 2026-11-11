"""Build directly-follows graphs and simplify them using significance and correlation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


START_NODE = "__START__"
END_NODE = "__END__"


@dataclass(slots=True)
class FuzzyMinerConfig:
    """Control graph simplification and the retention of local alternatives."""

    min_cases: int = 20
    min_edge_count: int = 2
    max_nodes: int = 14
    max_edges: int = 24
    preserve_top_k: int = 2
    edge_utility_quantile: float = 0.62
    significance_weight: float = 0.65
    correlation_weight: float = 0.35
    label_top_edges: int = 10


def _trace_from_case(case: dict[str, Any]) -> list[str]:
    activities = case.get("activities", [])
    return [str(activity) for activity in activities if str(activity)]


def _edge_correlation(
    count: int,
    source_total_out: int,
    target_total_in: int,
) -> float:
    denominator = sqrt(max(1, source_total_out) * max(1, target_total_in))
    return float(count / denominator) if denominator else 0.0


def discover_fuzzy_process(
    cases: list[dict[str, Any]],
    cluster_id: int | str,
    config: FuzzyMinerConfig | None = None,
) -> dict[str, Any]:
    """Discover a simplified fuzzy directly-follows graph for one cluster."""
    config = config or FuzzyMinerConfig()
    traces = [_trace_from_case(case) for case in cases]
    traces = [trace for trace in traces if trace]

    node_counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()
    start_counts: Counter[str] = Counter()
    end_counts: Counter[str] = Counter()
    position_sum: defaultdict[str, float] = defaultdict(float)
    position_seen: Counter[str] = Counter()
    trace_variants: Counter[tuple[str, ...]] = Counter()

    for trace in traces:
        trace_variants[tuple(trace)] += 1
        node_counts.update(trace)
        start_counts[trace[0]] += 1
        end_counts[trace[-1]] += 1
        edge_counts.update(zip(trace, trace[1:], strict=False))
        denominator = max(1, len(trace) - 1)
        for index, activity in enumerate(trace):
            position_sum[activity] += index / denominator
            position_seen[activity] += 1

    total_activities = sum(node_counts.values())
    total_edges = sum(edge_counts.values())
    max_node_count = max(node_counts.values(), default=1)
    max_edge_count = max(edge_counts.values(), default=1)

    retained_nodes = {activity for activity, _ in node_counts.most_common(config.max_nodes)}
    retained_nodes.update(activity for activity, _ in start_counts.most_common(3))
    retained_nodes.update(activity for activity, _ in end_counts.most_common(3))

    out_totals: Counter[str] = Counter()
    in_totals: Counter[str] = Counter()
    for (source, target), count in edge_counts.items():
        out_totals[source] += count
        in_totals[target] += count

    edge_rows: list[dict[str, Any]] = []
    utilities: list[float] = []
    for (source, target), count in edge_counts.items():
        retained_candidate = source in retained_nodes and target in retained_nodes and count >= config.min_edge_count
        significance = float(count / max_edge_count)
        correlation = _edge_correlation(count, out_totals[source], in_totals[target])
        utility = config.significance_weight * significance + config.correlation_weight * correlation
        if retained_candidate:
            utilities.append(utility)
        edge_rows.append(
            {
                "cluster_id": cluster_id,
                "source": source,
                "target": target,
                "count": int(count),
                "significance": significance,
                "correlation": correlation,
                "utility": utility,
                "retained": False,
            }
        )

    threshold = float(np.quantile(np.asarray(utilities, dtype=float), config.edge_utility_quantile)) if utilities else float("inf")
    edge_frame = pd.DataFrame(edge_rows)
    retained_edge_keys: set[tuple[str, str]] = set()

    if not edge_frame.empty:
        candidate_frame = edge_frame.loc[
            (edge_frame["count"] >= config.min_edge_count)
            & (edge_frame["source"].isin(retained_nodes))
            & (edge_frame["target"].isin(retained_nodes))
        ].copy()
        for row in candidate_frame.itertuples(index=False):
            if float(row.utility) >= threshold:
                retained_edge_keys.add((str(row.source), str(row.target)))

        # Keep local best alternatives around every activity. This follows the
        # Fuzzy Miner goal of simplifying the graph without erasing branches.
        for _, group in candidate_frame.sort_values(
            ["source", "utility", "count"], ascending=[True, False, False]
        ).groupby("source", sort=False):
            for row in group.head(config.preserve_top_k).itertuples(index=False):
                retained_edge_keys.add((str(row.source), str(row.target)))
        for _, group in candidate_frame.sort_values(
            ["target", "utility", "count"], ascending=[True, False, False]
        ).groupby("target", sort=False):
            for row in group.head(config.preserve_top_k).itertuples(index=False):
                retained_edge_keys.add((str(row.source), str(row.target)))

        ordered_retained = sorted(
            retained_edge_keys,
            key=lambda key: (
                -float(candidate_frame.loc[
                    (candidate_frame["source"] == key[0]) & (candidate_frame["target"] == key[1]),
                    "utility",
                ].max()),
                -int(candidate_frame.loc[
                    (candidate_frame["source"] == key[0]) & (candidate_frame["target"] == key[1]),
                    "count",
                ].max()),
                key[0],
                key[1],
            ),
        )
        retained_edge_keys = set(ordered_retained[: config.max_edges])
        edge_frame["retained"] = [
            (str(row.source), str(row.target)) in retained_edge_keys
            for row in edge_frame.itertuples(index=False)
        ]

    retained_nodes_from_edges = set()
    for source, target in retained_edge_keys:
        retained_nodes_from_edges.add(source)
        retained_nodes_from_edges.add(target)
    if retained_nodes_from_edges:
        retained_nodes = retained_nodes_from_edges | set(start_counts) | set(end_counts)

    node_rows = []
    for activity, count in node_counts.most_common():
        avg_position = position_sum[activity] / max(1, position_seen[activity])
        node_rows.append(
            {
                "cluster_id": cluster_id,
                "activity": activity,
                "count": int(count),
                "significance": float(count / max_node_count),
                "start_count": int(start_counts.get(activity, 0)),
                "end_count": int(end_counts.get(activity, 0)),
                "avg_position": float(avg_position),
                "retained": activity in retained_nodes,
            }
        )
    node_frame = pd.DataFrame(node_rows)

    retained_edge_frame = edge_frame.loc[edge_frame["retained"]].copy() if not edge_frame.empty else pd.DataFrame()
    retained_edge_coverage = float(retained_edge_frame["count"].sum() / total_edges) if total_edges and not retained_edge_frame.empty else 0.0
    retained_node_count = int(node_frame["retained"].sum()) if not node_frame.empty else 0
    retained_node_coverage = float(node_frame.loc[node_frame["retained"], "count"].sum() / total_activities) if total_activities and not node_frame.empty else 0.0

    summary = {
        "cluster_id": cluster_id,
        "case_count": int(len(traces)),
        "activity_count": int(total_activities),
        "trace_variant_count": int(len(trace_variants)),
        "node_count": int(len(node_frame)),
        "retained_node_count": retained_node_count,
        "edge_count": int(len(edge_frame)),
        "retained_edge_count": int(len(retained_edge_frame)),
        "retained_node_coverage": retained_node_coverage,
        "retained_edge_coverage": retained_edge_coverage,
        "edge_utility_threshold": threshold,
        "top_trace": " -> ".join(trace_variants.most_common(1)[0][0]) if trace_variants else "",
        "top_trace_count": int(trace_variants.most_common(1)[0][1]) if trace_variants else 0,
    }

    return {
        "cluster_id": cluster_id,
        "summary": summary,
        "nodes": node_frame,
        "edges": edge_frame.sort_values(["retained", "utility", "count"], ascending=[False, False, False]) if not edge_frame.empty else edge_frame,
        "starts": dict(start_counts),
        "ends": dict(end_counts),
        "config": config,
    }


def fuzzy_graph_tables(graphs: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert discovered fuzzy graphs into exportable summary/node/edge tables."""
    summaries = pd.DataFrame([graph["summary"] for graph in graphs])
    node_frames = [graph["nodes"] for graph in graphs if not graph["nodes"].empty]
    edge_frames = [graph["edges"] for graph in graphs if not graph["edges"].empty]
    nodes = pd.concat(node_frames, ignore_index=True) if node_frames else pd.DataFrame()
    edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    return summaries, nodes, edges


def _layout_nodes(node_frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Place retained activities in flow-position layers to reduce overlap."""
    retained = node_frame.loc[node_frame["retained"]].copy()
    if retained.empty:
        return {START_NODE: (0.04, 0.0), END_NODE: (0.96, 0.0)}

    retained = retained.sort_values(["avg_position", "count", "activity"], ascending=[True, False, True]).reset_index(drop=True)
    n_nodes = len(retained)
    layer_count = max(2, min(5, int(ceil(sqrt(n_nodes)))))
    positions: dict[str, tuple[float, float]] = {START_NODE: (0.04, 0.0), END_NODE: (0.96, 0.0)}

    retained["layer"] = np.clip(
        np.floor(retained["avg_position"].astype(float) * layer_count).astype(int),
        0,
        layer_count - 1,
    )
    if retained["layer"].nunique() == 1 and n_nodes > 2:
        retained["layer"] = np.linspace(0, layer_count - 1, n_nodes).round().astype(int)

    x_positions = np.linspace(0.18, 0.82, layer_count)
    for layer, group in retained.groupby("layer", sort=True):
        group = group.sort_values(["count", "activity"], ascending=[False, True])
        y_values = np.linspace(0.62, -0.62, len(group)) if len(group) > 1 else np.array([0.0])
        for y, row in zip(y_values, group.itertuples(index=False), strict=False):
            positions[str(row.activity)] = (float(x_positions[int(layer)]), float(y))
    return positions


def _edge_linewidth(count: int, max_edge_count: int) -> float:
    """Scale edge thickness by path frequency, not by node frequency."""
    return 0.7 + 5.2 * sqrt(max(0.0, count / max(1, max_edge_count)))


def _node_size(count: int, max_node_count: int) -> float:
    return 760 + 2150 * sqrt(max(0.0, count / max(1, max_node_count)))


def render_fuzzy_process_graph(
    graph: dict[str, Any],
    output_path: Path,
    title: str | None = None,
) -> Path:
    """Render one simplified fuzzy process graph as a PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = graph["nodes"]
    edges = graph["edges"]
    retained_edges = edges.loc[edges["retained"]].copy() if not edges.empty else pd.DataFrame()
    positions = _layout_nodes(nodes)
    max_node_count = max([1] + ([int(value) for value in nodes.get("count", pd.Series(dtype=int)).tolist()] if not nodes.empty else []))
    max_edge_count = max([1] + ([int(value) for value in retained_edges.get("count", pd.Series(dtype=int)).tolist()] if not retained_edges.empty else []))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.set_axis_off()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.95, 0.95)
    graph_title = title or f"Cluster {graph['cluster_id']} fuzzy process graph"
    ax.set_title(graph_title, fontsize=13, pad=12)

    def draw_arrow(source: str, target: str, count: int, color: str, alpha: float = 0.68, rank: int = 0) -> None:
        if source not in positions or target not in positions:
            return
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        linewidth = _edge_linewidth(count, max_edge_count)
        if source == target:
            loop = FancyArrowPatch(
                (x1 - 0.055, y1 + 0.08),
                (x1 + 0.055, y1 + 0.08),
                connectionstyle="arc3,rad=1.75",
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=linewidth,
                color=color,
                alpha=alpha,
                zorder=1,
            )
            ax.add_patch(loop)
            return
        vertical_delta = abs(y2 - y1)
        same_layer = abs(x2 - x1) < 0.04
        if same_layer:
            rad = 0.35 + 0.08 * (rank % 3)
        else:
            rad = 0.10 + 0.04 * (rank % 4)
            if y1 > y2:
                rad = -rad
            if vertical_delta > 0.55:
                rad *= 0.6
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=linewidth,
            color=color,
            alpha=alpha,
            shrinkA=22,
            shrinkB=22,
            zorder=1,
        )
        ax.add_patch(arrow)

    retained_edges = retained_edges.sort_values(["count", "utility"], ascending=[True, True])
    for rank, row in enumerate(retained_edges.itertuples(index=False)):
        draw_arrow(str(row.source), str(row.target), int(row.count), "#4C78A8", 0.70, rank)

    for activity, count in graph["starts"].items():
        if activity in positions:
            draw_arrow(START_NODE, activity, int(count), "#59A14F", 0.50)
    for activity, count in graph["ends"].items():
        if activity in positions:
            draw_arrow(activity, END_NODE, int(count), "#E15759", 0.50)

    label_edges = retained_edges.sort_values(["count", "utility"], ascending=[False, False]).head(graph["config"].label_top_edges)
    for row in label_edges.itertuples(index=False):
        source = str(row.source)
        target = str(row.target)
        if source not in positions or target not in positions:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        if source == target:
            lx, ly = x1, y1 + 0.18
        else:
            lx, ly = (x1 + x2) / 2, (y1 + y2) / 2 + 0.045
        ax.text(
            lx,
            ly,
            str(int(row.count)),
            fontsize=8,
            color="#2F4B7C",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "#D6DEE8", "alpha": 0.86},
            zorder=5,
        )

    retained_nodes = nodes.loc[nodes["retained"]].copy() if not nodes.empty else pd.DataFrame()
    for special, label, color in [(START_NODE, "START", "#59A14F"), (END_NODE, "END", "#E15759")]:
        x, y = positions[special]
        ax.scatter([x], [y], s=620, color=color, alpha=0.96, edgecolor="white", linewidth=1.4, zorder=3)
        ax.text(x, y, label, fontsize=9, color="white", weight="bold", ha="center", va="center", zorder=4)

    for row in retained_nodes.itertuples(index=False):
        activity = str(row.activity)
        x, y = positions[activity]
        ax.scatter(
            [x],
            [y],
            s=_node_size(int(row.count), max_node_count),
            color="#F28E2B",
            alpha=0.96,
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        label = activity.replace("_", "\n")
        ax.text(x, y, label, fontsize=8.3, color="black", ha="center", va="center", zorder=4)

    summary = graph["summary"]
    footer = (
        f"cases={summary['case_count']}, variants={summary['trace_variant_count']}, "
        f"retained edges={summary['retained_edge_count']}/{summary['edge_count']}, "
        f"edge coverage={summary['retained_edge_coverage']:.2f}"
    )
    ax.text(0.5, -0.9, footer, fontsize=9, color="#555555", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=165)
    plt.close(fig)
    return output_path


def render_fuzzy_process_contact_sheet(
    diagram_table: pd.DataFrame,
    output_path: Path,
    columns: int = 2,
    thumbnail_width: int = 760,
    padding: int = 28,
    title_height: int = 42,
    include_task_subtype: bool = True,
) -> Path:
    """Render all per-cluster process diagrams into a compact comparison sheet."""
    from PIL import Image, ImageDraw, ImageFont

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if diagram_table.empty:
        Image.new("RGB", (thumbnail_width, title_height + padding), "white").save(output_path)
        return output_path

    rows = diagram_table.sort_values(["case_count", "cluster_id"], ascending=[False, True]).to_dict("records")
    thumbnails: list[tuple[dict[str, Any], Image.Image]] = []
    thumb_height = None
    for row in rows:
        image = Image.open(row["diagram_path"]).convert("RGB")
        scale = thumbnail_width / image.width
        height = int(image.height * scale)
        resized = image.resize((thumbnail_width, height), Image.Resampling.LANCZOS)
        thumb_height = thumb_height or height
        thumbnails.append((row, resized))

    columns = max(1, int(columns))
    grid_rows = ceil(len(thumbnails) / columns)
    cell_width = thumbnail_width + padding
    cell_height = int(thumb_height or 1) + title_height + padding
    sheet_width = columns * cell_width + padding
    sheet_height = grid_rows * cell_height + padding
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for index, (row, image) in enumerate(thumbnails):
        grid_row = index // columns
        grid_col = index % columns
        x = padding + grid_col * cell_width
        y = padding + grid_row * cell_height
        label_parts = [f"Cluster {row['cluster_id']}", f"cases={row['case_count']}"]
        if not include_task_subtype:
            if "cluster_probability_mean" in row and pd.notna(row["cluster_probability_mean"]):
                label_parts.append(f"p={float(row['cluster_probability_mean']):.2f}")
            if "median_activity_count" in row and pd.notna(row["median_activity_count"]):
                label_parts.append(f"med_acts={float(row['median_activity_count']):.1f}")
        if include_task_subtype and row.get("task_subtype"):
            label_parts.append(str(row["task_subtype"]))
        label = " | ".join(label_parts)
        draw.text((x, y), label, fill=(40, 40, 40), font=font)
        sheet.paste(image, (x, y + title_height))

    sheet.save(output_path)
    return output_path

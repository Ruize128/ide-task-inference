"""Render paper process maps from saved aggregate fuzzy-process tables.

Defaults to reference clusters 5 and 21. Does not run clustering or training.
"""
from __future__ import annotations
import argparse
from math import sqrt
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project_paths import get_paths
PATHS = get_paths()
OUTPUT_DIR = PATHS.reporting_figures / "process_maps"


ACTIVITY_LABELS = {
    "EDIT_BURST": "Edit",
    "CODE_COMPLETION": "Complete",
    "NAVIGATION_BURST": "Navigate",
    "SEARCH_SESSION": "Search",
    "BUILD": "Build",
    "TEST_RUN": "Test",
    "DEBUG_SESSION": "Debug",
    "VCS_COMMIT": "VCS\ncommit",
    "VCS_SYNC": "VCS\nsync",
    "VCS_ACTIVITY": "VCS",
    "IDE_COMMAND_ACTIVITY": "IDE\ncommand",
    "SYSTEM_ACTIVITY": "System",
    "LIFECYCLE": "Lifecycle",
    "INACTIVITY": "Inactive",
}

def _activity_label(activity: str) -> str:
    return ACTIVITY_LABELS.get(activity, activity.replace("_", "\n").title())

def _save_png(
    fig: plt.Figure,
    filename: str,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Save a process-map figure at publication resolution."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    plt.close(fig)
    return output_path

def _cluster_rows(frame: pd.DataFrame, cluster_id: int) -> pd.DataFrame:
    cluster_ids = pd.to_numeric(frame["cluster_id"], errors="coerce")
    return frame.loc[cluster_ids == cluster_id].copy()

def _select_process_content(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    max_nodes: int = 9,
    max_edges: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    retained_nodes = nodes.loc[nodes["retained"].fillna(False).astype(bool)].copy()
    retained_edges = edges.loc[edges["retained"].fillna(False).astype(bool)].copy()

    retained_nodes["visual_score"] = (
        pd.to_numeric(retained_nodes["count"], errors="coerce").fillna(0)
        + 2.0 * pd.to_numeric(retained_nodes["start_count"], errors="coerce").fillna(0)
        + 2.0 * pd.to_numeric(retained_nodes["end_count"], errors="coerce").fillna(0)
    )
    retained_nodes = retained_nodes.sort_values(
        ["visual_score", "count", "activity"],
        ascending=[False, False, True],
    ).head(max_nodes)
    visible_activities = set(retained_nodes["activity"].astype(str))

    retained_edges = retained_edges.loc[
        retained_edges["source"].astype(str).isin(visible_activities)
        & retained_edges["target"].astype(str).isin(visible_activities)
    ]
    retained_edges = retained_edges.sort_values(
        ["utility", "count", "source", "target"],
        ascending=[False, False, True, True],
    ).head(max_edges)
    return retained_nodes, retained_edges

def _compact_process_positions(nodes: pd.DataFrame) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {
        "__START__": (0.08, 0.0),
        "__END__": (0.92, 0.0),
    }
    if nodes.empty:
        return positions

    arranged = nodes.sort_values(
        ["avg_position", "count", "activity"],
        ascending=[True, False, True],
    ).copy()
    layer_count = min(3, len(arranged))
    if layer_count == 1:
        x_positions = np.array([0.50])
    elif layer_count == 2:
        x_positions = np.array([0.36, 0.64])
    else:
        x_positions = np.array([0.28, 0.50, 0.72])
    for layer, row_indices in enumerate(
        np.array_split(np.arange(len(arranged)), layer_count)
    ):
        group = arranged.iloc[row_indices]
        y_positions = (
            np.linspace(0.52, -0.52, len(group))
            if len(group) > 1
            else np.array([0.0])
        )
        for y_position, row in zip(y_positions, group.itertuples(index=False), strict=True):
            positions[str(row.activity)] = (float(x_positions[layer]), float(y_position))
    return positions

CANONICAL_ACTIVITY_POSITIONS = {
    # Reuse positions across clusters for visual comparison.
    "EDIT_BURST": (0.28, 0.44),
    "CODE_COMPLETION": (0.48, 0.44),
    "BUILD": (0.70, 0.44),
    "TEST_RUN": (0.84, 0.44),
    "DEBUG_SESSION": (0.58, 0.00),
    "VCS_ACTIVITY": (0.78, 0.00),
    "VCS_COMMIT": (0.84, -0.22),
    "LIFECYCLE": (0.16, -0.44),
    "IDE_COMMAND_ACTIVITY": (0.34, -0.44),
    "SEARCH_SESSION": (0.52, -0.44),
    "NAVIGATION_BURST": (0.70, -0.44),
    "VCS_SYNC": (0.86, -0.44),
}

CANONICAL_FALLBACK_POSITIONS = (
    (0.18, 0.00),
    (0.40, 0.00),
    (0.66, 0.00),
    (0.48, -0.22),
)

def _canonical_process_positions(nodes: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Place known activity types in stable positions across all process maps."""
    positions: dict[str, tuple[float, float]] = {
        "__START__": (0.08, 0.0),
        "__END__": (0.92, 0.0),
    }
    unknown_activities: list[str] = []
    for row in nodes.sort_values(["activity", "count"], ascending=[True, False]).itertuples(index=False):
        activity = str(row.activity)
        fixed_position = CANONICAL_ACTIVITY_POSITIONS.get(activity)
        if fixed_position is None:
            unknown_activities.append(activity)
        else:
            positions[activity] = fixed_position

    for index, activity in enumerate(unknown_activities):
        positions[activity] = CANONICAL_FALLBACK_POSITIONS[
            index % len(CANONICAL_FALLBACK_POSITIONS)
        ]
    return positions

def render_process_map(
    cluster_id: int,
    summary_table: pd.DataFrame,
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    *,
    output_dir: Path = OUTPUT_DIR,
    filename: str | None = None,
    figure_size: tuple[float, float] = (3.0, 2.6),
    max_nodes: int = 9,
    max_edges: int = 12,
    node_label_fontsize: float = 7.2,
    terminal_label_fontsize: float = 5.8,
    terminal_node_size: float = 330.0,
    node_size_base: float = 260.0,
    node_size_scale: float = 340.0,
    y_limits: tuple[float, float] = (-0.74, 0.74),
    position_mode: str = "local",
) -> Path:
    """Render one process map from aggregate fuzzy-process tables.

    Canonical layout gives each activity a consistent position across clusters.
    """
    summary_rows = _cluster_rows(summary_table, cluster_id)
    nodes = _cluster_rows(node_table, cluster_id)
    edges = _cluster_rows(edge_table, cluster_id)
    if summary_rows.empty or nodes.empty:
        raise RuntimeError(f"Archived fuzzy graph data is missing for cluster {cluster_id}.")

    nodes, edges = _select_process_content(
        nodes,
        edges,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )
    if position_mode == "local":
        positions = _compact_process_positions(nodes)
    elif position_mode == "canonical":
        positions = _canonical_process_positions(nodes)
    else:
        raise ValueError("position_mode must be 'local' or 'canonical'.")
    max_node_count = max(1, int(pd.to_numeric(nodes["count"]).max()))
    max_edge_count = max(
        [1] + [int(value) for value in pd.to_numeric(edges.get("count", pd.Series(dtype=int))).tolist()]
    )

    fig, ax = plt.subplots(figsize=figure_size)
    ax.set_axis_off()
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(*y_limits)

    def draw_arrow(
        source: str,
        target: str,
        count: int,
        color: str,
        alpha: float,
        rank: int = 0,
    ) -> None:
        if source not in positions or target not in positions:
            return
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        linewidth = 0.45 + 2.0 * sqrt(max(0.0, count / max_edge_count))
        if source == target:
            arrow = FancyArrowPatch(
                (x1 - 0.04, y1 + 0.07),
                (x1 + 0.04, y1 + 0.07),
                connectionstyle="arc3,rad=1.55",
                arrowstyle="-|>",
                mutation_scale=7,
                linewidth=linewidth,
                color=color,
                alpha=alpha,
                zorder=1,
            )
        else:
            same_layer = abs(x2 - x1) < 0.03
            if same_layer:
                radius = 0.28 + 0.06 * (rank % 3)
            else:
                radius = 0.08 + 0.035 * (rank % 4)
                if y1 > y2:
                    radius = -radius
            arrow = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                connectionstyle=f"arc3,rad={radius}",
                arrowstyle="-|>",
                mutation_scale=7,
                linewidth=linewidth,
                color=color,
                alpha=alpha,
                shrinkA=13,
                shrinkB=13,
                zorder=1,
            )
        ax.add_patch(arrow)

    edges_to_draw = edges.sort_values(["count", "utility"], ascending=[True, True])
    for rank, row in enumerate(edges_to_draw.itertuples(index=False)):
        draw_arrow(str(row.source), str(row.target), int(row.count), "#4C78A8", 0.72, rank)

    start_nodes = nodes.loc[pd.to_numeric(nodes["start_count"]) > 0].sort_values(
        "start_count", ascending=False
    ).head(3)
    end_nodes = nodes.loc[pd.to_numeric(nodes["end_count"]) > 0].sort_values(
        "end_count", ascending=False
    ).head(3)
    for row in start_nodes.itertuples(index=False):
        draw_arrow("__START__", str(row.activity), int(row.start_count), "#59A14F", 0.58)
    for row in end_nodes.itertuples(index=False):
        draw_arrow(str(row.activity), "__END__", int(row.end_count), "#E15759", 0.58)

    for special, label, color in (
        ("__START__", "Start", "#59A14F"),
        ("__END__", "End", "#E15759"),
    ):
        x_position, y_position = positions[special]
        ax.scatter(
            [x_position],
            [y_position],
            s=terminal_node_size,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.text(
            x_position,
            y_position,
            label,
            fontsize=terminal_label_fontsize,
            color="white",
            weight="bold",
            ha="center",
            va="center",
            zorder=4,
        )

    for row in nodes.itertuples(index=False):
        activity = str(row.activity)
        x_position, y_position = positions[activity]
        node_size = node_size_base + node_size_scale * sqrt(
            max(0.0, int(row.count) / max_node_count)
        )
        ax.scatter(
            [x_position],
            [y_position],
            s=node_size,
            color="#F28E2B",
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        ax.text(
            x_position,
            y_position,
            _activity_label(activity),
            fontsize=node_label_fontsize,
            color="black",
            ha="center",
            va="center",
            linespacing=0.9,
            zorder=4,
        )

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    return _save_png(
        fig,
        filename or f"cluster_{cluster_id:02d}_fuzzy_process_clean.png",
        output_dir=output_dir,
    )

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=PATHS.paper_reference / 'discovery',
                        help='Directory containing aggregate fuzzy summary, node and edge tables.')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--clusters', type=int, nargs='+', default=[5, 21])
    args = parser.parse_args()
    names = ['context_cluster_fuzzy_summary.parquet', 'context_cluster_fuzzy_nodes.parquet',
             'context_cluster_fuzzy_edges.parquet']
    missing = [str(args.input_dir / n) for n in names if not (args.input_dir / n).is_file()]
    if missing:
        raise FileNotFoundError('Missing aggregate tables: ' + ', '.join(missing))
    summary, nodes, edges = [pd.read_parquet(args.input_dir / n) for n in names]
    for cluster_id in args.clusters:
        path = render_process_map(
            cluster_id, summary, nodes, edges, output_dir=args.output_dir,
            filename=f'cluster_{cluster_id:02d}_process_map.png',
            figure_size=(4.5, 1.8), max_nodes=9, max_edges=12,
            node_label_fontsize=9.0, terminal_label_fontsize=7.2,
            terminal_node_size=360.0, node_size_base=330.0, node_size_scale=430.0,
            y_limits=(-0.70, 0.70), position_mode='canonical',
        )
        print('Generated:', path)

if __name__ == '__main__':
    main()

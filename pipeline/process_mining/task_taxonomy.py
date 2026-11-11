"""Store the manual cluster interpretations used as classifier labels.

The discovery notebook exports this mapping. The training notebook loads it
or uses these definitions directly when the export is missing. Review the
cluster evidence after rerunning discovery, since IDs are specific to a run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TAXONOMY_SCHEMA_VERSION = "task_taxonomy_v4_16_subtasks"

# Which 02 run the cluster map below was derived from. Update together with
# RUN_CLUSTER_TO_SUBTASK after each full re-run.
RUN_ID = "full_data_8321_sessions_24_clusters_length_invariant_2026-07-03"


@dataclass(frozen=True)
class Subtask:
    task_family: str
    subtask: str
    description: str


TASK_FAMILIES: list[str] = [
    "Code Understanding / Investigation",
    "Code Modification",
    "Runtime Feedback",
    "Project Maintenance / Version Control",
    "Workspace / Context Management",
]

# Test-file context supports a test-code editing label. Test execution,
# refactoring, and dependency management did not form distinct clusters.
SUBTASKS: list[Subtask] = [
    Subtask(
        "Code Understanding / Investigation",
        "Context / File Navigation",
        "Short or medium NAVIGATION_BURST and IDE-command loops with little editing; moving through files, views, or nearby code context.",
    ),
    Subtask(
        "Code Understanding / Investigation",
        "Symbol / Usage Investigation",
        "Navigation-dominant traces with find-usages-like symbol/context signals plus some edit/completion support.",
    ),
    Subtask(
        "Code Understanding / Investigation",
        "Search / Query Investigation",
        "SEARCH_SESSION-centered investigation with navigation and IDE-command support; querying the codebase rather than following references.",
    ),
    Subtask(
        "Code Understanding / Investigation",
        "Runtime-Aware Investigation",
        "Navigation-dominant investigation where debugger/runtime activity is a major part of the trace.",
    ),
    Subtask(
        "Code Modification",
        "Completion-Assisted Editing",
        "Strong CODE_COMPLETION and EDIT_BURST alternation; active code writing with frequent completion support.",
    ),
    Subtask(
        "Code Modification",
        "Edit-Completion-Navigation Refinement",
        "Repeated interaction among edit, completion, and navigation; iterative refinement across nearby locations or related symbols.",
    ),
    Subtask(
        "Code Modification",
        "Command-Assisted Editing",
        "Edit-heavy traces with strong IDE-command participation.",
    ),
    Subtask(
        "Code Modification",
        "Test-Code Editing / Refinement",
        "Edit/navigation/completion loops distinguished by test-file context (file_role=test); writing or maintaining test code.",
    ),
    Subtask(
        "Code Modification",
        "Debug-Guided Edit Repair",
        "EDIT_BURST and DEBUG_SESSION repeatedly interact; code changes driven by runtime/debug feedback.",
    ),
    Subtask(
        "Runtime Feedback",
        "Build Feedback / Diagnostics",
        "BUILD-centered feedback connected to navigation, edit, and IDE-command activity.",
    ),
    Subtask(
        "Runtime Feedback",
        "Tight Debugger Interaction",
        "Repeated DEBUG_SESSION activity with IDE-command control; stepping, inspecting, or controlling debugger state.",
    ),
    Subtask(
        "Runtime Feedback",
        "Debug-Edit-Navigate Runtime Investigation",
        "Broad loop among debugging, editing, and navigation; runtime investigation and repair cycles.",
    ),
    Subtask(
        "Project Maintenance / Version Control",
        "VCS Inspection / Sync Maintenance",
        "VCS_ACTIVITY-dominant status or change-list inspection, typically closing with a sync; this run shows one combined VCS behavior group.",
    ),
    Subtask(
        "Workspace / Context Management",
        "General IDE Tool / Command Operation",
        "IDE-command and navigation loops with little direct editing; IDE tools, panels, or command-driven workspace movement.",
    ),
    Subtask(
        "Workspace / Context Management",
        "Workspace-Assisted Editing / Context Switching",
        "IDE commands alternate with edit/completion/navigation; workspace operations supporting active coding or context switching.",
    ),
    Subtask(
        "Workspace / Context Management",
        "Command-Driven Debug Interaction",
        "IDE-command and debug activity tightly connected; command-driven interaction with a running debug session rather than direct code repair.",
    ),
]

SUBTASK_BY_NAME: dict[str, Subtask] = {item.subtask: item for item in SUBTASKS}
SUBTASK_TO_FAMILY: dict[str, str] = {item.subtask: item.task_family for item in SUBTASKS}

# Cluster map for the run named in RUN_ID (24 non-noise clusters, derived from
# activity shares, top-lift features, and fuzzy top traces of that run).
RUN_CLUSTER_TO_SUBTASK: dict[int, str] = {
    22: "Context / File Navigation",              # NAV48/IDE41, N->I->N loop, low membership prob
    23: "Symbol / Usage Investigation",           # NAV46/IDE27/EDIT20, find_usages on source files
    10: "Search / Query Investigation",           # SEARCH32/NAV22/IDE18, I->N->SEARCH
    12: "Runtime-Aware Investigation",            # NAV39/DEBUG21/IDE19, N->I->DEBUG
    1: "Completion-Assisted Editing",             # COMP45/EDIT36, C->E->C loop
    2: "Edit-Completion-Navigation Refinement",   # COMP42/NAV28/EDIT25, C->N->C
    9: "Edit-Completion-Navigation Refinement",   # NAV40/COMP27/EDIT23, N->E->C, find_usages
    3: "Command-Assisted Editing",                # EDIT44/IDE32, E->I->E in .xaml/UI files
    6: "Command-Assisted Editing",                # EDIT45/IDE23, E->I->E core editing loop (largest cluster)
    5: "Test-Code Editing / Refinement",          # EDIT37/NAV26/COMP17, file_role=test
    8: "Test-Code Editing / Refinement",          # NAV38/EDIT24/COMP23, file_role=test, editing-family dominant
    21: "Test-Code Editing / Refinement",         # NAV41/IDE23/EDIT21, file_role=test, N->C->N (navigation-lean)
    4: "Debug-Guided Edit Repair",                # EDIT38/DEBUG25, E->I->D, edit-led repair
    15: "Debug-Guided Edit Repair",               # DEBUG53/EDIT28, D->E->D, debug-led repair
    11: "Build Feedback / Diagnostics",           # BUILD48/NAV21/EDIT16, BUILD->I->N
    16: "Tight Debugger Interaction",             # DEBUG74/IDE24, D->D->D, very long durations
    14: "Debug-Edit-Navigate Runtime Investigation",  # DEBUG41/NAV25/IDE16/EDIT15, D->N->D
    0: "VCS Inspection / Sync Maintenance",       # VCS_ACTIVITY84/VCS_SYNC16, V->V->V->SYNC
    18: "General IDE Tool / Command Operation",   # IDE99, pure command bursts
    19: "General IDE Tool / Command Operation",   # IDE51/NAV46, I->N->I, no editing
    7: "Workspace-Assisted Editing / Context Switching",   # IDE37/EDIT31/COMP16, I->E->C
    17: "Workspace-Assisted Editing / Context Switching",  # IDE46/NAV30/EDIT22, I->N->I with edits
    20: "Workspace-Assisted Editing / Context Switching",  # IDE51/EDIT46, I->E->I
    13: "Command-Driven Debug Interaction",       # IDE44/DEBUG30/NAV14, I->D->I
}


def taxonomy_payload(
    cluster_map: dict[int, str] | None = None,
    run_id: str = RUN_ID,
    observed_cluster_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Build the JSON-serializable taxonomy payload exported by notebook 02."""
    cluster_map = RUN_CLUSTER_TO_SUBTASK if cluster_map is None else cluster_map
    unknown = sorted(subtask for subtask in set(cluster_map.values()) if subtask not in SUBTASK_BY_NAME)
    if unknown:
        raise ValueError(f"Cluster map references subtasks missing from SUBTASKS: {unknown}")

    payload: dict[str, Any] = {
        "schema_version": TAXONOMY_SCHEMA_VERSION,
        "run_id": run_id,
        "task_families": list(TASK_FAMILIES),
        "subtasks": [
            {
                "task_family": item.task_family,
                "subtask": item.subtask,
                "description": item.description,
            }
            for item in SUBTASKS
        ],
        "cluster_to_subtask": {str(cluster_id): subtask for cluster_id, subtask in sorted(cluster_map.items())},
    }
    if observed_cluster_ids is not None:
        observed = sorted(int(value) for value in observed_cluster_ids)
        payload["observed_cluster_ids"] = observed
        payload["unmapped_cluster_ids"] = sorted(set(observed) - set(cluster_map))
        payload["stale_cluster_ids"] = sorted(set(cluster_map) - set(observed))
    return payload


def export_taxonomy(
    path: Path,
    cluster_map: dict[int, str] | None = None,
    run_id: str = RUN_ID,
    observed_cluster_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Write the taxonomy payload to ``path`` and warn about cluster drift."""
    payload = taxonomy_payload(cluster_map=cluster_map, run_id=run_id, observed_cluster_ids=observed_cluster_ids)
    if payload.get("unmapped_cluster_ids"):
        print(
            "WARNING: this run produced non-noise clusters that RUN_CLUSTER_TO_SUBTASK does not map: "
            f"{payload['unmapped_cluster_ids']}. Re-derive the map in process_mining/task_taxonomy.py "
            "from the new cluster evidence before training on these labels."
        )
    if payload.get("stale_cluster_ids"):
        print(
            "WARNING: RUN_CLUSTER_TO_SUBTASK maps cluster ids that this run did not produce: "
            f"{payload['stale_cluster_ids']}. The map likely belongs to an older run."
        )
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_taxonomy(path: Path) -> dict[str, Any]:
    """Load a taxonomy payload previously written by :func:`export_taxonomy`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))

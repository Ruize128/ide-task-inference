"""Configuration for the standalone process-mining pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


from project_paths import get_paths

PATHS = get_paths()
PACKAGE_DIR = Path(__file__).resolve().parent
CODE_DIR = PATHS.root


"""
Central configuration shared by the notebook and command-line runner.
"""
@dataclass(slots=True)
class ProcessMiningConfig:

    base_dir: Path = PATHS.discovery
    sessions_path: Path = PATHS.preprocessing / "sessions.jsonl"
    canonical_events_path: Path = PATHS.preprocessing / "canonical_events.jsonl"

    # The discovery notebook overrides these sampling defaults for the full run.
    max_sessions: int | None = 500
    min_events_per_session: int = 5
    session_offset: int = 0
    session_stride: int = 25
    session_sample_fraction: float | None = None
    session_sample_seed: int = 42

    # Activity abstraction: nearby events with compatible semantics become one
    # higher-level activity burst, which is the main mining granularity.
    burst_gap_seconds: float = 30.0
    max_burst_events: int = 80

    # Pseudo-case segmentation: split long IDE streams into task-like episodes.
    inactivity_gap_seconds: float = 10 * 60.0
    context_shift_gap_seconds: float = 90.0
    min_case_activities: int = 2
    max_case_activities: int = 30
    close_case_after_activities: tuple[str, ...] = (
        "VCS_COMMIT",
        "VCS_SYNC",
    )

    # Discovery and reporting.
    min_edge_count: int = 2
    top_n_traces: int = 25
    top_n_patterns: int = 40
    max_clusters: int = 8
    random_seed: int = 42

    # Scratch paths stay under data/intermediate; paper exports use outputs/.
    data_raw_index_dir: Path = field(init=False)
    data_interim_dir: Path = field(init=False)
    data_processed_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    tables_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.sessions_path = Path(self.sessions_path)
        self.canonical_events_path = Path(self.canonical_events_path)
        self.data_raw_index_dir = self.base_dir / "data" / "raw_index"
        self.data_interim_dir = self.base_dir / "data" / "interim"
        self.data_processed_dir = self.base_dir / "data" / "processed"
        self.outputs_dir = PATHS.discovery_results
        self.figures_dir = self.outputs_dir / "figures"
        self.tables_dir = self.outputs_dir / "tables"
        self.logs_dir = self.outputs_dir / "logs"
        self.artifacts_dir = self.base_dir / "artifacts"

    def ensure_dirs(self) -> None:
        """Create the isolated process-mining output directories."""
        for path in (
            self.data_raw_index_dir,
            self.data_interim_dir,
            self.data_processed_dir,
            self.outputs_dir,
            self.figures_dir,
            self.tables_dir,
            self.logs_dir,
            self.artifacts_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

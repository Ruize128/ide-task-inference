"""Repository-relative paths shared by notebooks, preprocessing and reporting.

Edit configs/paths.json to use external storage. Relative paths resolve from
this repository, never from the shell's current working directory.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw_data: Path
    intermediate: Path
    generated_outputs: Path
    paper_reference: Path

    @property
    def preprocessing(self): return self.intermediate / 'preprocessing'
    @property
    def discovery(self): return self.intermediate / 'discovery'
    @property
    def training(self): return self.intermediate / 'training'
    @property
    def discovery_results(self): return self.generated_outputs / 'discovery'
    @property
    def classification_results(self): return self.generated_outputs / 'classification'
    @property
    def models(self): return self.generated_outputs / 'models' / 'task_classifier'
    @property
    def reporting_figures(self): return self.generated_outputs / 'figures'

    def ensure_runtime_dirs(self):
        for directory in (self.preprocessing, self.discovery, self.training,
                          self.discovery_results, self.classification_results,
                          self.models, self.reporting_figures):
            directory.mkdir(parents=True, exist_ok=True)

def get_paths():
    config = json.loads((ROOT / 'configs' / 'paths.json').read_text(encoding='utf-8'))
    def resolve(key):
        value = Path(config[key]).expanduser()
        return value.resolve() if value.is_absolute() else (ROOT / value).resolve()
    return ProjectPaths(ROOT, *(resolve(k) for k in
                        ('raw_data', 'intermediate', 'generated_outputs', 'paper_reference')))

# Only aggregate research outputs are published from the discovery workspace.
# Segment/boundary/representative tables and context traces remain intermediate.
DISCOVERY_RESULT_FILES = (
    'context_pipeline_summary.json', 'task_taxonomy.json',
    'context_cluster_quality.parquet', 'context_cluster_feature_rankings.parquet',
    'context_cluster_activity_matrix.parquet', 'context_cluster_process_summary.parquet',
    'context_cluster_fuzzy_summary.parquet', 'context_cluster_fuzzy_nodes.parquet',
    'context_cluster_fuzzy_edges.parquet',
)

def export_discovery_results(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in DISCOVERY_RESULT_FILES:
        path = source / name
        if path.is_file():
            shutil.copyfile(path, destination / name)

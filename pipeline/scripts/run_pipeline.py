"""Run the paper notebooks in order; all generated files use project_paths.py."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from project_paths import get_paths

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['preprocess', 'discover', 'train', 'all'])
    parser.add_argument('--max-zips', type=int, help='Optional preprocessing-only subset; not a paper reproduction.')
    args = parser.parse_args()
    paths = get_paths()
    paths.ensure_runtime_dirs()
    phases = ['preprocess', 'discover', 'train'] if args.phase == 'all' else [args.phase]
    for phase in phases:
        if phase == 'preprocess':
            command = [sys.executable, '-m', 'pipeline.preprocessing.run_preprocessing']
            if args.max_zips is not None:
                command += ['--max-zips', str(args.max_zips)]
            subprocess.run(command, cwd=ROOT, check=True)
            continue
        if phase == 'discover' and not (paths.preprocessing / 'sessions.jsonl').is_file():
            raise FileNotFoundError('Run preprocessing first: ' + str(paths.preprocessing / 'sessions.jsonl'))
        if phase == 'train':
            required = ['context_clustered_segments.parquet', 'context_cluster_process_summary.parquet',
                        'context_pipeline_summary.json', 'task_taxonomy.json']
            missing = [str(paths.discovery / n) for n in required if not (paths.discovery / n).is_file()]
            if missing:
                raise FileNotFoundError('Run discovery first. Missing: ' + ', '.join(missing))
        import nbformat
        from nbclient import NotebookClient
        filename = '02_task_discovery.ipynb' if phase == 'discover' else '03_training_classifier.ipynb'
        notebook = nbformat.read(ROOT / 'pipeline' / filename, as_version=4)
        client = NotebookClient(notebook, timeout=None, kernel_name='python3',
                                resources={'metadata': {'path': str(ROOT)}})
        destination = paths.intermediate / 'executed_notebooks' / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            client.execute()
        finally:
            nbformat.write(notebook, destination)
        print('Executed notebook:', destination)

if __name__ == '__main__':
    main()

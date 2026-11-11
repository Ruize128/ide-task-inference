# IDE Task Inference

Discover developer task categories from IDE interaction logs and predict them
from partially observed activity.

## Paper results

The experiment used 8,321 sessions with 11,372,894 canonical events. The pipeline
produced 2,754,487 activity items and 144,118 candidate segments. Manual
interpretation of 24 clusters yielded 16 subtasks and 5 task families.

| Metric | Result |
| --- | ---: |
| Subtask accuracy | 79.7% |
| Subtask top-3 accuracy | 96.3% |
| Task family accuracy | 85.4% |

## Run the pipeline

The three notebooks and their supporting code are in `pipeline/`. The
`preprocessing/` and `process_mining/` subdirectories contain the processing
modules. The `scripts/` subdirectory contains the command-line entry points.

Create and activate a Python 3.10+ environment. Set up the dataset as described
below, then run these commands from the repository root.

    python -m pip install -r requirements.txt
    python -m ipykernel install --user --name python3
    python pipeline/scripts/run_pipeline.py preprocess
    python pipeline/scripts/run_pipeline.py discover
    python pipeline/scripts/run_pipeline.py train

You can also run the three notebooks in order in JupyterLab. In the first
notebook, set `RUN_PIPELINE = True` to run preprocessing.

To draw the paper's process maps from the saved cluster summaries, run:

    python pipeline/scripts/generate_process_maps.py

## Dataset and results

The paper uses the `Events-170301-2` release of
[Enriched Event Streams](https://www.kave.cc/datasets/).
Extract the outer archive to `data/raw/Events-170301-2/`, keeping the participant
ZIP files intact. To use an existing copy elsewhere, set `raw_data` in
`configs/paths.json`.

Saved paper results and models are in `outputs/reference/`. New runs write to
`outputs/runs/`.

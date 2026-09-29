# Engine Health & Degradation Prediction

Team Fast Code AI. A multi-output model on NASA C-MAPSS FD001 that predicts remaining useful life, a health score and a health stage, with a sensor attention layer for explainability.

Who does what, and the schedule: [WORK_DIVISION.md](WORK_DIVISION.md).

Branch rules and the history log: [instructions.md](instructions.md). Read it before your first commit.

## Setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest
```

Run everything from the repo root.

## Layout

| Path | Owner | Contents |
|---|---|---|
| `data/` | Member A | Download, preprocessing, windowing, labels, splits |
| `models/` | Member B | The three model stages, training loop |
| `evaluation/` | Member C | Metrics, plots, attention analysis |
| `config.py` | Shared | Constants: window, RUL cap, thresholds, sensor list |
| `labels.py` | Shared | Health score and health stage, derived from RUL |
| `contracts.py` | Shared | Validators for the three interfaces |
| `fakes.py` | Shared | Fake dataset and fake predictions in the contract formats |
| `tests/` | One file per member | `test_contracts.py` is shared |

## Working without each other's output

`python fakes.py` writes a fake dataset, a fake predictions file and a fake attention array to `artifacts/fake/`. Or import them:

```python
from fakes import fake_dataset, fake_predictions

batch = fake_dataset()          # dict of arrays, X is (N, 30, 14)
df, attn = fake_predictions()   # DataFrame and (N, 30, 14) attention
```

Validate at every handoff:

```python
from contracts import check_batch, check_model_output, check_predictions
```

## Rules

- Work on your own branch: `data`, `models` or `evaluation`. Nobody commits directly to `main`.
- Stay inside your own folder. Add tests as `tests/test_<your area>.py`.
- Shared files change only through a separate pull request that all three see.
- Raw data, arrays and checkpoints are gitignored. Share them outside git.
- Merge to `main` at the sync points: end of day 3 and end of day 5.

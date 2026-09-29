# History

The team's running log. Pull and read this before you implement anything. After you push a change on your own branch, add an entry at the bottom and push it here.

Full rules are in `instructions.md` on the `main` branch.

Entry format:

```
## YYYY-MM-DD | Member | branch

**Changed**
- What you added or changed, with file paths.

**Others need to know**
- Anything that affects another member.

**Commits:** short hashes
```

New entries go at the bottom. Do not edit or delete someone else's entry.

---

## 2026-09-28 | Member B | main

**Changed**
- Added `WORK_DIVISION.md`: who owns what (A data, B models, C evaluation), the three contracts, the day-by-day timeline, the two sync points.

**Others need to know**
- This was the first commit in the repo. Read it alongside the project context document.

**Commits:** 9ede3db

## 2026-09-28 | Member B | main

**Changed**
- Added the project skeleton.
- `config.py`: window 30, RUL cap 125, health exponent 0.7, stage thresholds 100 and 30, the 14 sensor names, raw file column names, predictions file columns.
- `labels.py`: `cap_rul`, `health_score`, `health_stage`, all computed from RUL.
- `contracts.py`: `check_batch`, `check_model_output`, `check_attention`, `check_predictions`.
- `fakes.py`: `fake_dataset()` and `fake_predictions()`. Running `python fakes.py` writes both to `artifacts/fake/`.
- `data/`, `models/`, `evaluation/`: one empty package per member.
- `tests/test_contracts.py`, `pytest.ini`, `requirements.txt`, `.gitignore`, `README.md`.
- `WORK_DIVISION.md`: framework changed from "Proposed: PyTorch" to "PyTorch".

**Others need to know**
- Framework is PyTorch.
- Run everything from the repo root. `pytest` should show 8 passing tests on a fresh clone.
- Member A: use `labels.py` for the three targets instead of rewriting them, and pass your output through `check_batch` before handing it over. Raw NASA files go in `data/raw/`, which is gitignored.
- Member B: `fake_dataset()` gives about 4,500 windows of shape (30, 14) with all three stages present. Three of the 14 fake sensors carry no signal.
- Member C: `fake_predictions()` returns a DataFrame in the predictions file format and an attention array of shape (N, 30, 14). The fake attention favours a different group of sensors in each stage.
- RUL of exactly 100 and exactly 30 both count as WARNING.
- `.npy`, `.npz`, `.pt`, `.pth`, `.ckpt`, `data/raw/` and `artifacts/` are gitignored. Share arrays and checkpoints outside git.
- `config.py`, `labels.py`, `contracts.py` and `fakes.py` are shared. Change them only through a pull request all three see.

**Commits:** 9244ab8

## 2026-09-29 | Member B | main

**Changed**
- Added `instructions.md`: branch rules and the history log workflow.
- `README.md`: added a link to `instructions.md`.

**Others need to know**
- From this point nobody pushes to `main`. Work in `data`, `models` or `evaluation`, and merge at the sync points.
- Keep two folders: your working clone, and a clone of the `history` branch. Setup commands are in `instructions.md`.

**Commits:** ee1f87d

## 2026-09-29 | Member B | history

**Changed**
- Created the `history` branch with this file.
- Added `.gitattributes` so entries added by two people at the same time merge automatically.

**Others need to know**
- This branch holds only the log and is never merged into `main`.
- If your push here is rejected, run `git pull` and push again.

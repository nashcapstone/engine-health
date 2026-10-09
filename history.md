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

## 2026-09-29 | Member B | main

**Changed**
- `instructions.md`: the history folder setup now includes `git config pull.rebase false`, and entries should have a blank line above and below.

**Others need to know**
- Run `git config pull.rebase false` once in your history folder. Without it, `git pull` stops with an error when two people have logged at the same time.

**Commits:** 5ea33e8

## 2026-09-29 | Member A | data

**Changed**
- `data/download.py`: `download()` fetches the NASA C-MAPSS archive and extracts `train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt` into `data/raw/`. It skips files that already exist. Run it with `python -m data.download`.
- `data/preprocess.py`: loads raw files, computes RUL, splits 80/20 by engine (seed 42), fits min-max on the 80 training engines only, builds 30-cycle sliding windows, and labels them with `labels.py`. The test set is the last 30-cycle window per test engine, with RUL taken from `RUL_FD001.txt`. `python -m data.preprocess` writes `train.npz`, `val.npz`, `test.npz` and `scaler.json` to `artifacts/data/`. Every split passes `check_batch`, and a leakage check confirms no engine is in both train and val.
- `data/loaders.py`: `get_dataloaders(batch_size)` returns `{"train", "val", "test"}` PyTorch DataLoaders. Only train is shuffled, and each batch is a dict of tensors in the contract 1 format.
- `tests/test_data.py`: 11 tests. The FD001 test is skipped when `data/raw/` is empty.

**Others need to know**
- Member B: the real data is ready. Run `python -m data.preprocess` once, then use `get_dataloaders()` or `data.preprocess.load_split("train")`. Train is (14070, 30, 14) from 80 engines, val is (3661, 30, 14) from 20 engines, test is (100, 30, 14). The train+val total is 17,731 windows, as expected.
- Stage counts (HEALTHY/WARNING/CRITICAL): train 5992/5678/2400, val 1641/1420/600, test 33/42/25. CRITICAL is about 17% of windows, so consider class weights for the stage head.
- Train X lies in [0, 1]. Val and test can fall slightly outside it (val: -0.03 to 1.03) because the scaler is not refitted on them.
- Test y_rul is capped at 125 like everything else. Test `cycle` is the last recorded cycle of each test engine.
- No shared files changed. `VAL_FRACTION = 0.2` is in `data/preprocess.py`, not `config.py`.

**Commits:** 11908ab

## 2026-09-29 | Member B | models

**Changed**
- Created the `models` branch from `main` and merged `data` into it (fast-forward), so the models train on Member A's dataloaders.
- `models/heads.py`: `MultiHead`, the three output heads. Every model ends in it, so the contract 2 dict is built in one place.
- `models/cnn.py`: `CNNBaseline` (stage 1). `models/cnn_lstm.py`: `CNNLSTM` (stage 2). Both return `attn = None`.
- `models/losses.py`: `MultiTaskLoss` and `stage_class_weights`.
- `models/train.py`: `train`, `evaluate`, `predict`, `make_predictions`, `write_predictions`, `save_checkpoint`, `load_checkpoint`. Run with `python -m models.train --model cnn --data real`.
- `models/__init__.py`: `build_model(name)` with names `cnn` and `cnn_lstm`.
- `tests/test_models.py`: 10 tests. Both models pass the three checks: contract 2 output, finite loss and gradients, overfitting one batch of 32.

**Others need to know**
- First results on real FD001, 30 epochs, untuned. CNN: val RMSE 13.30, test RMSE 13.45, test stage accuracy 0.80. CNN+LSTM: val RMSE 13.34, test RMSE 12.70, test stage accuracy 0.82. The test set is 100 windows, so the gap between the two is within noise.
- Member C: each run writes `best.pt`, `history.csv` and `predictions.csv` to `artifacts/runs/<model>_<data>/`. The predictions are for the test split and pass `check_predictions`. `attn.npy` is written only for models with attention, so neither of these runs has one. `rul_pred` is clipped to [0, 125] and `health_pred` to [0, 100]. `stage_pred` comes from the stage head, not from `rul_pred`, so the two can disagree.
- Member C: `models.train.load_checkpoint(path)` returns the model ready for inference, and `make_predictions(model, batch)` returns the DataFrame and attention for any contract 1 dict.
- Loss: both regression targets are scaled to 0-1 inside the loss, weights are rul 1.0, health 1.0, stage 0.1, and the stage head uses inverse-frequency class weights. These are first guesses, not tuned.
- Member A: no changes to your files. `models/train.py` imports `EngineWindows` from `data/loaders.py` and `load_split` from `data/preprocess.py`.
- No shared files changed.

**Commits:** a34cbbc

## 2026-09-29 | Member C | evaluation

**Changed**
- Created the `evaluation` branch from `models`, so it has the data pipeline and both models.
- `evaluation/metrics.py`: `rmse`, `mae`, `nasa_score`, `stage_accuracy`, `confusion_matrix`, `stage_recall`, `head_agreement`, and `summarize(df)`, which returns every headline metric for one predictions file.
- `evaluation/plots.py`: predicted vs actual RUL, RUL error histogram, stage confusion matrix, health score over life and RUL over life for one engine or a grid of engines.
- `evaluation/report.py`: `python -m evaluation.report cnn_real cnn_lstm_real` writes `metrics.json`, `val_predictions.csv` and the plots to `artifacts/runs/<run>/eval/`, then prints a table comparing the runs.
- `tests/test_evaluation.py`: 12 tests, including hand-checked RMSE, MAE and NASA score values.

**Others need to know**
- Reproduced B's test results exactly. CNN: RMSE 13.45, MAE 9.97, health RMSE 8.34, stage accuracy 0.80. CNN+LSTM: RMSE 12.70, MAE 9.35, health RMSE 7.94, stage accuracy 0.82.
- First NASA scores on the 100-row test split: CNN 334.3, CNN+LSTM 303.0. The score is a sum over rows, not a mean, so compare it only on the same split.
- Both models get about 0.97 recall on HEALTHY and 1.00 on CRITICAL, but only 0.55 (CNN) and 0.60 (CNN+LSTM) on WARNING. Most stage errors are WARNING engines called HEALTHY or CRITICAL.
- Member B: no need to write validation predictions in every training run. The report builds them from `best.pt` with `load_checkpoint` and `make_predictions`. The predictions file format works as is for everything so far.
- Member A: `python -m data.preprocess` hung on the download on this machine (system Python 3.9 on macOS). Fetching the zip with curl into `data/raw/` worked. No change made to `data/`.
- No shared files changed.

**Commits:** d14243b

## 2026-09-29 | Member C | evaluation

**Changed**
- `evaluation/attention.py`: `attention_by_stage` (mean attention per sensor in each health stage), `attention_over_time`, `top_sensors`, `stage_shift` (CRITICAL minus HEALTHY per sensor), and heatmaps: one window, by stage, across the window, and over one engine's life. Built and tested on `fake_predictions()`.
- `evaluation/metrics.py`: `errors_by_stage`, which gives n, RMSE, MAE and bias per true stage.
- `evaluation/plots.py`: `plot_error_by_rul`, which plots bias and RMSE in 10-cycle bins of actual RUL.
- `evaluation/report.py`: adds the error breakdown and `error_by_rul.png`. When a run has `attn.npy`, it also writes the attention plots, `val_attn.npy` and the top sensors per stage. It also accepts a folder path, so `python fakes.py && python -m evaluation.report artifacts/fake --no-val` works.
- `tests/test_evaluation.py`: 6 more tests. The whole suite is 47 passed.

**Others need to know**
- Evaluation of the two real models (day 4 task). RUL error by true stage on the validation split, CNN+LSTM (CNN is similar):
  - HEALTHY: RMSE 12.5, bias -7.5 (predicts too low; true RUL is capped flat at 125).
  - WARNING: RMSE 16.5, bias +7.1 (predicts too late, which the NASA score punishes most).
  - CRITICAL: RMSE 3.7, bias +0.5.
- WARNING is the weak spot for both RUL and stage. The worst bins are true RUL 60 to 100, with RMSE about 19 and bias about +12. Stage recall on WARNING is about 0.55 to 0.60, against 0.97 or more for the other two stages.
- CNN vs CNN+LSTM: CNN+LSTM is slightly better on the 100-row test split (RMSE 12.70 vs 13.45, NASA 303 vs 334), but the two are level on the 3,661-row validation split (13.34 vs 13.30). Treat them as equal for now. Validation was also used to choose `best.pt`, so validation numbers are slightly optimistic.
- Member B: the attention code expects `attn.npy` exactly as described in contract 3, with weights that sum to 1 across sensors. If the weights are multiplied by 14 inside the model, keep the file as the weights that sum to 1. Nothing else is needed from you yet.
- No shared files changed.

**Commits:** 99af80b

## 2026-10-09 | Member C | models

**Changed**
- Member C is now finishing the whole project alone. Each part still goes on its own branch: model work on `models`, evaluation on `evaluation`.
- `models/attention.py`: stage 3. `SensorAttention` scores the 14 sensors at each timestep (a conv over time, then a softmax over sensors). `AttentionCNNLSTM` multiplies the input by the weights times 14, so uniform attention leaves the input unchanged, then runs the CNN+LSTM and the three heads. `attn` in the output is the weights that sum to 1.
- `models/__init__.py`: registered the model as `attention`. Train it with `python -m models.train --model attention --data real`.
- `tests/test_models.py`: 3 attention tests: weights sum to 1, uniform weights leave the input unchanged, and a run writes `attn.npy`. The overfit test now checks the best loss reached, not the last step. The attention model reached 0.0019 by step 400, but a brief Adam spike put step 600 at 0.022.
- 36 tests pass on `models`.

**Others need to know**
- First attention run, 30 epochs, untuned, one seed:

  | Model | Val RMSE | Test RMSE | Test NASA | Test stage acc |
  |---|---|---|---|---|
  | CNN | 13.30 | 13.45 | 334 | 0.80 |
  | CNN+LSTM | 13.34 | 12.70 | 303 | 0.82 |
  | Attention | 12.99 | 12.88 | 302 | 0.81 |

  The gaps are within noise. Whether attention really helps is for the ablation runs with several seeds.
- No shared files changed.

**Commits:** ee7426e

## 2026-10-09 | Member C | evaluation

**Changed**
- Merged `models` into `evaluation`, so the evaluation code can load the attention model.
- `evaluation/attention.py`: the attention-over-life plot keeps about 10 cycle ticks.
- 54 tests pass on `evaluation`.

**Others need to know**
- Attention on the validation split: s14 (corrected core speed) gets the most attention in every stage (about 0.19 against a uniform 0.071). s11 (static pressure at the HPC outlet) rises toward failure: 0.099 HEALTHY, 0.113 WARNING, 0.135 CRITICAL. s15 (bypass ratio) is mostly ignored (0.013 to 0.018).
- The project-context hypothesis that s2 and s7 warn early is not supported so far, since both stay below uniform.
- No shared files changed.

**Commits:** 454927c, 89b6802

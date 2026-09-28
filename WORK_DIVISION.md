WORK DIVISION: Engine Health & Degradation Prediction
Team: Fast Code AI (3 members)
Read alongside the project context document. This covers only who does what and how we work in parallel.

---

THE IDEA

We split by pipeline layer, one owner each:
  Member A: Data
  Member B: Models
  Member C: Evaluation and explainability

Normally B would wait for A's data and C would wait for B's model.
We avoid that by agreeing three interfaces ("contracts") in the first
hour. After that, each member builds against a fake version of the
input they need, and swaps in the real one when it arrives.

Budget: 2 hrs/day each, 7 days.

---

HOUR 1, TOGETHER: FREEZE THE CONTRACTS

Contract 1: Data -> Model
  X         shape (N, 30, 14), float32   30 cycles x 14 sensors
  y_rul     shape (N,), float            capped at 125
  y_health  shape (N,), float            0 to 100
  y_stage   shape (N,), int              0 = HEALTHY, 1 = WARNING, 2 = CRITICAL
  engine_id shape (N,), int
  cycle     shape (N,), int              cycle of the last row in the window

Contract 2: Model -> Evaluation
  Every model (including baselines) returns a dict:
    rul           shape (N,)
    health        shape (N,)
    stage_logits  shape (N, 3)
    attn          shape (N, 30, 14), or None for models without attention

Contract 3: Predictions file (one per training run)
  predictions.csv with columns:
    engine_id, cycle, rul_true, rul_pred, health_true, health_pred,
    stage_true, stage_pred
  attn.npy with shape (N, 30, 14), rows in the same order as the CSV

Also agree in hour 1:
  - Framework: PyTorch.
  - Repo layout and one branch per member (see README.md).
  - One shared config file holding: window = 30, RUL cap = 125,
    health exponent = 0.7, stage thresholds (100 and 30), the 14
    sensor names. Nobody hardcodes these anywhere else.

Changing a contract after hour 1 needs all three to agree.

---

MEMBER A: DATA

Owns:
  - Download FD001 (Kaggle or NASA repository)
  - Drop sensors s1, s5, s6, s10, s16, s18, s19
  - Min-max normalization, fitted on training engines only
  - Sliding windows of 30 cycles
  - The three labels: RUL, health score, health stage
  - Train/validation split
  - Test set: last 30-cycle window per test engine, matched to
    RUL_FD001.txt
  - From day 4: running experiments and ablations, results table

Must get right:
  - Split train/validation BY ENGINE, not by window. Windows from the
    same engine overlap heavily, so a window-level split leaks.
  - Fit the normalizer on training engines only, then apply it to
    validation and test.

Expected size: FD001 has 20,631 training rows, giving about 17,700
windows before the validation split.

Day 1 deliverable for B: rough, unvalidated X.npy and label arrays in
the contract shapes. The careful version follows on day 3.

---

MEMBER B: MODELS

Owns:
  - Stage 1: 1D-CNN baseline
  - Stage 2: CNN + LSTM
  - Stage 3: attention + CNN + LSTM + three output heads
  - Training loop, multi-task loss, checkpointing
  - Writing the predictions file (contract 3)
  - Inference function for the final output card

Works without real data by:
  - Day 1: synthetic tensors in the contract shapes, with a fake
    degradation trend so the loss can actually fall
  - Day 2 onward: A's rough arrays

Correctness checks to pass before real training:
  - Forward pass returns the contract 2 dict with the right shapes
  - Loss computes, backward runs, no NaNs
  - Model can overfit one small batch to near-zero loss
  - Attention weights sum to 1 across the 14 sensors at each timestep

Decisions B owns:
  - Loss weights for the three heads. RUL and health are on different
    scales (0-125 and 0-100) from cross-entropy, so scale the targets
    or weight the losses.
  - Softmax attention weights sum to 1, which shrinks inputs to about
    1/14 of their size. Consider multiplying the weights by 14.

Must do: hand C an early, untuned stage 3 checkpoint by day 5 at the
latest, so C's attention work is not blocked by tuning.

---

MEMBER C: EVALUATION AND EXPLAINABILITY

Owns:
  - Metrics: RMSE, MAE, NASA asymmetric score, with unit tests
  - Plots: health score over engine life, predicted vs actual RUL
  - Attention analysis: average attention per sensor in each health
    stage (HEALTHY, WARNING, CRITICAL)
  - Sensor failure demo: corrupt one sensor, compare the change in
    prediction against that sensor's attention weight
  - Early warning analysis: which sensors get high attention while
    RUL is still high
  - Final output card:
        Engine 42 - Cycle 150
        Health Score   68 / 100
        Stage          WARNING
        Estimated RUL  38 cycles

Works without real predictions by:
  - Generating a fake predictions.csv and attn.npy in the contract 3
    format, and building all plots and analysis against those

Must get right:
  - The sensor patterns in the project context (s2/s7 early, s11/s14/s17
    late) are hypotheses. Report what the trained model actually shows,
    even if it differs.

---

TIMELINE

Day 1
  A: contracts, download, explore data, deliver rough arrays to B
  B: contracts, CNN baseline on synthetic data, correctness checks
  C: contracts, metrics module with unit tests

Day 2
  A: preprocessing, windowing, labels
  B: training loop, first baseline run on rough arrays, start CNN+LSTM
  C: plotting functions on fake predictions

Day 3
  A: deliver final dataloaders with leakage checks        [SYNC 1]
  B: train baseline on final data, write first predictions file
  C: attention heatmap code on fake attention

Day 4
  A: run stage 1 and stage 2 experiments, log results
  B: build stage 3 model
  C: evaluate B's real baseline predictions

Day 5
  A: ablations (with/without attention, with/without stage head,
     loss weights)
  B: tune stage 3, hand early checkpoint to C              [SYNC 2]
  C: sensor failure simulation, early warning analysis

Day 6
  A: results table across all three stages
  B: inference function for the output card
  C: attention-by-stage plots on the real model

Day 7
  All: integration, final results, write-up, presentation

---

SYNC POINTS

Sync 1, end of day 3: A's final data replaces B's rough arrays.
  Check: shapes match contract 1, no engine appears in both train
  and validation.

Sync 2, end of day 5: B's stage 3 model feeds C's analysis.
  Check: predictions file and attn.npy match contract 3.

Daily: a 5-minute message each. What I finished, what I am doing
next, whether I am blocked.

---

RISKS

  - Stage 3 slips: C has nothing real to analyse. Mitigation: the
    early checkpoint on day 5, tuned or not.
  - Attention shows no clear stage pattern: still a valid result.
    The ablation table (does attention improve RMSE?) carries the
    novelty claim either way.
  - Health score adds little: it is a direct function of RUL, so
    most multi-task gain should come from the stage head. The
    ablations will show this.

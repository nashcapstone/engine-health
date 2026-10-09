"""The final output card, from raw sensor readings.

    python -m evaluation.card --engine 42 --cycle 150          one engine, test file
    python -m evaluation.card --engine 42 --split train        a run-to-failure engine
    python -m evaluation.card --fleet                          every test engine, most urgent first

The model is artifacts/runs/attention_real/ unless --run says otherwise.
"""

import argparse
import os

import pandas as pd

import config

STAGE_MARKS = {"HEALTHY": "[ OK ]", "WARNING": "[ ! ]", "CRITICAL": "[!!!]"}


def format_card(result, true_rul=None):
    """The card as text. true_rul, if known, is shown for comparison."""
    lines = [
        f"Engine {result['engine_id']} - Cycle {result['cycle']}",
        f"Health Score   {result['health']:.0f} / 100",
        f"Stage          {result['stage_name']}  {STAGE_MARKS[result['stage_name']]}",
        f"Estimated RUL  {result['rul']:.0f} cycles",
    ]
    if true_rul is not None:
        lines.append(f"Actual RUL     {min(true_rul, config.RUL_CAP):.0f} cycles")
    if result["top_sensors"]:
        watched = ", ".join(f"{s} ({w / (1 / config.N_SENSORS):.1f}x)" for s, w in result["top_sensors"])
        lines.append(f"Watching       {watched}")
    width = max(len(line) for line in lines) + 2
    border = "+" + "-" * width + "+"
    return "\n".join([border] + [f"| {line.ljust(width - 1)}|" for line in lines] + [border])


def fleet_table(predictor, raw, true_rul=None):
    """One row per engine at its last recorded cycle, most urgent (lowest RUL) first."""
    rows = []
    for eid, rows_e in raw.groupby("engine_id", sort=True):
        r = predictor.predict(rows_e)
        rows.append(
            {
                "engine_id": eid,
                "cycle": r["cycle"],
                "health": round(r["health"], 1),
                "stage": r["stage_name"],
                "rul_pred": round(r["rul"], 1),
                "top_sensor": r["top_sensors"][0][0] if r["top_sensors"] else "",
            }
        )
    table = pd.DataFrame(rows)
    if true_rul is not None:
        table["rul_true"] = table["engine_id"].map(true_rul).clip(upper=config.RUL_CAP)
    return table.sort_values("rul_pred").reset_index(drop=True)


def _load_raw(split):
    """Raw rows and, when known, true RUL at each engine's last cycle (a Series by engine)."""
    from data.preprocess import add_test_rul, add_train_rul, load_raw, load_test_rul

    raw = load_raw(split)
    if split == "test":
        raw = add_test_rul(raw, load_test_rul())
    else:
        raw = add_train_rul(raw)
    return raw


if __name__ == "__main__":
    from models.inference import Predictor

    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=int)
    parser.add_argument("--cycle", type=int, help="defaults to the engine's last recorded cycle")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--run", default=os.path.join(config.ARTIFACTS_DIR, "runs", "attention_real"))
    parser.add_argument("--fleet", action="store_true", help="table of every engine instead of one card")
    args = parser.parse_args()
    assert args.fleet or args.engine is not None, "give --engine N or --fleet"

    predictor = Predictor.load(args.run)
    raw = _load_raw(args.split)
    if args.fleet:
        last = raw.sort_values("cycle").groupby("engine_id").tail(1).set_index("engine_id")["rul"]
        table = fleet_table(predictor, raw.drop(columns="rul"), last)
        print(table.to_string(index=False))
        print(f"\n{table['stage'].value_counts().to_dict()} across {len(table)} engines")
    else:
        rows = raw[raw["engine_id"] == args.engine]
        assert len(rows), f"no engine {args.engine} in the {args.split} file"
        result = predictor.predict(rows.drop(columns="rul"), args.cycle)
        true_rul = rows.loc[rows["cycle"] == result["cycle"], "rul"].iloc[0]
        print(format_card(result, true_rul))

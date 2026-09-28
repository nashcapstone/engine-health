"""Shared constants. Frozen after hour 1: change only through a PR all three members see."""

DATASET = "FD001"
SEED = 42

WINDOW = 30
RUL_CAP = 125
HEALTH_EXPONENT = 0.7

# Stage boundaries on RUL: > HEALTHY_ABOVE is HEALTHY, < CRITICAL_BELOW is CRITICAL, else WARNING
HEALTHY_ABOVE = 100
CRITICAL_BELOW = 30
STAGE_NAMES = ("HEALTHY", "WARNING", "CRITICAL")
N_STAGES = len(STAGE_NAMES)

# Near-zero variance in FD001
DROPPED_SENSORS = ("s1", "s5", "s6", "s10", "s16", "s18", "s19")
SENSORS = tuple(f"s{i}" for i in range(1, 22) if f"s{i}" not in DROPPED_SENSORS)
N_SENSORS = len(SENSORS)

RAW_COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

# Both directories are gitignored
RAW_DIR = "data/raw"
ARTIFACTS_DIR = "artifacts"

PREDICTION_COLUMNS = (
    "engine_id",
    "cycle",
    "rul_true",
    "rul_pred",
    "health_true",
    "health_pred",
    "stage_true",
    "stage_pred",
)

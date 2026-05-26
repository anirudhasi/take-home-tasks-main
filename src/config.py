"""
config.py — Central configuration for the CODVO pump failure prediction pipeline.

Every tunable constant lives here. Change here, effects propagate everywhere.
No magic numbers anywhere else in the codebase.
"""
from pathlib import Path

# ─────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
DOCS_DIR  = ROOT / "docs"
ARTIFACTS_DIR = ROOT / "submission" / "model_artifacts"
PREDS_DIR     = ROOT / "submission" / "predictions"

TRAIN_PATH = DATA_DIR / "train.csv"
UNITS_PATH = DATA_DIR / "train_units.csv"
SAMPLE_TEST_PATH = DATA_DIR / "sample_test_input.csv"

# ─────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────
RANDOM_SEED = 42

# ─────────────────────────────────────────────────
# Data schema (update if column names differ from
# what data_dictionary.md specifies)
# ─────────────────────────────────────────────────
UNIT_ID_COL   = "unit_id"
CYCLE_COL     = "cycle"
EVENT_COL     = "event_flag"

# These will be auto-detected from train.csv in data_loader.py
# Override here if you want to fix them explicitly
SENSOR_COLS: list = []        # Populated at runtime
OPERATIONAL_COLS: list = []   # Settings / setpoints (not condition indicators)

# Sentinel values used by SCADA for missing readings
SENTINEL_VALUES = [-999, -9999, 999999]

# ─────────────────────────────────────────────────
# Label engineering
# ─────────────────────────────────────────────────
RUL_MAX = 125                   # Piecewise cap — overridden if auto-selected
AUTO_SELECT_RUL_MAX = True      # Empirically search for best cap
RUL_MAX_SEARCH_RANGE = (50, 201, 5)   # (start, stop, step) in cycles
BASELINE_WINDOW = 20            # First N cycles used to establish healthy baseline
CENSORING_STRATEGY = "exclude"  # Options: "exclude" | "conservative_bound"
CENSORED_WEIGHT = 0.0           # sample_weight for censored rows (0 = excluded)

# ─────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────
ROLL_WINDOWS   = [5, 10, 20, 30]        # Rolling stat window sizes (cycles)
TREND_WINDOWS  = [10, 20, 30]           # OLS slope / curvature windows
EWMA_SPANS     = [0.1, 0.3, 0.5]       # EWMA alpha values
USE_CROSS_SENSOR = True                 # Build physically-motivated interaction features
MAX_NAN_PCT = 0.30                      # Drop features with > 30% NaN after engineering

# ─────────────────────────────────────────────────
# Feature selection
# ─────────────────────────────────────────────────
TOP_N_FEATURES = 150       # Keep top N by XGBoost 'gain' importance
MIN_IMPORTANCE = 0.0001    # Drop features below this threshold

# ─────────────────────────────────────────────────
# Model hyperparameters
# (defaults — Optuna will refine these)
# ─────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":      500,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "min_child_weight":  5,
    "tree_method":       "hist",   # CPU-efficient
    "random_state":      RANDOM_SEED,
    "n_jobs":            -1,
}

LGBM_PARAMS = {
    "n_estimators":      500,
    "num_leaves":        31,
    "max_depth":         -1,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "min_child_samples": 10,
    "random_state":      RANDOM_SEED,
    "n_jobs":            -1,
    "verbose":           -1,
}

ENSEMBLE_XGB_WEIGHT  = 0.5   # Fraction allocated to XGBoost predictions
ENSEMBLE_LGBM_WEIGHT = 0.5   # Must sum to 1 with above
USE_ASYMMETRIC_LOSS  = True  # PHM asymmetric loss for XGBoost

# ─────────────────────────────────────────────────
# Optuna tuning
# ─────────────────────────────────────────────────
OPTUNA_N_TRIALS  = 80
OPTUNA_TIMEOUT   = 3600       # seconds
OPTUNA_DIRECTION = "minimize" # minimise RMSE

# ─────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────
CV_STRATEGY   = "group_kfold"   # "group_kfold" | "logo" (leave-one-group-out)
CV_N_SPLITS   = 5
LAST_N_CYCLES = 5               # Use median of last N cycle preds as unit-level RUL

# ─────────────────────────────────────────────────
# Inference / alerting
# ─────────────────────────────────────────────────
WARNING_THRESHOLDS = {
    "RED":   30,   # Immediate maintenance — failure likely within 30 cycles
    "AMBER": 60,   # Plan maintenance — degrading, 31-60 cycles
    "GREEN": 9999, # Healthy
}
MIN_CYCLES_FOR_PREDICTION = 10   # Need at least 10 cycles to trust features

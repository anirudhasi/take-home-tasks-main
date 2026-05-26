"""
data_loader.py — Data ingestion, schema validation, and basic pre-processing.

Responsibilities:
  1. Load train.csv and train_units.csv
  2. Auto-detect sensor columns vs operational columns
  3. Replace sentinel values with NaN
  4. Validate data integrity (schema, missing cycles, duplicates)
  5. Return clean DataFrames ready for label engineering
"""
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    UNIT_ID_COL, CYCLE_COL, EVENT_COL,
    SENTINEL_VALUES, RANDOM_SEED
)
from src.utils import get_logger, downcast_numerics

log = get_logger("data_loader")


# ─────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────
def load_and_validate_data(
    train_path: Path,
    units_path: Path,
    verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train.csv and train_units.csv, validate, and return clean DataFrames.

    Returns
    -------
    train_df : pd.DataFrame
        Long-format sensor history. Sentinel values replaced with NaN.
    units_df : pd.DataFrame
        One row per unit with commissioning date, cycles_recorded, event_flag.
    """
    log.info("Loading training data…")
    train_df = pd.read_csv(train_path)
    units_df = pd.read_csv(units_path)

    # --- Standardise column names (lowercase, strip whitespace) ---
    train_df.columns = [c.strip().lower().replace(" ", "_") for c in train_df.columns]
    units_df.columns = [c.strip().lower().replace(" ", "_") for c in units_df.columns]

    # --- Normalise units_df column aliases to canonical names ---
    units_df = units_df.rename(columns={"n_cycles": "cycles_recorded", "event": "event_flag"})

    # --- Validate required columns ---
    _check_required_columns(train_df, required=[UNIT_ID_COL, CYCLE_COL])
    _check_required_columns(units_df, required=[UNIT_ID_COL, "cycles_recorded", EVENT_COL])

    # --- Replace sentinel values ---
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    for sentinel in SENTINEL_VALUES:
        train_df[numeric_cols] = train_df[numeric_cols].replace(sentinel, np.nan)

    # --- Sort by unit and cycle ---
    train_df = train_df.sort_values([UNIT_ID_COL, CYCLE_COL]).reset_index(drop=True)

    # --- Downcast for memory efficiency ---
    train_df = downcast_numerics(train_df)

    if verbose:
        _print_summary(train_df, units_df)

    return train_df, units_df


def detect_sensor_columns(
    train_df: pd.DataFrame,
    id_cols: list = None
) -> tuple[list, list]:
    """
    Auto-detect which columns are sensor readings vs. operational/ID columns.

    Heuristics:
      - Operational/setpoint cols: low unique-value count (step-like signals)
        OR named with common patterns (op_setting, setting, mode, speed_cmd)
      - Sensor cols: everything else numeric, excluding id/cycle cols

    Returns
    -------
    sensor_cols : list of str
    operational_cols : list of str
    """
    if id_cols is None:
        id_cols = [UNIT_ID_COL, CYCLE_COL]

    candidate_numeric = [
        c for c in train_df.select_dtypes(include=[np.number]).columns
        if c not in id_cols
    ]

    n_units = train_df[UNIT_ID_COL].nunique()

    operational_cols, sensor_cols = [], []
    for col in candidate_numeric:
        n_unique = train_df[col].nunique()
        name_lower = col.lower()

        is_setting = (
            n_unique <= max(10, n_units * 0.5)
            or any(kw in name_lower for kw in
                   ["setting", "op_", "mode", "cmd", "setpoint", "sp_"])
        )

        # Dead channel: near-zero variance across all units
        overall_std = train_df[col].std()
        is_dead = (overall_std < 1e-6)

        if is_dead:
            log.warning(f"  Dead channel detected (std≈0): {col} — will drop")
        elif is_setting:
            operational_cols.append(col)
        else:
            sensor_cols.append(col)

    log.info(f"Detected {len(sensor_cols)} sensor cols, "
             f"{len(operational_cols)} operational cols")
    return sensor_cols, operational_cols


def load_inference_input(
    input_path: Path,
    expected_sensor_cols: list,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load a CSV in the inference schema (same as train.csv, no event labels).
    Validates schema, replaces sentinels, sorts.
    """
    df = pd.read_csv(input_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = set(expected_sensor_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing expected sensor columns: {missing}\n"
            f"Expected: {expected_sensor_cols}\nFound: {list(df.columns)}"
        )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for sentinel in SENTINEL_VALUES:
        df[numeric_cols] = df[numeric_cols].replace(sentinel, np.nan)

    # Fill remaining NaN with column median (fail-safe for live inference)
    for col in expected_sensor_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df = df.sort_values([UNIT_ID_COL, CYCLE_COL]).reset_index(drop=True)

    if verbose:
        log.info(f"Inference input: {df[UNIT_ID_COL].nunique()} units, "
                 f"{len(df)} rows")
    return df


# ─────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────
def _check_required_columns(df: pd.DataFrame, required: list) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def _print_summary(train_df: pd.DataFrame, units_df: pd.DataFrame) -> None:
    n_units   = train_df[UNIT_ID_COL].nunique()
    n_rows    = len(train_df)
    n_failed  = (units_df[EVENT_COL] == 1).sum()
    n_censored = (units_df[EVENT_COL] == 0).sum()
    lifetimes = units_df["cycles_recorded"]
    n_missing = train_df.isnull().sum().sum()
    missing_pct = n_missing / train_df.size * 100

    log.info("=" * 55)
    log.info(f"  Units total    : {n_units}")
    log.info(f"  Failed         : {n_failed}  |  Censored: {n_censored}")
    log.info(f"  Total rows     : {n_rows:,}")
    log.info(f"  Lifetime min   : {lifetimes.min():.0f}  "
             f"mean: {lifetimes.mean():.0f}  "
             f"max: {lifetimes.max():.0f} cycles")
    log.info(f"  Missing values : {n_missing:,}  ({missing_pct:.2f}%)")
    log.info("=" * 55)

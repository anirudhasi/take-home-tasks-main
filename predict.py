#!/usr/bin/env python3
"""
predict.py — LIVE INFERENCE SCRIPT for CODVO pump RUL prediction.

Usage:
    python predict.py --input data/sample_test_input.csv
    python predict.py --input /path/to/live_test.csv --output results.csv
    python predict.py --input data/sample_test_input.csv --shap  # show SHAP for top unit

Output columns:
    unit_id             — pump identifier
    predicted_rul       — estimated cycles to failure (PRIMARY output)
    failure_prob_30     — P(failure within 30 cycles) from calibration curve
    health_index        — [0=healthy, 1=critical] continuous score
    prediction_std      — std of last-N-cycle predictions (confidence proxy)
    warning_level       — GREEN / AMBER / RED based on predicted_rul
    last_cycle          — last cycle observed in input data
    n_cycles_observed   — total cycles of history provided

Designed to run in < 3 minutes on CPU with no internet access.
All artefacts are loaded locally from submission/model_artifacts/.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── path setup ──
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import (
    ARTIFACTS_DIR, UNIT_ID_COL, CYCLE_COL,
    WARNING_THRESHOLDS, MIN_CYCLES_FOR_PREDICTION, LAST_N_CYCLES
)
from src.data_loader import load_inference_input
from src.utils import get_logger, load_pickle

log = get_logger("predict")


# ─────────────────────────────────────────────────
# Artefact loading
# ─────────────────────────────────────────────────
def load_artefacts():
    """Load all serialised pipeline artefacts. Fails fast with clear errors."""
    required = [
        "ensemble_model.pkl",
        "feature_pipeline.pkl",
        "feature_cols.pkl",
        "train_config.pkl",
    ]
    for fname in required:
        path = ARTIFACTS_DIR / fname
        if not path.exists():
            sys.exit(
                f"\n[ERROR] Artefact not found: {path}\n"
                f"Run 'python train.py' first to generate model artefacts.\n"
            )

    model         = load_pickle(ARTIFACTS_DIR / "ensemble_model.pkl")
    feature_eng   = load_pickle(ARTIFACTS_DIR / "feature_pipeline.pkl")
    feature_cols  = load_pickle(ARTIFACTS_DIR / "feature_cols.pkl")
    train_config  = load_pickle(ARTIFACTS_DIR / "train_config.pkl")

    return model, feature_eng, feature_cols, train_config


# ─────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────
def failure_prob(predicted_rul: float, calibration_params: dict) -> float:
    """
    P(true RUL < 30 | predicted_rul = x) from isotonic calibration curve.
    Falls back to sigmoid approximation if calibration not available.
    """
    if calibration_params and "pred_bins" in calibration_params:
        return float(np.interp(
            predicted_rul,
            calibration_params["pred_bins"],
            calibration_params["true_probs"],
        ))
    # Sigmoid fallback: centred at RUL=30, width=10
    return float(1.0 / (1.0 + np.exp((predicted_rul - 30.0) / 10.0)))


# ─────────────────────────────────────────────────
# Warning level
# ─────────────────────────────────────────────────
def assign_warning(rul: float) -> str:
    if rul <= WARNING_THRESHOLDS["RED"]:
        return "RED"
    if rul <= WARNING_THRESHOLDS["AMBER"]:
        return "AMBER"
    return "GREEN"


# ─────────────────────────────────────────────────
# Main prediction function
# ─────────────────────────────────────────────────
def run_inference(input_path: str, output_path: str, show_shap: bool = False) -> pd.DataFrame:
    t0 = time.perf_counter()

    # 1. Load artefacts
    log.info("Loading artefacts…")
    model, feature_eng, feature_cols, cfg = load_artefacts()
    sensor_cols        = cfg["sensor_cols"]
    rul_max            = cfg["rul_max"]
    calibration_params = cfg.get("calibration_params", {})

    # 2. Load & validate input
    log.info(f"Loading input: {input_path}")
    test_df = load_inference_input(
        Path(input_path),
        expected_sensor_cols=sensor_cols,
    )

    # 3. Feature engineering (transform only — no refit)
    log.info("Engineering features…")
    test_feat = feature_eng.transform(test_df)

    # Guard: ensure all required feature cols exist
    missing_feat = [c for c in feature_cols if c not in test_feat.columns]
    if missing_feat:
        log.warning(f"  {len(missing_feat)} feature cols missing — filling with 0")
        for c in missing_feat:
            test_feat[c] = 0.0

    X = test_feat[feature_cols].fillna(0.0).values

    # 4. Row-level predictions
    log.info("Running predictions…")
    row_preds = model.predict(X)

    # 5. Aggregate to unit level
    results = []
    unit_ids_arr = test_feat[UNIT_ID_COL].values

    for uid in test_df[UNIT_ID_COL].unique():
        unit_mask = unit_ids_arr == uid
        unit_df   = test_feat[unit_mask].sort_values(CYCLE_COL)
        n_cycles  = len(unit_df)

        if n_cycles < MIN_CYCLES_FOR_PREDICTION:
            log.warning(
                f"  Unit {uid}: only {n_cycles} cycles observed — "
                f"prediction may be unreliable."
            )

        # Indices of this unit's rows in the full test array
        unit_row_idx = np.where(unit_mask)[0]
        unit_preds   = row_preds[unit_row_idx]

        last_n  = min(LAST_N_CYCLES, len(unit_preds))
        tail    = unit_preds[-last_n:]
        pred_rul = float(max(0.0, np.median(tail)))
        pred_std = float(np.std(tail)) if last_n > 1 else 0.0

        health  = round(1.0 - min(pred_rul / rul_max, 1.0), 4)
        prob_30 = round(failure_prob(pred_rul, calibration_params), 4)
        warn    = assign_warning(pred_rul)

        results.append({
            UNIT_ID_COL:          uid,
            "predicted_rul":      round(pred_rul, 1),
            "failure_prob_30":    prob_30,
            "health_index":       health,
            "prediction_std":     round(pred_std, 2),
            "warning_level":      warn,
            "last_cycle":         int(unit_df[CYCLE_COL].iloc[-1]),
            "n_cycles_observed":  n_cycles,
        })

    results_df = (
        pd.DataFrame(results)
        .sort_values("predicted_rul")
        .reset_index(drop=True)
    )

    # 6. Enrich with actual RUL when ground truth is available (train.csv)
    results_df = _add_actual_rul(results_df, rul_max)

    # 7. Save predictions + accuracy stats
    elapsed = time.perf_counter() - t0
    _save_with_stats(results_df, output_path, elapsed)

    # 8. Console summary
    log.info("=" * 55)
    log.info(f"Inference complete in {elapsed:.1f}s")
    log.info(f"Results saved -> {output_path}")
    log.info("")
    log.info("Fleet alert summary:")
    for level in ["RED", "AMBER", "GREEN"]:
        count = (results_df["warning_level"] == level).sum()
        log.info(f"  {level:6s}: {count} units")
    log.info("")
    log.info("Top 5 highest-risk units:")
    top5 = results_df.head(5)[
        ["unit_id", "predicted_rul", "failure_prob_30", "warning_level"]
    ]
    log.info("\n" + top5.to_string(index=False))
    log.info("=" * 55)

    # 9. Optional SHAP explanation for highest-risk unit
    if show_shap:
        _shap_explanation(
            model, test_feat, feature_cols, results_df, rul_max
        )

    return results_df


# ─────────────────────────────────────────────────
# Actual RUL lookup (when train.csv is available)
# ─────────────────────────────────────────────────
def _add_actual_rul(results_df: pd.DataFrame, rul_max: float) -> pd.DataFrame:
    """
    Try to join actual remaining-useful-life from train.csv + train_units.csv.
    Adds columns: actual_rul, error, abs_error.
    Silently skips if files are not found or unit is not in training data.
    """
    from src.config import TRAIN_PATH, UNITS_PATH
    if not TRAIN_PATH.exists() or not UNITS_PATH.exists():
        return results_df

    try:
        train_df = pd.read_csv(TRAIN_PATH)
        units_df = pd.read_csv(UNITS_PATH)
        train_df.columns = [c.strip().lower().replace(" ", "_") for c in train_df.columns]
        units_df.columns = [c.strip().lower().replace(" ", "_") for c in units_df.columns]
        units_df = units_df.rename(columns={"n_cycles": "cycles_recorded", "event": "event_flag"})

        # True last cycle per unit
        true_last = train_df.groupby("unit_id")["cycle"].max().to_dict()
        event_map = units_df.set_index("unit_id")["event_flag"].to_dict()

        actual_ruls = []
        for _, row in results_df.iterrows():
            uid = row["unit_id"]
            last_obs = row["last_cycle"]
            total = true_last.get(uid)
            failed = event_map.get(uid, 0) == 1

            if total is None or not failed:
                actual_ruls.append(None)   # unknown / censored
            else:
                raw = max(0, int(total) - int(last_obs))
                actual_ruls.append(min(raw, int(rul_max)))

        results_df = results_df.copy()
        results_df.insert(2, "actual_rul", actual_ruls)

        valid = results_df["actual_rul"].notna()
        results_df["error"]     = np.nan
        results_df["abs_error"] = np.nan
        diff = (
            results_df.loc[valid, "predicted_rul"] - results_df.loc[valid, "actual_rul"]
        ).round(1)
        results_df.loc[valid, "error"]     = diff
        results_df.loc[valid, "abs_error"] = diff.abs()

        n_known = valid.sum()
        if n_known > 0:
            log.info(f"Actual RUL added for {n_known} units (from train.csv).")
    except Exception as exc:
        log.warning(f"Could not load actual RUL: {exc}")

    return results_df


# ─────────────────────────────────────────────────
# Save predictions + accuracy summary in one CSV
# ─────────────────────────────────────────────────
def _save_with_stats(results_df: pd.DataFrame, output_path: str, elapsed: float) -> None:
    """
    Write predictions CSV, then append an accuracy summary block at the bottom.
    The summary rows use a leading '#' so tools can filter them easily.
    """
    import io, csv
    from src.model import phm_score as _phm

    valid = results_df["actual_rul"].notna() if "actual_rul" in results_df.columns else pd.Series([], dtype=bool)
    has_actuals = valid.any() if len(valid) else False

    # Build accuracy stats
    stats_lines = [
        "",
        "# --- ACCURACY SUMMARY ---",
        f"# Generated,{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"# Inference time (s),{elapsed:.1f}",
        f"# Units predicted,{len(results_df)}",
    ]
    if has_actuals:
        v = results_df[valid]
        y_pred = v["predicted_rul"].values.astype(float)
        y_true = v["actual_rul"].values.astype(float)
        rmse      = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        mae       = float(np.mean(np.abs(y_pred - y_true)))
        w30       = float(np.mean(np.abs(y_pred - y_true) <= 30) * 100)
        w20       = float(np.mean(np.abs(y_pred - y_true) <= 20) * 100)
        phm       = _phm(y_pred, y_true)
        late_pct  = float(np.mean(y_pred > y_true) * 100)
        stats_lines += [
            f"# Units with known actuals,{int(valid.sum())}",
            f"# RMSE (cycles),{rmse:.1f}",
            f"# MAE (cycles),{mae:.1f}",
            f"# Within 30 cycles (%),{w30:.0f}",
            f"# Within 20 cycles (%),{w20:.0f}",
            f"# PHM Score (lower=better),{phm:.1f}",
            f"# Late predictions (%),{late_pct:.0f}",
        ]
    for level in ["RED", "AMBER", "GREEN"]:
        cnt = (results_df["warning_level"] == level).sum()
        stats_lines.append(f"# {level},{cnt} units")

    with open(output_path, "w", newline="") as f:
        results_df.to_csv(f, index=False)
        f.write("\n")
        f.write("\n".join(stats_lines) + "\n")


# ─────────────────────────────────────────────────
# SHAP explanation for top-risk unit
# ─────────────────────────────────────────────────
def _shap_explanation(model, test_feat, feature_cols, results_df, rul_max):
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("shap / matplotlib not installed — skipping SHAP plot")
        return

    top_uid = results_df.iloc[0][UNIT_ID_COL]
    log.info(f"\nGenerating SHAP waterfall for highest-risk unit: {top_uid}")

    unit_feat = (
        test_feat[test_feat[UNIT_ID_COL] == top_uid]
        .sort_values(CYCLE_COL)
        .tail(LAST_N_CYCLES)
    )
    X_unit = unit_feat[feature_cols].fillna(0.0).values

    explainer = shap.TreeExplainer(model.xgb_model_)
    shap_vals = explainer.shap_values(X_unit)

    # Use last cycle's SHAP values
    explanation = shap.Explanation(
        values=shap_vals[-1],
        base_values=explainer.expected_value,
        data=X_unit[-1],
        feature_names=feature_cols,
    )
    shap.plots.waterfall(explanation, max_display=15, show=False)

    # waterfall internally calls plt.ioff(); restore interactive mode so show() works
    plt.ion()
    # suptitle targets the figure, not the last twin-axis that waterfall leaves as gca
    plt.gcf().suptitle(f"SHAP explanation — Unit {top_uid} (last cycle)", y=1.02, fontsize=12)

    shap_path = Path("submission") / "predictions" / f"shap_{top_uid}.png"
    plt.savefig(shap_path, dpi=150, bbox_inches="tight")
    log.info(f"SHAP plot saved → {shap_path}")
    plt.show()


# ─────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict Remaining Useful Life for pump fleet"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV (same schema as train.csv, no event label)"
    )
    parser.add_argument(
        "--output", default="data/predictions.csv",
        help="Path for output predictions CSV (default: data/predictions.csv)"
    )
    parser.add_argument(
        "--shap", action="store_true",
        help="Generate SHAP waterfall for highest-risk unit"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = run_inference(args.input, args.output, show_shap=args.shap)

    # Print minimal CSV to stdout (for piping / scoring)
    print(results[["unit_id", "predicted_rul"]].to_csv(index=False), end="")

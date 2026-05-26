#!/usr/bin/env python3
"""
train.py — End-to-end training pipeline for CODVO pump RUL prediction.

Usage:
    python train.py                        # uses defaults from config.py
    python train.py --tune                 # run Optuna hyperparameter search
    python train.py --censoring conservative_bound

Steps:
    1. Load & validate data
    2. Detect sensor columns
    3. Label engineering (piecewise RUL + censoring handling)
    4. [Optional] Auto-select RUL_MAX
    5. Feature engineering
    6. Feature selection
    7. Cross-validation (GroupKFold-5)
    8. Final model training on full dataset
    9. Probability calibration
    10. Save all artefacts
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Pin all randomness FIRST before any library imports
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

from src.config import (
    TRAIN_PATH, UNITS_PATH, ARTIFACTS_DIR,
    RUL_MAX, AUTO_SELECT_RUL_MAX, RUL_MAX_SEARCH_RANGE,
    CENSORING_STRATEGY, TOP_N_FEATURES,
    XGB_PARAMS, LGBM_PARAMS,
    ENSEMBLE_XGB_WEIGHT, ENSEMBLE_LGBM_WEIGHT,
    OPTUNA_N_TRIALS, OPTUNA_TIMEOUT,
    CV_STRATEGY, CV_N_SPLITS,
    RANDOM_SEED,
)
from src.data_loader   import load_and_validate_data, detect_sensor_columns
from src.label_engineer import RULLabelEngineer
from src.feature_engineer import PumpFeatureEngineer, select_features_by_importance
from src.model          import RULEnsemble, tune_hyperparameters, calibrate_failure_prob
from src.validation     import UnitLevelValidator
from src.eda_utils      import select_rul_max_empirically
from src.utils          import get_logger, save_pickle

log = get_logger("train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train pump RUL ensemble")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter optimisation")
    parser.add_argument("--censoring",
                        choices=["exclude", "conservative_bound"],
                        default=CENSORING_STRATEGY)
    parser.add_argument("--rul-max", type=int, default=None,
                        help="Fix RUL_MAX (skip auto-selection)")
    parser.add_argument("--cv-splits", type=int, default=CV_N_SPLITS)
    return parser.parse_args()


def main():
    args = parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────
    log.info("=" * 60)
    log.info("STEP 1: Loading data")
    train_df, units_df = load_and_validate_data(TRAIN_PATH, UNITS_PATH)

    # ── 2. Detect sensor columns ──────────────────
    log.info("STEP 2: Detecting sensor columns")
    sensor_cols, operational_cols = detect_sensor_columns(train_df)
    log.info(f"  Sensors    : {sensor_cols}")
    log.info(f"  Operational: {operational_cols}")

    # ── 3. Label engineering ──────────────────────
    log.info("STEP 3: Label engineering")
    rul_max = args.rul_max or RUL_MAX

    if AUTO_SELECT_RUL_MAX and args.rul_max is None:
        log.info("  Auto-selecting RUL_MAX…")
        rul_max = select_rul_max_empirically(
            train_df, units_df, sensor_cols,
            search_range=RUL_MAX_SEARCH_RANGE,
            plot=False,
        )

    label_eng = RULLabelEngineer(
        rul_max=rul_max,
        censoring_strategy=args.censoring,
    )
    df_labeled = label_eng.fit_transform(train_df, units_df)
    log.info(f"  RUL_MAX used: {rul_max}")

    # ── 4. Feature engineering ────────────────────
    log.info("STEP 4: Feature engineering")
    fe = PumpFeatureEngineer(
        sensor_cols=sensor_cols,
    )
    fe.fit(df_labeled)
    df_features = fe.transform(df_labeled)

    # Drop high-NaN features
    df_features, dropped_cols = fe.drop_high_nan_features(df_features)
    log.info(f"  After drop: {len(df_features.columns)} columns remain")

    # ── 5. Feature selection ──────────────────────
    log.info("STEP 5: Feature selection")
    meta_cols = ["unit_id", "cycle", "rul", "raw_rul",
                 "sample_weight", "in_degradation_window"]
    candidate_features = [
        c for c in df_features.columns if c not in meta_cols
    ]
    y_sel = df_features["rul"].values
    w_sel = df_features["sample_weight"].values

    feature_cols = select_features_by_importance(
        df_features[candidate_features],
        y_sel, w_sel,
        top_n=TOP_N_FEATURES,
        random_state=RANDOM_SEED,
    )
    log.info(f"  Selected {len(feature_cols)} features")

    # ── 6. Hyperparameter tuning (optional) ───────
    xgb_p  = XGB_PARAMS.copy()
    lgbm_p = LGBM_PARAMS.copy()
    xgb_w  = ENSEMBLE_XGB_WEIGHT
    lgbm_w = ENSEMBLE_LGBM_WEIGHT

    if args.tune:
        log.info("STEP 6: Optuna hyperparameter search")
        best = tune_hyperparameters(
            df_features, feature_cols,
            n_trials=OPTUNA_N_TRIALS,
            timeout=OPTUNA_TIMEOUT,
            random_state=RANDOM_SEED,
        )
        # Unpack Optuna params back to dicts
        xgb_keys  = [k for k in best if k.startswith("xgb_") and k != "xgb_w"]
        lgbm_keys = [k for k in best if k.startswith("lgbm_")]
        for k in xgb_keys:
            xgb_p[k.replace("xgb_", "")] = best[k]
        for k in lgbm_keys:
            lgbm_p[k.replace("lgbm_", "")] = best[k]
        xgb_w  = best.get("xgb_w", xgb_w)
        lgbm_w = 1.0 - xgb_w
    else:
        log.info("STEP 6: Skipping hyperparameter tuning (use --tune to enable)")

    # ── 7. Cross-validation ───────────────────────
    log.info("STEP 7: Cross-validation")
    model_cv = RULEnsemble(
        xgb_params=xgb_p, lgbm_params=lgbm_p,
        xgb_weight=xgb_w, lgbm_weight=lgbm_w,
    )
    validator = UnitLevelValidator(strategy=CV_STRATEGY, n_splits=args.cv_splits)
    cv_results = validator.evaluate_model(df_features, feature_cols, model_cv)

    log.info("─" * 50)
    log.info("CV Results Summary:")
    log.info(f"  RMSE         : {cv_results['rmse'].mean():.2f} "
             f"± {cv_results['rmse'].std():.2f}")
    log.info(f"  MAE          : {cv_results['mae'].mean():.2f}")
    log.info(f"  PHM Score    : {cv_results['phm_score'].mean():.1f}")
    log.info(f"  Within-30    : {cv_results['within_30'].mean():.1f}%")
    log.info(f"  Late pred %  : {cv_results['late_pct'].mean():.1f}%")
    log.info("─" * 50)

    cv_results.to_csv(ARTIFACTS_DIR / "cv_results.csv", index=False)

    # ── 8. Final model on full dataset ────────────
    log.info("STEP 8: Training final model on full dataset")
    X_all = df_features[feature_cols].fillna(0).values
    y_all = df_features["rul"].values
    w_all = df_features["sample_weight"].values

    final_model = RULEnsemble(
        xgb_params=xgb_p, lgbm_params=lgbm_p,
        xgb_weight=xgb_w, lgbm_weight=lgbm_w,
    )
    final_model.fit(X_all, y_all, sample_weight=w_all)

    # ── 9. Probability calibration ────────────────
    log.info("STEP 9: Calibrating failure probability")
    # Use last fold's OOF predictions for calibration
    last_fold = cv_results["fold"].max()
    # We do a simple refit on first 60 units, evaluate on last 20 for calibration
    failed_ids = units_df[units_df["event_flag"] == 1]["unit_id"].tolist()
    cal_ids    = failed_ids[-20:]  # hold-out 20 units
    cal_train  = df_features[~df_features["unit_id"].isin(cal_ids)]
    cal_test   = df_features[df_features["unit_id"].isin(cal_ids)]

    cal_model = RULEnsemble(xgb_params=xgb_p, lgbm_params=lgbm_p)
    valid_cal  = ~np.isnan(cal_train["rul"].values)
    cal_model.fit(
        cal_train[feature_cols].fillna(0).values[valid_cal],
        cal_train["rul"].values[valid_cal],
        sample_weight=cal_train["sample_weight"].values[valid_cal],
    )
    cal_preds = cal_model.predict(cal_test[feature_cols].fillna(0).values)
    cal_true  = cal_test["rul"].values
    mask_valid = ~np.isnan(cal_true)
    calibration_params = calibrate_failure_prob(
        cal_preds[mask_valid], cal_true[mask_valid], horizon=30
    )

    # ── 10. Save artefacts ────────────────────────
    log.info("STEP 10: Saving artefacts")
    final_model.save(ARTIFACTS_DIR / "ensemble_model.pkl")
    save_pickle(fe,              ARTIFACTS_DIR / "feature_pipeline.pkl")
    save_pickle(feature_cols,    ARTIFACTS_DIR / "feature_cols.pkl")
    save_pickle(label_eng,       ARTIFACTS_DIR / "label_engineer.pkl")
    save_pickle({
        "sensor_cols":          sensor_cols,
        "operational_cols":     operational_cols,
        "rul_max":              rul_max,
        "baseline_window":      fe.baseline_window,
        "censoring_strategy":   args.censoring,
        "xgb_weight":           xgb_w,
        "lgbm_weight":          lgbm_w,
        "calibration_params":   calibration_params,
        "cv_rmse_mean":         float(cv_results["rmse"].mean()),
        "cv_rmse_std":          float(cv_results["rmse"].std()),
        "cv_within30_mean":     float(cv_results["within_30"].mean()),
    }, ARTIFACTS_DIR / "train_config.pkl")

    log.info("=" * 60)
    log.info("Training complete!")
    log.info(f"Artefacts saved to: {ARTIFACTS_DIR}")
    log.info(f"Final CV RMSE: {cv_results['rmse'].mean():.2f} ± {cv_results['rmse'].std():.2f}")
    log.info(f"Within-30 accuracy: {cv_results['within_30'].mean():.1f}%")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

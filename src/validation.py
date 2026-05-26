"""
validation.py — Unit-level validation harness.

Critical principle: NO pump appears in both train and test in the same fold.
All feature engineering is refitted on train-fold data only.

Provides:
  - UnitLevelValidator: GroupKFold / LeaveOneGroupOut CV
  - Progressive accuracy analysis (accuracy vs. warning horizon)
  - All PHM-Society metrics
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.base import clone

from src.config import (
    UNIT_ID_COL, CYCLE_COL,
    CV_STRATEGY, CV_N_SPLITS, LAST_N_CYCLES
)
from src.utils import get_logger

log = get_logger("validation")


# ─────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────
def phm_score(errors: np.ndarray) -> float:
    """
    PHM asymmetric score (lower is better).
    d = pred - true
    L(d) = exp(-d/13)-1  if d<0  (early prediction: gentle)
           exp( d/10)-1  if d>0  (late  prediction: steep)
    """
    return float(
        np.where(errors < 0,
                 np.exp(-errors / 13.0) - 1,
                 np.exp( errors / 10.0) - 1).sum()
    )


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Full metric suite at unit level."""
    errors = y_pred - y_true
    abs_e  = np.abs(errors)
    return {
        "rmse":           float(np.sqrt(np.mean(errors ** 2))),
        "mae":            float(np.mean(abs_e)),
        "median_ae":      float(np.median(abs_e)),
        "phm_score":      phm_score(errors),
        "late_pct":       float((errors > 0).mean() * 100),
        "within_10":      float((abs_e <= 10).mean() * 100),
        "within_30":      float((abs_e <= 30).mean() * 100),
        "within_50":      float((abs_e <= 50).mean() * 100),
        "n_units":        int(len(y_pred)),
    }


# ─────────────────────────────────────────────────
# Aggregation: row-level → unit-level
# ─────────────────────────────────────────────────
def aggregate_to_unit_level(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    last_n: int = LAST_N_CYCLES,
) -> pd.DataFrame:
    """
    Attach row-level predictions to df, then take median of last `last_n`
    cycles per unit as the unit-level RUL estimate.

    Returns DataFrame with [unit_id, pred_rul, true_rul].
    """
    df = df.copy()
    df["_pred"] = y_pred
    df["_true"] = df["rul"].values

    rows = []
    for uid, grp in df.groupby(UNIT_ID_COL):
        grp_s = grp.sort_values(CYCLE_COL)
        tail  = grp_s.tail(last_n)
        rows.append({
            UNIT_ID_COL: uid,
            "pred_rul":  float(tail["_pred"].median()),
            "true_rul":  float(grp_s["_true"].iloc[-1]),
            "last_cycle": int(grp_s[CYCLE_COL].iloc[-1]),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────
# Main validator class
# ─────────────────────────────────────────────────
class UnitLevelValidator:
    """
    Cross-validation with strict unit-level group splits.

    Parameters
    ----------
    strategy : str  "group_kfold" | "logo"
    n_splits : int  (used only for group_kfold)
    """

    def __init__(
        self,
        strategy: str = CV_STRATEGY,
        n_splits: int = CV_N_SPLITS,
    ):
        self.strategy = strategy
        self.n_splits = n_splits

    def evaluate_model(
        self,
        df_labeled: pd.DataFrame,
        feature_cols: list,
        model,
        target_col: str  = "rul",
        weight_col: str  = "sample_weight",
    ) -> pd.DataFrame:
        """
        Run full CV loop.  Feature engineering is intentionally NOT refitted here
        (call evaluate_pipeline for that).  Use when feature matrix is already built.

        Returns per-fold metric DataFrame.
        """
        unit_ids = df_labeled[UNIT_ID_COL].values
        X        = df_labeled[feature_cols].fillna(0).values
        y        = df_labeled[target_col].values
        w        = df_labeled[weight_col].values

        splits = self._get_splits(df_labeled, unit_ids)
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            Xtr, ytr, wtr = X[train_idx], y[train_idx], w[train_idx]
            Xte, yte      = X[test_idx],  y[test_idx]
            df_te         = df_labeled.iloc[test_idx].copy()

            # Only train on labeled rows
            labeled = ~np.isnan(ytr) & (wtr > 0)
            m = clone(model)
            m.fit(Xtr[labeled], ytr[labeled], sample_weight=wtr[labeled])

            pred_all = m.predict(Xte)

            # Aggregate to unit level
            unit_df = aggregate_to_unit_level(df_te, pred_all)
            unit_df = unit_df.dropna(subset=["true_rul"])

            metrics = compute_metrics(
                unit_df["pred_rul"].values,
                unit_df["true_rul"].values,
            )
            metrics["fold"] = fold_idx
            fold_results.append(metrics)
            log.info(
                f"  Fold {fold_idx+1} | RMSE={metrics['rmse']:.2f} "
                f"| PHM={metrics['phm_score']:.1f} "
                f"| within30={metrics['within_30']:.1f}%"
            )

        return pd.DataFrame(fold_results)

    def evaluate_pipeline(
        self,
        df_labeled: pd.DataFrame,
        feature_engineer,
        feature_cols: list,
        model,
        target_col: str  = "rul",
        weight_col: str  = "sample_weight",
    ) -> pd.DataFrame:
        """
        Full pipeline CV including feature engineering refitting.
        More expensive but eliminates any possible leakage via baselines.
        """
        unit_ids = df_labeled[UNIT_ID_COL].values
        splits   = self._get_splits(df_labeled, unit_ids)
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            df_tr = df_labeled.iloc[train_idx].copy()
            df_te = df_labeled.iloc[test_idx].copy()

            # Refit feature engineer on train fold ONLY
            fe = clone(feature_engineer)
            fe.fit(df_tr)
            df_tr_feat = fe.transform(df_tr)
            df_te_feat = fe.transform(df_te)

            # Determine feature cols that exist after transform
            valid_feat_cols = [c for c in feature_cols if c in df_tr_feat.columns]

            Xtr = df_tr_feat[valid_feat_cols].fillna(0).values
            ytr = df_tr_feat[target_col].values
            wtr = df_tr_feat[weight_col].values
            Xte = df_te_feat[valid_feat_cols].fillna(0).values

            labeled = ~np.isnan(ytr) & (wtr > 0)
            m = clone(model)
            m.fit(Xtr[labeled], ytr[labeled], sample_weight=wtr[labeled])

            pred_all = m.predict(Xte)
            unit_df  = aggregate_to_unit_level(df_te_feat, pred_all)
            unit_df  = unit_df.dropna(subset=["true_rul"])

            metrics = compute_metrics(
                unit_df["pred_rul"].values,
                unit_df["true_rul"].values,
            )
            metrics["fold"] = fold_idx
            fold_results.append(metrics)
            log.info(
                f"  Fold {fold_idx+1} | RMSE={metrics['rmse']:.2f} "
                f"| PHM={metrics['phm_score']:.1f} "
                f"| within30={metrics['within_30']:.1f}%"
            )

        return pd.DataFrame(fold_results)

    def _get_splits(self, df, unit_ids):
        if self.strategy == "group_kfold":
            cv = GroupKFold(n_splits=self.n_splits)
        elif self.strategy == "logo":
            cv = LeaveOneGroupOut()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        return list(cv.split(df, groups=unit_ids))


# ─────────────────────────────────────────────────
# Progressive accuracy analysis
# ─────────────────────────────────────────────────
def progressive_accuracy_analysis(
    df_te: pd.DataFrame,
    y_pred: np.ndarray,
    rul_bins: list = None,
) -> pd.DataFrame:
    """
    Compute metrics within RUL buckets (e.g. [0-25], [25-50], [50-75] …).

    Answers: "How accurate is the model as a function of remaining life?"
    Expected: accuracy improves (RMSE drops) as RUL → 0.

    Parameters
    ----------
    df_te    : test DataFrame with 'rul' column (true labels)
    y_pred   : row-level predictions (same index order as df_te)
    rul_bins : list of (low, high) tuples — default 5 equal-width bins 0–125

    Returns pd.DataFrame with metrics per bucket.
    """
    if rul_bins is None:
        rul_bins = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 125)]

    df = df_te.copy()
    df["_pred"] = y_pred
    results = []

    for (lo, hi) in rul_bins:
        mask = (df["rul"] >= lo) & (df["rul"] < hi) & df["rul"].notna()
        subset = df[mask]
        if len(subset) < 5:
            continue
        errors = subset["_pred"].values - subset["rul"].values
        m = compute_metrics(subset["_pred"].values, subset["rul"].values)
        m["rul_bucket"]  = f"{lo}–{hi}"
        m["bucket_rows"] = int(mask.sum())
        results.append(m)

    return pd.DataFrame(results)

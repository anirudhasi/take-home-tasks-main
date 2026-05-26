"""
feature_engineer.py — Physics-informed feature engineering pipeline.

Feature categories (all per sensor channel):
  1. Rolling statistics  : mean, std, min, max, range, CV  (w ∈ {5,10,20,30})
  2. Trend features      : OLS slope + curvature            (w ∈ {10,20,30})
  3. EWMA features       : level + deviation                (α ∈ {0.1,0.3,0.5})
  4. Baseline deviation  : z-score vs own healthy baseline  (first 20 cycles)
  5. Cross-sensor        : physically motivated interactions (post-EDA)

Total candidates ≈ 790 before selection.

Design:
  - sklearn BaseEstimator/TransformerMixin → pkl-serialisable, pipeline-composable
  - fit() learns per-unit baselines from TRAIN data only
  - transform() is side-effect free; safe to call on test data
  - All operations are per-unit to respect temporal ordering
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.config import (
    UNIT_ID_COL, CYCLE_COL,
    ROLL_WINDOWS, TREND_WINDOWS, EWMA_SPANS,
    BASELINE_WINDOW, USE_CROSS_SENSOR, MAX_NAN_PCT
)
from src.utils import get_logger

log = get_logger("feature_engineer")

# ─────────────────────────────────────────────────
# Helper: OLS slope + curvature over a window
# ─────────────────────────────────────────────────
def _poly2_features(values: np.ndarray, window: int):
    """
    Compute quadratic polynomial fit over a sliding window.
    Returns (slope_array, curvature_array) of same length as `values`.
    First `window-1` entries are NaN (insufficient history).
    """
    n = len(values)
    slopes     = np.full(n, np.nan)
    curvatures = np.full(n, np.nan)
    x = np.arange(window, dtype=float)

    for i in range(window - 1, n):
        y = values[i - window + 1: i + 1].copy()
        nans = np.isnan(y)
        if nans.sum() > window * 0.3:      # >30% NaN → skip
            continue
        y[nans] = np.nanmean(y)            # fill sparse NaN
        try:
            coeffs = np.polyfit(x, y, deg=2)
            slopes[i]     = coeffs[1]      # linear coeff
            curvatures[i] = coeffs[0]      # quadratic coeff (acceleration)
        except (np.linalg.LinAlgError, ValueError):
            pass

    return slopes, curvatures


# ─────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────
class PumpFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible feature engineering pipeline for pump RUL prediction.

    Parameters
    ----------
    sensor_cols : list of str
        Sensor column names (set from EDA; required before fit).
    roll_windows : list of int
    trend_windows : list of int
    ewma_spans : list of float   (alpha values, NOT span/halflife)
    baseline_window : int        Number of first cycles treated as healthy baseline.
    use_cross_sensor : bool      Build physically-motivated interaction features.
    """

    def __init__(
        self,
        sensor_cols: list = None,
        roll_windows: list = None,
        trend_windows: list = None,
        ewma_spans: list = None,
        baseline_window: int = BASELINE_WINDOW,
        use_cross_sensor: bool = USE_CROSS_SENSOR,
    ):
        self.sensor_cols     = sensor_cols or []
        self.roll_windows    = roll_windows  or ROLL_WINDOWS
        self.trend_windows   = trend_windows or TREND_WINDOWS
        self.ewma_spans      = ewma_spans    or EWMA_SPANS
        self.baseline_window = baseline_window
        self.use_cross_sensor = use_cross_sensor

    # ── Fit ───────────────────────────────────────
    def fit(self, df: pd.DataFrame, y=None):
        """
        Learn per-unit healthy baselines from the FIRST `baseline_window` cycles.
        Global fallback baselines are computed for unseen units at inference time.
        """
        self.unit_baselines_ = {}

        for uid, grp in df.groupby(UNIT_ID_COL):
            early = grp.sort_values(CYCLE_COL).head(self.baseline_window)
            self.unit_baselines_[uid] = {
                col: {
                    "mean": float(early[col].mean()),
                    "std":  max(float(early[col].std()), 1e-6),
                }
                for col in self.sensor_cols
            }

        # Global baseline: population mean/std of early cycles
        early_all = df.groupby(UNIT_ID_COL).apply(
            lambda g: g.sort_values(CYCLE_COL).head(self.baseline_window)
        ).reset_index(drop=True)
        self.global_baseline_ = {
            col: {
                "mean": float(early_all[col].mean()),
                "std":  max(float(early_all[col].std()), 1e-6),
            }
            for col in self.sensor_cols
        }

        self.feature_names_out_ = self._build_feature_names()
        log.info(
            f"PumpFeatureEngineer fitted | "
            f"{len(self.unit_baselines_)} units | "
            f"~{len(self.feature_names_out_)} candidate features"
        )
        return self

    # ── Transform ─────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "unit_baselines_")
        frames = []
        for uid, grp in df.groupby(UNIT_ID_COL):
            grp = grp.sort_values(CYCLE_COL).copy()
            baseline = self.unit_baselines_.get(uid, self.global_baseline_)

            grp = self._rolling_features(grp)
            grp = self._trend_features(grp)
            grp = self._ewma_features(grp)
            grp = self._baseline_deviation_features(grp, baseline)
            if self.use_cross_sensor:
                grp = self._cross_sensor_features(grp)

            frames.append(grp)

        result = pd.concat(frames, ignore_index=True)
        return result

    # ── Drop high-NaN features ─────────────────────
    def drop_high_nan_features(
        self, df: pd.DataFrame, threshold: float = MAX_NAN_PCT
    ) -> tuple:
        """
        After transform(), drop columns where NaN fraction > threshold.
        Returns (cleaned_df, list_of_dropped_cols).
        """
        nan_frac = df.isnull().mean()
        drop_cols = nan_frac[nan_frac > threshold].index.tolist()
        # Never drop id or target columns
        drop_cols = [
            c for c in drop_cols
            if c not in [UNIT_ID_COL, CYCLE_COL, "rul", "raw_rul",
                         "sample_weight", "in_degradation_window"]
        ]
        log.info(f"Dropping {len(drop_cols)} high-NaN features (>{threshold*100:.0f}%)")
        return df.drop(columns=drop_cols), drop_cols

    # ─── Private: feature blocks ──────────────────

    def _rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.sensor_cols:
            vals = df[col]
            for w in self.roll_windows:
                roll = vals.rolling(window=w, min_periods=max(1, w // 2))
                p    = f"{col}_r{w}"
                df[f"{p}_mean"]  = roll.mean()
                df[f"{p}_std"]   = roll.std()
                df[f"{p}_min"]   = roll.min()
                df[f"{p}_max"]   = roll.max()
                df[f"{p}_range"] = df[f"{p}_max"] - df[f"{p}_min"]
                # Coefficient of variation (dimensionless noise measure)
                df[f"{p}_cv"]    = df[f"{p}_std"] / (df[f"{p}_mean"].abs() + 1e-6)
        return df

    def _trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.sensor_cols:
            vals = df[col].values.astype(float)
            for w in self.trend_windows:
                slopes, curvs = _poly2_features(vals, w)
                df[f"{col}_slope{w}"] = slopes
                df[f"{col}_curv{w}"]  = curvs
        return df

    def _ewma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.sensor_cols:
            vals = df[col]
            for alpha in self.ewma_spans:
                tag  = str(alpha).replace(".", "")
                ewma = vals.ewm(alpha=alpha, adjust=False).mean()
                df[f"{col}_ewma{tag}"]    = ewma
                df[f"{col}_ewmadev{tag}"] = vals - ewma   # deviation from smoothed level
        return df

    def _baseline_deviation_features(self, df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
        for col in self.sensor_cols:
            mu    = baseline[col]["mean"]
            sigma = baseline[col]["std"]
            df[f"{col}_devbase"]  = df[col] - mu
            df[f"{col}_zscore"]   = (df[col] - mu) / sigma
        return df

    def _cross_sensor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Physically motivated cross-sensor features.
        Guards existence of columns before computing.
        Only adds features that are meaningful for centrifugal pumps:
          - Pressure ratio (discharge / suction)
          - Efficiency proxy (flow / power)
          - Specific vibration (vibration / flow)
          - Thermal efficiency (temp / current)
        Column names are inferred heuristically from the sensor list.
        """
        cols = df.columns.tolist()

        def find(keywords):
            """Return first col whose name contains any keyword."""
            for kw in keywords:
                for c in self.sensor_cols:
                    if kw.lower() in c.lower():
                        return c
            return None

        p_out  = find(["disch", "outlet_p", "p_out", "pressure_out", "hp"])
        p_in   = find(["suct",  "inlet_p",  "p_in",  "pressure_in",  "lp"])
        flow   = find(["flow", "flowrate", "q_"])
        power  = find(["power", "watt", "kw"])
        vib    = find(["vib", "vibr", "accel"])
        temp   = find(["temp", "temperature", "t_"])
        curr   = find(["current", "amp", "ia"])

        if p_out and p_in:
            df["feat_pressure_ratio"] = df[p_out] / (df[p_in].abs() + 1e-6)

        if flow and power:
            df["feat_efficiency_proxy"] = df[flow] / (df[power].abs() + 1e-6)

        if vib and flow:
            df["feat_specific_vib"] = df[vib] / (df[flow].abs() + 1e-6)

        if temp and curr:
            df["feat_thermal_eff"] = df[temp] / (df[curr].abs() + 1e-6)

        return df

    def _build_feature_names(self) -> list:
        names = list(self.sensor_cols)
        for col in self.sensor_cols:
            for w in self.roll_windows:
                for stat in ["mean", "std", "min", "max", "range", "cv"]:
                    names.append(f"{col}_r{w}_{stat}")
            for w in self.trend_windows:
                names += [f"{col}_slope{w}", f"{col}_curv{w}"]
            for alpha in self.ewma_spans:
                tag = str(alpha).replace(".", "")
                names += [f"{col}_ewma{tag}", f"{col}_ewmadev{tag}"]
            names += [f"{col}_devbase", f"{col}_zscore"]
        names += [CYCLE_COL]
        if self.use_cross_sensor:
            names += [
                "feat_pressure_ratio", "feat_efficiency_proxy",
                "feat_specific_vib", "feat_thermal_eff"
            ]
        return names


# ─────────────────────────────────────────────────
# Feature selection utility
# ─────────────────────────────────────────────────
def select_features_by_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    top_n: int = 150,
    random_state: int = 42,
) -> list:
    """
    Fit a quick XGBoost model to get feature importances (gain),
    return top_n feature names.

    Uses a reduced n_estimators for speed — this is selection only.
    """
    import xgboost as xgb

    valid_mask = ~np.isnan(y)
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(
        X[valid_mask].fillna(0),
        y[valid_mask],
        sample_weight=sample_weight[valid_mask],
    )

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    selected = importance.head(top_n).index.tolist()
    log.info(
        f"Feature selection: {len(X.columns)} → {len(selected)} features "
        f"| top gain: {importance.iloc[0]:.5f} | "
        f"bottom kept: {importance[selected[-1]]:.5f}"
    )
    return selected

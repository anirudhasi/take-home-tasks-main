"""
model.py — XGBoost + LightGBM weighted ensemble for pump RUL prediction.

Key design decisions:
  1. Custom asymmetric loss for XGBoost (PHM Society scoring function)
     — Late predictions penalised more than early ones, matching business cost.
  2. LightGBM with standard RMSE as complementary learner.
  3. Unit-level RUL derived from median of last-N-cycle predictions
     (robust to sensor noise spikes at the final cycle).
  4. Optuna-driven hyperparameter tuning with unit-level GroupKFold CV.
  5. Calibration curve for failure_prob_30 output.
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from src.config import (
    UNIT_ID_COL, CYCLE_COL,
    XGB_PARAMS, LGBM_PARAMS,
    ENSEMBLE_XGB_WEIGHT, ENSEMBLE_LGBM_WEIGHT,
    USE_ASYMMETRIC_LOSS, LAST_N_CYCLES, RANDOM_SEED
)
from src.utils import get_logger, save_pickle, load_pickle

log = get_logger("model")


# ─────────────────────────────────────────────────
# Asymmetric PHM Loss
# ─────────────────────────────────────────────────
def _phm_grad_hess(y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None):
    """
    Custom XGBoost objective: PHM Society asymmetric loss.

    d = y_pred - y_true

    Loss  L(d) = exp(-d/13) - 1   if d < 0  (early: under-predict RUL)
                 exp( d/10) - 1   if d > 0  (late:  over-predict RUL) ← steeper!

    Gradient and Hessian derived analytically.

    The asymmetry (10 vs 13) means late predictions (missed failure)
    accumulate penalty ~30% faster than early predictions (false alarms).

    NOTE: XGBoost 2+ sklearn API calls objective as func(y_true, y_pred).
    """
    d = y_pred - y_true

    # Gradient  ∂L/∂ŷ
    grad = np.where(
        d < 0,
        -(1.0 / 13.0) * np.exp(-d / 13.0),
         (1.0 / 10.0) * np.exp( d / 10.0),
    )
    # Hessian  ∂²L/∂ŷ² (must be positive for stability)
    hess = np.where(
        d < 0,
        (1.0 / 13.0 ** 2) * np.exp(-d / 13.0),
        (1.0 / 10.0 ** 2) * np.exp( d / 10.0),
    )
    if sample_weight is not None:
        grad = grad * sample_weight
        hess = hess * sample_weight
    return grad, hess


def phm_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    PHM Society evaluation metric (lower is better).
    Used for model comparison and final reporting.
    """
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(s.sum())


# ─────────────────────────────────────────────────
# Ensemble model
# ─────────────────────────────────────────────────
class RULEnsemble(BaseEstimator, RegressorMixin):
    """
    Weighted ensemble of XGBoost (asymmetric loss) + LightGBM (RMSE).

    Parameters
    ----------
    xgb_params : dict    XGBoost hyperparameters.
    lgbm_params : dict   LightGBM hyperparameters.
    xgb_weight : float   Weight for XGBoost predictions (0–1).
    lgbm_weight : float  Weight for LightGBM predictions (must sum to 1).
    use_asymmetric_loss : bool
    """

    def __init__(
        self,
        xgb_params: dict  = None,
        lgbm_params: dict = None,
        xgb_weight: float  = ENSEMBLE_XGB_WEIGHT,
        lgbm_weight: float = ENSEMBLE_LGBM_WEIGHT,
        use_asymmetric_loss: bool = USE_ASYMMETRIC_LOSS,
    ):
        self.xgb_params  = xgb_params  or XGB_PARAMS.copy()
        self.lgbm_params = lgbm_params or LGBM_PARAMS.copy()
        self.xgb_weight  = xgb_weight
        self.lgbm_weight = lgbm_weight
        self.use_asymmetric_loss = use_asymmetric_loss

    # ── Fit ───────────────────────────────────────
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray = None,
    ):
        # Drop rows with NaN target or zero sample weight
        valid = ~np.isnan(y)
        if sample_weight is not None:
            valid = valid & (sample_weight > 0)
        X_fit = X[valid]
        y_fit = y[valid]
        sw    = sample_weight[valid] if sample_weight is not None else None

        # ── XGBoost ──
        log.info("Training XGBoost…")
        xgb_kw = self.xgb_params.copy()
        if self.use_asymmetric_loss:
            objective = _phm_grad_hess
            xgb_kw.setdefault("base_score", float(np.mean(y_fit)))
        else:
            objective = "reg:squarederror"

        self.xgb_model_ = xgb.XGBRegressor(objective=objective, **xgb_kw)
        self.xgb_model_.fit(X_fit, y_fit, sample_weight=sw)

        # ── LightGBM ──
        log.info("Training LightGBM…")
        self.lgbm_model_ = lgb.LGBMRegressor(**self.lgbm_params)
        self.lgbm_model_.fit(X_fit, y_fit, sample_weight=sw)

        # Store feature count for shape checking
        self.n_features_in_ = X_fit.shape[1]
        log.info(
            f"Ensemble trained | XGB weight={self.xgb_weight:.2f} "
            f"| LGBM weight={self.lgbm_weight:.2f}"
        )
        return self

    # ── Predict (row-level) ───────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "xgb_model_")
        X_safe = np.nan_to_num(X, nan=0.0)
        xgb_pred  = self.xgb_model_.predict(X_safe)
        lgbm_pred = self.lgbm_model_.predict(X_safe)
        ensemble  = self.xgb_weight * xgb_pred + self.lgbm_weight * lgbm_pred
        return np.clip(ensemble, 0.0, None)   # RUL ≥ 0

    # ── Predict (unit-level) ──────────────────────
    def predict_unit_rul(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        last_n: int = LAST_N_CYCLES,
    ) -> pd.DataFrame:
        """
        Aggregate row-level predictions to unit-level RUL estimate.

        Uses median of the LAST `last_n` cycle predictions per unit.
        Median is more robust than the final-cycle-only prediction because:
          - A single sensor spike at the last cycle could distort the estimate.
          - Median of 5 cycles is still representative of current condition.

        Returns
        -------
        pd.DataFrame with columns:
            unit_id, predicted_rul, pred_std, last_cycle
        """
        check_is_fitted(self, "xgb_model_")
        X = df[feature_cols].values
        preds = self.predict(X)

        rows = []
        for uid, grp in df.groupby(UNIT_ID_COL):
            grp_sorted = grp.sort_values(CYCLE_COL)
            idx = grp_sorted.index
            unit_preds = preds[
                [i for i, gi in enumerate(df.index) if gi in set(idx)]
            ]
            tail = unit_preds[-last_n:] if len(unit_preds) >= last_n else unit_preds
            rows.append({
                UNIT_ID_COL:     uid,
                "predicted_rul": float(np.median(tail)),
                "pred_std":       float(np.std(tail)) if len(tail) > 1 else 0.0,
                "last_cycle":    int(grp_sorted[CYCLE_COL].iloc[-1]),
                "n_cycles":      len(grp_sorted),
            })

        return pd.DataFrame(rows).sort_values("predicted_rul").reset_index(drop=True)

    # ── Feature importance ────────────────────────
    def feature_importance_df(self, feature_cols: list) -> pd.DataFrame:
        check_is_fitted(self, "xgb_model_")
        xgb_imp  = self.xgb_model_.feature_importances_
        lgbm_imp = self.lgbm_model_.feature_importances_

        # Normalise each to [0,1] then average
        xgb_norm  = xgb_imp  / (xgb_imp.max()  + 1e-9)
        lgbm_norm = lgbm_imp / (lgbm_imp.max() + 1e-9)
        combined  = 0.5 * xgb_norm + 0.5 * lgbm_norm

        df_imp = pd.DataFrame({
            "feature":   feature_cols,
            "xgb_gain":  xgb_imp,
            "lgbm_gain": lgbm_imp,
            "combined":  combined,
        }).sort_values("combined", ascending=False).reset_index(drop=True)
        return df_imp

    # ── Persistence ───────────────────────────────
    def save(self, path: Path) -> None:
        save_pickle(self, path)

    @classmethod
    def load(cls, path: Path) -> "RULEnsemble":
        return load_pickle(path)


# ─────────────────────────────────────────────────
# Optuna hyperparameter tuning
# ─────────────────────────────────────────────────
def tune_hyperparameters(
    df_labeled: pd.DataFrame,
    feature_cols: list,
    n_trials: int = 80,
    timeout: int = 3600,
    random_state: int = RANDOM_SEED,
) -> dict:
    """
    Run Optuna Bayesian optimisation to find best XGBoost + LGBM hyperparameters.
    Optimises unit-level RMSE under GroupKFold(5).

    Returns best_params dict ready to pass to RULEnsemble.
    """
    import optuna
    from src.validation import UnitLevelValidator
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        xgb_p = {
            "n_estimators":     trial.suggest_int("xgb_n_est", 200, 800),
            "max_depth":        trial.suggest_int("xgb_depth", 3, 8),
            "learning_rate":    trial.suggest_float("xgb_lr", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("xgb_sub", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_col", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("xgb_alpha", 1e-4, 2.0, log=True),
            "reg_lambda":       trial.suggest_float("xgb_lam",   1e-4, 5.0, log=True),
            "min_child_weight": trial.suggest_int("xgb_mcw", 3, 20),
            "tree_method": "hist", "random_state": random_state, "n_jobs": -1,
        }
        lgbm_p = {
            "n_estimators": trial.suggest_int("lgbm_n_est", 200, 800),
            "num_leaves":   trial.suggest_int("lgbm_leaves", 15, 63),
            "learning_rate": trial.suggest_float("lgbm_lr", 0.01, 0.15, log=True),
            "subsample":     trial.suggest_float("lgbm_sub", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("lgbm_col", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("lgbm_mcs", 5, 30),
            "random_state": random_state, "n_jobs": -1, "verbose": -1,
        }
        xgb_w = trial.suggest_float("xgb_w", 0.3, 0.7)

        model = RULEnsemble(
            xgb_params=xgb_p, lgbm_params=lgbm_p,
            xgb_weight=xgb_w, lgbm_weight=1 - xgb_w,
        )
        validator = UnitLevelValidator(strategy="group_kfold", n_splits=5)
        results = validator.evaluate_model(df_labeled, feature_cols, model)
        return float(results["rmse"].mean())

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    log.info(
        f"Optuna complete | best RMSE: {study.best_value:.3f} "
        f"| trials: {len(study.trials)}"
    )
    return study.best_params


# ─────────────────────────────────────────────────
# Probability calibration
# ─────────────────────────────────────────────────
def calibrate_failure_prob(
    val_preds: np.ndarray,
    val_true_rul: np.ndarray,
    horizon: int = 30,
    n_bins: int = 20,
) -> dict:
    """
    Build a calibration curve: P(true RUL < horizon | predicted_RUL = x).
    Fitted via isotonic regression on validation set predictions.

    Returns dict with pred_bins and true_probs for np.interp at inference time.
    """
    from sklearn.isotonic import IsotonicRegression

    # Binary label: did the unit fail within `horizon` cycles?
    binary_true = (val_true_rul <= horizon).astype(float)

    iso = IsotonicRegression(out_of_bounds="clip", increasing=False)
    iso.fit(val_preds, binary_true)

    # Build lookup table
    pred_bins  = np.linspace(val_preds.min(), val_preds.max(), 200)
    true_probs = iso.predict(pred_bins)

    log.info(
        f"Calibration fitted | horizon={horizon} | "
        f"base rate: {binary_true.mean():.3f}"
    )
    return {
        "pred_bins":  pred_bins.tolist(),
        "true_probs": true_probs.tolist(),
        "horizon":    horizon,
    }

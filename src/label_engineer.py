"""
label_engineer.py — Piecewise-linear RUL label generation with
censoring-aware treatment.

Key design decisions:
  - Failed units:   raw_rul = max_cycle - current_cycle, then capped at RUL_MAX
  - Censored units: excluded from supervised target (sample_weight=0)
                    OR assigned conservative lower-bound RUL
  - RUL_MAX:        auto-selected empirically OR fixed via config
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    UNIT_ID_COL, CYCLE_COL, EVENT_COL,
    RUL_MAX, CENSORING_STRATEGY, CENSORED_WEIGHT, BASELINE_WINDOW
)
from src.utils import get_logger

log = get_logger("label_engineer")


class RULLabelEngineer(BaseEstimator, TransformerMixin):
    """
    Generates piecewise-linear RUL labels from run-to-failure data.

    Parameters
    ----------
    rul_max : int
        Maximum RUL cap.  Observations with true RUL > rul_max get label=rul_max.
        This encodes the "healthy plateau" — sensors carry no information about
        how far a pump is from failure when it still has a long life ahead.
    censoring_strategy : str
        "exclude"            — censored rows get sample_weight=0 (default)
        "conservative_bound" — censored rows get synthetic RUL based on
                               the p90 empirical lifetime, with reduced weight
    censored_weight : float
        Weight assigned to censored rows under conservative_bound strategy.
    """

    def __init__(
        self,
        rul_max: int = RUL_MAX,
        censoring_strategy: str = CENSORING_STRATEGY,
        censored_weight: float = CENSORED_WEIGHT,
    ):
        self.rul_max = rul_max
        self.censoring_strategy = censoring_strategy
        self.censored_weight = censored_weight

    # ── Fit ───────────────────────────────────────
    def fit(self, train_df: pd.DataFrame, units_df: pd.DataFrame):
        """
        Learn unit-level metadata from training data.
        Must be called before transform().
        """
        self.failed_ids_   = set(
            units_df[units_df[EVENT_COL] == 1][UNIT_ID_COL].tolist()
        )
        self.censored_ids_ = set(
            units_df[units_df[EVENT_COL] == 0][UNIT_ID_COL].tolist()
        )
        # True last cycle per unit (= failure cycle for failed units)
        self.max_cycles_ = (
            train_df.groupby(UNIT_ID_COL)[CYCLE_COL].max().to_dict()
        )
        # Empirical lifetime distribution (failed units only) for conservative bound
        self.failed_lifetimes_ = np.array([
            self.max_cycles_[uid] for uid in self.failed_ids_
            if uid in self.max_cycles_
        ])
        self.p90_lifetime_ = (
            float(np.percentile(self.failed_lifetimes_, 90))
            if len(self.failed_lifetimes_) > 0 else 300.0
        )

        log.info(
            f"RULLabelEngineer fitted | rul_max={self.rul_max} | "
            f"failed={len(self.failed_ids_)} | "
            f"censored={len(self.censored_ids_)} | "
            f"p90_lifetime={self.p90_lifetime_:.0f}"
        )
        return self

    # ── Transform ─────────────────────────────────
    def transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns df with additional columns:
          raw_rul              — actual cycles to failure (NaN for censored)
          rul                  — piecewise-capped label (training target)
          sample_weight        — 1.0 for failed rows, 0.0 (or reduced) for censored
          in_degradation_window — bool, True when rul < rul_max
        """
        df = train_df.copy()
        df["raw_rul"]       = np.nan
        df["rul"]           = np.nan
        df["sample_weight"] = 0.0

        # ── Failed units ──
        for uid in self.failed_ids_:
            mask = df[UNIT_ID_COL] == uid
            if not mask.any():
                continue
            max_cyc = self.max_cycles_.get(uid, df.loc[mask, CYCLE_COL].max())
            raw = (max_cyc - df.loc[mask, CYCLE_COL]).clip(lower=0).astype(float)
            df.loc[mask, "raw_rul"]      = raw
            df.loc[mask, "rul"]          = raw.clip(upper=self.rul_max)
            df.loc[mask, "sample_weight"] = 1.0

        # ── Censored units ──
        for uid in self.censored_ids_:
            mask = df[UNIT_ID_COL] == uid
            if not mask.any():
                continue

            if self.censoring_strategy == "exclude":
                # raw_rul stays NaN; sample_weight stays 0 → excluded from fit
                pass

            elif self.censoring_strategy == "conservative_bound":
                # Assign synthetic RUL based on p90 empirical lifetime
                # Rationale: the pump *at least* survived to its last cycle;
                # we assume it would have failed around the fleet's p90 lifetime.
                # This is a lower bound, deliberately conservative (under-estimates RUL).
                max_cyc = self.max_cycles_.get(uid, df.loc[mask, CYCLE_COL].max())
                synthetic_raw = (self.p90_lifetime_ - df.loc[mask, CYCLE_COL]).clip(lower=0)
                df.loc[mask, "raw_rul"]       = synthetic_raw
                df.loc[mask, "rul"]           = synthetic_raw.clip(upper=self.rul_max)
                df.loc[mask, "sample_weight"] = self.censored_weight
            else:
                raise ValueError(
                    f"Unknown censoring_strategy: '{self.censoring_strategy}'. "
                    "Use 'exclude' or 'conservative_bound'."
                )

        # ── Degradation window flag ──
        df["in_degradation_window"] = df["rul"] < self.rul_max

        # ── Summary ──
        labeled_rows   = (df["sample_weight"] > 0).sum()
        unlabeled_rows = (df["sample_weight"] == 0).sum()
        log.info(
            f"Labels generated | labeled rows: {labeled_rows:,} | "
            f"excluded (censored): {unlabeled_rows:,} | "
            f"in degradation window: {df['in_degradation_window'].sum():,}"
        )
        return df

    # ── Convenience ───────────────────────────────
    def fit_transform(self, train_df: pd.DataFrame, units_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df, units_df).transform(train_df)

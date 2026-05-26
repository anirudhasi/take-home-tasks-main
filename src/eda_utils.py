"""
eda_utils.py — Reusable EDA functions for pump degradation analysis.

All functions return data structures AND (optionally) matplotlib figures
so they can be used both in notebooks and headless.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu, kstest
from scipy.signal import welch

from src.config import UNIT_ID_COL, CYCLE_COL, EVENT_COL
from src.utils import get_logger

log = get_logger("eda_utils")

# Colour palette
FAIL_COLOR    = "#E24B4A"
CENSOR_COLOR  = "#3B8BD4"
HEALTHY_COLOR = "#1D9E75"


# ─────────────────────────────────────────────────
# 1. Signal Quality Classification
# ─────────────────────────────────────────────────
def classify_signal_quality(
    train_df: pd.DataFrame,
    units_df: pd.DataFrame,
    sensor_cols: list,
    degradation_window_pct: float = 0.30
) -> pd.DataFrame:
    """
    For every sensor column, compute:
      - variance (global)
      - mean Spearman(sensor, cycles_to_failure) in last `degradation_window_pct` of life
      - autocorrelation at lag=1 (stationarity proxy)
      - missing pct

    Returns
    -------
    pd.DataFrame with columns:
      channel, variance, degradation_score, autocorr_lag1, missing_pct, quality_tier
    """
    failed_ids = units_df[units_df[EVENT_COL] == 1][UNIT_ID_COL].tolist()
    results = []

    for col in sensor_cols:
        corrs, autocorrs = [], []

        for uid in failed_ids:
            unit_data = (
                train_df[train_df[UNIT_ID_COL] == uid]
                .sort_values(CYCLE_COL)[col]
                .dropna()
                .values
            )
            n = len(unit_data)
            if n < 15:
                continue

            # Degradation window: last `degradation_window_pct` of life
            cut = int((1 - degradation_window_pct) * n)
            y = unit_data[cut:]
            x = np.arange(len(y))
            if len(y) > 3:
                corr, _ = spearmanr(x, y)
                if not np.isnan(corr):
                    corrs.append(corr)

            # Autocorrelation lag-1
            if n > 2:
                ac = np.corrcoef(unit_data[:-1], unit_data[1:])[0, 1]
                autocorrs.append(ac)

        variance     = train_df[col].var()
        deg_score    = float(np.nanmean(corrs)) if corrs else 0.0
        autocorr_l1  = float(np.nanmean(autocorrs)) if autocorrs else 0.0
        missing_pct  = train_df[col].isna().mean() * 100

        # Tier classification
        if variance < 1e-6:
            tier = "dead"
        elif missing_pct > 50:
            tier = "mostly_missing"
        elif abs(deg_score) > 0.25:
            tier = "degradation_signal"
        elif abs(autocorr_l1) < 0.1:
            tier = "noise_dominant"
        else:
            tier = "operational_or_weak"

        results.append({
            "channel":        col,
            "variance":       round(variance, 6),
            "degradation_score": round(deg_score, 4),
            "autocorr_lag1":  round(autocorr_l1, 4),
            "missing_pct":    round(missing_pct, 2),
            "quality_tier":   tier,
        })

    df_quality = pd.DataFrame(results)
    df_quality = df_quality.sort_values(
        "degradation_score", key=abs, ascending=False
    ).reset_index(drop=True)

    log.info("Signal quality summary:")
    for tier, grp in df_quality.groupby("quality_tier"):
        log.info(f"  {tier:25s}: {len(grp)} channels")

    return df_quality


# ─────────────────────────────────────────────────
# 2. Fleet Degradation Profile (aligned to failure)
# ─────────────────────────────────────────────────
def plot_fleet_degradation_profile(
    train_df: pd.DataFrame,
    units_df: pd.DataFrame,
    sensor_col: str,
    rul_max: int = 125,
    ax=None
):
    """
    Aligns all failed units to failure (x = cycles_to_failure),
    plots mean ± 1-std band.  Vertical dashed line at -rul_max
    marks where the piecewise label starts.
    """
    failed_ids = units_df[units_df[EVENT_COL] == 1][UNIT_ID_COL].tolist()
    max_life = train_df[train_df[UNIT_ID_COL].isin(failed_ids)].groupby(UNIT_ID_COL)[CYCLE_COL].max()

    # Align each unit to cycles_to_failure
    aligned = {}
    for uid in failed_ids:
        unit = train_df[train_df[UNIT_ID_COL] == uid].sort_values(CYCLE_COL)
        rul  = max_life[uid] - unit[CYCLE_COL].values
        vals = unit[sensor_col].values
        for r, v in zip(rul, vals):
            aligned.setdefault(int(r), []).append(v)

    # Only keep RUL values with >= 5 observations
    rul_vals = sorted([r for r, vs in aligned.items() if len(vs) >= 5])
    means = [np.nanmean(aligned[r]) for r in rul_vals]
    stds  = [np.nanstd(aligned[r])  for r in rul_vals]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    ax.fill_between(
        rul_vals,
        np.array(means) - np.array(stds),
        np.array(means) + np.array(stds),
        alpha=0.25, color=FAIL_COLOR, label="±1 std"
    )
    ax.plot(rul_vals, means, color=FAIL_COLOR, lw=2, label="Fleet mean")
    ax.axvline(rul_max, color="gray", ls="--", lw=1.2, label=f"RUL_MAX={rul_max}")
    ax.invert_xaxis()
    ax.set_xlabel("Cycles to failure (RUL)")
    ax.set_ylabel(sensor_col)
    ax.set_title(f"Fleet degradation profile — {sensor_col}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return ax


# ─────────────────────────────────────────────────
# 3. RUL_MAX Empirical Selection
# ─────────────────────────────────────────────────
def select_rul_max_empirically(
    train_df: pd.DataFrame,
    units_df: pd.DataFrame,
    sensor_cols: list,
    search_range: tuple = (50, 201, 5),
    plot: bool = True
) -> int:
    """
    For each candidate RUL_MAX, compute mean |Spearman(sensor, rul)| across all
    degradation-window observations of failed units.

    Returns the RUL_MAX that maximises signal-to-noise ratio.
    """
    from src.label_engineer import RULLabelEngineer

    failed_ids = units_df[units_df[EVENT_COL] == 1][UNIT_ID_COL].tolist()
    candidates = range(*search_range)
    results = []

    for rul_max in candidates:
        le = RULLabelEngineer(rul_max=rul_max, censoring_strategy="exclude")
        le.fit(train_df, units_df)
        df_lab = le.transform(train_df)

        dw = df_lab[
            (df_lab["in_degradation_window"]) &
            (df_lab[UNIT_ID_COL].isin(failed_ids))
        ]
        if len(dw) < 50:
            results.append({"rul_max": rul_max, "mean_abs_corr": 0.0})
            continue

        corrs = []
        for col in sensor_cols:
            if dw[col].isna().all():
                continue
            c, _ = spearmanr(
                dw[col].fillna(dw[col].median()).values,
                dw["rul"].values
            )
            if not np.isnan(c):
                corrs.append(abs(c))

        results.append({
            "rul_max":       rul_max,
            "mean_abs_corr": float(np.mean(corrs)) if corrs else 0.0
        })

    df_res = pd.DataFrame(results)
    best_rul_max = int(df_res.loc[df_res["mean_abs_corr"].idxmax(), "rul_max"])
    log.info(f"Empirically selected RUL_MAX = {best_rul_max} "
             f"(mean |Spearman| = "
             f"{df_res['mean_abs_corr'].max():.4f})")

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_res["rul_max"], df_res["mean_abs_corr"], lw=2, color=FAIL_COLOR)
        ax.axvline(best_rul_max, color="gray", ls="--",
                   label=f"Best: {best_rul_max} cycles")
        ax.set_xlabel("Candidate RUL_MAX (cycles)")
        ax.set_ylabel("Mean |Spearman corr| in degradation window")
        ax.set_title("Empirical RUL_MAX selection")
        ax.legend(); plt.tight_layout(); plt.show()

    return best_rul_max


# ─────────────────────────────────────────────────
# 4. Censoring EDA (Kaplan-Meier + distributional test)
# ─────────────────────────────────────────────────
def censoring_eda(
    train_df: pd.DataFrame,
    units_df: pd.DataFrame,
    sensor_cols: list,
    cycle_window: tuple = (50, 100),
    significance: float = 0.05
) -> dict:
    """
    1. Kaplan-Meier survival estimate.
    2. Mann-Whitney U test: are censored and failed units distinguishable
       at matched cycle windows?

    Returns dict with KMF fitter and per-channel test results.
    """
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        log.warning("lifelines not installed — skipping KM estimate")
        return {}

    kmf = KaplanMeierFitter()
    kmf.fit(
        units_df["cycles_recorded"],
        event_observed=units_df[EVENT_COL],
        label="Fleet survival"
    )

    log.info("Kaplan-Meier median survival time: "
             f"{kmf.median_survival_time_:.0f} cycles")

    failed_ids   = units_df[units_df[EVENT_COL] == 1][UNIT_ID_COL].tolist()
    censored_ids = units_df[units_df[EVENT_COL] == 0][UNIT_ID_COL].tolist()

    test_results = []
    for col in sensor_cols[:10]:   # Limit to avoid clutter
        fail_vals = train_df[
            (train_df[UNIT_ID_COL].isin(failed_ids)) &
            (train_df[CYCLE_COL].between(*cycle_window))
        ][col].dropna()

        cens_vals = train_df[
            (train_df[UNIT_ID_COL].isin(censored_ids)) &
            (train_df[CYCLE_COL].between(*cycle_window))
        ][col].dropna()

        if len(fail_vals) < 5 or len(cens_vals) < 3:
            continue

        stat, p = mannwhitneyu(fail_vals, cens_vals, alternative="two-sided")
        test_results.append({
            "channel": col,
            "mw_stat": round(stat, 2),
            "p_value": round(p, 4),
            "significant": p < significance,
        })

    df_tests = pd.DataFrame(test_results)
    n_sig = df_tests["significant"].sum() if len(df_tests) else 0
    log.info(f"Censoring test: {n_sig}/{len(df_tests)} channels "
             f"significantly different at cycles {cycle_window}")

    return {"kmf": kmf, "channel_tests": df_tests}


# ─────────────────────────────────────────────────
# 5. Correlation Heatmap
# ─────────────────────────────────────────────────
def plot_sensor_correlation_heatmap(
    train_df: pd.DataFrame,
    sensor_cols: list,
    figsize=(12, 10)
) -> pd.DataFrame:
    """
    Spearman correlation matrix across all sensor channels.
    Highlights clusters of collinear signals.
    """
    corr = train_df[sensor_cols].dropna().corr(method="spearman")

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=False, cmap="coolwarm",
        vmin=-1, vmax=1, center=0, linewidths=0.3,
        ax=ax
    )
    ax.set_title("Sensor Spearman correlation matrix", fontsize=14, pad=12)
    plt.tight_layout()
    plt.show()
    return corr


# ─────────────────────────────────────────────────
# 6. Unit Lifetime Distribution
# ─────────────────────────────────────────────────
def plot_lifetime_distribution(units_df: pd.DataFrame, ax=None):
    """Histogram + KDE of unit lifetimes, coloured by event type."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    failed   = units_df[units_df[EVENT_COL] == 1]["cycles_recorded"]
    censored = units_df[units_df[EVENT_COL] == 0]["cycles_recorded"]

    ax.hist(failed, bins=20, color=FAIL_COLOR, alpha=0.7, label=f"Failed (n={len(failed)})")
    ax.hist(censored, bins=10, color=CENSOR_COLOR, alpha=0.7,
            label=f"Censored (n={len(censored)})")
    ax.axvline(failed.median(), color=FAIL_COLOR, ls="--", lw=1.5,
               label=f"Median failed: {failed.median():.0f}")
    ax.set_xlabel("Recorded lifetime (cycles)")
    ax.set_ylabel("Count")
    ax.set_title("Unit lifetime distribution by event type")
    ax.legend()
    plt.tight_layout()
    return ax

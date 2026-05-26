"""
evaluate.py  --  Adds actual_rul + error metrics to predictions.csv
                 and prints overall accuracy summary.
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')

from src.utils import load_pickle
from src.config import ARTIFACTS_DIR
from src.data_loader import load_and_validate_data
from src.validation import compute_metrics
from src.model import phm_score

# ── load artifacts ──
cfg   = load_pickle(ARTIFACTS_DIR / 'train_config.pkl')
model = load_pickle(ARTIFACTS_DIR / 'ensemble_model.pkl')
fe    = load_pickle(ARTIFACTS_DIR / 'feature_pipeline.pkl')
fcols = load_pickle(ARTIFACTS_DIR / 'feature_cols.pkl')
rul_max = cfg['rul_max']

# ── load ground-truth (train.csv has true failure cycle for each unit) ──
train_df, units_df = load_and_validate_data(
    'data/train.csv', 'data/train_units.csv', verbose=False
)
units_df.columns = [c.strip().lower().replace(' ', '_') for c in units_df.columns]
units_df = units_df.rename(columns={'n_cycles': 'cycles_recorded', 'event': 'event_flag'})

# True total cycles per failed unit
true_total = (
    train_df.groupby('unit_id')['cycle'].max()
    .rename('true_total_cycles')
    .reset_index()
)
units_info = units_df.merge(true_total, on='unit_id')

# ── load predictions ──
preds = pd.read_csv('data/predictions.csv')

# ── compute actual RUL for each unit ──
rows = []
for _, row in preds.iterrows():
    uid = row['unit_id']
    last_obs = row['last_cycle']
    info = units_info[units_info['unit_id'] == uid]

    if len(info) == 0 or info.iloc[0]['event_flag'] != 1:
        actual_raw = None  # censored — true failure unknown
    else:
        true_end = int(info.iloc[0]['true_total_cycles'])
        actual_raw = max(0, true_end - last_obs)

    # Cap at rul_max (same as model training)
    actual_capped = min(actual_raw, rul_max) if actual_raw is not None else None
    rows.append({'unit_id': uid, 'actual_rul_raw': actual_raw, 'actual_rul': actual_capped})

actual_df = pd.DataFrame(rows)
result = preds.merge(actual_df, on='unit_id')

# ── compute per-unit error ──
valid = result['actual_rul'].notna()
result['error'] = np.nan
result['abs_error'] = np.nan
result['error_pct'] = np.nan

result.loc[valid, 'error']     = (result.loc[valid, 'predicted_rul'] - result.loc[valid, 'actual_rul']).round(1)
result.loc[valid, 'abs_error'] = result.loc[valid, 'error'].abs().round(1)
result.loc[valid, 'error_pct'] = (
    result.loc[valid, 'abs_error'] / (result.loc[valid, 'actual_rul'] + 1e-6) * 100
).round(1)

# ── reorder columns ──
col_order = [
    'unit_id', 'predicted_rul', 'actual_rul', 'actual_rul_raw',
    'error', 'abs_error', 'error_pct',
    'failure_prob_30', 'health_index', 'prediction_std',
    'warning_level', 'last_cycle', 'n_cycles_observed'
]
result = result[col_order]

# ── save ──
result.to_csv('data/predictions.csv', index=False)
print("Updated data/predictions.csv with actual_rul and error columns.\n")
print(result[['unit_id','predicted_rul','actual_rul','error','abs_error','warning_level']].to_string(index=False))

# ── overall accuracy summary ──
v = result[valid]
y_pred = v['predicted_rul'].values
y_true = v['actual_rul'].values

rmse         = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
mae          = float(np.mean(np.abs(y_pred - y_true)))
within_30    = float(np.mean(np.abs(y_pred - y_true) <= 30) * 100)
within_20    = float(np.mean(np.abs(y_pred - y_true) <= 20) * 100)
phm          = phm_score(y_pred, y_true)
late_pct     = float(np.mean(y_pred > y_true) * 100)

print(f"""
{'='*48}
MODEL ACCURACY SUMMARY  (n={len(v)} units)
{'='*48}
  RMSE                : {rmse:.1f} cycles
  MAE                 : {mae:.1f} cycles
  Within 30 cycles    : {within_30:.0f}%   (predictions within ±30 of truth)
  Within 20 cycles    : {within_20:.0f}%   (predictions within ±20 of truth)
  PHM Score           : {phm:.1f}          (lower is better)
  Late predictions    : {late_pct:.0f}%   (model over-estimated RUL → risky)
{'='*48}
Note: rul_max = {rul_max} cycles (degradation window cap)
""")

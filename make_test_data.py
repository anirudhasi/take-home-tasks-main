import sys; sys.path.insert(0, '.')
import pandas as pd, warnings; warnings.filterwarnings('ignore')

train_df = pd.read_csv('data/train.csv')
train_df.columns = [c.strip().lower().replace(' ', '_') for c in train_df.columns]

# Cut each unit so remaining = total_cycles - cut_at_cycle
# Negative offset means "cut N cycles before the end"
targets = {
    # RED: <=30 cycles remaining — cut right at the end
    'PUMP_005': -3,    # 165 total -> cut at 162 ->  3 remaining
    'PUMP_011': -5,    # 156 total -> cut at 151 ->  5 remaining
    'PUMP_008': -8,    # 180 total -> cut at 172 ->  8 remaining
    'PUMP_014': -12,   # 128 total -> cut at 116 -> 12 remaining
    'PUMP_009': -25,   # 231 total -> cut at 206 -> 25 remaining
    # AMBER: 31-60 cycles remaining
    'PUMP_002': -40,   # 197 total -> cut at 157 -> 40 remaining
    'PUMP_001': -55,   # 292 total -> cut at 237 -> 55 remaining
    'PUMP_015': -50,   # 222 total -> cut at 172 -> 50 remaining
    # GREEN: >60 cycles remaining
    'PUMP_004': -80,   # 319 total -> cut at 239 -> 80 remaining
    'PUMP_006': -90,   # 315 total -> cut at 225 -> 90 remaining
    'PUMP_010': -70,   # 237 total -> cut at 167 -> 70 remaining
}

frames = []
for uid, offset in targets.items():
    unit = train_df[train_df['unit_id'] == uid].sort_values('cycle')
    total = len(unit)
    cut = total + offset
    remaining = total - cut
    level = 'RED' if remaining <= 30 else 'AMBER' if remaining <= 60 else 'GREEN'
    print(f'{uid}: cut at cycle {cut}/{total}  remaining={remaining}  expected={level}')
    frames.append(unit.iloc[:cut])

test_df = pd.concat(frames, ignore_index=True)
drop_cols = [c for c in ['timestamp'] if c in test_df.columns]
test_df = test_df.drop(columns=drop_cols)
test_df.to_csv('data/my_test_input.csv', index=False)
print(f'\nSaved {len(test_df)} rows, {test_df["unit_id"].nunique()} pumps -> data/my_test_input.csv')

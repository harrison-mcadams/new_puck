import pandas as pd
import sys
from pathlib import Path

path = Path('analysis/nested_xgs/test_predictions.csv')
if not path.exists():
    print("File not found")
    sys.exit(1)

df = pd.read_csv(path)
df = df[(df['event'] != 'blocked-shot') & (df['mp_xGoal'].notna())].copy()
df['xg_diff'] = df['xG'] - df['mp_xGoal']

# 1. Dist Bins
print("--- Distance Bins ---")
df['dist_bin'] = pd.cut(df['distance'], bins=[0, 10, 20, 30, 40, 50, 60, 100])
print(df.groupby('dist_bin', observed=True)[['xG', 'mp_xGoal', 'xg_diff']].mean())

# 2. Shot Type
print("\n--- Shot Types ---")
st = df.groupby('shot_type')[['xG', 'mp_xGoal', 'xg_diff']].agg(['mean', 'count'])
print(st[st[('xG', 'count')] > 100])

# 3. Rebound
if 'is_rebound' in df.columns:
    print("\n--- Rebounds ---")
    print(df.groupby('is_rebound')[['xG', 'mp_xGoal', 'xg_diff']].mean())

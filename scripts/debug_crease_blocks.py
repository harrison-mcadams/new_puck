import pandas as pd
import os

# Try probable paths
paths = [
    'scripts/analysis/nested_xgs/test_predictions.csv',
    'analysis/nested_xgs/test_predictions.csv',
    'test_predictions.csv'
]

f = None
for p in paths:
    if os.path.exists(p):
        f = p
        break

if not f:
    print("Could not find test_predictions.csv")
    exit(1)

print(f"Reading {f}...")
df = pd.read_csv(f)

# Filter to blocked
blk = df[df['is_blocked'] == 1]

# Sort by xG
top_blk = blk.sort_values('xG', ascending=False).head(20)

print('\n--- TOP 20 HIGH xG BLOCKED SHOTS ---')
cols = ['xG', 'distance', 'angle_deg', 'prob_finish']
# Check for coordinate columns
coords = []
if 'x' in df.columns: coords.extend(['x', 'y'])
if 'imputed_x' in df.columns: coords.extend(['imputed_x', 'imputed_y'])

print(top_blk[cols + coords].to_string(index=False))

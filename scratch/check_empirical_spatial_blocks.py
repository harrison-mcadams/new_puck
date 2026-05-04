import pandas as pd
import numpy as np

df = pd.read_csv('data/20252026/20252026_df.csv', low_memory=False)
df = df[df['event'].isin(['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal'])]

# Standardize coordinates to attacking right
df['x'] = pd.to_numeric(df['x'], errors='coerce')
df['y'] = pd.to_numeric(df['y'], errors='coerce')
df.loc[df['x'] < 0, 'y'] *= -1
df.loc[df['x'] < 0, 'x'] *= -1

# Bin the offensive zone
# X from 25 to 89, bins of 10
# Y from -42.5 to 42.5, bins of 10
df['x_bin'] = pd.cut(df['x'], bins=np.arange(25, 95, 10))
df['y_bin'] = pd.cut(df['y'], bins=np.arange(-45, 55, 10))

summary = df.groupby(['x_bin', 'y_bin'])['event'].agg(
    total='count',
    blocks=lambda x: (x == 'blocked-shot').sum()
).reset_index()

summary['block_rate'] = summary['blocks'] / summary['total']
summary['block_rate'] = summary['block_rate'].fillna(0)

# Print as a nice table
print(summary.pivot(index='y_bin', columns='x_bin', values='block_rate').round(3))

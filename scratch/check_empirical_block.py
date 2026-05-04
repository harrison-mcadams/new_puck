import pandas as pd

df = pd.read_csv('data/20252026/20252026_df.csv', low_memory=False)
df = df[df['event'].isin(['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal'])]

print(f"Total shots: {len(df)}")
block_rate = (df['event'] == 'blocked-shot').mean()
print(f"Overall block rate: {block_rate:.3f}")

z1 = df[(df['distance'] < 30) & (df['distance'] >= 10)]
print(f"Block rate (10-30ft): {(z1['event'] == 'blocked-shot').mean():.3f} (N={len(z1)})")

z2 = df[(df['distance'] < 60) & (df['distance'] >= 30)]
print(f"Block rate (30-60ft): {(z2['event'] == 'blocked-shot').mean():.3f} (N={len(z2)})")

z3 = df[(df['distance'] >= 60)]
print(f"Block rate (>60ft): {(z3['event'] == 'blocked-shot').mean():.3f} (N={len(z3)})")

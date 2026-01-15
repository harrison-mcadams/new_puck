
import pandas as pd
import numpy as np

path = 'analysis/debug_imputation_pipeline.csv'
print(f"Loading {path}...")
df = pd.read_csv(path)

print(f"Total rows: {len(df)}")
df_blocked = df[df['event'] == 'blocked-shot']
print(f"Blocked shots: {len(df_blocked)}")

# Inspect "Defensive Zone" blocks (assuming x is standard 0-100 scale? No, NHL is -100 to 100)
# But here we have 'x' and 'x_adj'.
# Let's check distribution of x.
print("\n--- 'x' Distribution (Raw?) ---")
print(df_blocked['x'].describe())

print("\n--- 'x_adj' Distribution (Adjusted) ---")
print(df_blocked['x_adj'].describe())

print("\n--- 'distance' Distribution ---")
print(df_blocked['distance'].describe())

# Check for "High Danger" blocks that shouldn't be high danger
# Low distance blocks
df_close = df_blocked[df_blocked['distance'] < 20]
print(f"\n--- Close Range Blocks (< 20ft): {len(df_close)} ---")
if not df_close.empty:
    print(df_close[['x', 'x_adj', 'y', 'y_adj', 'distance']].head(10))

# Check for "Defensive Zone" raw coords
# If raw x is relative to center (0), -25 to -100 is defensive?
# Or if it's 0-200?
# Usually NHL API is -100 to +100.
# If we have positive x only (already flipped?), then x < 25 is defensive.
# Let's see the range of 'x'.

# Look for correlation between x_adj and distance.
# If x_adj is near 89 (net), distance should be small.

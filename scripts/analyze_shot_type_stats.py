import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from puck import data_pipeline, fit_xgs

print("Loading data...")
try:
    df = fit_xgs.load_data()  # Might need load_all_seasons if available, but this is checking logic
except:
    # Fallback to loading data directory manually if needed, 
    # but fit_xgs.load_data() usually works for default
    df = pd.read_csv('data/20232024.csv') # Fallback example

if df.empty:
    print("No data found.")
    sys.exit(0)

# Apply preprocessing to get valid columns
print("Preprocessing...")
df = data_pipeline.preprocess_features(df, is_training=False, apply_imputation=True) # Use imputation for stats

# Filter to shots
df_shots = df[df['event'].isin(['shot-on-goal', 'goal', 'missed-shot'])]

print("\n--- Shot Type Stats ---")
stats = df_shots.groupby('shot_type')['event'].apply(
    lambda x: pd.Series({
        'count': len(x), 
        'goals': (x == 'goal').sum(),
        'freq': len(x) / len(df_shots)
    })
).unstack()

stats['obs_shooting_percentage'] = stats['goals'] / stats['count']

print(stats.sort_values('freq', ascending=False))

print("\n--- Implied Marginalized Shooting % ---")
# Weighted average of Obs Sum
weighted_avg = (stats['obs_shooting_percentage'] * stats['freq']).sum()
print(f"Weighted Average Sh%: {weighted_avg:.4f}")

wrist_sh = stats.loc['wrist', 'obs_shooting_percentage'] if 'wrist' in stats.index else 0
print(f"Wrist Sh%: {wrist_sh:.4f}")

if weighted_avg > wrist_sh:
    print(f"\nRESULT: Marginalized ({weighted_avg:.4f}) is HIGHER than Wrist ({wrist_sh:.4f}).")
    print("This explains why 'Unknown' xG appears higher than the standard 'Wrist' baseline.")
else:
    print(f"\nRESULT: Marginalized ({weighted_avg:.4f}) is LOWER/EQUAL to Wrist.")

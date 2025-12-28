
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import analyze
from puck import fit_xgs

print("Loading Data...")
if os.path.exists('data/20252026.csv'):
    df = pd.read_csv('data/20252026.csv')
elif os.path.exists('data/20252026/20252026_df.csv'):
    df = pd.read_csv('data/20252026/20252026_df.csv')
else:
    raise FileNotFoundError("Could not find season data CSV")
print(f"Data Loaded: {len(df)} rows")

if 'shot_type' not in df.columns:
    print("WARNING: shot_type missing from CSV")
else:
    print(f"shot_type present. NaNs: {df['shot_type'].isna().sum()}")

print("Running _predict_xgs...")
try:
    df_pred, clf, meta = analyze._predict_xgs(df)
    
    if 'xgs' in df_pred.columns:
        valid_xg = df_pred['xgs'].dropna()
        print(f"Prediction Complete. xG Clean Count: {len(valid_xg)}")
        print(f"Total xG: {valid_xg.sum()}")
        print(f"Mean xG: {valid_xg.mean()}")
        
        # Check goals
        goals = df_pred[df_pred['event'] == 'goal'].shape[0]
        print(f"Total Goals: {goals}")
        
        print("\n--- xG by Event Type ---")
        breakdown = df_pred.groupby('event')['xgs'].agg(['count', 'sum', 'mean']).sort_values('sum', ascending=False)
        print(breakdown)

        
    else:
        print("Error: 'xgs' column not created.")
except Exception as e:
    print(f"CRASH in _predict_xgs: {e}")
    import traceback
    traceback.print_exc()


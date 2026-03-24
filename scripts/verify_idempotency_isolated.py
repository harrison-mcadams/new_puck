
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from puck import data_pipeline

def verify():
    print("--- Isolated Verification of Idempotency ---")
    
    # Load sample raw-ish data (re-fetch a few games or load existing)
    csv_path = 'data/20252026.csv'
    df = pd.read_csv(csv_path)
    
    # Filter for a few shots
    shots = df[df.event.isin(['shot-on-goal','goal'])].head(10).copy()
    print(f"Testing on {len(shots)} shots.")
    
    # Pass 1
    print("\nPass 1 execution...")
    p1 = data_pipeline.preprocess_features(shots, is_training=False, verbose=True)
    
    # Pass 2
    print("\nPass 2 execution...")
    p2 = data_pipeline.preprocess_features(p1.copy(), is_training=False, verbose=True)
    
    # Compare
    diffs = np.abs(p1['x'].values - p2['x'].values)
    print(f"\nMax X Diff: {diffs.max()}")
    if diffs.max() > 1e-3:
        idx = np.argmax(diffs)
        print(f"Row {idx} mismatched!")
        print(f"  P1: x={p1.iloc[idx]['x']}, x_adj={p1.iloc[idx]['x_adj']}, side={p1.iloc[idx]['home_team_defending_side']}, is_home={p1.iloc[idx]['is_home']}")
        print(f"  P2: x={p2.iloc[idx]['x']}, x_adj={p2.iloc[idx]['x_adj']}, side={p2.iloc[idx]['home_team_defending_side']}, is_home={p2.iloc[idx]['is_home']}")
    else:
        print("SUCCESS: Idempotency verified.")

if __name__ == "__main__":
    verify()

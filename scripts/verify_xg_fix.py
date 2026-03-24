
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from puck import analyze
from puck import config

def verify():
    print("--- Verifying xG Inflation Fix ---")
    
    season = '20252026'
    csv_path = 'data/20252026.csv'
    
    if not os.path.exists(csv_path):
        print(f"Data file {csv_path} not found. Running daily update to generate it components...")
        return

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")
    
    # Check orientation
    shot_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    df_shots = df[df['event'].isin(shot_events)].copy()
    
    total_shots = len(df_shots)
    if total_shots == 0:
        print("No shots found. Aborting.")
        return

    pos_x = (df_shots['x'] > 0).sum()
    neg_x = (df_shots['x'] < 0).sum()
    
    print(f"Total Shots: {total_shots}")
    print(f"Shots on Right (X > 0): {pos_x} ({100*pos_x/total_shots:.1f}%)")
    print(f"Shots on Left (X < 0): {neg_x} ({100*neg_x/total_shots:.1f}%)")
    
    # Check xG values
    if 'xgs' in df_shots.columns:
        total_xg = df_shots['xgs'].sum()
        total_goals = (df_shots['event'] == 'goal').sum()
        ratio = total_xg / total_goals if total_goals > 0 else 0
        
        print(f"Total xG: {total_xg:.2f}")
        print(f"Total Goals: {total_goals}")
        print(f"xG/Goal Ratio: {ratio:.2f}")
        
        if 0.5 < ratio < 1.5:
            print("SUCCESS: xG/Goal ratio is within normal range.")
        else:
            print(f"WARNING: xG/Goal ratio ({ratio:.2f}) is still potentially abnormal.")
    else:
        print("Column 'xgs' not found.")

    # Test Idempotency
    print("\nTesting Idempotency of Preprocessing Pipeline...")
    from puck import data_pipeline
    
    # Sample a few shots
    sample_df = df_shots.head(100).copy()
    
    # Pass 1
    processed_1 = data_pipeline.preprocess_features(sample_df, is_training=False)
    # Pass 2
    processed_2 = data_pipeline.preprocess_features(processed_1.copy(), is_training=False)
    
    # Compare
    diff_x = np.abs(processed_1['x'].values - processed_2['x'].values).max()
    diff_dist = np.abs(processed_1['distance'].values - processed_2['distance'].values).max()
    
    print(f"Max difference in X after second pass: {diff_x:.6f}")
    print(f"Max difference in distance after second pass: {diff_dist:.6f}")
    
    if diff_x < 1e-6 and diff_dist < 1e-6:
        print("SUCCESS: Pipeline is idempotent.")
    else:
        print("FAILURE: Pipeline is NOT idempotent.")

if __name__ == "__main__":
    verify()

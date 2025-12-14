
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_nested_xgs

def analyze_unknown():
    print("Loading data...")
    # Load just one season for speed if possible, or all
    # We can try loading 20252026 
    df = pd.read_csv('data/20252026/20252026_df.csv')
    print(f"Loaded {len(df)} rows.")

    # Mimic fit logic
    df['is_blocked'] = (df['event'] == 'blocked-shot').astype(int)
    
    # Accuracy model is trained on UNBLOCKED shots
    df_unblocked = df[df['is_blocked'] == 0].copy()
    print(f"Unblocked shots: {len(df_unblocked)}")

    if 'shot_type' not in df_unblocked.columns:
        print("shot_type column missing!")
        return

    # Fill NA
    df_unblocked['shot_type'] = df_unblocked['shot_type'].fillna('Unknown')
    
    # Target: is_on_net (shot-on-goal or goal)
    df_unblocked['is_on_net'] = df_unblocked['event'].isin(['shot-on-goal', 'goal']).astype(int)

    print("\n--- Shot Type Distribution (Unblocked) ---")
    dist = df_unblocked['shot_type'].value_counts()
    print(dist)

    print("\n--- Accuracy (Is On Net) by Shot Type ---")
    stats = df_unblocked.groupby('shot_type')['is_on_net'].agg(['count', 'mean', 'sum'])
    stats.columns = ['Total', 'OnNet_Rate', 'OnNet_Count']
    stats = stats.sort_values('OnNet_Rate', ascending=False)
    print(stats)
    
    print("\n--- Events mapped to 'Unknown' Shot Type ---")
    unknowns = df_unblocked[df_unblocked['shot_type'] == 'Unknown']
    if not unknowns.empty:
        print(unknowns['event'].value_counts())
    else:
        print("No 'Unknown' shot types found in unblocked data??")

if __name__ == "__main__":
    analyze_unknown()

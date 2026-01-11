
import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.getcwd())
from puck import fit_xgs, data_pipeline

def audit():
    print("Loading data...")
    df = fit_xgs.load_all_seasons_data() 
    df_p = data_pipeline.preprocess_features(df, apply_imputation=True)
    
    valid_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    df_shots = df_p[df_p['event'].isin(valid_events)].copy()
    df_shots['is_blk'] = (df_shots['event'] == 'blocked-shot').astype(int)
    
    # Check Binary Features
    print("\n--- Feature Leakage: Binary Flags ---")
    for col in ['is_rebound', 'is_rush']:
        print(f"\n{col}:")
        print(df_shots.groupby('is_blk')[col].value_counts(normalize=True))
        
    # Check Angle Concentration
    print("\n--- Angle Distribution (Blocks vs Shots) ---")
    df_shots['angle_bin'] = (df_shots['angle_deg'] // 5) * 5
    ang = df_shots.groupby(['is_blk', 'angle_bin']).size().unstack(fill_value=0)
    ang_pct = ang.div(ang.sum(axis=1), axis=0)
    print(ang_pct.T.head(10))

if __name__ == "__main__":
    audit()

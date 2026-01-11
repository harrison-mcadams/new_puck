
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
    
    # Check nulls in shooter_role
    print("\n--- shooter_role Null Counts ---")
    df_shots['role_is_null'] = df_shots['shooter_role'].isna()
    print(df_shots.groupby('is_blk')['role_is_null'].value_counts(normalize=True))
    
    # Check value distributions including 'Unknown' or whatever it is
    print("\n--- shooter_role Value Counts (Normalized) ---")
    print(df_shots.groupby('is_blk')['shooter_role'].value_counts(normalize=True, dropna=False))

if __name__ == "__main__":
    audit()

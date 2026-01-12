
import pandas as pd
import numpy as np
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import fit_xgs

def check_quality():
    print("Loading data for Refined QC check...")
    # fit_xgs.load_data() loads all seasons in data/
    df = fit_xgs.load_data()
    
    # Filter for SHOTS only
    shot_events = ['shot-on-goal', 'blocked-shot', 'missed-shot', 'goal']
    df_shots = df[df['event'].isin(shot_events)].copy()
    
    print(f"Total Rows: {len(df)}")
    print(f"Shot Rows: {len(df_shots)}")
    
    # metrics to check
    # Note: 'player_role' is derived, check 'primary_position' instead
    cols_to_check = ['shoots_catches', 'shot_type', 'primary_position']
    
    print("\n--- Data Quality Report (Shot Events Only) ---")
    
    for c in cols_to_check:
        if c not in df_shots.columns:
            print(f"Column {c} missing!")
            continue
            
        n_total = len(df_shots)
        
        # Count 'Unknown', 'None', or NaNs
        # Normalize to lower string
        s = df_shots[c].astype(str).str.lower().str.strip()
        bad_mask = s.isin(['unknown', 'nan', 'none', ''])
        
        bad_count = bad_mask.sum()
        pct = (bad_count / n_total) * 100
        
        print(f"Feature '{c}':")
        print(f"  Valid: {n_total - bad_count}")
        print(f"  Missing/Unknown: {bad_count} ({pct:.2f}%)")
        print(f"  Top 5 Values:\n{df_shots[c].value_counts().head(5)}")
        print("-" * 30)

if __name__ == "__main__":
    check_quality()

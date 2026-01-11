
import pandas as pd
import numpy as np
import sys, os

# Ensure we can import puck
sys.path.append(os.getcwd())
from puck import fit_xgs, data_pipeline

def audit():
    print("Loading data...")
    # Load 1 season for speed, or a few.
    df = fit_xgs.load_data() # Usually current/prev season
    
    print("Preprocessing (with Imputation)...")
    df_p = data_pipeline.preprocess_features(
        df, 
        is_training=True, 
        apply_imputation=True, 
        apply_arena_adjustments=True
    )
    
    # Filter to shots
    shot_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    df_shots = df_p[df_p['event'].isin(shot_events)].copy()
    
    # Bin by distance
    df_shots['dist_bin'] = (df_shots['distance'] // 5) * 5
    
    # Calculate Ratio by Role and Dist
    summary = df_shots.groupby(['shooter_role', 'dist_bin', 'event']).size().unstack(fill_value=0)
    
    # Calculate P(Blocked) = Blocks / (Blocks + Unblocked)
    summary['unblocked'] = summary.get('shot-on-goal', 0) + summary.get('missed-shot', 0) + summary.get('goal', 0)
    summary['blocks'] = summary.get('blocked-shot', 0)
    summary['total'] = summary['blocks'] + summary['unblocked']
    summary['p_blocked_raw'] = summary['blocks'] / summary['total']
    
    print("\nAuditing P(Blocked) by Distance (Forwards):")
    print(summary.loc['F'][['blocks', 'unblocked', 'total', 'p_blocked_raw']].query('total > 100').head(20))
    
    print("\nAuditing P(Blocked) by Distance (Defensemen):")
    print(summary.loc['D'][['blocks', 'unblocked', 'total', 'p_blocked_raw']].query('total > 100').head(20))

if __name__ == "__main__":
    audit()

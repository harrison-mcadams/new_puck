
import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.getcwd())
from puck import fit_xgs, data_pipeline

def find_clusters():
    print("Loading data...")
    df = fit_xgs.load_all_seasons_data()
    
    print("Preprocessing...")
    df_p = data_pipeline.preprocess_features(df, apply_imputation=True)
    
    valid_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    df_shots = df_p[df_p['event'].isin(valid_events)].copy()
    df_shots['is_blk'] = (df_shots['event'] == 'blocked-shot').astype(int)
    
    # Bins
    df_shots['dist_bin'] = (df_shots['distance'] // 10) * 10
    df_shots['angle_bin'] = (df_shots['angle_deg'] // 10) * 10
    
    # Group by potential splitting features
    cols = ['shooter_role', 'dist_bin', 'angle_bin', 'game_state']
    
    print("\n--- Top High-Probability Segments ---")
    res = df_shots.groupby(cols)['is_blk'].agg(['mean', 'count'])
    
    # Filter to segments with significant volume
    res_high = res[res['count'] > 1000].sort_values('mean', ascending=False)
    print(res_high.head(30))

if __name__ == "__main__":
    find_clusters()

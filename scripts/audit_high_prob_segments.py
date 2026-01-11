
import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.getcwd())
from puck import fit_xgs, data_pipeline

def audit():
    print("Loading data...")
    df = fit_xgs.load_data()
    df_p = data_pipeline.preprocess_features(df, apply_imputation=True)
    
    valid_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    df_shots = df_p[df_p['event'].isin(valid_events)].copy()
    df_shots['is_blk'] = (df_shots['event'] == 'blocked-shot').astype(int)
    
    # Check Role + Distance
    df_shots['dist_bin'] = (df_shots['distance'] // 10) * 10
    
    print("\n--- P(Blocked) by Role and Distance ---")
    res = df_shots.groupby(['shooter_role', 'dist_bin'])['is_blk'].agg(['mean', 'count'])
    print(res[res['count'] > 500])
    
    # Check Role + Distance + Game State
    print("\n--- P(Blocked) by Role, Distance, and Game State ---")
    res_gs = df_shots.groupby(['shooter_role', 'dist_bin', 'game_state'])['is_blk'].agg(['mean', 'count'])
    print(res_gs[res_gs['mean'] > 0.6].query('count > 100'))
    
    # Check Role + Distance + Last Event
    print("\n--- P(Blocked) by Role, Distance, and Last Event ---")
    res_le = df_shots.groupby(['shooter_role', 'dist_bin', 'last_event_type'])['is_blk'].agg(['mean', 'count'])
    print(res_le[res_le['mean'] > 0.6].query('count > 100'))

if __name__ == "__main__":
    audit()

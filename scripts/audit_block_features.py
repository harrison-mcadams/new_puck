
import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.getcwd())
from puck import fit_xgs, data_pipeline

def audit():
    print("Loading data...")
    df = fit_xgs.load_data().sample(200000, random_state=42)
    
    print("Preprocessing...")
    df_p = data_pipeline.preprocess_features(df, apply_imputation=True)
    
    valid_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    df_shots = df_p[df_p['event'].isin(valid_events)].copy()
    df_shots['is_blk'] = (df_shots['event'] == 'blocked-shot').astype(int)
    
    # Check top features
    print("\n--- Distribution of last_event_type ---")
    dist = df_shots.groupby(['is_blk', 'last_event_type']).size().unstack(fill_value=0)
    dist_pct = dist.div(dist.sum(axis=1), axis=0)
    print(dist_pct.T.sort_values(by=1, ascending=False).head(10))
    
    print("\n--- Distribution of game_state ---")
    gs = df_shots.groupby(['is_blk', 'game_state']).size().unstack(fill_value=0)
    gs_pct = gs.div(gs.sum(axis=1), axis=0)
    print(gs_pct.T.sort_values(by=1, ascending=False).head(10))
    
    print("\n--- Distribution of shot_type ---")
    if 'shot_type' in df_shots.columns:
        st = df_shots.groupby(['is_blk', 'shot_type']).size().unstack(fill_value=0)
        st_pct = st.div(st.sum(axis=1), axis=0)
        print(st_pct.T.sort_values(by=1, ascending=False))

    # Check for feature uniqueness
    print("\n--- Check for Leakage: Features with high correlation to is_blk ---")
    # For categoricals, we look at the Block Rate per category
    # If any category has Block Rate > 0.8, it's a leaker.
    
    for col in ['last_event_type', 'game_state', 'shooter_role']:
        rate = df_shots.groupby(col)['is_blk'].mean()
        count = df_shots.groupby(col)['is_blk'].count()
        leakers = rate[(rate > 0.7) & (count > 50)]
        if not leakers.empty:
            print(f"POTENTIAL LEAKER in {col}:")
            print(pd.concat([rate, count], axis=1).loc[leakers.index])

if __name__ == "__main__":
    audit()

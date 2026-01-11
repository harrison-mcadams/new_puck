
import pandas as pd
import numpy as np
import sys, os, joblib

sys.path.append(os.getcwd())
from puck import fit_xgs, data_pipeline

def inspect_test():
    model_path = 'analysis/xgs/xg_model_nested.joblib'
    model = joblib.load(model_path)
    
    print("Loading data...")
    df = fit_xgs.load_data()
    df_p = data_pipeline.preprocess_features(df, is_training=False, apply_imputation=True)
    
    valid_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    df_shots = df_p[df_p['event'].isin(valid_events)].copy()
    
    print("Predicting Block Prob...")
    df_shots['p_block'] = model.predict_proba_layer(df_shots, layer='block')
    df_shots['is_blk'] = (df_shots['event'] == 'blocked-shot').astype(int)
    
    # Filter to Point Shots (dist 60-70)
    df_point = df_shots[(df_shots['distance'] >= 60) & (df_shots['distance'] <= 70) & (df_shots['shooter_role'] == 'F')].copy()
    
    print(f"\nAudit of Forwards at 60-70ft:")
    print(f"Total: {len(df_point)}")
    print(f"Actual Block Rate: {df_point['is_blk'].mean():.4f}")
    print(f"Pred Block Prob:   {df_point['p_block'].mean():.4f}")
    
    # Look at high p_block rows
    print("\n--- Top 20 Predictions at 60-70ft (Forwards) ---")
    cols = ['p_block', 'is_blk', 'distance', 'angle_deg', 'game_state', 'last_event_type', 'score_diff']
    print(df_point[cols].sort_values('p_block', ascending=False).head(20))
    
    # Look at LOW p_block rows at the same distance
    print("\n--- Bottom 20 Predictions at 60-70ft (Forwards) ---")
    print(df_point[cols].sort_values('p_block', ascending=True).head(20))

if __name__ == "__main__":
    inspect_test()


import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.getcwd())
from puck import fit_xgs, data_pipeline

def check_ratio():
    print("Loading all seasons...")
    df = fit_xgs.load_all_seasons_data()
    
    print("Preprocessing...")
    df_p = data_pipeline.preprocess_features(df, apply_imputation=True)
    
    valid_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    df_shots = df_p[df_p['event'].isin(valid_events)].copy()
    
    # 5x5 bins (same as dashboard?)
    # Dashboard uses contourf, but we check raw 5x5 counts
    df_shots['bx'] = (df_shots['x'] // 5) * 5
    df_shots['by'] = (df_shots['y'] // 5) * 5
    
    for role in ['F', 'D']:
        subset = df_shots[df_shots['shooter_role'] == role]
        counts = subset.groupby(['bx', 'by', 'event']).size().unstack(fill_value=0)
        
        counts['unblocked'] = counts.get('shot-on-goal', 0) + counts.get('missed-shot', 0) + counts.get('goal', 0)
        counts['blocks'] = counts.get('blocked-shot', 0)
        counts['total'] = counts['blocks'] + counts['unblocked']
        counts['ratio'] = counts['blocks'] / counts['total']
        
        print(f"\n--- Top 10 High-Ratio Bins (Role {role}) ---")
        print(counts.query('total > 100').sort_values('ratio', ascending=False).head(10))
        
        # Check specific point: High Slot (x=65, y=0)
        print(f"\nRatio at High Slot (x=65, y=0) for Role {role}:")
        try:
             print(counts.loc[(65, 0)][['blocks', 'unblocked', 'ratio']])
        except:
             print("Data point missing or sparse.")

if __name__ == "__main__":
    check_ratio()

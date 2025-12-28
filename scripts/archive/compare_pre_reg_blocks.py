import sys
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_nested_xgs, fit_xgs, config

def analyze_subset(df, name):
    if df.empty:
        print(f"\n--- {name}: NO DATA ---")
        return
        
    print(f"\n--- {name} Analysis ---")
    print(f"Sample size: {len(df)}")
    
    # MANUAL Preprocess
    valid_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    df_proc = df[df['event'].isin(valid_events)].copy()
    if df_proc.empty:
        print("No valid shot events.")
        return
        
    df_proc['is_blocked'] = (df_proc['event'] == 'blocked-shot').astype(int)
    y = df_proc['is_blocked']
    
    # Empty Net Stats
    if 'is_net_empty' in df_proc.columns:
        en = df_proc[df_proc['is_net_empty'] == True]
        print(f"Empty Net Shots: {len(en)} ({len(en)/len(df_proc):.1%})")
        if not en.empty:
            print(f"Empty Net Block Rate: {en['is_blocked'].mean():.1%}")
    else:
        print("is_net_empty COLUMN MISSING")

    print(f"Total Block Rate: {y.mean():.1%}")
    
    # Model Perf
    clf_mini = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    df_mini = df_proc.copy()
    # OHE game_state
    for gs in ['5v5', '5v4', '4v5', '4v4', '3v3']:
        df_mini[f'game_state_{gs}'] = (df_mini['game_state'] == gs).astype(int)
    
    feats = ['distance', 'angle_deg'] + [f'game_state_{gs}' for gs in ['5v5', '4v4', '3v3']]
    clf_mini.fit(df_mini[feats], df_mini['is_blocked'])
    
    p = clf_mini.predict_proba(df_mini[feats])[:, 1]
    auc = roc_auc_score(df_mini['is_blocked'], p)
    print(f"Mini-Model AUC: {auc:.4f}")
    
    # Correlations
    corr = df_mini[feats + ['is_blocked']].corr()['is_blocked'].sort_values(ascending=False)
    print("\nCorrelations with is_blocked:")
    print(corr)

def main():
    print("Loading all seasons data...")
    df_all = fit_xgs.load_all_seasons_data()
    # DO NOT FILTER EMPTY NET YET
    df_all['game_id_str'] = df_all['game_id'].astype(str)
    
    df_pre = df_all[df_all['game_id_str'].str[4:6] == '01'].copy()
    df_reg = df_all[df_all['game_id_str'].str[4:6] == '02'].copy()
    df_post = df_all[df_all['game_id_str'].str[4:6] == '03'].copy()
    
    analyze_subset(df_pre, "Pre-season (TT=01)")
    analyze_subset(df_reg, "Regular Season (TT=02)")
    analyze_subset(df_post, "Playoffs (TT=03)")

if __name__ == "__main__":
    main()

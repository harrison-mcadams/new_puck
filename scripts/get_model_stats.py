import sys
import os
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_xgs, fit_nested_xgs, impute

def get_stats():
    print("Gathering Model Statistics...")
    
    # Paths
    path_single = 'analysis/xgs/xg_model_single.joblib'
    path_nested = 'analysis/xgs/xg_model_nested.joblib'
    
    # Load Models
    clf_single = joblib.load(path_single)
    clf_nested = joblib.load(path_nested)
    
    # Load a chunk of data for evaluation (latest season)
    df = pd.read_csv('data/20252026/20252026_df.csv')
    
    # Filter 
    valid_events = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
    mask = df['event'].isin(valid_events)
    if 'is_net_empty' in df.columns:
        mask &= (df['is_net_empty'] != 1)
    mask &= ~df['game_state'].isin(['1v0', '0v1'])
    
    df_eval = df[mask].copy()
    y_true = (df_eval['event'] == 'goal').astype(int).values
    
    print(f"Evaluating on {len(df_eval)} samples from 2025-26 season.")
    
    # Single Model Substats
    feats_single = ['distance', 'angle_deg', 'game_state']
    # Load meta to get exact feature names
    import json
    with open(path_single + '.meta.json', 'r') as f:
        meta_s = json.load(f)
    final_feats_s = meta_s['final_features']
    cat_map_s = meta_s['categorical_levels_map']
    
    df_s_eval, _, _ = fit_xgs.clean_df_for_model(df_eval.copy(), feats_single, fixed_categorical_levels=cat_map_s)
    # y_true_s might be slightly different if clean_df dropped any NaNs (unlikely here but safe)
    y_true_s = df_s_eval['is_goal'].values
    p_single = clf_single.predict_proba(df_s_eval[final_feats_s])[:, 1]
    
    # Nested Model Substats
    df_n_eval = df_eval.loc[df_s_eval.index].copy()
    df_n_imp = impute.impute_blocked_shot_origins(df_n_eval, method='mean_6')
    p_nested = clf_nested.predict_proba(df_n_imp)[:, 1]
    
    def print_metrics(name, y, p):
        auc = roc_auc_score(y, p)
        brier = brier_score_loss(y, p)
        print(f"\nModel: {name}")
        print(f"  AUC:   {auc:.4f}")
        print(f"  Brier: {brier:.4f}")

    print_metrics("Single (Baseline)", y_true_s, p_single)
    print_metrics("Nested (Optimal)", y_true_s, p_nested)

if __name__ == "__main__":
    get_stats()

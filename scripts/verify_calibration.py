
import pandas as pd
import numpy as np
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import config, analyze, fit_xgs

def verify(model_name='nested_all'):
    season = 20252026
    
    model_path = os.path.join(config.ANALYSIS_DIR, 'xgs', f'xg_model_{model_name}.joblib')
    print(f"Loading model from {model_path}...")
    
    print(f"Loading data for {season}...")
    df = fit_xgs.load_all_seasons_data(config.DATA_DIR)
    
    # Filter 5v5 for comparison with daily output
    print("Filtering 5v5...")
    print(f"Unique game_state: {df['game_state'].unique()}")
    print(f"Unique is_net_empty: {df['is_net_empty'].unique()}")
    
    # Ensure correct types for filtering
    # Handle is_net_empty being potentially string or int or float
    if df['is_net_empty'].dtype == object:
        # try to cast? or just check values
         pass
         
    mask_5v5 = (df['game_state'] == '5v5') & (df['is_net_empty'] == 0)
    df_5v5 = df[mask_5v5].copy()
    
    print(f"Data shape: {df_5v5.shape}")
    
    # Predict
    print("Predicting xG...")
    # We use analyze._predict_xgs directly to match daily logic
    # It handles bio enrichment now (thanks to my fix)
    df_pred, clf, _ = analyze._predict_xgs(df_5v5, model_path=model_path, behavior='load')
    
    if 'xgs' not in df_pred.columns:
        print("Error: xgs column not created.")
        return

    # Filter Valid Attempts for Summation (same as analyze.xgs_map)
    attempt_types = ['goal', 'shot-on-goal', 'missed-shot', 'blocked-shot']
    mask_attempts = df_pred['event'].isin(attempt_types)
    
    total_xg = df_pred.loc[mask_attempts, 'xgs'].sum()
    total_goals = (df_pred['event'] == 'goal').sum()
    
    print(f"\n--- Results (5v5) ---")
    print(f"Total xG: {total_xg:.2f}")
    print(f"Total Goals: {total_goals}")
    print(f"Ratio: {total_xg / total_goals:.3f}")
    
    if hasattr(clf, 'classes_'):
        print(f"Model Classes: {clf.classes_}")
        
    print("\nShot Type Analysis:")
    if 'shot_type' in df_pred.columns:
        print(df_pred['shot_type'].value_counts().head(10))
    else:
        print("shot_type column missing!")
        
    # Compare row counts with Strict Training Filters
    print("\nApplying Strict Training Preprocessing (preprocess_features)...")
    from puck import fit_nested_xgs
    
    # We apply it to a fresh copy of the loaded data filtered to 5v5
    # Note: preprocess_features expects raw data structure
    df_strict = fit_nested_xgs.preprocess_features(df_5v5.copy())
    print(f"Rows after strict training filters: {len(df_strict)}")
    print(f"Rows in current verification set: {len(df_5v5)}")
    
    if len(df_strict) < len(df_5v5):
        print(f"DISCREPANCY: Training filters remove {len(df_5v5) - len(df_strict)} more rows!")
        # Re-calc xG on strict set
        df_pred_strict, _, _ = analyze._predict_xgs(df_strict, behavior='overwrite')
        
        attempt_types = ['goal', 'shot-on-goal', 'missed-shot', 'blocked-shot']
        mask_attempts_s = df_pred_strict['event'].isin(attempt_types)
        xg_strict = df_pred_strict.loc[mask_attempts_s, 'xgs'].sum()
        goals_strict = (df_pred_strict['event'] == 'goal').sum()
        print(f"Strict Filter Ratio: {xg_strict / goals_strict:.4f}")
    else:
        print("Filtering is consistent.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='nested_all')
    args = parser.parse_args()
    verify(args.model)
